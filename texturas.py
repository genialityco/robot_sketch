#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Zonas de textura del boceto (pelo, barba, rellenos): se dibujan a proposito en
vez de vectorizarlas tal cual.

El problema: una zona con muchas rayitas pegadas, binarizada, es una mancha con
huecos; su esqueleto es una malla de lineas conectadas (la "red"). Aca:

1. se detectan las zonas donde hay muchas lineas juntas: cuantas lineas cruzan
   una ventana de unos mm (largo del esqueleto / lado de la ventana), asi no
   depende del grosor del trazo (una linea sola, aunque gruesa, no es textura);
2. se borran del boceto antes de vectorizar (split_textures devuelve el boceto
   limpio);
3. se redibujan: el contorno de la zona y un rayado de trazos cortos que siguen
   la direccion del pelo (la del tensor de estructura del propio boceto),
   parejos y separados entre si (lineas de flujo, como Jobard-Lefer), mas
   densos donde el boceto es mas oscuro.

Todo en pixeles de la imagen de trabajo (draw_person.load_image, ~800 px),
con Y hacia abajo; quien lo usa invierte Y como el resto de los moves.
"""

import math

import cv2
import numpy as np

import pipeline as pl

# deteccion
WINDOW_MM = 3.0            # lado de la ventana donde se cuentan las lineas
MIN_LINES = 3.2            # lineas que cruzan la ventana para que sea textura
CLOSE_MM = 1.2             # se juntan zonas de textura separadas por menos que esto
MIN_AREA_MM2 = 30.0        # zonas mas chicas no se tocan (una ceja, un detalle)
KEEP_OUT_EDGE_MM = 0.0
# rayado
SEP_DARK_MM, SEP_LIGHT_MM = 1.1, 2.4     # separacion entre trazos: zona oscura / clara
LEN_MM = (2.5, 7.0)                       # largo de cada trazo (al azar entre estos)
MIN_LEN_MM = 1.2
MAX_TURN_DEG = 35                         # el trazo se corta si la direccion gira mas que esto de golpe
OUTLINE = False                           # dibujar el contorno de cada zona (queda como una burbuja: la
                                          # forma ya la dan los trazos del rayado)
OUTLINE_SMOOTH_PX = 2.0
SEED = 7


def _texture_regions(gray, ink, mm_per_px):
    skel = pl.dp.thin(ink).astype(np.float32)
    w = max(5, int(round(WINDOW_MM / mm_per_px)) | 1)
    lines = cv2.boxFilter(skel, -1, (w, w), normalize=False) / w   # lineas que cruzan la ventana
    region = (lines >= MIN_LINES).astype(np.uint8)
    k = max(1, int(round(CLOSE_MM / mm_per_px)))
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
    region = cv2.morphologyEx(region, cv2.MORPH_CLOSE, ker)
    region = cv2.dilate(region, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (w // 2 * 2 + 1, w // 2 * 2 + 1)))
    region &= cv2.dilate(ink.astype(np.uint8), ker)          # no mas alla de la tinta
    region = cv2.morphologyEx(region, cv2.MORPH_CLOSE, ker)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(region, connectivity=8)
    min_area = MIN_AREA_MM2 / (mm_per_px ** 2)
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
    region = keep[labels].astype(np.uint8)
    # rellenar huecos chicos dentro de cada zona
    return cv2.morphologyEx(region, cv2.MORPH_CLOSE, ker), lines


def _orientation(gray):
    """Direccion de las lineas en cada pixel (radianes, eje: 0 = horizontal):
    la del autovector chico del tensor de estructura."""
    g = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 1.2)
    gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    s = 4.0
    jxx, jyy, jxy = (cv2.GaussianBlur(a, (0, 0), s) for a in (gx * gx, gy * gy, gx * gy))
    return 0.5 * np.arctan2(2 * jxy, jxx - jyy) + np.pi / 2


def _hatch(region, darkness, theta, mm_per_px, rng):
    """Lineas de flujo parejas dentro de la zona, siguiendo theta."""
    h, w = region.shape
    sep_d, sep_l = SEP_DARK_MM / mm_per_px, SEP_LIGHT_MM / mm_per_px
    cell = sep_d / 2.0
    grid = {}   # celda -> puntos ya dibujados (para la separacion)

    def near(x, y, dmin):
        cx, cy = int(x // cell), int(y // cell)
        r = int(math.ceil(dmin / cell))
        for gx in range(cx - r, cx + r + 1):
            for gy in range(cy - r, cy + r + 1):
                for qx, qy in grid.get((gx, gy), ()):
                    if (qx - x) ** 2 + (qy - y) ** 2 < dmin * dmin:
                        return True
        return False

    def sep_at(x, y):
        return sep_l + (sep_d - sep_l) * float(np.clip(darkness[int(y), int(x)], 0, 1))

    def trace(x, y, sign, max_len):
        pts, (px, py), prev, length = [], (x, y), None, 0.0
        step = 0.7
        while length < max_len:
            a = theta[int(py), int(px)]
            dx, dy = math.cos(a), math.sin(a)
            if prev is None:
                dx, dy = dx * sign, dy * sign
            elif dx * prev[0] + dy * prev[1] < 0:
                dx, dy = -dx, -dy
            if prev is not None and dx * prev[0] + dy * prev[1] < math.cos(math.radians(MAX_TURN_DEG)):
                break
            nx, ny = px + dx * step, py + dy * step
            if not (0 <= nx < w - 1 and 0 <= ny < h - 1) or not region[int(ny), int(nx)]:
                break
            if near(nx, ny, 0.5 * sep_at(nx, ny)):
                break
            pts.append((nx, ny))
            prev, (px, py), length = (dx, dy), (nx, ny), length + step
        return pts

    ys, xs = np.nonzero(region)
    if not len(xs):
        return []
    order = rng.permutation(len(xs))[: max(1, int(len(xs) / (cell * cell)) * 2)]
    strokes = []
    for i in order:
        x, y = float(xs[i]) + rng.random(), float(ys[i]) + rng.random()
        if near(x, y, sep_at(x, y)):
            continue
        max_len = rng.uniform(*LEN_MM) / mm_per_px / 2
        back, fwd = trace(x, y, -1, max_len), trace(x, y, 1, max_len)
        line = back[::-1] + [(x, y)] + fwd
        if len(line) < 2 or len(line) * 0.7 * mm_per_px < MIN_LEN_MM:
            continue
        strokes.append(np.asarray(line))
        for qx, qy in line[::2]:
            grid.setdefault((int(qx // cell), int(qy // cell)), []).append((qx, qy))
    return strokes


def _outlines(region, mm_per_px):
    import sketch_svg
    cnts, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = []
    for c in cnts:
        p = c.reshape(-1, 2).astype(np.float64)
        if len(p) < 8:
            continue
        p = np.vstack([p, p[:1]])
        p = sketch_svg.smooth_path(p, OUTLINE_SMOOTH_PX, 0.5)
        if len(p) >= 2:
            out.append(p)
    return out


def split_textures(sketch_path, mm_per_px, clean_path, exclude=None):
    """Detecta las zonas de textura del boceto, las borra (clean_path: el boceto
    sin ellas, a su resolucion original) y devuelve (trazos que las reemplazan en
    pixeles de trabajo con Y hacia abajo, mascara de las zonas, info).
    exclude: poligonos (pixeles de trabajo, Y hacia abajo) que nunca son
    textura: ojos, cejas, nariz y boca (sus rayitas son el rasgo)."""
    img = pl.dp.load_image(sketch_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ink = pl.dp.sketch_to_mask(img) > 0
    region, lines = _texture_regions(gray, ink, mm_per_px)
    if exclude:
        cv2.fillPoly(region, [np.round(z).astype(np.int32) for z in exclude], 0)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(region, connectivity=8)
        keep = np.zeros(n, bool)
        keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= MIN_AREA_MM2 / (mm_per_px ** 2)
        region = keep[labels].astype(np.uint8)
    full = cv2.imread(sketch_path, cv2.IMREAD_COLOR)
    big = cv2.resize(region, (full.shape[1], full.shape[0]), interpolation=cv2.INTER_NEAREST)
    full[big > 0] = 255
    cv2.imwrite(clean_path, full)
    if not region.any():
        return [], region, dict(zones=0, strokes=0, area_mm2=0.0)
    # oscuridad local (para la densidad del rayado): cuanta tinta hay alrededor
    k = max(3, int(round(WINDOW_MM / mm_per_px)) | 1)
    darkness = cv2.boxFilter(ink.astype(np.float32), -1, (k, k)) / 0.45
    theta = _orientation(gray)
    rng = np.random.default_rng(SEED)
    strokes = _hatch(region, darkness, theta, mm_per_px, rng)
    if OUTLINE:
        strokes = _outlines(region, mm_per_px) + strokes
    n = cv2.connectedComponents(region)[0] - 1
    return strokes, region, dict(zones=n, strokes=len(strokes), area_mm2=round(float(region.sum()) * mm_per_px ** 2, 1))


# Rasgos que nunca son textura (landmarks de MediaPipe): ojos con las cejas, nariz y boca
_EXCLUDE_GROUPS = [
    (33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7, 70, 63, 105, 66, 107, 55, 65, 52, 53, 46),
    (263, 466, 388, 387, 386, 385, 384, 398, 362, 382, 381, 380, 374, 373, 390, 249, 300, 293, 334, 296, 336, 285, 295, 282,
     283, 276),
    (48, 64, 98, 97, 2, 326, 327, 294, 278, 4, 1, 5),
    (61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146),
]
EXCLUDE_GROW = 1.3


def feature_zones(points_up, img_height):
    """Poligonos (pixeles de trabajo, Y hacia abajo) de ojos+cejas, nariz y
    boca, agrandados, desde los landmarks con Y hacia arriba (pl.locate_face)."""
    if not points_up or len(points_up) < 468:
        return []
    zones = []
    for idx in _EXCLUDE_GROUPS:
        p = np.array([(points_up[i][0], img_height - points_up[i][1]) for i in idx], np.float32)
        hull = cv2.convexHull(p).reshape(-1, 2)
        c = hull.mean(axis=0)
        zones.append(c + (hull - c) * EXCLUDE_GROW)
    return zones
