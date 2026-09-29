#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Retrato por puntos (puntillismo) para el myCobot 280.

Un punto es un toque vertical: la punta nunca se mueve de lado apoyada, asi
que el juego de J1 y la friccion (lo que arruinaba las lineas) no deforman
nada. Este modulo:

  1. convierte la imagen (foto o boceto) en un mapa de densidad: donde la
     imagen es oscura van mas puntos. Niveles + gamma para que la piel clara
     quede casi limpia, bordes finos sumados para que ojos, cejas, gafas y
     labios no se pierdan, y fondo blanco = sin puntos;
  2. reparte los puntos segun esa densidad, con uno de tres metodos:
       - 'voronoi': punteado de Voronoi ponderado (Secord 2002). Puntos
         organicos y parejos, el aspecto mas "de artista". Es el default.
       - 'atkinson' / 'floyd': difusion de error (dithering) sobre una grilla
         con celda = separacion minima. Aspecto de trama o grabado, mas
         geometrico. Atkinson descarta parte del error: luces mas limpias y
         mas contraste que Floyd-Steinberg.
     En los tres, ningun par de puntos queda mas cerca que MIN_SPACING_MM
     (los puntos no se funden en rayas);
  3. ordena los puntos por bandas (distancia a la base) y, dentro de cada
     banda, con la base girando siempre en el mismo sentido. A todos los
     puntos se llega igual, el juego de J1 los corre a todos lo mismo y la
     imagen queda nitida (el mismo principio que j1_unidireccional);
  4. lo convierte en moves para lib.draw_moves (plan_to_moves): cada punto
     es bajar y subir en el mismo lugar, con un levantado corto entre puntos
     (lib.DOTS_PEN_LIFT_MM) y la bajada/subida de siempre.

En la app: prueba "Rostro real con puntos" de la calibracion (vista previa y
dibujo). El plan se dibuja con lib.draw_moves (plan_to_moves), asi usa lo mismo
que el resto: velocidades, bajada/subida, compensaciones y la tableta.

Uso por linea de comandos (con el server apagado, el puerto serie tiene que estar libre):
    python puntillismo.py foto.jpg --minutos 6 --preview preview.png
    python puntillismo.py foto.jpg --minutos 6 --metodo atkinson --preview p.png --densidad d.png
    python puntillismo.py foto.jpg --minutos 6 --dibujar

La mejor entrada es una version en escala de grises con FONDO BLANCO (ver
GEMINI_PROMPT_TONOS): el fondo blanco se descarta solo (WHITE_CUT) y no gasta
puntos. Tambien acepta el boceto de lineas de Gemini: los puntos siguen las
lineas, parejos.
"""

import argparse
import math
import time

import numpy as np

import pipeline as pl

lib = pl.lib

# Sugerencia de prompt para pedirle a Gemini la imagen de entrada (en vez del
# boceto de lineas): tonos para el puntillismo, sin fondo que gaste puntos.
GEMINI_PROMPT_TONOS = (
    'Convierte esta foto en un retrato en escala de grises de alto contraste, estilo ilustracion '
    'de grabado: solo cabeza y hombros, centrado, con el fondo completamente blanco. Conserva '
    'fielmente los rasgos, las gafas y el peinado de la persona. Sombras definidas, piel clara y '
    'limpia, sin texto ni marcos.'
)

# --- imagen -> densidad ---------------------------------------------------------
WORK_PX = 500           # lado mayor al que se lleva la imagen para calcular
DARKS_Q = 0.02          # cuantil (del primer plano) que pasa a negro total
LIGHTS_Q = 0.80         # cuantil que pasa a blanco: el 20% mas claro no lleva puntos
GAMMA = 1.4             # >1 aclara los medios tonos: menos puntos en la piel
TONE_MAX = 0.35         # techo del tono: una zona oscura grande (ropa, pelo negro) no se
                        # come todo el presupuesto de puntos; lo que sobra va a los rasgos
EDGE_WEIGHT = 0.9       # cuanto suman los bordes finos (ojos, cejas, gafas, labios)
DENSITY_FLOOR = 0.12    # por debajo de esto, sin puntos: evita puntos sueltos en zonas casi blancas
FACE_FOCUS = 0.1        # peso de la densidad lejos de la cara (1 = sin enfoque): con pocos
                        # cientos de puntos, cada punto en la ropa es uno menos en los ojos
EDGE_SIGMA_FRAC = 1 / 250   # escala de esos bordes, como fraccion del lado mayor
WHITE_CUT = 0.94        # en la imagen original, mas claro que esto = fondo, sin puntos
CLAHE_CLIP = 2.0        # contraste local (0 = desactivado)

# --- densidad -> puntos ---------------------------------------------------------
MIN_SPACING_MM = 1.7    # distancia minima entre centros de puntos (bajar a 1.5 si los puntos
                        # salen redondos; subir si se tocan)
VORONOI_ITERATIONS = 40
DOT_DIAMETER_MM = 0.9   # solo para el preview: diametro de un punto del marcador

# --- orden y ejecucion ----------------------------------------------------------
DIRECTION = 1           # sentido de giro de la base dentro de cada banda (+1 antihorario desde arriba)
BAND_MM = 6.0           # alto de cada banda, en distancia a la base
APPROACH_MM = 9.0       # llegada por el aire al inicio de cada banda: mas que el juego de J1
# (altura de los saltos, velocidades y bajada/subida: las de lib.draw_moves;
# el levantado entre puntos es lib.DOTS_PEN_LIFT_MM)

# Estimacion de tiempo, con el ciclo de draw_moves (viaje + bajada en dos pasos
# + subida, cada uno esperando al brazo). El --dibujar de la linea de comandos
# imprime los segundos reales por punto: ajustar con ese valor.
SECONDS_PER_DOT = 2.0
SECONDS_PER_BAND = 2.5


# -------------------------------------------------------------------------------
# 1. Imagen -> densidad
# -------------------------------------------------------------------------------
def _face_box(gray_u8):
    """(x, y, ancho, alto) del ovalo de la cara con MediaPipe (ya lo usa el
    pipeline; el detector Haar de OpenCV no viene en opencv 5), o None."""
    import cv2
    try:
        pts = pl.dp.face_landmark_points(cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2BGR))
        if not pts:
            return None
        from mediapipe.tasks.python import vision
        idx = {i for c in vision.FaceLandmarksConnections.FACE_LANDMARKS_FACE_OVAL for i in (c.start, c.end)}
    except Exception:
        return None
    xs = [pts[i][0] for i in idx]
    ys = [pts[i][1] for i in idx]
    return min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)


# Recorte a cabeza y hombros antes de calcular (margen alrededor del ovalo de
# la cara, en fracciones de su ancho/alto): en una foto comun la cara ocupa
# poco del cuadro y en la hoja quedaria chica, con pocos puntos en los rasgos.
CROP_SIDE, CROP_TOP, CROP_BOTTOM = 0.4, 0.55, 0.35


def _crop_head(gray_u8, box):
    x, y, w, h = box
    H, W = gray_u8.shape
    x0, x1 = int(max(0, x - CROP_SIDE * w)), int(min(W, x + w + CROP_SIDE * w))
    y0, y1 = int(max(0, y - CROP_TOP * h)), int(min(H, y + h + CROP_BOTTOM * h))
    return gray_u8[y0:y1, x0:x1], (x - x0, y - y0, w, h)


def _face_weight(shape, box, focus):
    """Peso por zona: 1 en la cara (y el pelo y menton cercanos), bajando
    hasta `focus` lejos de ella, para que ropa y fondo no se lleven los
    puntos. box: ovalo de la cara (ver _face_box)."""
    x, y, w, h = box
    cx, cy, rx, ry = x + w / 2, y + h * 0.45, w * 0.68, h * 1.0   # angosta: los bordes del fondo a los costados no
    Y, X = np.mgrid[0:shape[0], 0:shape[1]]
    d = np.sqrt(((X - cx) / rx) ** 2 + ((Y - cy) / ry) ** 2)
    return focus + (1 - focus) * np.clip((1.3 - d) / 0.4, 0, 1)   # pleno hasta d=0.9, minimo desde 1.3


def load_density(path, work_px=WORK_PX, darks_q=DARKS_Q, lights_q=LIGHTS_Q, gamma=GAMMA,
                 tone_max=TONE_MAX, edge_weight=EDGE_WEIGHT, density_floor=DENSITY_FLOOR,
                 face_focus=FACE_FOCUS, white_cut=WHITE_CUT, clahe_clip=CLAHE_CLIP, rotation_deg=0,
                 crop_head=True):
    """Imagen -> (array con filas de arriba a abajo y valores 0..1: cuanta tinta
    va en cada lugar; si se encontro la cara). Con crop_head se recorta a
    cabeza y hombros. rotation_deg gira el resultado igual que
    pipeline.rotate_moves (antihorario, multiplo de 90)."""
    import cv2
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError('no se pudo leer la imagen: %s' % path)
    box = _face_box(img)
    if box is not None and crop_head:
        img, box = _crop_head(img, box)
    h, w = img.shape
    f = work_px / max(h, w)
    img = cv2.resize(img, (max(1, round(w * f)), max(1, round(h * f))),
                     interpolation=cv2.INTER_AREA if f < 1 else cv2.INTER_CUBIC)
    if box is not None:
        box = tuple(v * f for v in box)

    background = img / 255.0 > white_cut
    weight = _face_weight(img.shape, box, face_focus) if (face_focus < 1 and box is not None) else None
    if clahe_clip > 0:
        img = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8)).apply(img)
    g = img.astype(np.float64) / 255.0
    fg = g[~background] if (~background).sum() > 100 else g.ravel()

    # niveles + gamma: el cuantil lights_q pasa a blanco (sin puntos), darks_q a
    # negro; con techo tone_max para que las zonas oscuras grandes no dominen
    lo, hi = np.quantile(fg, darks_q), np.quantile(fg, lights_q)
    tone = tone_max * np.clip((hi - g) / max(hi - lo, 1e-6), 0, 1) ** gamma

    # bordes finos: diferencia de gaussianas, solo donde el centro es mas oscuro
    # que su entorno (ojos, cejas, marco de gafas, labios, comisuras)
    s = max(0.8, max(g.shape) * EDGE_SIGMA_FRAC)
    edge = np.clip(cv2.GaussianBlur(g, (0, 0), 2.5 * s) - cv2.GaussianBlur(g, (0, 0), s), 0, None)
    ref = edge[~background] if (~background).sum() > 100 else edge.ravel()
    edge = np.clip(edge / max(np.quantile(ref, 0.99), 1e-6), 0, 1)

    density = np.clip(tone + edge_weight * edge, 0, 1)
    if weight is not None:
        density *= weight
    else:
        print('puntillismo: no se encontro la cara, sin enfoque', flush=True)
    density[(density < density_floor) | background] = 0

    turns = (int(rotation_deg) // 90) % 4
    return (np.ascontiguousarray(np.rot90(density, turns)) if turns else density), box is not None


# -------------------------------------------------------------------------------
# 2. Densidad -> puntos (en pixeles de la imagen, origen arriba a la izquierda)
# -------------------------------------------------------------------------------
def _voronoi_stipple(density, n, iterations=VORONOI_ITERATIONS, seed=1):
    """Punteado de Voronoi ponderado: cada punto se mueve al centro de masa
    (pesado por la densidad) de la zona que le queda mas cerca, hasta que
    los puntos quedan repartidos parejo segun la densidad."""
    from scipy.spatial import cKDTree
    rng = np.random.default_rng(seed)
    ys, xs = np.nonzero(density > 1e-3)
    if len(xs) == 0 or n <= 0:
        return np.zeros((0, 2))
    pix = np.column_stack([xs + 0.5, ys + 0.5])
    wts = density[ys, xs]
    p = wts / wts.sum()
    pts = pix[rng.choice(len(pix), size=n, p=p)] + rng.uniform(-0.5, 0.5, (n, 2))
    for _ in range(iterations):
        _, owner = cKDTree(pts).query(pix)
        wsum = np.bincount(owner, weights=wts, minlength=n)
        ok = wsum > 0
        for axis in (0, 1):
            acc = np.bincount(owner, weights=wts * pix[:, axis], minlength=n)
            pts[ok, axis] = acc[ok] / wsum[ok]
        if (~ok).any():   # puntos que se quedaron sin zona: reubicar segun la densidad
            pts[~ok] = pix[rng.choice(len(pix), size=int((~ok).sum()), p=p)]
    return pts


def _enforce_spacing(pts, min_px):
    """Saca puntos hasta que ningun par quede a menos de min_px."""
    from scipy.spatial import cKDTree
    if len(pts) == 0:
        return pts
    tree = cKDTree(pts)
    keep = np.ones(len(pts), bool)
    for i in range(len(pts)):
        if keep[i]:
            for j in tree.query_ball_point(pts[i], min_px):
                if j > i:
                    keep[j] = False
    return pts[keep]


_KERNELS = {  # (dx, dy, peso), divisor
    'floyd': ([(1, 0, 7), (-1, 1, 3), (0, 1, 5), (1, 1, 1)], 16),
    'atkinson': ([(1, 0, 1), (2, 0, 1), (-1, 1, 1), (0, 1, 1), (1, 1, 1), (0, 2, 1)], 8),
}


def _error_diffusion(grid, method):
    kernel, div = _KERNELS[method]
    g = grid.copy()
    gh, gw = g.shape
    out = np.zeros(g.shape, bool)
    for y in range(gh):
        sgn = 1 if y % 2 == 0 else -1          # recorrido serpentina: sin vetas diagonales
        for x in (range(gw) if sgn > 0 else range(gw - 1, -1, -1)):
            new = 1.0 if g[y, x] >= 0.5 else 0.0
            out[y, x] = new > 0
            err = g[y, x] - new
            for dx, dy, wgt in kernel:
                xx, yy = x + dx * sgn, y + dy
                if 0 <= xx < gw and yy < gh:
                    g[yy, xx] += err * wgt / div
    return out


def _dither_stipple(density, cell_px, method, n_target):
    """Dithering sobre una grilla de celda = separacion minima (un punto como
    mucho por celda). Se ajusta la exposicion para acercarse a n_target."""
    import cv2
    h, w = density.shape
    gw, gh = max(1, int(w / cell_px)), max(1, int(h / cell_px))
    grid = cv2.resize(density, (gw, gh), interpolation=cv2.INTER_AREA)
    lo, hi = 0.05, 50.0
    best = None
    for _ in range(20):
        gain = math.sqrt(lo * hi)
        out = _error_diffusion(np.clip(grid * gain, 0, 1), method)
        if best is None or abs(out.sum() - n_target) < abs(best.sum() - n_target):
            best = out
        if out.sum() > n_target:
            hi = gain
        else:
            lo = gain
    ys, xs = np.nonzero(best)
    return np.column_stack([(xs + 0.5) * w / gw, (ys + 0.5) * h / gh])


def plan_from_image(path, n_dots=None, minutes=None, method='voronoi', max_size_mm=None,
                    min_spacing_mm=MIN_SPACING_MM, rotation_deg=0, **density_kw):
    """
    Imagen -> plan de puntos: dict con 'dots' [(x, y), ...] en coordenadas del
    dibujo (y hacia arriba, como strokes_to_moves) y 'bounds' fijos para el
    mapeo a la hoja (preserve_aspect=True).

    n_dots: cuantos puntos pedir; si no se da, se calcula para que el dibujo
    tarde `minutes` (default 6) segun SECONDS_PER_DOT / SECONDS_PER_BAND.
    max_size_mm: lado mayor del retrato en la hoja (None = llenar el area).
    Ningun par de puntos queda a menos de min_spacing_mm.
    """
    density, face = load_density(path, rotation_deg=rotation_deg, **density_kw)
    H, W = density.shape
    quad_w, quad_h = lib._quad_edge_lengths()
    scale = min(quad_w / W, quad_h / H)                  # mm por pixel
    if max_size_mm:
        scale = min(scale, max_size_mm / max(W, H))
    half_w, half_h = quad_w / scale / 2, quad_h / scale / 2
    bounds = (W / 2 - half_w, W / 2 + half_w, H / 2 - half_h, H / 2 + half_h)

    if n_dots is None:
        bands = H * scale / BAND_MM
        n_dots = max(50, int(((minutes or 6) * 60 - bands * SECONDS_PER_BAND) / SECONDS_PER_DOT))
    min_px = min_spacing_mm / scale

    if method == 'voronoi':
        pts = _enforce_spacing(_voronoi_stipple(density, n_dots), min_px)
    elif method in _KERNELS:
        pts = _dither_stipple(density, min_px, method, n_dots)
    else:
        raise ValueError('metodo desconocido: %s (voronoi, atkinson o floyd)' % method)

    dots = [(float(x), float(H - y)) for x, y in pts]
    print('puntillismo: %d puntos pedidos, %d quedaron (%s, separacion minima %.1f mm, %.2f mm/px)'
          % (n_dots, len(dots), method, min_spacing_mm, scale), flush=True)
    return dict(dots=dots, bounds=bounds, method=method, mm_per_px=scale, density=density, face=face)


# -------------------------------------------------------------------------------
# 3. Orden por bandas, con la base siempre girando en el mismo sentido
# -------------------------------------------------------------------------------
def order_dots(dots, bounds):
    """-> (orden [(indice, empieza_banda), ...], coords del robot por indice).
    Usa la calibracion activa en el momento de llamarla."""
    moves = [(x, y, True) for x, y in dots]
    coords = [c for *_, c in lib.iter_target_coords(moves, preserve_aspect=True, bounds=bounds)]
    if not coords:
        return [], coords
    r = np.array([math.hypot(c[0], c[1]) for c in coords])
    th = np.array([math.atan2(c[1], c[0]) for c in coords])
    mean = math.atan2(np.sin(th).mean(), np.cos(th).mean())
    th = (th - mean + math.pi) % (2 * math.pi) - math.pi     # sin saltos en +-180 grados
    band = np.floor((r.max() - r) / BAND_MM).astype(int)     # banda 0 = la mas lejana
    order = sorted(range(len(coords)), key=lambda i: (band[i], DIRECTION * th[i]))
    out, prev = [], None
    for i in order:
        out.append((i, band[i] != prev))
        prev = band[i]
    return out, coords


def estimate_minutes(order):
    bands = sum(1 for _, new in order if new)
    return (len(order) * SECONDS_PER_DOT + bands * SECONDS_PER_BAND) / 60.0


# -------------------------------------------------------------------------------
# 4. Preview (sin robot)
# -------------------------------------------------------------------------------
def render_preview(plan, out_path, density_path=None):
    """PNG de como quedan los puntos sobre la hoja (arriba = esquinas 'arriba'),
    con el diametro real del marcador. Devuelve estadisticas."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import EllipseCollection
    from matplotlib.patches import Rectangle

    quad_w, quad_h = lib._quad_edge_lengths()
    moves = [(x, y, True) for x, y in plan['dots']]
    xy = np.array([(u * quad_w, v * quad_h) for *_, u, v in
                   lib.iter_sheet_uv(moves, preserve_aspect=True, bounds=plan['bounds'])]).reshape(-1, 2)
    order, _ = order_dots(plan['dots'], plan['bounds'])
    minutes = estimate_minutes(order)
    bands = sum(1 for _, new in order if new)

    fig, ax = plt.subplots(figsize=(6, 6.5))
    ax.add_patch(Rectangle((0, 0), quad_w, quad_h, fill=False, edgecolor='tab:blue',
                           linestyle='--', linewidth=0.8))
    ax.add_collection(EllipseCollection(DOT_DIAMETER_MM, DOT_DIAMETER_MM, 0, units='xy',
                                        offsets=xy, offset_transform=ax.transData, facecolors='black'))
    ax.set_xlim(-3, quad_w + 3)
    ax.set_ylim(-3, quad_h + 3)
    ax.set_aspect('equal')
    ax.set_xlabel('ancho de la hoja (mm)')
    ax.set_title('Puntillismo (%s): %d puntos, %d bandas, ~%.1f min estimado'
                 % (plan['method'], len(xy), bands, minutes), fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)

    if density_path is not None:
        import cv2
        cv2.imwrite(density_path, (255 * (1 - plan['density'])).astype(np.uint8))
    return dict(points=len(xy), bands=bands, estimated_minutes=round(minutes, 1),
                area_mm=[round(quad_w, 1), round(quad_h, 1)])


# -------------------------------------------------------------------------------
# 5. Dibujar
# -------------------------------------------------------------------------------
def _approach_point(x, y, bounds):
    """Punto del dibujo desde el cual, yendo en linea recta hasta (x, y), la
    base gira APPROACH_MM de arco en el sentido DIRECTION (se llega por el
    aire al inicio de cada banda con J1 ya apoyado del lado correcto)."""
    def theta(px, py):
        for *_, c in lib.iter_target_coords([(px, py, True)], preserve_aspect=True, bounds=bounds):
            return math.atan2(c[1], c[0]), math.hypot(c[0], c[1])
    eps = (bounds[1] - bounds[0]) * 1e-3
    t0, r = theta(x, y)
    gx = (theta(x + eps, y)[0] - t0) / eps
    gy = (theta(x, y + eps)[0] - t0) / eps
    g = math.hypot(gx, gy)
    if g < 1e-12:
        return x, y
    step = APPROACH_MM / (g * r)
    return x - DIRECTION * step * gx / g, y - DIRECTION * step * gy / g


def plan_to_moves(plan):
    """
    Plan -> (moves, bounds) para lib.draw_moves / pipeline.draw con
    preserve_aspect=True: los puntos en el orden por bandas (order_dots), cada
    uno como (x, y, arriba), (x, y, abajo) -- un punto: la pluma baja y sube en
    el mismo lugar --, y al empezar cada banda una llegada por el aire desde
    APPROACH_MM antes en el giro de la base. Dibujarlo con
    unidirectional=False (el orden ya es el de la base en un solo sentido) y
    un levantado corto (lib.DOTS_PEN_LIFT_MM).
    """
    order, _coords = order_dots(plan['dots'], plan['bounds'])
    moves = []
    for i, new_band in order:
        x, y = plan['dots'][i]
        if new_band:
            ax, ay = _approach_point(x, y, plan['bounds'])
            moves.append((ax, ay, False))
        moves += [(x, y, False), (x, y, True)]
    return moves, plan['bounds']


# -------------------------------------------------------------------------------
# Linea de comandos
# -------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description='Retrato por puntos para el myCobot 280')
    ap.add_argument('imagen')
    ap.add_argument('--metodo', default='voronoi', choices=['voronoi', 'atkinson', 'floyd'])
    ap.add_argument('--puntos', type=int, help='cantidad de puntos (si no, se calcula con --minutos)')
    ap.add_argument('--minutos', type=float, default=6.0, help='tiempo objetivo del dibujo')
    ap.add_argument('--tamano-mm', type=float, help='lado mayor del retrato en la hoja')
    ap.add_argument('--separacion-mm', type=float, default=MIN_SPACING_MM)
    ap.add_argument('--gamma', type=float, default=GAMMA)
    ap.add_argument('--bordes', type=float, default=EDGE_WEIGHT, help='peso de los bordes finos')
    ap.add_argument('--luces', type=float, default=LIGHTS_Q, help='cuantil que queda sin puntos')
    ap.add_argument('--preview', help='PNG con el preview en la hoja')
    ap.add_argument('--densidad', help='PNG con el mapa de densidad (para ajustar parametros)')
    ap.add_argument('--dibujar', action='store_true', help='dibujar en el robot')
    args = ap.parse_args()

    import calibration as cal
    cal.bootstrap()
    plan = plan_from_image(args.imagen, n_dots=args.puntos, minutes=args.minutos, method=args.metodo,
                           max_size_mm=args.tamano_mm, min_spacing_mm=args.separacion_mm,
                           rotation_deg=cal.active_rotation_deg(), gamma=args.gamma,
                           edge_weight=args.bordes, lights_q=args.luces)
    if args.preview:
        print(render_preview(plan, args.preview, args.densidad))
    if args.dibujar:
        import robot
        moves, bounds = plan_to_moves(plan)
        with robot.use('puntillismo') as mc:
            t0 = time.time()
            pl.draw(mc, moves, preserve_aspect=True, bounds=bounds, unidirectional=False,
                    pen_lift_mm=lib.DOTS_PEN_LIFT_MM)
            print('%.2f s por punto (ajustar SECONDS_PER_DOT)' % ((time.time() - t0) / max(1, len(plan['dots']))))


if __name__ == '__main__':
    main()
