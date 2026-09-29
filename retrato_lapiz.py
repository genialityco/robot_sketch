#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Dibujo a lapiz de lineas para la tableta.

En la tableta (pagina /tablet) el brazo no deja tinta: la pagina dibuja, como
trazos de grafito, exactamente el recorrido del brazo, a medida que avanza.
Para que se vea que lo esta dibujando (no que se descubre una imagen), lo que
se ve y lo que se descarga es ese mismo recorrido, sin sombras ni fundidos:

  - la IMAGEN de partida: un dibujo de lineas a lapiz, sin sombras y mas
    detallado que el de "trazo simple" (estilo 'lapiz' en gemini_sketch.STYLES);
    si Gemini falla, opencv_pencil saca las lineas de la foto (tecnica "dodge").
  - el RECORRIDO: las lineas de esa imagen con el mismo vectorizador que
    "trazo simple" (pipeline.line_art_svg_moves: lineas largas y suaves), con
    un nivel de detalle propio (LINE_ART_LEVELS['lapiz']: trazos mas cortos y
    menos suavizado), y un tope de tiempo (limit_to_seconds): si no entra en
    TARGET_SECONDS en modo tableta, se dejan afuera los trazos menos importantes.
  - DOS VERSIONES (ver robot_version y display_sync): la tableta muestra el
    dibujo detallado, sin tope de tiempo, y el brazo recorre una version simple
    y mas rapida; la tableta va dibujando lo detallado por donde pasa la punta.
"""

import math

import cv2
import numpy as np

import pipeline as pl
import puntillismo as pt

lib = pl.lib

TARGET_SECONDS = 75.0     # duracion maxima del recorrido (en modo tableta)


# -------------------------------------------------------------------------------
# Imagen
# -------------------------------------------------------------------------------
PENCIL_STRENGTH = 2.2    # cuanto se oscurecen las lineas de grafito


def opencv_pencil(photo_path, out_path, side_px=900):
    """Dibujo de lineas a lapiz con OpenCV (respaldo de Gemini): recorte a
    cabeza y hombros, lineas de grafito (dodge), sin sombras, y lo lejos de la
    cara se lleva a blanco."""
    gray = cv2.imread(photo_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError('no se pudo leer la foto: %s' % photo_path)
    box = pt._face_box(gray)
    if box is not None:
        gray, box = pt._crop_head(gray, box)
    f = side_px / max(gray.shape)
    gray = cv2.resize(gray, None, fx=f, fy=f, interpolation=cv2.INTER_AREA if f < 1 else cv2.INTER_CUBIC)
    blur = cv2.GaussianBlur(255 - gray, (0, 0), 6)
    pencil = cv2.divide(gray, 255 - blur, scale=256).astype(np.float64)
    pencil = 255 - np.clip((255 - pencil) * PENCIL_STRENGTH, 0, 255)   # grafito mas marcado
    out = pencil
    if box is not None:
        weight = pt._face_weight(gray.shape, tuple(v * f for v in box), 0.0)
        out = 255 - (255 - out) * weight
    cv2.imwrite(out_path, np.clip(out, 0, 255).astype(np.uint8))
    return out_path


# -------------------------------------------------------------------------------
# Rasgos (ojos, cejas, nariz, boca) con mas detalle
# -------------------------------------------------------------------------------
# El vectorizador trabaja con toda la imagen a ~800 px: un ojo ocupa ~60 px, y
# el adelgazado y el suavizado funden el iris con el parpado, y los trazos de
# menos de 2 mm (fosas nasales, comisuras) se descartan. La zona de los rasgos
# (ubicada con MediaPipe) se vuelve a vectorizar desde la imagen original,
# FEATURE_UPSCALE veces mas grande, con menos suavizado y trazos mas cortos.
FEATURE_UPSCALE = 3          # respecto de la imagen de trabajo (800 px)
FEATURE_MARGIN = 0.12        # margen alrededor de los rasgos (fraccion de su tamano)
FEATURE_SMOOTH_PX = 1.4      # suavizado, en pixeles de la zona ampliada
FEATURE_MIN_MM = 1.0         # trazos mas cortos no se dibujan (el resto del dibujo: 2 mm)
FEATURE_STEP_MM = 1.0        # un punto cada tanto a lo largo de los rasgos (curvas finas)


def _cut_box(strokes, box):
    """Saca de los trazos lo que cae dentro del rectangulo (x0, x1, y0, y1)."""
    x0, x1, y0, y1 = box
    out = []
    for st in strokes:
        cur = []
        for x, y in st:
            if x0 <= x <= x1 and y0 <= y <= y1:
                if len(cur) >= 2:
                    out.append(cur)
                cur = []
            else:
                cur.append((x, y))
        if len(cur) >= 2:
            out.append(cur)
    return out


def _feature_strokes(source, box_up, work_h, work_scale):
    """Trazos de la zona de rasgos, vectorizada a mas resolucion. box_up en
    pixeles de la imagen de trabajo con y hacia arriba; devuelve trazos en esas
    mismas coordenadas."""
    import sketch_svg
    x0, x1, y0, y1 = box_up
    top, bottom = work_h - y1, work_h - y0                   # a filas de la imagen
    full = cv2.imread(source, cv2.IMREAD_COLOR)
    if full is None:
        return []
    # recorte desde la imagen ORIGINAL (tiene mas detalle que la de trabajo)
    fx0, fx1 = int(max(0, x0 / work_scale)), int(min(full.shape[1], x1 / work_scale))
    fy0, fy1 = int(max(0, top / work_scale)), int(min(full.shape[0], bottom / work_scale))
    if fx1 - fx0 < 8 or fy1 - fy0 < 8:
        return []
    k = FEATURE_UPSCALE * work_scale                          # pixeles originales -> zona ampliada
    crop = cv2.resize(full[fy0:fy1, fx0:fx1], None, fx=k, fy=k, interpolation=cv2.INTER_CUBIC)
    mask = pl.dp.sketch_to_mask(crop)
    out = []
    for path in sketch_svg.skeleton_paths(mask):
        p = sketch_svg.smooth_path(path, FEATURE_SMOOTH_PX, 0.4)
        if len(p) < 2:
            continue
        xs = fx0 * work_scale + p[:, 0] / FEATURE_UPSCALE      # -> imagen de trabajo
        ys = fy0 * work_scale + p[:, 1] / FEATURE_UPSCALE
        out.append(np.column_stack([xs, work_h - ys]))        # y hacia arriba
    return out


def with_detailed_features(source, fine_moves, rotation_deg=0, detail='lapiz', line_mm=1.0, eyes_from='sketch'):
    """
    Moves del vectorizador (pipeline.line_art_svg_moves) -> moves finales
    (como pipeline.finish_line_art_moves), con la zona de los rasgos
    reemplazada por una version mas detallada, los ojos con iris y pupila
    limpios (_clean_irises) y los contornos cortados empalmados (_bridge_gaps).
    line_mm: grosor del trazo en la tableta (para rellenar la pupila).
    eyes_from: 'sketch' = los ojos del boceto (Gemini los dibuja bien);
    'landmarks' = dibujados desde los landmarks, con la mirada de los dos ojos
    igualada (el boceto local los deja flojos).
    Devuelve (moves, se_encontro_la_cara, zona de los rasgos (x0, x1, y0, y1)
    en las coordenadas de los moves, o None): en la tableta esa zona se dibuja
    con un trazo mas fino (FEATURE_LINE_FACTOR).
    """
    import sketch_svg
    base = pl.finish_line_art_moves(fine_moves, detail, rotation_deg)
    scale = pl._px_to_mm_scale(fine_moves, rotation_deg)    # mm por pixel de trabajo
    core_bbox, eyes, face_pts = pl.locate_face(source, with_points=True)
    if core_bbox is None:
        return pl._flat_strokes_to_moves(_bridge_gaps(_strokes(base), BRIDGE_MM / scale)), False, None
    img = pl.dp.load_image(source)
    work_h = img.shape[0]
    full = cv2.imread(source, cv2.IMREAD_GRAYSCALE)
    work_scale = img.shape[1] / full.shape[1] if full is not None else 1.0
    x0, x1, y0, y1 = core_bbox
    mx, my = FEATURE_MARGIN * (x1 - x0), FEATURE_MARGIN * (y1 - y0)
    box = (x0 - mx, x1 + mx, y0 - my, y1 + my)

    features = [sketch_svg.resample(f, FEATURE_STEP_MM / scale) for f in
                _feature_strokes(source, box, work_h, work_scale)
                if _length(f) * scale >= FEATURE_MIN_MM]
    features = _bridge_gaps(features, FEATURE_BRIDGE_MM / scale)
    landmark_eyes = eyes_from == 'landmarks' and bool(eyes)
    if landmark_eyes:
        eyes = harmonize_gaze(eyes)
    # ojos: los del boceto, vectorizados con cuidado (ver _sketch_eyes)
    if eyes and SKETCH_EYES and not landmark_eyes:
        eye_strokes, eye_zones = _sketch_eyes(source, eyes, work_h, work_scale,
                                              line_mm * FEATURE_LINE_FACTOR / scale)
        if eye_strokes:
            polys = [z.astype(np.float32) for z in eye_zones]
            outside = lambda q: all(cv2.pointPolygonTest(z, (float(q[0]), float(q[1])), False) < 0 for z in polys)
            kept = []
            for st in features:
                kept += _runs_inside([tuple(q) for q in st], outside)
            features = kept + [sketch_svg.resample(st, FEATURE_STEP_MM / scale * 0.6) for st in eye_strokes]
    # boca y base de la nariz: desde los landmarks (ver _landmark_features)
    features = _landmark_features(features, face_pts, eyes, FEATURE_STEP_MM / scale,
                                  LANDMARK_FEATURES + (('ojos',) if landmark_eyes else ()))
    if eyes and (CLEAN_IRISES or landmark_eyes):
        features = _clean_irises(features, eyes, FEATURE_STEP_MM / scale, line_mm * FEATURE_LINE_FACTOR / scale)
    rest = _bridge_gaps([np.asarray(st) for st in _cut_box(_strokes(base), box)], BRIDGE_MM / scale)
    if not features:
        return pl._flat_strokes_to_moves(pl.dp.order_strokes(rest)), True, None
    strokes = pl.dp.order_strokes(rest + features)
    return pl._flat_strokes_to_moves(strokes), True, box


# -------------------------------------------------------------------------------
# Ojos, boca y base de la nariz desde los landmarks
# -------------------------------------------------------------------------------
# Lo que el boceto dibuja en esas zonas, vectorizado, depende de la tinta (lineas
# cortadas, dobles, rellenos). Los landmarks de MediaPipe dan esas formas exactas:
# se dibujan como curvas suaves por esos puntos, y se borra lo que habia dibujado
# el boceto en esas zonas (sin lineas dobles). Las cejas y los costados de la
# nariz quedan del boceto (ahi se ven mas naturales que una linea geometrica).
LANDMARK_FEATURES = ('boca', 'nariz')   # 'ojos' tambien se puede; sin el, los ojos quedan los del boceto
LOWER_LID_SPAN = (0.12, 0.92)      # el parpado de abajo, parcial (desde el lagrimal), como lo dibujaria alguien
LOWER_LIP_SPAN = (0.22, 0.78)      # el borde de abajo del labio inferior, parcial
MOUTH_OPEN = 0.10                  # boca abierta (separacion / ancho): las dos lineas de adentro; si no, una
ERASE_EYE = 1.35                   # zona que se borra del boceto: el ojo agrandado esto (desde su centro)
ERASE_MOUTH = 1.18
ERASE_NOSE_PX = 0.35               # alrededor de la base de la nariz: esto x el ancho de la nariz
NOSE_SKETCH_ENOUGH = 1.5           # si el boceto ya dibujo en la base de la nariz mas que esto x su ancho,
                                   # queda la del boceto (suele ser mejor: fosas, punta); si no, los landmarks

_LIP_UPPER = (61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291)
_LIP_LOWER = (291, 375, 321, 405, 314, 17, 84, 181, 91, 146, 61)
_LIP_IN_UPPER = (78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308)
_LIP_IN_LOWER = (78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308)
_NOSE_BASE = (64, 98, 97, 2, 326, 327, 294)     # ala, fosa, columela, fosa, ala (zona que se borra)
# cada ala de la nariz curvandose hacia su fosa (dos ganchos, sin la linea del medio)
_NOSE_WINGS = ((48, 64, 98, 97), (278, 294, 327, 326))


def _spline(pts, step_px):
    """Curva suave (Catmull-Rom) por los puntos, muestreada cada ~step_px."""
    p = np.asarray(pts, float)
    if len(p) < 3:
        return p
    ext = np.vstack([2 * p[0] - p[1], p, 2 * p[-1] - p[-2]])
    out = [p[0]]
    for i in range(1, len(ext) - 2):
        p0, p1, p2, p3 = ext[i - 1], ext[i], ext[i + 1], ext[i + 2]
        n = max(2, int(np.linalg.norm(p2 - p1) / max(step_px, 1e-6)) + 1)
        for t in np.linspace(0, 1, n)[1:]:
            t2, t3 = t * t, t * t * t
            out.append(0.5 * (2 * p1 + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    return np.asarray(out)


def _span(curve, a, b):
    """Tramo de la curva entre las fracciones a y b de su largo."""
    d = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(curve, axis=0).T))])
    if d[-1] <= 0:
        return curve
    keep = (d >= a * d[-1]) & (d <= b * d[-1])
    return curve[keep] if keep.sum() >= 2 else curve


def _scaled_poly(pts, k):
    p = np.asarray(pts, np.float32)
    c = p.mean(axis=0)
    return c + (p - c) * k


def _landmark_features(strokes, pts, eyes, step_px, which=None):
    """Saca del boceto lo que dibujo en ojos, boca y base de la nariz y agrega
    esas formas desde los landmarks (pts: los 478, Y invertido)."""
    if not pts or len(pts) < 468:
        return strokes
    which = LANDMARK_FEATURES if which is None else which
    P = lambda idx: [pts[i] for i in idx]
    zones, new = [], []
    if 'ojos' in which and eyes:
        for eye in eyes:
            c = eye['contour']                          # 16 puntos: parpado de arriba (9) y de abajo
            upper = c[:9]                               # de la esquina de afuera al lagrimal
            lower = [c[8]] + c[9:] + [c[0]]              # del lagrimal a la esquina de afuera
            new.append(_spline(upper, step_px))
            new.append(_span(_spline(lower, step_px), *LOWER_LID_SPAN))
            zones.append(_scaled_poly(c, ERASE_EYE))
    if 'boca' in which:
        width = math.dist(pts[61], pts[291])
        opening = math.dist(pts[13], pts[14]) / max(width, 1e-6)
        new.append(_spline(P(_LIP_UPPER), step_px))
        new.append(_span(_spline(P(_LIP_LOWER), step_px), *LOWER_LIP_SPAN))
        if opening > MOUTH_OPEN:
            new.append(_spline(P(_LIP_IN_UPPER), step_px))
            new.append(_spline(P(_LIP_IN_LOWER), step_px))
        else:   # boca cerrada: la linea entre los labios
            mid = [((a[0] + b[0]) / 2, (a[1] + b[1]) / 2) for a, b in zip(P(_LIP_IN_UPPER), P(_LIP_IN_LOWER))]
            new.append(_spline([pts[61]] + mid[1:-1] + [pts[291]], step_px))
        zones.append(_scaled_poly(P(_LIP_UPPER) + P(_LIP_LOWER), ERASE_MOUTH))
    if 'nariz' in which:
        base = P(_NOSE_BASE)
        w = math.dist(pts[64], pts[294])
        b = np.asarray(base, np.float32)
        x0, y0 = b.min(axis=0) - ERASE_NOSE_PX * w * np.array([0.3, 1.0])
        x1, y1 = b.max(axis=0) + ERASE_NOSE_PX * w * np.array([0.3, 0.5])
        zone = np.array([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], np.float32)
        # cuanto dibujo el boceto ahi
        inked = 0.0
        for st in strokes:
            for a, c in zip(st, st[1:]):
                if (cv2.pointPolygonTest(zone, (float(a[0]), float(a[1])), False) >= 0 and
                        cv2.pointPolygonTest(zone, (float(c[0]), float(c[1])), False) >= 0):
                    inked += math.dist(a, c)
        if inked < NOSE_SKETCH_ENOUGH * w:
            for wing in _NOSE_WINGS:
                new.append(_spline(P(wing), step_px))
            zones.append(zone)
    if not new:
        return strokes
    polys = [z.astype(np.float32) for z in zones]

    def outside(q):
        return all(cv2.pointPolygonTest(z, (float(q[0]), float(q[1])), False) < 0 for z in polys)
    kept = []
    for st in strokes:
        kept += _runs_inside([tuple(q) for q in st], outside)
    return kept + [np.asarray(c) for c in new if len(c) >= 2]


# -------------------------------------------------------------------------------
# Ojos del boceto, vectorizados con cuidado
# -------------------------------------------------------------------------------
# Los ojos son los del boceto (no se reemplazan). Pero vectorizados como el resto
# (esqueleto de la tinta) pierden la pupila (una mancha rellena: su esqueleto es un
# garabato) y el iris se vuelve una cruz (sus rayitas se unen con el circulo). Aca
# cada ojo se recorta del boceto original, ampliado, y:
#   - lo relleno (pupila, delineado grueso) se dibuja como relleno: su contorno y
#     anillos hacia adentro, respetando los huecos (el brillo);
#   - las lineas (parpados, circulo del iris, pestanas) se vectorizan aparte;
#   - solo se descartan las rayitas cortas DENTRO del iris (textura, no rasgo).
# Los landmarks solo ubican el ojo; la forma es la del boceto.
SKETCH_EYES = True
EYE_CROP_WIDTH_PX = 420       # el ojo se amplia a este ancho para vectorizarlo
EYE_CROP_SIDE = 0.45          # recorte: margen a los costados (en anchos del ojo)
EYE_CROP_UP = 0.62            # arriba, en distancias a la ceja (sin llegar a la ceja)
EYE_CROP_DOWN = 0.55          # abajo, en anchos del ojo
EYE_SOLID = 2.4               # relleno: mas grueso que esto x el grosor de las lineas del ojo
IRIS_TEXTURE_MAX = 0.75       # rayitas dentro del iris mas cortas que esto x su radio: textura
EYE_MIN_STROKE_PENS = 2.2     # trazos mas cortos que esto x el grosor del trazo (pestanas sueltas): con
                              # el grosor de la tableta quedan como manchas sobre el parpado


def _sketch_eyes(source, eyes, work_h, work_scale, pen_px):
    """Trazos de los ojos del boceto (pixeles de trabajo, Y hacia arriba) y las
    zonas que reemplazan (poligonos), o ([], [])."""
    import sketch_svg
    full = cv2.imread(source, cv2.IMREAD_GRAYSCALE)
    if full is None:
        return [], []
    strokes, zones = [], []
    for eye in eyes:
        c = np.asarray(eye['contour'], float)                       # trabajo, Y hacia arriba
        w = float(eye['width'])
        cx, cy = eye['center']
        x0, x1 = c[:, 0].min() - EYE_CROP_SIDE * w, c[:, 0].max() + EYE_CROP_SIDE * w
        y_top = cy + EYE_CROP_UP * eye['brow_dist']
        y_bot = c[:, 1].min() - EYE_CROP_DOWN * w
        # a pixeles del boceto original (Y hacia abajo)
        fx0, fx1 = int(max(0, x0 / work_scale)), int(min(full.shape[1], x1 / work_scale))
        fy0, fy1 = int(max(0, (work_h - y_top) / work_scale)), int(min(full.shape[0], (work_h - y_bot) / work_scale))
        if fx1 - fx0 < 8 or fy1 - fy0 < 8:
            continue
        up = EYE_CROP_WIDTH_PX / float(fx1 - fx0)
        crop = cv2.resize(full[fy0:fy1, fx0:fx1], None, fx=up, fy=up, interpolation=cv2.INTER_CUBIC)
        _, ink = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        ink = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)) > 0
        if ink.sum() < 20:
            continue
        dist = cv2.distanceTransform(ink.astype(np.uint8), cv2.DIST_L2, 3)
        thin = pl.dp.thin(ink)
        half = float(np.median(dist[thin > 0])) if thin.any() else 1.5    # medio grosor tipico de las lineas
        core = dist > max(3.0, EYE_SOLID * half)
        solid = np.zeros_like(ink)
        if core.any():   # lo relleno, reconstruido desde su nucleo
            r = int(round(max(3.0, EYE_SOLID * half))) + 1
            solid = cv2.dilate(core.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))) > 0
            solid &= ink
        to_work = lambda p: np.column_stack([(fx0 + p[:, 0] / up) * work_scale, work_h - (fy0 + p[:, 1] / up) * work_scale])
        out = []
        # relleno: contorno y anillos hacia adentro (los huecos, como el brillo, quedan)
        pen = max(1.0, pen_px / work_scale * up)
        cur = solid.astype(np.uint8)
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(2 * pen) | 1, int(2 * pen) | 1))
        for _ in range(40):
            if not cur.any():
                break
            cnts, _h = cv2.findContours(cur, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
            for cnt in cnts:
                p = cnt.reshape(-1, 2).astype(float)
                if len(p) >= 3:
                    out.append(np.vstack([p, p[:1]]))
            nxt = cv2.erode(cur, ker)
            # lo que desaparece en este paso (el centro de la pupila): un punto en su medio
            n, lab = cv2.connectedComponents(cur)
            for k in range(1, n):
                comp = lab == k
                if not (nxt & comp).any():
                    dd = cv2.distanceTransform(comp.astype(np.uint8), cv2.DIST_L2, 3)
                    yy, xx = np.unravel_index(int(dd.argmax()), dd.shape)
                    out.append(np.array([(xx - 0.5, yy), (xx + 0.5, yy)], float))
            cur = nxt
        # lineas: el resto de la tinta
        lines = ink & ~(cv2.dilate(solid.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0)
        iris = None
        if eye.get('iris_c') is not None and eye.get('iris_r'):
            ic = ((eye['iris_c'][0] / work_scale - fx0) * up, ((work_h - eye['iris_c'][1]) / work_scale - fy0) * up)
            iris = (ic, eye['iris_r'] / work_scale * up)
        for path in sketch_svg.skeleton_paths(lines.astype(np.uint8) * 255):
            p = sketch_svg.smooth_path(path, 1.2, 0.3)
            if len(p) < 2:
                continue
            length = float(np.sum(np.hypot(*np.diff(p, axis=0).T)))
            if length < EYE_MIN_STROKE_PENS * pen:
                continue   # pestana suelta
            if iris is not None:
                (icx, icy), ir = iris
                inside = np.hypot(p[:, 0] - icx, p[:, 1] - icy) < 0.92 * ir
                length = float(np.sum(np.hypot(*np.diff(p, axis=0).T)))
                if inside.all() and length < IRIS_TEXTURE_MAX * ir:
                    continue   # rayita del iris
            out.append(p)
        strokes += [to_work(np.asarray(p, float)) for p in out]
        zone = np.array([(x0, y_bot), (x1, y_bot), (x1, y_top), (x0, y_top)], np.float32)
        zones.append(zone)
    return strokes, zones


# -------------------------------------------------------------------------------
# Mirada igualada (ojos desde los landmarks)
# -------------------------------------------------------------------------------
# El punto del iris de MediaPipe tiene un error de 1-2 px en fotos de camara web;
# en un ojo de ~12 mm eso ya hace que cada ojo "mire" para otro lado (estrabismo).
# Las personas miran con los dos ojos igual: se mide la mirada de cada uno
# respecto de sus comisuras, se usa el promedio para los dos (limitado a un
# rango natural) y el mismo tamano de iris.
GAZE_MAX_X = 0.14        # corrimiento maximo del iris a los costados (fraccion del ancho del ojo)
GAZE_MAX_Y = 0.06        # arriba/abajo
IRIS_RADIUS = (0.19, 0.25)   # radio del iris permitido (fraccion del ancho del ojo)


def harmonize_gaze(eyes):
    import copy
    eyes = copy.deepcopy(eyes)
    usable = [e for e in eyes if e.get('iris_c') is not None and e.get('iris_r')]
    if len(usable) < 2:
        return eyes
    # ejes comunes: x de izquierda a derecha de la imagen (por las comisuras), y perpendicular
    frames, gx, gy, rr = [], [], [], []
    for e in usable:
        a, b = sorted([e['inner'], e['outer']], key=lambda p: p[0])
        ex = np.subtract(b, a) / max(math.dist(a, b), 1e-6)
        ey = np.array([-ex[1], ex[0]])
        c = np.array(e['center'], float)
        d = np.subtract(e['iris_c'], c)
        frames.append((c, ex, ey, e['width']))
        gx.append(float(d @ ex) / e['width'])
        gy.append(float(d @ ey) / e['width'])
        rr.append(e['iris_r'] / e['width'])
    ux = float(np.clip(np.mean(gx), -GAZE_MAX_X, GAZE_MAX_X))
    uy = float(np.clip(np.mean(gy), -GAZE_MAX_Y, GAZE_MAX_Y))
    r = float(np.clip(np.mean(rr), *IRIS_RADIUS))
    for e, (c, ex, ey, w) in zip(usable, frames):
        e['iris_c'] = tuple(c + (ux * ex + uy * ey) * w)
        e['iris_r'] = r * w
    return eyes


# -------------------------------------------------------------------------------
# Ojos: iris y pupila limpios
# -------------------------------------------------------------------------------
# Gemini dibuja el iris con textura (rayitas radiales, brillo, pupila rellena):
# vectorizado da una cruz o un garabato. Dentro del ojo abierto (contorno de
# MediaPipe) se saca lo que dibujo el boceto en el iris y se dibuja un iris
# limpio: su circulo (cortado por los parpados) y la pupila rellena con una
# espiral, dejando el brillo.
FEATURE_LINE_FACTOR = 0.5    # los rasgos, con la mitad del grosor de la tableta (y de la descarga)
IRIS_KEEP_MARGIN = 0.18      # lo del boceto a menos de esto (x radio del iris) del borde del ojo se deja (parpados)
PUPIL_FRACTION = 0.42        # radio de la pupila / radio del iris
HIGHLIGHT = (0.38, 0.38, 0.34)   # brillo: centro (hacia afuera-arriba) y radio, en radios de pupila
PUPIL_FILL = 0.8             # separacion de las vueltas de la espiral, en grosores de trazo
CLEAN_IRISES = False         # False: el iris y la pupila quedan como los dibujo el boceto


def _runs_inside(pts, keep, closed=False):
    """Tramos consecutivos de pts donde keep(p) es verdadero (closed: el
    ultimo tramo sigue en el primero)."""
    flags = [bool(keep(p)) for p in pts]
    runs, cur = [], []
    for p, ok in zip(pts, flags):
        if ok:
            cur.append(p)
        else:
            if len(cur) >= 2:
                runs.append(cur)
            cur = []
    if cur:
        if closed and flags[0] and runs and not all(flags):
            runs[0] = cur + runs[0]   # el tramo del final sigue en el del principio
        elif closed and all(flags):
            runs.append(cur + [cur[0]])
        elif len(cur) >= 2:
            runs.append(cur)
    return [np.asarray(r, float) for r in runs if len(r) >= 2]


def _clean_irises(strokes, eyes, step_px, pen_px):
    out = list(strokes)
    for eye in eyes:
        c, r = eye.get('iris_c'), eye.get('iris_r')
        if c is None or not r or r < 2:
            continue
        poly = np.asarray(eye['contour'], np.float32)
        cx, cy = c

        def inside_eye(p, margin=0.0):
            return cv2.pointPolygonTest(poly, (float(p[0]), float(p[1])), True) > margin

        # lo que el boceto dibujo dentro del iris (sin tocar los parpados)
        def texture(p):
            return math.hypot(p[0] - cx, p[1] - cy) <= r * 1.05 and inside_eye(p, IRIS_KEEP_MARGIN * r)
        cleaned = []
        for st in out:
            cleaned += _runs_inside([tuple(q) for q in st], lambda q: not texture(q))
        out = cleaned

        # circulo del iris, cortado por los parpados
        n = max(24, int(2 * math.pi * r / (step_px * 0.6)))
        circle = [(cx + r * math.cos(t), cy + r * math.sin(t)) for t in np.linspace(0, 2 * math.pi, n, endpoint=False)]
        out += _runs_inside(circle, lambda q: inside_eye(q, -0.5), closed=True)

        # pupila: espiral desde el centro, sin el brillo
        rp = PUPIL_FRACTION * r
        gap = max(0.5, PUPIL_FILL * pen_px)
        hx, hy, hr = cx + HIGHLIGHT[0] * rp, cy + HIGHLIGHT[1] * rp, HIGHLIGHT[2] * rp
        spiral, th = [], 0.0
        while True:
            rad = gap * th / (2 * math.pi)
            if rad > rp:
                break
            spiral.append((cx + rad * math.cos(th), cy + rad * math.sin(th)))
            th += min(0.6, (step_px * 0.5) / max(rad, gap))
        spiral += [(cx + rp * math.cos(t), cy + rp * math.sin(t)) for t in np.linspace(th, th + 2 * math.pi, 32)]
        out += _runs_inside(spiral, lambda q: math.hypot(q[0] - hx, q[1] - hy) > hr and inside_eye(q, 0.0))
    return out


# -------------------------------------------------------------------------------
# Contornos cortados
# -------------------------------------------------------------------------------
# El vectorizador corta las lineas finas o claras del boceto en pedazos (el
# contorno de la cabeza sale "punteado"). Se empalman los extremos que quedan
# cerca y siguen la misma direccion.
BRIDGE_MM = 2.0
FEATURE_BRIDGE_MM = 1.2      # en los rasgos, huecos mas chicos (no unir pestanas ni pelos de las cejas)
BRIDGE_MAX_DEG = 40


def _bridge_gaps(strokes, gap_px, max_deg=BRIDGE_MAX_DEG):
    strokes = [np.asarray(s, float) for s in strokes if len(s) >= 2]
    if len(strokes) < 2 or gap_px <= 0:
        return strokes
    cos_min = math.cos(math.radians(max_deg))
    ends, dirs = [], []            # extremo e: trazo e // 2, e % 2 = 0 inicio / 1 final; dir hacia afuera
    for st in strokes:
        for tip, prev in ((st[0], st[min(3, len(st) - 1)]), (st[-1], st[max(0, len(st) - 4)])):
            d = tip - prev
            n = math.hypot(d[0], d[1])
            ends.append(tip)
            dirs.append(d / n if n > 1e-9 else np.zeros(2))
    ends, dirs = np.asarray(ends), np.asarray(dirs)
    diff = ends[None, :, :] - ends[:, None, :]          # diff[a, b] = ends[b] - ends[a]
    dist = np.hypot(diff[..., 0], diff[..., 1])
    idx = np.arange(len(ends))
    cand = []
    for a, b in zip(*np.nonzero((dist <= gap_px) & (idx[:, None] < idx[None, :]))):
        if a // 2 == b // 2:
            continue
        dd = dist[a, b]
        v = diff[a, b] / max(dd, 1e-9)
        # sale de a hacia b, y b sale hacia a
        if dd < 1e-6 or (dirs[a] @ v >= cos_min and dirs[b] @ -v >= cos_min):
            cand.append((dd, int(a), int(b)))
    cand.sort()
    parent = list(range(len(strokes)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    link = {}
    for _d, a, b in cand:
        if a in link or b in link or root(a // 2) == root(b // 2):
            continue
        link[a], link[b] = b, a
        parent[root(a // 2)] = root(b // 2)
    out, used = [], set()
    for k in range(len(strokes)):
        if k in used:
            continue
        # ir hasta un extremo libre de la cadena (sin ciclos: ver root)
        e = 2 * k
        while e in link:
            e = link[e] ^ 1
        # recorrerla desde ahi
        chain = []
        while True:
            i = e // 2
            used.add(i)
            chain.append(strokes[i] if e % 2 == 0 else strokes[i][::-1])
            if (e ^ 1) not in link:
                break
            e = link[e ^ 1]
        out.append(np.vstack(chain))
    return out


def _length(pts):
    return float(sum(math.dist(a, b) for a, b in zip(pts, pts[1:])))


# -------------------------------------------------------------------------------
# Tope de tiempo
# -------------------------------------------------------------------------------
def _strokes(moves):
    out, cur = [], []
    for x, y, pen_down in moves:
        if not pen_down:
            if cur:
                out.append(cur)
            cur = []
        else:
            cur.append((x, y))
    if cur:
        out.append(cur)
    return out


def tablet_seconds(moves, mm_per_px):
    """Duracion en modo tableta, como la cuenta draw_moves: cada comando espera
    su recorrido a TABLET_SPEED_MM_S, no menos que TABLET_MIN_PACING (en la
    tableta los tramos no se parten)."""
    seconds = 0.0
    for a, b in zip(moves, moves[1:]):
        seg = math.dist(a[:2], b[:2]) * mm_per_px
        seconds += max(lib.TABLET_MIN_PACING, seg / lib.TABLET_SPEED_MM_S)
    return seconds


def limit_to_seconds(moves, mm_per_px, target_seconds=TARGET_SECONDS):
    """Si el recorrido dura mas que target_seconds, se sacan trazos empezando
    por los mas cortos (los que menos se notan) hasta que entre. Devuelve
    (moves, segundos, trazos, trazos que habia)."""
    strokes = _strokes(moves)
    total = len(strokes)
    seconds = tablet_seconds(moves, mm_per_px)
    if seconds > target_seconds:
        by_len = sorted(range(len(strokes)), key=lambda i: sum(
            math.dist(p, q) for p, q in zip(strokes[i], strokes[i][1:])))
        keep = set(range(len(strokes)))
        for i in by_len:
            if seconds <= target_seconds or len(keep) <= 1:
                break
            keep.discard(i)
            moves = pl._flat_strokes_to_moves([strokes[j] for j in sorted(keep)])
            seconds = tablet_seconds(moves, mm_per_px)
        strokes = [strokes[j] for j in sorted(keep)]
    return moves, seconds, len(strokes), total


# -------------------------------------------------------------------------------
# Dos versiones: la tableta muestra la detallada, el brazo recorre una parecida
# -------------------------------------------------------------------------------
# El brazo no deja tinta en la tableta: la pagina muestra un dibujo detallado y el
# brazo recorre una version mas simple. Para que se vea que la punta lo esta
# dibujando, las dos tienen que parecerse:
#   - el brazo recorre casi todo (trazos de mas de ROBOT_MIN_STROKE_MM), siempre
#     los rasgos (ojos, cejas, nariz, boca), y el rayado del pelo/barba en zigzag
#     continuo, pasando por cada trazo (robot_version);
#   - la tableta no muestra lo que el brazo no recorre cerca (prune_display);
#   - la tableta solo avanza mientras la punta esta abajo, y cada trazo se dibuja
#     mientras la punta pasa por su zona (display_sync).
ROBOT_TARGET_SECONDS = 55.0   # duracion maxima del recorrido del brazo (modo tableta)
ROBOT_SIMPLIFY_MM = 0.5       # tolerancia de la simplificacion (Douglas-Peucker)
ROBOT_MIN_STROKE_MM = 2.5     # el brazo no recorre trazos mas cortos (salvo los rasgos)
FEATURE_MIN_ROBOT_MM = 0.8    # en la zona de los rasgos, hasta trazos de este largo
ROBOT_COVER_MM = 3.0          # si hay que recortar tiempo, primero los trazos que ya tienen otro a menos de esto
ROBOT_TRAVEL_GUESS_MM = 8.0   # viaje promedio entre trazos (para estimar el tiempo al recortar)
HATCH_JOIN_MM = 4.5           # rayado (pelo): trazos a menos de esto se unen sin levantar (zigzag)
HATCH_ROBOT_SPACING_MM = 2.6  # el brazo recorre los trazos del rayado separados al menos esto (uno de
                              # cada dos, mas o menos): la tableta muestra igual los del medio (ver prune_display)
ROBOT_MAX_OVER = 1.35         # si pasa de target_seconds x esto, se recortan trazos (no los rasgos)
DISPLAY_COVER_MM = 3.0        # la tableta muestra un trazo si el brazo pasa a menos de esto...
DISPLAY_COVER_FRACTION = 0.6  # ...en al menos esta parte de su largo
SYNC_SAMPLE_MM = 1.0
SYNC_MIN_SPEED_RATIO = 0.0    # un tramo de la tableta se dibuja mientras la punta avanza al menos esto de su largo
SYNC_MAX_DRIFT_MM = 4.0       # ...salvo que la tableta ya vaya esto atras de donde la ubica el brazo
SYNC_CHUNK_MM = 3.0  # la tableta ubica cada tramo de este largo por separado en el recorrido del brazo:
                              # una linea larga se va dibujando junto a la punta (entera, aparecia de golpe)


def _resample_px(st, step):
    pts = [tuple(st[0])]
    for a, b in zip(st, st[1:]):
        d = math.dist(a, b)
        if d <= 0:
            continue
        n = int(d // step)
        for k in range(1, n + 1):
            t = k * step / d
            pts.append((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t))
    if math.dist(pts[-1], st[-1]) > 1e-9:
        pts.append(tuple(st[-1]))
    return pts


def _stroke_seconds(st, mm_per_px):
    """Tiempo aproximado de un trazo en modo tableta (como tablet_seconds), con su viaje."""
    segs = [math.dist(a, b) * mm_per_px for a, b in zip(st, st[1:])]
    return (sum(max(lib.TABLET_MIN_PACING, d / lib.TABLET_SPEED_MM_S) for d in segs)
            + 2 * lib.TABLET_MIN_PACING + ROBOT_TRAVEL_GUESS_MM / lib.TABLET_SPEED_MM_S)


class _NearGrid:
    """Puntos en una grilla, para preguntar rapido si hay alguno cerca."""
    def __init__(self, cell):
        self.cell, self.g = cell, {}

    def add(self, pts, tag=0):
        for x, y in pts:
            self.g.setdefault((int(x // self.cell), int(y // self.cell)), []).append((x, y, tag))

    def near(self, x, y, r, skip=None, alive=None):
        cx, cy = int(x // self.cell), int(y // self.cell)
        for gx in (cx - 1, cx, cx + 1):
            for gy in (cy - 1, cy, cy + 1):
                for qx, qy, t in self.g.get((gx, gy), ()):
                    if t != skip and (alive is None or alive[t]) and (qx - x) ** 2 + (qy - y) ** 2 <= r * r:
                        return True
        return False


def _drop_covered(strokes, fixed_seconds, mm_per_px, target_seconds):
    """Saca trazos (los mas cortos primero) mientras el tiempo pase de
    target_seconds, pero solo los que siguen cubiertos por otros que quedan (a
    menos de ROBOT_COVER_MM en casi todo su largo): el brazo sigue pasando
    cerca de todo."""
    secs = [_stroke_seconds(st, mm_per_px) for st in strokes]
    total = fixed_seconds + sum(secs)
    if total <= target_seconds:
        return strokes
    r = ROBOT_COVER_MM / mm_per_px
    grid = _NearGrid(r)
    samples = [_resample_px(st, SYNC_SAMPLE_MM / mm_per_px) for st in strokes]
    for i, sm in enumerate(samples):
        grid.add(sm, i)
    alive = [True] * len(strokes)
    for i in sorted(range(len(strokes)), key=lambda k: _length(strokes[k])):
        if total <= target_seconds:
            break
        pts = samples[i]
        if sum(grid.near(x, y, r, skip=i, alive=alive) for x, y in pts) >= 0.9 * len(pts):
            alive[i] = False
            total -= secs[i]
    return [st for st, k in zip(strokes, alive) if k]


def _hatch_chains(hatch, mm_per_px):
    """Rayado (pelo, barba) -> cadenas en zigzag: cada trazo, seguido del mas
    cercano (dado vuelta si conviene), sin levantar si esta a menos de HATCH_JOIN_MM."""
    left = [list(map(tuple, np.asarray(h, float))) for h in hatch if len(h) >= 2]
    chains = []
    gap = HATCH_JOIN_MM / mm_per_px
    while left:
        cur = left.pop(0)
        while left:
            end = cur[-1]
            best, bi, rev = None, -1, False
            for i, h in enumerate(left):
                for r_, p in ((False, h[0]), (True, h[-1])):
                    d = math.dist(end, p)
                    if best is None or d < best:
                        best, bi, rev = d, i, r_
            if best > gap:
                break
            h = left.pop(bi)
            cur += h[::-1] if rev else h
        chains.append(cur)
    return chains


def robot_version(moves, mm_per_px, target_seconds=ROBOT_TARGET_SECONDS, feature_box=None, hatch=None):
    """Recorrido del brazo a partir del detallado (mismas coordenadas). moves:
    el detallado SIN el rayado; hatch: los trazos del rayado (ver texturas.py),
    que el brazo recorre en zigzag; feature_box (x0, x1, y0, y1): la zona de los
    rasgos, cuyos trazos se recorren siempre. Devuelve (moves, segundos, trazos,
    trazos que habia)."""
    eps = ROBOT_SIMPLIFY_MM / mm_per_px

    def simplify(st):
        arr = np.asarray(st, np.float32).reshape(-1, 1, 2)
        simp = cv2.approxPolyDP(arr, eps, False).reshape(-1, 2) if len(st) >= 3 else arr.reshape(-1, 2)
        return [(float(x), float(y)) for x, y in simp]

    def in_features(st):
        if not feature_box:
            return False
        x0, x1, y0, y1 = feature_box
        return all(x0 <= x <= x1 and y0 <= y <= y1 for x, y in st)

    fixed, optional = [], []
    for st in _strokes(moves):
        length = _length(st) * mm_per_px
        if in_features(st):
            if length >= FEATURE_MIN_ROBOT_MM:
                fixed.append(simplify(st))
        elif length >= ROBOT_MIN_STROKE_MM:
            optional.append(simplify(st))
    fixed += [simplify(c) for c in _hatch_chains(_thin_hatch(hatch or [], mm_per_px), mm_per_px)]
    total = len(fixed) + len(optional)
    if not fixed and not optional:   # dibujo de puros trazos cortos: el mas largo
        optional = [max(_strokes(moves), key=_length)]
    fixed_s = sum(_stroke_seconds(st, mm_per_px) for st in fixed)
    optional = _drop_covered(optional, fixed_s, mm_per_px, target_seconds)
    # recorrido optimizado (vpype): menos bajadas de la punta y menos viaje
    robot = pl.optimize_moves(pl._flat_strokes_to_moves(fixed + optional), mm_per_px)
    seconds = tablet_seconds(robot, mm_per_px)
    if seconds > target_seconds * ROBOT_MAX_OVER and optional:
        # ultimo recurso: los trazos opcionales mas cortos, aunque queden sin cubrir (los rasgos y el rayado quedan)
        opt, _s, _n, _t = limit_to_seconds(pl._flat_strokes_to_moves(optional), mm_per_px,
                                           max(1.0, target_seconds * ROBOT_MAX_OVER - fixed_s))
        robot = pl.optimize_moves(pl._flat_strokes_to_moves(fixed + [list(map(tuple, s)) for s in _strokes(opt)]),
                                  mm_per_px)
        seconds = tablet_seconds(robot, mm_per_px)
    return robot, seconds, len(_strokes(robot)), total


def _thin_hatch(hatch, mm_per_px):
    """Del rayado, los trazos separados al menos HATCH_ROBOT_SPACING_MM (los largos primero)."""
    r = HATCH_ROBOT_SPACING_MM / mm_per_px
    grid = _NearGrid(r)
    kept = []
    for h in sorted(hatch, key=lambda h: -_length(h)):
        pts = _resample_px([tuple(p) for p in h], SYNC_SAMPLE_MM / mm_per_px)
        if sum(grid.near(x, y, r) for x, y in pts) < 0.5 * len(pts):
            kept.append(h)
            grid.add(pts)
    return kept


def prune_display(display, robot, mm_per_px):
    """Saca del detallado los trazos por donde el brazo no pasa cerca (a menos de
    DISPLAY_COVER_MM en DISPLAY_COVER_FRACTION de su largo): la tableta no dibuja
    lejos de la punta. Devuelve (moves, trazos que se sacaron)."""
    r = DISPLAY_COVER_MM / mm_per_px
    grid = _NearGrid(r)
    for st in _strokes(robot):
        grid.add(_resample_px(st, SYNC_SAMPLE_MM / mm_per_px))
    kept, dropped = [], 0
    for st in _strokes(display):
        pts = _resample_px(st, SYNC_SAMPLE_MM / mm_per_px)
        if sum(grid.near(x, y, r) for x, y in pts) >= DISPLAY_COVER_FRACTION * len(pts):
            kept.append(st)
        else:
            dropped += 1
    return pl._flat_strokes_to_moves(kept), dropped


def _chunks(strokes, w, h, chunk_mm):
    """Parte cada trazo [(u, v, grosor), ...] en tramos de ~chunk_mm (mm sobre la
    hoja de w x h); tramos consecutivos comparten el punto de union (sin huecos)."""
    out = []
    for st in strokes:
        cur, length = [st[0]], 0.0
        for a, b in zip(st, st[1:]):
            length += math.hypot((b[0] - a[0]) * w, (b[1] - a[1]) * h)
            cur.append(b)
            if length >= chunk_mm:
                out.append(cur)
                cur, length = [b], 0.0
        if len(cur) >= 2:
            if out and length < chunk_mm * 0.4 and out[-1][-1] == cur[0]:
                out[-1] = out[-1] + cur[1:]   # el ultimo pedacito, con el tramo anterior
            else:
                out.append(cur)
    return out


def display_sync(robot_pts, display_pts, area_mm):
    """
    robot_pts / display_pts: [(u, v, pluma_abajo), ...] sobre la hoja (ver
    lib.iter_sheet_uv, con el MISMO encuadre); display_pts puede traer un
    cuarto valor, el grosor relativo del trazo. Devuelve (puntos, mapper): los
    puntos detallados reordenados y orientados, y mapper(avance del brazo) ->
    avance en esos puntos (los dos, indices fraccionarios como en
    LivePublisher.reveal).

    Solo avanza mientras el brazo dibuja (con la punta arriba, la longitud
    dibujada del brazo no cambia), y cada trazo detallado se dibuja mientras
    la punta recorre la parte de su recorrido mas cercana a el: empieza cuando
    pasa por el punto mas cercano a su comienzo y termina cuando pasa por el mas
    cercano a su final (si hace falta se lo da vuelta, para que vaya en el mismo
    sentido que la punta).
    """
    w, h = float(area_mm[0]), float(area_mm[1])
    rob = np.array([(u * w, v * h) for u, v, _ in robot_pts], float)
    rpd = np.array([bool(pd) for *_, pd in robot_pts])
    n_disp = len(display_pts)
    seg = np.hypot(*np.diff(rob, axis=0).T) * (rpd[:-1] & rpd[1:]) if len(rob) > 1 else np.zeros(0)
    s_rob = np.concatenate([[0.0], np.cumsum(seg)])
    total_r = s_rob[-1]

    strokes, cur = [], []
    for p in display_pts:
        u, v, pd = p[:3]
        if pd:
            cur.append((u, v, p[3] if len(p) > 3 else 1))
        else:
            if len(cur) >= 2:
                strokes.append(cur)
            cur = []
    if len(cur) >= 2:
        strokes.append(cur)
    if total_r <= 0 or not strokes:   # sin nada que seguir: avance proporcional
        k = max(1, n_disp - 1) / max(1, len(robot_pts) - 1)
        return [tuple(p) for p in display_pts], lambda r: r * k
    strokes = _chunks(strokes, w, h, SYNC_CHUNK_MM) if SYNC_CHUNK_MM else strokes

    # muestras del recorrido del brazo con la punta abajo (cada ~1 mm) y su longitud dibujada
    sx, ss = [], []
    for k in np.nonzero(seg)[0]:
        n = max(1, int(seg[k] / SYNC_SAMPLE_MM))
        t = np.arange(n + 1) / n
        sx.append(rob[k] + (rob[k + 1] - rob[k]) * t[:, None])
        ss.append(s_rob[k] + seg[k] * t)
    samples, sample_s = np.concatenate(sx), np.concatenate(ss)

    def nearest_s(pts):
        pts = np.asarray(pts)
        out = np.empty(len(pts))
        for a in range(0, len(pts), 256):   # por bloques (memoria acotada)
            d = ((pts[a:a + 256, None, :] - samples[None, :, :]) ** 2).sum(-1)
            out[a:a + 256] = sample_s[d.argmin(1)]
        return out

    s_start = nearest_s([(st[0][0] * w, st[0][1] * h) for st in strokes])
    s_end = nearest_s([(st[-1][0] * w, st[-1][1] * h) for st in strokes])
    for j in range(len(strokes)):   # que vaya en el mismo sentido que la punta
        if s_end[j] < s_start[j]:
            strokes[j] = strokes[j][::-1]
            s_start[j], s_end[j] = s_end[j], s_start[j]
    order = np.argsort(s_start, kind='stable')

    points, starts, ends = [], [], []
    for j in order:
        st = strokes[j]
        points.append((st[0][0], st[0][1], 0, st[0][2]))
        starts.append(len(points))
        points.extend((u, v, 1, lw) for u, v, lw in st)
        ends.append(len(points) - 1)
    xy = np.array([(p[0] * w, p[1] * h) for p in points])
    pd = np.array([p[2] for p in points], bool)
    # longitud dibujada en la tableta hasta cada indice (los viajes suman casi nada:
    # asi la inversa longitud -> indice no queda ambigua)
    dseg = np.hypot(*np.diff(xy, axis=0).T) * (pd[:-1] & pd[1:]) + 1e-6
    s_disp = np.concatenate([[0.0], np.cumsum(dseg)])
    total_d = s_disp[-1]

    # nudos (longitud del brazo, longitud de la tableta): cada trazo va de su
    # comienzo a su final mientras la punta pasa por esa parte; entre un trazo y
    # el siguiente la tableta espera (no dibuja lejos de la punta)
    s0 = s_start[order]
    s1 = np.maximum(s_end[order], s0)
    xs, ys = [0.0], [0.0]
    prev_end = 0.0
    for i in range(len(order)):
        nxt = s0[i + 1] if i + 1 < len(order) else total_r
        start = max(s0[i], prev_end)
        # cada tramo se dibuja mientras la punta avanza al menos SYNC_MIN_SPEED_RATIO de su largo:
        # varios tramos en la misma zona (rayado, paralelos) se escalonan en vez de aparecer juntos,
        # salvo que la tableta ya vaya SYNC_MAX_DRIFT_MM atras de la punta
        chunk_len = s_disp[ends[i]] - s_disp[starts[i]]
        min_dur = SYNC_MIN_SPEED_RATIO * chunk_len if start - s0[i] < SYNC_MAX_DRIFT_MM else 0.0
        end = max(min(s1[i], nxt), start + min_dur)
        xs += [start, end]
        ys += [s_disp[starts[i]], s_disp[ends[i]]]
        prev_end = end
    xs.append(total_r)
    ys.append(total_d)
    xs = np.maximum.accumulate(np.asarray(xs))
    xs = xs + np.arange(len(xs)) * 1e-9 * max(1.0, total_r)   # estrictamente creciente
    ys = np.maximum.accumulate(np.asarray(ys))
    r_idx, d_idx = np.arange(len(s_rob)), np.arange(len(s_disp))

    def mapper(r):
        s = np.interp(r, r_idx, s_rob)
        if s >= total_r and r >= len(s_rob) - 1:
            return float(len(points) - 1)
        return float(np.interp(np.interp(s, xs, ys), s_disp, d_idx))

    return points, mapper
