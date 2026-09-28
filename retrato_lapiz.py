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


def with_detailed_features(source, fine_moves, rotation_deg=0, detail='lapiz', line_mm=1.0):
    """
    Moves del vectorizador (pipeline.line_art_svg_moves) -> moves finales
    (como pipeline.finish_line_art_moves), con la zona de los rasgos
    reemplazada por una version mas detallada, los ojos con iris y pupila
    limpios (_clean_irises) y los contornos cortados empalmados (_bridge_gaps).
    line_mm: grosor del trazo en la tableta (para rellenar la pupila).
    Devuelve (moves, se_encontro_la_cara, zona de los rasgos (x0, x1, y0, y1)
    en las coordenadas de los moves, o None): en la tableta esa zona se dibuja
    con un trazo mas fino (FEATURE_LINE_FACTOR).
    """
    import sketch_svg
    base = pl.finish_line_art_moves(fine_moves, detail, rotation_deg)
    scale = pl._px_to_mm_scale(fine_moves, rotation_deg)    # mm por pixel de trabajo
    core_bbox, eyes = pl.locate_face(source)
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
    if eyes:
        features = _clean_irises(features, eyes, FEATURE_STEP_MM / scale, line_mm * FEATURE_LINE_FACTOR / scale)
    rest = _bridge_gaps([np.asarray(st) for st in _cut_box(_strokes(base), box)], BRIDGE_MM / scale)
    if not features:
        return pl._flat_strokes_to_moves(pl.dp.order_strokes(rest)), True, None
    strokes = pl.dp.order_strokes(rest + features)
    return pl._flat_strokes_to_moves(strokes), True, box


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
# Dos versiones: la tableta muestra la detallada, el brazo recorre una simple
# -------------------------------------------------------------------------------
# El brazo no deja tinta en la tableta: la pagina puede mostrar un dibujo mas
# detallado que el recorrido del brazo. El brazo recorre una version simple
# (sin los trazos cortos, simplificada y con menos tiempo) y la tableta dibuja
# la detallada al mismo ritmo y por la misma zona por donde va la punta.
ROBOT_TARGET_SECONDS = 40.0   # duracion maxima del recorrido del brazo (modo tableta)
ROBOT_SIMPLIFY_MM = 0.8       # tolerancia de la simplificacion (Douglas-Peucker)
ROBOT_MIN_STROKE_MM = 6.0     # el brazo no recorre trazos mas cortos (la tableta si los dibuja)
SYNC_WEIGHT = 0.7             # 1 = cada trazo detallado se dibuja justo cuando la punta pasa por
                              # su zona (donde el brazo no dibuja nada, la tableta dibujaria
                              # de golpe); 0 = la tableta avanza parejo, sin seguir a la punta
SYNC_SAMPLE_MM = 1.0


def robot_version(moves, mm_per_px, target_seconds=ROBOT_TARGET_SECONDS):
    """Recorrido simple del brazo a partir del detallado (mismas coordenadas).
    Devuelve (moves, segundos, trazos, trazos que habia) como limit_to_seconds."""
    eps = ROBOT_SIMPLIFY_MM / mm_per_px
    strokes = []
    for st in _strokes(moves):
        if _length(st) * mm_per_px < ROBOT_MIN_STROKE_MM:
            continue
        arr = np.asarray(st, np.float32).reshape(-1, 1, 2)
        simp = cv2.approxPolyDP(arr, eps, False).reshape(-1, 2) if len(st) >= 3 else arr.reshape(-1, 2)
        strokes.append([(float(x), float(y)) for x, y in simp])
    if not strokes:   # dibujo de puros trazos cortos: el mas largo
        strokes = [max(_strokes(moves), key=_length)]
    return limit_to_seconds(pl._flat_strokes_to_moves(strokes), mm_per_px, target_seconds)


def display_sync(robot_pts, display_pts, area_mm, weight=SYNC_WEIGHT):
    """
    robot_pts / display_pts: [(u, v, pluma_abajo), ...] sobre la hoja (ver
    lib.iter_sheet_uv, con el MISMO encuadre); display_pts puede traer un
    cuarto valor, el grosor relativo del trazo. Devuelve (puntos, mapper):
    los puntos detallados reordenados (cada trazo, segun el punto del recorrido
    del brazo mas cercano a su mitad) y mapper(avance del brazo) -> avance en
    esos puntos (los dos, indices fraccionarios como en LivePublisher.reveal).
    """
    w, h = float(area_mm[0]), float(area_mm[1])
    rob = np.array([(u * w, v * h) for u, v, _ in robot_pts], float)
    rpd = np.array([bool(pd) for *_, pd in robot_pts])
    n_disp = len(display_pts)
    # longitud dibujada por el brazo hasta cada indice
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

    # muestras del recorrido del brazo (cada ~1 mm), con su longitud dibujada
    sx, ss = [rob[-1:]], [s_rob[-1:]]
    for k in np.nonzero(seg)[0]:
        n = max(1, int(seg[k] / SYNC_SAMPLE_MM))
        t = np.arange(n) / n
        sx.append(rob[k] + (rob[k + 1] - rob[k]) * t[:, None])
        ss.append(s_rob[k] + seg[k] * t)
    samples, sample_s = np.concatenate(sx), np.concatenate(ss)

    mids = np.array([(st[len(st) // 2][0] * w, st[len(st) // 2][1] * h) for st in strokes])
    anchor = np.empty(len(strokes))
    for a in range(0, len(mids), 256):   # mas cercano, por bloques (memoria acotada)
        d = ((mids[a:a + 256, None, :] - samples[None, :, :]) ** 2).sum(-1)
        anchor[a:a + 256] = sample_s[d.argmin(1)]
    order = np.argsort(anchor, kind='stable')

    points, starts = [], []
    for j in order:
        st = strokes[j]
        points.append((st[0][0], st[0][1], 0, st[0][2]))
        starts.append(len(points))
        points.extend((u, v, 1, lw) for u, v, lw in st)
    xy = np.array([(p[0] * w, p[1] * h) for p in points])
    pd = np.array([p[2] for p in points], bool)
    # longitud dibujada en la tableta hasta cada indice (los viajes suman casi nada:
    # asi la inversa longitud -> indice no queda ambigua)
    dseg = np.hypot(*np.diff(xy, axis=0).T) * (pd[:-1] & pd[1:]) + 1e-6
    s_disp = np.concatenate([[0.0], np.cumsum(dseg)])
    total_d = s_disp[-1]

    # cada trazo detallado empieza cuando el brazo llega a su zona (mezclado
    # con un avance parejo, para no dibujar de golpe donde el brazo no pasa)
    d_start = s_disp[starts]
    knot_s = weight * anchor[order] + (1 - weight) * d_start / total_d * total_r
    knot_s = np.maximum.accumulate(np.clip(knot_s, 0, total_r))
    xs = np.concatenate([[0.0], knot_s, [total_r]])
    ys = np.concatenate([[0.0], d_start, [total_d]])
    xs = xs + np.arange(len(xs)) * 1e-9 * max(1.0, total_r)   # estrictamente creciente
    ys = np.maximum.accumulate(ys)
    r_idx, d_idx = np.arange(len(s_rob)), np.arange(len(s_disp))

    def mapper(r):
        s = np.interp(r, r_idx, s_rob)
        if s >= total_r and r >= len(s_rob) - 1:
            return float(len(points) - 1)
        return float(np.interp(np.interp(s, xs, ys), s_disp, d_idx))

    return points, mapper
