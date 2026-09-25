#!/usr/bin/env python
# coding=utf-8
"""
Foto de una persona -> quitar fondo -> contornos -> G-code -> preview -> myCobot 280 M5.

Uso rapido:
    python draw_person.py process foto.jpg                 # genera foto.gcode + preview
    python draw_person.py process foto.jpg --detail 2      # mas detalle interior
    python draw_person.py process foto.jpg --no-face-boost # sin refuerzo automatico en la cara
    python draw_person.py sketch boceto.png                # boceto/line-art (p.ej. de IA) -> G-code + preview
    python draw_person.py preview foto.gcode               # solo ver un G-code
    python draw_person.py pose --port COM3                 # leer coords (para calibrar)
    python draw_person.py corners foto.gcode --port COM3   # recorre esquinas con lapiz arriba
    python draw_person.py run foto.gcode --port COM3       # dibuja (pide confirmacion)
    python draw_person.py run foto.gcode --dry-run         # simula sin robot

Dependencias: opencv-python, numpy, matplotlib, pymycobot (solo para run/corners/pose).
Opcional: `pip install rembg` para un recorte de fondo mucho mejor (si no, usa GrabCut).

Sistema de coordenadas: base del robot = origen, X hacia adelante, Y hacia la izquierda
(vista desde la base). La parte SUPERIOR de la imagen queda LEJOS del robot.
"""
import argparse
import math
import os
import re
import sys
import time

import cv2
import numpy as np

# --------------------------------------------------------------------------- #
# Configuracion por defecto (AJUSTAR con `pose` antes de dibujar de verdad)
# --------------------------------------------------------------------------- #
DEFAULTS = dict(
    center_x=44.6,       # mm, centro del area de dibujo -- calibrado 2026-09-22 con `pose_zero` (COM5,
    center_y=-213.5,      # base rotada mirando a la izquierda, ver nota mas abajo) -- corrido 10mm a la derecha
    z_down=256.9,        # mm, Z con el lapiz tocando el papel -- calibrado 2026-09-22, VERIFICAR contacto
    orientation=(27.05, -39.1, 97.53),  # rx, ry, rz -- calibrado 2026-09-22 con pose_zero (reproducible)
    width=138.0,         # mm, tamano maximo del dibujo (Y) -- hoja A5 (148mm) con 5mm de margen por lado
    height=200.0,        # mm, tamano maximo del dibujo (X) -- hoja A5 (210mm) con 5mm de margen por lado
    z_up=438.4,          # mm, Z con el lapiz levantado (z_down + 20mm de margen)
    feed=1000,           # mm/min, solo informativo en el G-code / estimacion de tiempo
)
# NOTA: la base fisica del robot quedo rotada mirando hacia la izquierda (no de frente
# al papel) porque la articulacion 1 no tiene rango para llegar "hacia atras" desde la
# orientacion original -- si se vuelve a mover la base, hay que recalibrar con `pose`.


# --------------------------------------------------------------------------- #
# Esquinas calibradas a mano (alternativa a calcular el rectangulo desde
# center_x/center_y/width/height + una sola orientacion): cada esquina se
# joguea por separado hasta el papel real (como `pose`) y guarda sus 6
# coordenadas propias -- mas robusto que un solo punto + medidas si el papel
# no queda perfectamente plano/alineado con la orientacion calculada.
#
# CALIBRAR con:  python draw_person.py corners-set <nombre> --port COM5
# Verificar con: python draw_person.py corners-goto --port COM5
# --------------------------------------------------------------------------- #
CORNERS = {
    "arriba_izquierda": None,  # (x, y, z, rx, ry, rz) -- CALIBRAR
    "arriba_derecha": None,
    "abajo_izquierda": None,
    "abajo_derecha": None,
}


# --------------------------------------------------------------------------- #
# Esquinas en ANGULOS de articulacion (J1..J6), medidas a mano con
# send_angles - evita la cinematica inversa por completo, tanto para llegar
# a las esquinas como para dibujar (ver corner_angles_at/run_angles mas
# abajo). Se usa en vez de CORNERS/DEFAULTS cuando la IK por coordenadas
# cartesianas daba problemas ("enroscarse").
#
# PEN_UP_ANGLES: punto de referencia con el lapiz levantado, centrado sobre
# el papel, sin tocar la superficie -- se usa para calcular un desplazamiento
# de "levantar" relativo, aplicado sobre cualquier punto interpolado (ver
# pen_up_angles_at). Ambos calibrados a mano 2026-09-22.
# --------------------------------------------------------------------------- #
CORNER_ANGLES = {
    "arriba_izquierda": [10, -25, 25, -5, 0, 0],
    "arriba_derecha": [-15, -25, 25, -5, 0, 0],
    "abajo_izquierda": [30, 35, -70, 20, 5, 0],
    "abajo_derecha": [-15, 35, -70, 20, 5, 0],
}
PEN_UP_ANGLES = [0, 20, -35, 10, 0, 0]
ANGLE_SPEED = 30  # velocidad por defecto para send_angles (baja, espacio publico)
# multiplicador sobre el desplazamiento medido de "lapiz arriba" (ver
# pen_up_angles_at) -- subilo si el lapiz roza el papel al viajar, bajalo si
# levanta demasiado / tarda mucho en bajar. 1.0 = tal cual PEN_UP_ANGLES (medido).
PEN_LIFT_SCALE = 1.0
# fraccion CHICA del mismo desplazamiento, aplicada tambien al lapiz ABAJO
# (ver pen_down_angles_at) -- alivia la presion contra el papel sin llegar a
# despegarlo. Subilo un poco si sigue presionando de mas, bajalo (o 0) si
# deja de tocar.
PEN_DOWN_LIFT_SCALE = 0.12

# proporcion ancho/alto FISICA real del papel dentro de esas 4 esquinas (para
# no deformar el dibujo al ajustarlo) -- ver DEFAULTS width/height (hoja A5)
CORNER_ASPECT_WH = DEFAULTS["width"] / DEFAULTS["height"]


def _lerp6(a, b, t):
    return [float(ai + (bi - ai) * t) for ai, bi in zip(a, b)]


def corner_angles_at(u, v):
    """Interpola (bilineal, en espacio de articulaciones) los 6 angulos para
    la posicion (u, v) dentro del rectangulo de dibujo: u=0..1 izquierda a
    derecha, v=0..1 arriba a abajo. Con el lapiz TOCANDO el papel."""
    top = _lerp6(CORNER_ANGLES["arriba_izquierda"], CORNER_ANGLES["arriba_derecha"], u)
    bottom = _lerp6(CORNER_ANGLES["abajo_izquierda"], CORNER_ANGLES["abajo_derecha"], u)
    return _lerp6(top, bottom, v)


def _lift_delta(scale):
    """Desplazamiento (por articulacion) entre PEN_UP_ANGLES y el centro de
    la zona de dibujo, escalado por `scale` -- se asume aproximadamente
    constante en toda el area (razonable para una hoja chica como A5 frente
    al alcance del brazo). Compartido por pen_up_angles_at/pen_down_angles_at."""
    center_down = corner_angles_at(0.5, 0.5)
    return [(up - c) * scale for up, c in zip(PEN_UP_ANGLES, center_down)]


def pen_up_angles_at(u, v):
    """Angulos con el lapiz LEVANTADO en (u, v), para viajar entre trazos."""
    delta = _lift_delta(PEN_LIFT_SCALE)
    down = corner_angles_at(u, v)
    return [d + off for d, off in zip(down, delta)]


def pen_down_angles_at(u, v):
    """Angulos con el lapiz TOCANDO en (u, v) pero levemente aliviado
    (PEN_DOWN_LIFT_SCALE) para que no presione tanto contra el papel -- se
    usa para los trazos en vez de corner_angles_at() tal cual."""
    delta = _lift_delta(PEN_DOWN_LIFT_SCALE)
    down = corner_angles_at(u, v)
    return [d + off for d, off in zip(down, delta)]


def strokes_to_uv(strokes, aspect_wh=CORNER_ASPECT_WH):
    """Normaliza trazos (en pixeles) a fracciones u,v en [0,1] del
    rectangulo de dibujo, manteniendo la proporcion del dibujo (no lo
    deforma) segun aspect_wh = ancho/alto fisico real del papel."""
    allp = np.vstack(strokes)
    x0, y0 = allp.min(axis=0)
    x1, y1 = allp.max(axis=0)
    w, h = max(x1 - x0, 1.0), max(y1 - y0, 1.0)
    img_aspect = w / h
    if img_aspect > aspect_wh:
        scale_u, scale_v = 1.0, aspect_wh / img_aspect
    else:
        scale_u, scale_v = img_aspect / aspect_wh, 1.0
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    out = []
    for s in strokes:
        u = 0.5 + (s[:, 0] - cx) / w * scale_u
        v = 0.5 + (s[:, 1] - cy) / h * scale_v
        out.append(np.stack([u, v], axis=1))
    return out


def _joint_bounds():
    """Rango (min,max) por articulacion entre las referencias medidas a mano
    (CORNER_ANGLES + PEN_UP_ANGLES) -- cualquier angulo interpolado que se
    salga bastante de esto (ver _check_bounds) es sospechoso, no fue
    verificado como alcanzable contra el robot real."""
    pts = list(CORNER_ANGLES.values()) + [PEN_UP_ANGLES]
    return [min(p[j] for p in pts) for j in range(6)], [max(p[j] for p in pts) for j in range(6)]


_JOINT_MIN, _JOINT_MAX = _joint_bounds()
_BOUNDS_MARGIN = 8.0  # grados de margen antes de recortar (las 5 mediciones no cubren el 100% del area real)


def _check_bounds(angles):
    warn = []
    for j, a in enumerate(angles):
        if a < _JOINT_MIN[j] - _BOUNDS_MARGIN or a > _JOINT_MAX[j] + _BOUNDS_MARGIN:
            warn.append("J%d=%.1f (rango medido %.1f..%.1f)" % (j + 1, a, _JOINT_MIN[j], _JOINT_MAX[j]))
    return warn


def _clamp_angles(angles):
    """Nunca manda un angulo mas alla del rango ya probado fisicamente
    (+ margen) -- evita pedir una pose sin verificar que puede trabar el
    robot (visto: J2/J3 se iban muy fuera de rango cerca de las esquinas de
    abajo al aplicar el desplazamiento de 'lapiz arriba')."""
    return [min(max(a, _JOINT_MIN[j] - _BOUNDS_MARGIN), _JOINT_MAX[j] + _BOUNDS_MARGIN)
            for j, a in enumerate(angles)]


def run_angles(strokes_uv, args):
    """Dibuja mandando angulos de articulacion directo (send_angles),
    interpolados bilinealmente entre CORNER_ANGLES/PEN_UP_ANGLES -- sin
    pasar por cinematica inversa en ningun momento."""
    speed = args.speed

    def goto(mc, angles, label=""):
        warn = _check_bounds(angles)
        angles = _clamp_angles(angles)
        rounded = [round(a, 1) for a in angles]
        tag = "  [recortado, pedia: %s]" % "; ".join(warn) if warn else ""
        print("  %s send_angles(%s, %d)%s" % (label, rounded, speed, tag))
        if mc is not None:
            mc.sync_send_angles(angles, speed, timeout=6)

    mc = None
    if not args.dry_run:
        mc = connect(args.port, args.baud)
        mc.set_color(0, 0, 255)
        print("Angulos actuales:", mc.get_angles())

    total = len(strokes_uv)
    for i, s in enumerate(strokes_uv):
        print("-- trazo %d/%d (%d puntos) --" % (i, total, len(s)))
        u0, v0 = s[0]
        goto(mc, pen_up_angles_at(u0, v0), "arriba")
        goto(mc, pen_down_angles_at(u0, v0), "abajo")
        for u, v in s[1:]:
            goto(mc, pen_down_angles_at(u, v), "abajo")
        goto(mc, pen_up_angles_at(s[-1][0], s[-1][1]), "arriba")

    goto(mc, PEN_UP_ANGLES, "final")
    if mc:
        mc.set_color(0, 255, 0)
    print("Listo.")


# --------------------------------------------------------------------------- #
# 1. Quitar fondo
# --------------------------------------------------------------------------- #
def load_image(path, max_side=800):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit("No se pudo leer la imagen: %s" % path)
    h, w = img.shape[:2]
    s = max_side / max(h, w)
    if s < 1:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return img


def _add_nvidia_dll_dirs():
    """En Windows, los paquetes pip nvidia-cudnn-cuXX/nvidia-cublas-cuXX no dejan
    sus DLL en una carpeta que el loader busque por defecto. Las agregamos a mano
    para que onnxruntime-gpu (usado por rembg) encuentre cudnn64_9.dll etc.
    cuDNN 9 resuelve sus propias sub-DLL (cudnn_graph64_9.dll, ...) con su propio
    cargador interno, que solo mira el PATH del proceso (no os.add_dll_directory),
    asi que hay que anteponerlas tambien ahi."""
    if sys.platform != "win32":
        return
    try:
        import glob
        import sysconfig
        base = os.path.join(sysconfig.get_paths()["purelib"], "nvidia")
        bin_dirs = glob.glob(os.path.join(base, "*", "bin"))
        for bin_dir in bin_dirs:
            if hasattr(os, "add_dll_directory"):
                os.add_dll_directory(bin_dir)
        if bin_dirs:
            os.environ["PATH"] = os.pathsep.join(bin_dirs) + os.pathsep + os.environ.get("PATH", "")
    except Exception:
        pass


def remove_background(img, method="auto"):
    """Devuelve mascara uint8 (255 = persona)."""
    if method in ("auto", "rembg"):
        try:
            _add_nvidia_dll_dirs()
            from rembg import remove  # type: ignore
            rgba = remove(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            mask = (np.asarray(rgba)[..., 3] > 127).astype(np.uint8) * 255
            return _clean_mask(mask)
        except ImportError:
            if method == "rembg":
                raise SystemExit("rembg no esta instalado: pip install rembg")
            print("[info] rembg no instalado, usando GrabCut (menos preciso).")

    h, w = img.shape[:2]
    mx, my = int(w * 0.04), int(h * 0.04)
    rect = (mx, my, w - 2 * mx, h - 2 * my)
    mask = np.zeros((h, w), np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    cv2.grabCut(img, mask, rect, bgd, fgd, 6, cv2.GC_INIT_WITH_RECT)
    mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    return _clean_mask(mask)


def _clean_mask(mask):
    """Rellena huecos, quita motas y se queda con las regiones grandes."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if n <= 1:
        return mask
    biggest = stats[1:, cv2.CC_STAT_AREA].max()
    out = np.zeros_like(mask)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= 0.1 * biggest:
            out[labels == i] = 255
    return out


# --------------------------------------------------------------------------- #
# 2. Bordes -> esqueleto -> trazos
# --------------------------------------------------------------------------- #
def face_region_mask(mask, top_frac=0.2):
    """Aproxima la zona del rostro como la parte superior del recuadro de la persona
    (cabeza), que es donde cae la cara en un retrato tipico (opencv-python 5.x ya no
    trae CascadeClassifier para deteccion real de caras)."""
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return None
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    y_face = y0 + int((y1 - y0) * top_frac)
    region = np.zeros_like(mask)
    region[y0:y_face + 1, x0:x1 + 1] = 255
    return cv2.bitwise_and(region, mask)


def extract_edges(img, mask, detail=1, face_boost=True):
    """Silueta + (opcional) detalle interior. detail: 0 = solo silueta, 1..3 = mas detalle.
    face_boost: aplica ademas un Canny mas fino solo sobre la zona del rostro (rasgos: ojos,
    nariz, boca, cejas), util para retratos donde la cara domina la imagen."""
    silhouette = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    edges = silhouette > 0
    inner = cv2.erode(mask, np.ones((9, 9), np.uint8)) > 0  # evita bordes residuales del fondo
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if detail > 0:
        gray_smooth = cv2.bilateralFilter(gray, 9, 60, 60)
        lo, hi = {1: (60, 140), 2: (40, 100), 3: (25, 70)}[min(detail, 3)]
        edges |= (cv2.Canny(gray_smooth, lo, hi) > 0) & inner

    if face_boost:
        face_mask = face_region_mask(mask)
        if face_mask is not None:
            gray_fine = cv2.bilateralFilter(gray, 5, 40, 40)  # menos suavizado: mas rasgos finos
            edges |= (cv2.Canny(gray_fine, 15, 45) > 0) & inner & (face_mask > 0)

    edges = cv2.morphologyEx(edges.astype(np.uint8) * 255, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return edges > 0


# --------------------------------------------------------------------------- #
# 2a. Boceto ya hecho (p.ej. generado por IA) -> mascara de tinta
# --------------------------------------------------------------------------- #
def sketch_to_mask(img, invert=False, thresh=None):
    """Binariza un boceto/line-art (trazos oscuros sobre fondo claro, tipico de un
    dibujo a lapiz o generado por IA) en una mascara de tinta (255 = trazo). A
    diferencia de extract_edges, aqui el trazo YA es el dibujo: no se buscan bordes
    de una foto, se toma la tinta tal cual para ser lo mas fiel posible al boceto.
    invert: usalo si el boceto tiene trazos claros sobre fondo oscuro.
    thresh: umbral fijo 0-255; si es None se calcula automaticamente con Otsu."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kind = cv2.THRESH_BINARY if invert else cv2.THRESH_BINARY_INV
    if thresh is None:
        _, mask = cv2.threshold(gray, 0, 255, kind + cv2.THRESH_OTSU)
    else:
        _, mask = cv2.threshold(gray, thresh, 255, kind)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))


# --------------------------------------------------------------------------- #
# 2b. Rasgos faciales precisos (MediaPipe Face Landmarker)
# --------------------------------------------------------------------------- #
FACE_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
                   "face_landmarker/float16/latest/face_landmarker.task")
FACE_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                ".cache", "face_landmarker.task")


def _ensure_face_model():
    if not os.path.exists(FACE_MODEL_PATH):
        os.makedirs(os.path.dirname(FACE_MODEL_PATH), exist_ok=True)
        print("[info] descargando modelo de rostro de mediapipe (una sola vez, ~4 MB)...")
        import urllib.request
        urllib.request.urlretrieve(FACE_MODEL_URL, FACE_MODEL_PATH)
    return FACE_MODEL_PATH


def _connections_to_polylines(edges):
    """Agrupa pares (a, b) de un grafo disperso (cadenas de landmarks) en polilineas,
    igual que trace_skeleton pero para el grafo pequeno de MediaPipe en vez de pixeles."""
    adj = {}
    for a, b in edges:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)

    used = set()

    def walk(start, nxt):
        path = [start, nxt]
        used.add(frozenset((start, nxt)))
        prev, cur = start, nxt
        while len(adj[cur]) == 2:
            cand = [q for q in adj[cur] if q != prev and frozenset((cur, q)) not in used]
            if not cand:
                break
            q = cand[0]
            used.add(frozenset((cur, q)))
            path.append(q)
            prev, cur = cur, q
            if q == start:
                break
        return path

    paths = []
    nodes = [n for n, nb in adj.items() if len(nb) != 2]  # extremos y uniones (p.ej. ojo abierto)
    for n in nodes:
        for q in adj[n]:
            if frozenset((n, q)) not in used:
                paths.append(walk(n, q))
    for n in adj:  # lazos cerrados sin nodos de grado != 2 (p.ej. iris)
        for q in adj[n]:
            if frozenset((n, q)) not in used:
                paths.append(walk(n, q))
    return paths


def face_landmark_strokes(img):
    """Usa MediaPipe Face Landmarker para trazar ojos, cejas, iris, nariz y labios con
    precision anatomica. Canny no distingue un ojo de una arruga; los landmarks si.
    Devuelve lista de arrays (x, y) en pixeles, o [] si no hay mediapipe, falla la
    descarga del modelo, o no se detecta ninguna cara."""
    try:
        import mediapipe as mp
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.core.base_options import BaseOptions
    except ImportError:
        return []
    try:
        model_path = _ensure_face_model()
    except Exception as e:
        print("[info] no se pudo obtener el modelo de rostro de mediapipe:", e)
        return []

    h, w = img.shape[:2]
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    options = vision.FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path=model_path),
                                           num_faces=1)
    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        result = landmarker.detect(mp_img)
    if not result.face_landmarks:
        return []

    pts = [(p.x * w, p.y * h) for p in result.face_landmarks[0]]
    C = vision.FaceLandmarksConnections
    feature_sets = [C.FACE_LANDMARKS_LEFT_EYE, C.FACE_LANDMARKS_RIGHT_EYE,
                     C.FACE_LANDMARKS_LEFT_EYEBROW, C.FACE_LANDMARKS_RIGHT_EYEBROW,
                     C.FACE_LANDMARKS_LEFT_IRIS, C.FACE_LANDMARKS_RIGHT_IRIS,
                     C.FACE_LANDMARKS_LIPS, C.FACE_LANDMARKS_NOSE]
    strokes = []
    for conns in feature_sets:
        edges = [(c.start, c.end) for c in conns]
        for path in _connections_to_polylines(edges):
            arr = np.array([pts[i] for i in path], np.float64)
            if len(arr) >= 2:
                strokes.append(arr)
    return strokes


def thin(binary):
    """Adelgazamiento Zhang-Suen vectorizado con numpy (lineas de 1 px)."""
    img = np.pad(binary.astype(np.uint8), 1)
    while True:
        changed = False
        for step in (0, 1):
            P = [img[:-2, 1:-1], img[:-2, 2:], img[1:-1, 2:], img[2:, 2:],
                 img[2:, 1:-1], img[2:, :-2], img[1:-1, :-2], img[:-2, :-2]]  # N NE E SE S SW W NW
            B = sum(p.astype(np.int16) for p in P)
            A = sum(((P[i] == 0) & (P[(i + 1) % 8] == 1)).astype(np.int16) for i in range(8))
            if step == 0:
                c = ((P[0] * P[2] * P[4]) == 0) & ((P[2] * P[4] * P[6]) == 0)
            else:
                c = ((P[0] * P[2] * P[6]) == 0) & ((P[0] * P[4] * P[6]) == 0)
            rem = (img[1:-1, 1:-1] == 1) & (B >= 2) & (B <= 6) & (A == 1) & c
            if rem.any():
                img[1:-1, 1:-1][rem] = 0
                changed = True
        if not changed:
            break
    return img[1:-1, 1:-1] > 0


def trace_skeleton(skel):
    """Convierte un esqueleto de 1 px en polilineas [(x, y), ...] en pixeles."""
    ys, xs = np.nonzero(skel)
    pts = set(zip(xs.tolist(), ys.tolist()))
    offs = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, -1), (-1, 1), (1, 1)]

    def nbrs(p):
        out = []
        for dx, dy in offs:
            q = (p[0] + dx, p[1] + dy)
            if q not in pts:
                continue
            if dx and dy and ((p[0] + dx, p[1]) in pts or (p[0], p[1] + dy) in pts):
                continue  # diagonal redundante: ya conectado por un camino ortogonal
            out.append(q)
        return out

    adj = {p: nbrs(p) for p in pts}
    used = set()  # aristas usadas (frozenset de 2 pixeles)

    def walk(start, nxt):
        path = [start, nxt]
        used.add(frozenset((start, nxt)))
        prev, cur = start, nxt
        while len(adj[cur]) == 2:
            cand = [q for q in adj[cur] if q != prev and frozenset((cur, q)) not in used]
            if not cand:
                break
            q = cand[0]
            used.add(frozenset((cur, q)))
            path.append(q)
            prev, cur = cur, q
            if q == start:
                break
        return path

    paths = []
    nodes = [p for p in pts if len(adj[p]) != 2]  # extremos y uniones
    for n in nodes:
        for q in adj[n]:
            if frozenset((n, q)) not in used:
                paths.append(walk(n, q))
    for p in pts:  # lazos cerrados sin nodos
        for q in adj[p]:
            if frozenset((p, q)) not in used:
                paths.append(walk(p, q))
    return paths


def build_strokes(edges, min_len_px=12, eps_px=1.2):
    skel = thin(edges)
    strokes = []
    for path in trace_skeleton(skel):
        arr = np.array(path, np.float32).reshape(-1, 1, 2)
        length = cv2.arcLength(arr, False)
        if length < min_len_px:
            continue
        simp = cv2.approxPolyDP(arr, eps_px, False).reshape(-1, 2)
        if len(simp) >= 2:
            strokes.append(simp.astype(np.float64))
    return strokes


def order_strokes(strokes):
    """Vecino mas cercano, invirtiendo trazos si conviene (menos viajes con lapiz arriba)."""
    remaining = list(strokes)
    if not remaining:
        return []
    ordered, pos = [], np.zeros(2)
    while remaining:
        best, best_d, best_rev = 0, 1e18, False
        for i, s in enumerate(remaining):
            d0 = np.hypot(*(s[0] - pos))
            d1 = np.hypot(*(s[-1] - pos))
            if d0 < best_d:
                best, best_d, best_rev = i, d0, False
            if d1 < best_d:
                best, best_d, best_rev = i, d1, True
        s = remaining.pop(best)
        s = s[::-1] if best_rev else s
        ordered.append(s)
        pos = s[-1]
    return ordered


# --------------------------------------------------------------------------- #
# 3. Pixeles -> mm -> G-code
# --------------------------------------------------------------------------- #
def strokes_to_mm(strokes, img_shape, cfg):
    allp = np.vstack(strokes)
    x0, y0 = allp.min(axis=0)
    x1, y1 = allp.max(axis=0)
    w, h = max(x1 - x0, 1.0), max(y1 - y0, 1.0)
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0  # el dibujo se ajusta al area usando solo la persona
    scale = min(cfg["width"] / w, cfg["height"] / h)  # mm por pixel
    out = []
    for s in strokes:
        u = (s[:, 0] - cx) * scale            # derecha de la imagen
        v = (cy - s[:, 1]) * scale            # arriba de la imagen
        x = cfg["center_x"] + v               # arriba -> lejos del robot
        y = cfg["center_y"] - u               # derecha -> -Y
        out.append(np.stack([x, y], axis=1))
    return out


def strokes_to_gcode(strokes_mm, cfg):
    L = ["; Generado por draw_person.py", "G21 ; mm", "G90 ; absoluto",
         "G0 Z%.2f" % cfg["z_up"]]
    for s in strokes_mm:
        L.append("G0 X%.2f Y%.2f" % tuple(s[0]))
        L.append("G1 Z%.2f F%d" % (cfg["z_down"], cfg["feed"]))
        for p in s[1:]:
            L.append("G1 X%.2f Y%.2f F%d" % (p[0], p[1], cfg["feed"]))
        L.append("G0 Z%.2f" % cfg["z_up"])
    L.append("G0 X%.2f Y%.2f" % (cfg["center_x"], cfg["center_y"]))
    L.append("M2 ; fin")
    return "\n".join(L) + "\n"


def parse_gcode(text):
    """Devuelve lista de movimientos (cmd, x, y, z, f). Solo G0/G1 (modal)."""
    x = y = z = None
    f = None
    moves = []
    for raw in text.splitlines():
        line = raw.split(";")[0].strip().upper()
        if not line:
            continue
        m = re.match(r"G0?([01])\b", line)
        if not m:
            continue
        for ax, val in re.findall(r"([XYZF])(-?\d+\.?\d*)", line):
            val = float(val)
            if ax == "X": x = val
            elif ax == "Y": y = val
            elif ax == "Z": z = val
            else: f = val
        moves.append((int(m.group(1)), x, y, z, f))
    return moves


# --------------------------------------------------------------------------- #
# 4. Preview
# --------------------------------------------------------------------------- #
def preview(gcode_text, cfg=None, img=None, mask=None, save=None, show=True, mask_label="Sin fondo"):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    moves = parse_gcode(gcode_text)
    z_vals = [m[3] for m in moves if m[3] is not None]
    z_hi, z_lo = max(z_vals), min(z_vals)
    mid = (z_hi + z_lo) / 2.0

    pen_down_len = travel_len = 0.0
    n_strokes = 0
    segs_draw, segs_travel = [], []
    cur = None
    z = z_hi
    for cmd, x, y, zz, f in moves:
        if zz is not None:
            if z > mid >= zz:
                n_strokes += 1
            z = zz
        if x is None or y is None:
            continue
        if cur is not None and (x, y) != cur:
            d = math.hypot(x - cur[0], y - cur[1])
            if z <= mid and cmd == 1:
                segs_draw.append(((cur[1], cur[0]), (y, x)))
                pen_down_len += d
            else:
                segs_travel.append(((cur[1], cur[0]), (y, x)))
                travel_len += d
        cur = (x, y)

    ncols = 3 if img is not None else 1
    fig, axs = plt.subplots(1, ncols, figsize=(5 * ncols + 1, 6.5))
    axs = np.atleast_1d(axs)
    k = 0
    if img is not None:
        axs[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); axs[0].set_title("Original"); axs[0].axis("off")
        cut = img.copy()
        cut[mask == 0] = 255
        axs[1].imshow(cv2.cvtColor(cut, cv2.COLOR_BGR2RGB)); axs[1].set_title(mask_label); axs[1].axis("off")
        k = 2
    ax = axs[k]
    from matplotlib.collections import LineCollection
    ax.add_collection(LineCollection(segs_travel, colors="tab:red", linewidths=0.4, linestyles=":", alpha=0.5,
                                     label="viaje (lapiz arriba)"))
    ax.add_collection(LineCollection(segs_draw, colors="black", linewidths=1.1, label="dibujo (lapiz abajo)"))
    if cfg:
        hw, hh = cfg["width"] / 2, cfg["height"] / 2
        ax.plot([cfg["center_y"] - hw, cfg["center_y"] + hw, cfg["center_y"] + hw, cfg["center_y"] - hw,
                 cfg["center_y"] - hw],
                [cfg["center_x"] - hh, cfg["center_x"] - hh, cfg["center_x"] + hh, cfg["center_x"] + hh,
                 cfg["center_x"] - hh], color="tab:blue", lw=0.8, ls="--", label="area de dibujo")
    ax.autoscale()
    ax.invert_xaxis()  # +Y (izquierda del robot) a la izquierda, como se ve desde la base
    ax.set_aspect("equal")
    ax.set_xlabel("Y (mm)"); ax.set_ylabel("X (mm)  [lejos del robot ->]")
    est_min = (pen_down_len + travel_len) / max(cfg["feed"] if cfg else DEFAULTS["feed"], 1)
    ax.set_title("Preview robot: %d trazos, %.0f mm dibujo, %.0f mm viaje\n(~%.1f min a la velocidad teorica)"
                 % (n_strokes, pen_down_len, travel_len, est_min),
                 fontsize=9)
    ax.legend(loc="lower right", fontsize=7)
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=130)
        print("Preview guardado en", save)
    if show:
        plt.show()
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 5. Robot
# --------------------------------------------------------------------------- #
def connect(port, baud):
    try:
        from pymycobot import MyCobot280
    except ImportError:
        sys.path.insert(0, r"C:\Users\sagp7\Documents\pymycobot")
        from pymycobot import MyCobot280
    mc = MyCobot280(port, baud)
    mc.power_on()
    time.sleep(0.5)
    return mc


def run_gcode(text, args):
    moves = parse_gcode(text)
    rx, ry, rz = args.orientation
    speed = args.speed
    cur = None

    def goto(mc, x, y, z, mode):
        coords = [x, y, z, rx, ry, rz]
        if mc is None:
            return
        mc.sync_send_coords(coords, speed, mode, timeout=6)

    mc = None
    if not args.dry_run:
        mc = connect(args.port, args.baud)
        mc.set_color(0, 0, 255)
        print("Posicion actual:", mc.get_coords())
    total = len(moves)
    for i, (cmd, x, y, z, f) in enumerate(moves):
        cx, cy, cz = (cur if cur else (x, y, z))
        x = cx if x is None else x
        y = cy if y is None else y
        z = cz if z is None else z
        if z is None or x is None or y is None:
            continue
        # G0 (viaje / cambio de Z): mode 0 (angular, rapido).  G1 en plano: mode 1 (lineal, recto).
        goto(mc, x, y, z, 1 if cmd == 1 else 0)
        cur = (x, y, z)
        if i % 25 == 0:
            print("  %d/%d" % (i, total))
    if mc:
        mc.set_color(0, 255, 0)
    print("Listo.")


def confirm(msg):
    return input(msg + " [s/N]: ").strip().lower() in ("s", "si", "y", "yes")


def corners_set(name, args):
    """Joguea UNA esquina a mano (como `pose`) e imprime sus coords para
    pegar en CORNERS."""
    mc = connect(args.port, args.baud)
    mc.release_all_servos()
    input("Servos liberados. Mueve el lapiz hasta la esquina '%s' del papel y pulsa Enter..." % name)
    coords = mc.get_coords()
    print("\n%s = %s" % (name, coords))
    print("Copia esta linea dentro de CORNERS en draw_person.py:")
    print('    "%s": %s,' % (name, coords))


def corners_goto(args):
    """Recorre las 4 esquinas calibradas a mano (CORNERS) con el lapiz
    arriba, cada una con su propia orientacion (a diferencia de `corners`,
    que usa una sola orientacion fija para todo el rectangulo)."""
    missing = [name for name, c in CORNERS.items() if c is None]
    if missing:
        raise SystemExit("Faltan calibrar: %s (usa `corners-set <nombre> --port ...`)" % ", ".join(missing))

    safe_z = max(c[2] for c in CORNERS.values()) + 15  # por encima de la mas alta, margen de seguridad
    print("Esquinas calibradas (Z de viaje seguro: %.1f):" % safe_z)
    for name, c in CORNERS.items():
        print("  %s: %s" % (name, c))

    if not args.dry_run and not confirm("Voy a recorrer las 4 esquinas con el lapiz arriba. Continuar?"):
        return

    mc = None
    if not args.dry_run:
        mc = connect(args.port, args.baud)
        mc.set_color(0, 0, 255)
    for name, (x, y, _z, rx, ry, rz) in CORNERS.items():
        print("-> %s" % name)
        if mc is not None:
            mc.sync_send_coords([x, y, safe_z, rx, ry, rz], args.speed, 1, timeout=6)
    if mc is not None:
        mc.set_color(0, 255, 0)
    print("Listo.")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def add_pose_args(p):
    p.add_argument("--orientation", nargs=3, type=float, default=DEFAULTS["orientation"],
                   metavar=("RX", "RY", "RZ"), help="orientacion del efector con el lapiz hacia abajo")
    p.add_argument("--z-down", type=float, default=DEFAULTS["z_down"])
    p.add_argument("--z-up", type=float, default=DEFAULTS["z_up"])
    p.add_argument("--center", nargs=2, type=float, default=(DEFAULTS["center_x"], DEFAULTS["center_y"]),
                   metavar=("X", "Y"))
    p.add_argument("--width", type=float, default=DEFAULTS["width"], help="ancho maximo del dibujo (mm)")
    p.add_argument("--height", type=float, default=DEFAULTS["height"], help="alto maximo del dibujo (mm)")


def add_robot_args(p):
    p.add_argument("--port", default="COM3")
    p.add_argument("--baud", type=int, default=115200)
    p.add_argument("--speed", type=int, default=30, help="velocidad del robot 1-100")
    p.add_argument("--dry-run", action="store_true", help="no conecta con el robot")
    p.add_argument("--orientation", nargs=3, type=float, default=DEFAULTS["orientation"],
                   metavar=("RX", "RY", "RZ"))


def cfg_from(args):
    return dict(center_x=args.center[0], center_y=args.center[1], width=args.width, height=args.height,
                z_down=args.z_down, z_up=args.z_up, orientation=tuple(args.orientation), feed=DEFAULTS["feed"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("process", help="imagen -> G-code + preview")
    p.add_argument("image")
    p.add_argument("-o", "--out", help="archivo .gcode de salida (por defecto junto a la imagen)")
    p.add_argument("--detail", type=int, default=1, choices=[0, 1, 2, 3],
                   help="0 = solo silueta, 1-3 = mas detalle interior (default 1)")
    p.add_argument("--bg", default="auto", choices=["auto", "rembg", "grabcut"])
    p.add_argument("--no-face-boost", action="store_true",
                   help="no reforzar el detalle (ojos, nariz, boca, cejas) en la cara detectada")
    p.add_argument("--min-len", type=float, default=12, help="descarta trazos menores a N px")
    p.add_argument("--simplify", type=float, default=1.2, help="tolerancia de simplificacion (px)")
    p.add_argument("--no-show", action="store_true", help="no abrir ventana, solo guardar PNG")
    add_pose_args(p)

    p = sub.add_parser("sketch", help="boceto/line-art ya hecho (p.ej. generado por IA) -> G-code + preview")
    p.add_argument("image")
    p.add_argument("-o", "--out", help="archivo .gcode de salida (por defecto junto a la imagen)")
    p.add_argument("--threshold", type=int, default=None,
                   help="umbral fijo 0-255 para binarizar (por defecto: automatico con Otsu)")
    p.add_argument("--invert", action="store_true",
                   help="usa esto si el boceto tiene trazos claros sobre fondo oscuro")
    p.add_argument("--min-len", type=float, default=6, help="descarta trazos menores a N px")
    p.add_argument("--simplify", type=float, default=1.0, help="tolerancia de simplificacion (px)")
    p.add_argument("--no-show", action="store_true", help="no abrir ventana, solo guardar PNG")
    add_pose_args(p)

    p = sub.add_parser("sketch-angles",
                        help="boceto -> dibuja directo por angulos de articulacion (CORNER_ANGLES, sin IK)")
    p.add_argument("image")
    p.add_argument("--threshold", type=int, default=None,
                   help="umbral fijo 0-255 para binarizar (por defecto: automatico con Otsu)")
    p.add_argument("--invert", action="store_true",
                   help="usa esto si el boceto tiene trazos claros sobre fondo oscuro")
    p.add_argument("--min-len", type=float, default=6, help="descarta trazos menores a N px")
    p.add_argument("--simplify", type=float, default=1.0, help="tolerancia de simplificacion (px)")
    p.add_argument("--port", default="COM3"); p.add_argument("--baud", type=int, default=115200)
    p.add_argument("--speed", type=int, default=ANGLE_SPEED, help="velocidad 1-100")
    p.add_argument("--dry-run", action="store_true", help="no conecta con el robot, solo loguea los angulos")

    p = sub.add_parser("process-angles",
                        help="foto -> dibuja directo por angulos de articulacion (CORNER_ANGLES, sin IK)")
    p.add_argument("image")
    p.add_argument("--detail", type=int, default=1, choices=[0, 1, 2, 3],
                   help="0 = solo silueta, 1-3 = mas detalle interior (default 1)")
    p.add_argument("--bg", default="auto", choices=["auto", "rembg", "grabcut"])
    p.add_argument("--no-face-boost", action="store_true",
                   help="no reforzar el detalle (ojos, nariz, boca, cejas) en la cara detectada")
    p.add_argument("--min-len", type=float, default=12, help="descarta trazos menores a N px")
    p.add_argument("--simplify", type=float, default=1.2, help="tolerancia de simplificacion (px)")
    p.add_argument("--port", default="COM3"); p.add_argument("--baud", type=int, default=115200)
    p.add_argument("--speed", type=int, default=ANGLE_SPEED, help="velocidad 1-100")
    p.add_argument("--dry-run", action="store_true", help="no conecta con el robot, solo loguea los angulos")

    p = sub.add_parser("preview", help="ver un .gcode existente")
    p.add_argument("gcode")

    p = sub.add_parser("pose", help="leer coords actuales del robot (liberando servos para moverlo a mano)")
    p.add_argument("--port", default="COM3"); p.add_argument("--baud", type=int, default=115200)

    p = sub.add_parser("corners", help="recorre las esquinas del area (DEFAULTS) con el lapiz arriba")
    p.add_argument("gcode"); add_robot_args(p)

    p = sub.add_parser("corners-set", help="calibra UNA esquina a mano (como `pose`) e imprime sus coords")
    p.add_argument("corner", choices=list(CORNERS.keys()))
    p.add_argument("--port", default="COM3"); p.add_argument("--baud", type=int, default=115200)

    p = sub.add_parser("corners-goto",
                        help="recorre las 4 esquinas calibradas a mano (CORNERS), con el lapiz arriba")
    add_robot_args(p)

    p = sub.add_parser("run", help="dibuja el G-code con el robot")
    p.add_argument("gcode"); add_robot_args(p)

    args = ap.parse_args()

    if args.cmd == "process":
        cfg = cfg_from(args)
        img = load_image(args.image)
        mask = remove_background(img, args.bg)
        edges = extract_edges(img, mask, args.detail, face_boost=not args.no_face_boost)
        strokes = build_strokes(edges, args.min_len, args.simplify)
        if not args.no_face_boost:
            face_strokes = face_landmark_strokes(img)
            if face_strokes:
                print("[info] mediapipe: %d trazos de rasgos faciales (ojos, cejas, nariz, boca)"
                      % len(face_strokes))
            strokes += face_strokes
        strokes = order_strokes(strokes)
        if not strokes:
            raise SystemExit("No se encontraron contornos. Prueba --detail 2 o --bg rembg.")
        gcode = strokes_to_gcode(strokes_to_mm(strokes, img.shape, cfg), cfg)
        base = os.path.splitext(args.out or args.image)[0]
        out = args.out or base + ".gcode"
        with open(out, "w") as fh:
            fh.write(gcode)
        print("G-code: %s  (%d trazos, %d lineas)" % (out, len(strokes), gcode.count("\n")))
        preview(gcode, cfg, img, mask, save=base + "_preview.png", show=not args.no_show)

    elif args.cmd == "sketch":
        cfg = cfg_from(args)
        img = load_image(args.image)
        mask = sketch_to_mask(img, invert=args.invert, thresh=args.threshold)
        strokes = order_strokes(build_strokes(mask > 0, args.min_len, args.simplify))
        if not strokes:
            raise SystemExit("No se encontraron trazos. Prueba --invert o --threshold N.")
        gcode = strokes_to_gcode(strokes_to_mm(strokes, img.shape, cfg), cfg)
        base = os.path.splitext(args.out or args.image)[0]
        out = args.out or base + ".gcode"
        with open(out, "w") as fh:
            fh.write(gcode)
        print("G-code: %s  (%d trazos, %d lineas)" % (out, len(strokes), gcode.count("\n")))
        preview(gcode, cfg, img, mask, save=base + "_preview.png", show=not args.no_show,
                mask_label="Trazos detectados")

    elif args.cmd == "sketch-angles":
        img = load_image(args.image)
        mask = sketch_to_mask(img, invert=args.invert, thresh=args.threshold)
        strokes = order_strokes(build_strokes(mask > 0, args.min_len, args.simplify))
        if not strokes:
            raise SystemExit("No se encontraron trazos. Prueba --invert o --threshold N.")
        strokes_uv = strokes_to_uv(strokes)
        print("%d trazos. Modo angulos (sin IK): CORNER_ANGLES + PEN_UP_ANGLES." % len(strokes))
        if not args.dry_run and not confirm("Papel colocado. Continuar?"):
            return
        run_angles(strokes_uv, args)

    elif args.cmd == "process-angles":
        img = load_image(args.image)
        mask = remove_background(img, args.bg)
        edges = extract_edges(img, mask, args.detail, face_boost=not args.no_face_boost)
        strokes = build_strokes(edges, args.min_len, args.simplify)
        if not args.no_face_boost:
            face_strokes = face_landmark_strokes(img)
            if face_strokes:
                print("[info] mediapipe: %d trazos de rasgos faciales (ojos, cejas, nariz, boca)"
                      % len(face_strokes))
            strokes += face_strokes
        strokes = order_strokes(strokes)
        if not strokes:
            raise SystemExit("No se encontraron contornos. Prueba --detail 2 o --bg rembg.")
        strokes_uv = strokes_to_uv(strokes)
        print("%d trazos. Modo angulos (sin IK): CORNER_ANGLES + PEN_UP_ANGLES." % len(strokes))
        if not args.dry_run and not confirm("Papel colocado. Continuar?"):
            return
        run_angles(strokes_uv, args)

    elif args.cmd == "preview":
        preview(open(args.gcode).read())

    elif args.cmd == "pose":
        mc = connect(args.port, args.baud)
        mc.release_all_servos()
        input("Servos liberados. Mueve el brazo con el lapiz tocando el papel y pulsa Enter...")
        print("coords [x, y, z, rx, ry, rz] =", mc.get_coords())
        print("Usa x,y para --center, z para --z-down y rx ry rz para --orientation.")

    elif args.cmd == "corners-set":
        corners_set(args.corner, args)

    elif args.cmd == "corners-goto":
        corners_goto(args)

    elif args.cmd in ("corners", "run"):
        text = open(args.gcode).read()
        moves = parse_gcode(text)
        xs = [m[1] for m in moves if m[1] is not None]
        ys = [m[2] for m in moves if m[2] is not None]
        zs = [m[3] for m in moves if m[3] is not None]
        print("Area: X %.0f..%.0f  Y %.0f..%.0f  | Z lapiz abajo=%.1f  arriba=%.1f  | orient=%s  vel=%d"
              % (min(xs), max(xs), min(ys), max(ys), min(zs), max(zs), tuple(args.orientation), args.speed))
        if not args.dry_run and not confirm("Papel colocado y Z/orientacion calibrados. Continuar?"):
            return
        if args.cmd == "corners":
            corners = [(min(xs), min(ys)), (min(xs), max(ys)), (max(xs), max(ys)), (max(xs), min(ys)),
                       (min(xs), min(ys))]
            text = "G0 Z%.2f\n" % max(zs) + "".join("G0 X%.2f Y%.2f\n" % c for c in corners)
        run_gcode(text, args)


if __name__ == "__main__":
    main()
