#!/usr/bin/python
# -*- coding:utf-8 -*-
# Puente entre dos modulos, incluidos en lib/:
#   - draw_person.py (originalmente de C:\Users\prosopon\Documents\myCobot280):
#     extraccion de trazos desde una imagen (boceto -> mascara -> esqueleto ->
#     trazos).
#   - gcode_draw_lib.py (originalmente de pymycobot/demo/myCobot_280_demo):
#     calibracion/ejecucion en el robot real, depurada extensamente contra
#     fricciones/skew de esta maquina.
#
# Se copiaron a lib/ para que el repositorio funcione solo al clonarlo. Si
# se quiere usar otra copia (ej. la de su proyecto original), DRAW_PERSON_DIR
# y GCODE_LIB_DIR apuntan a la carpeta de cada uno.

import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

DRAW_PERSON_DIR = os.environ.get('DRAW_PERSON_DIR', os.path.join(HERE, 'lib'))
GCODE_LIB_DIR = os.environ.get('GCODE_LIB_DIR', os.path.join(HERE, 'lib'))

for _p in (DRAW_PERSON_DIR, os.path.abspath(GCODE_LIB_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import draw_person as dp          # noqa: E402  (extraccion de trazos)
import gcode_draw_lib as lib      # noqa: E402  (calibracion/ejecucion real)
import j1_unidireccional as j1u   # noqa: E402  (base en un solo sentido, ver ese modulo)
import eye_styles                 # noqa: E402  (estilos de ojo para los retratos)

# Interruptor temporal: con False, simplify_for_robot() devuelve los moves
# crudos de la extraccion tal cual (sin Douglas-Peucker ni filtrado de trazos
# cortos, sin priorizacion de cara). OJO: esto reintroduce el riesgo de
# friccion en pasos muy chicos que motivo toda la logica de mas abajo -- es
# para probar/comparar, no para dejarlo asi de forma permanente.
SIMPLIFICATION_ENABLED = True

# Con False se usa el sistema "simple" de draw_person.py -- un solo
# epsilon/min_len global, sin las 3 zonas de cara/ojos-nariz-boca. Se probo
# ese modo simple a pedido, pero un solo epsilon global no puede satisfacer a
# la vez "cara detallada" y "fondo/pelo con trazos largos y continuos": vuelto
# a True porque son objetivos opuestos que solo las zonas separan.
PRIORITIZE_FACE_ENABLED = True

# ---------------------------------------------------------------------------
# 1. Imagen (boceto) -> trazos en pixeles -> "moves" [(x, y, pluma_abajo), ...]
# ---------------------------------------------------------------------------
# Parametros de draw_person.build_strokes para la etapa de extraccion desde
# el boceto (mismos defaults que su comando `sketch`).
SKETCH_MIN_LEN_PX = 3
SKETCH_SIMPLIFY_PX = 1.0

# Zonas de tinta mucho mas gruesas que una linea normal del boceto (iris
# relleno, cejas o sombras pintadas) se dibujan por su CONTORNO en vez de por
# su linea central: el esqueleto de una mancha es un rombo/cruz que no se
# parece a nada (caso real: el iris relleno + parpado de un boceto daban dos
# rombos unidos por una raya en vez de un ojo). "Gruesa" = el radio de tinta
# supera THICK_FACTOR veces el de las lineas MAS gruesas normales de ESE
# boceto (percentil 90 del radio sobre el esqueleto), o THICK_MIN_PX. Con el
# radio tipico (mediana) en vez del p90, en un boceto de trazo fino y mucho
# sombreado el sombreado denso se volvia decenas de manchitas contorneadas.
# Medido en 4 bocetos reales (a 800px): p90 de 2.0-2.8px, iris de 7-8px.
THICK_FACTOR = 1.8
THICK_MIN_PX = 4.0


def _extract_strokes(mask, min_len_px, eps_px):
    """Mascara de tinta -> trazos: esqueleto para las lineas, contorno para
    las manchas gruesas (ver THICK_FACTOR)."""
    import cv2
    ink = mask > 0
    dist = cv2.distanceTransform(ink.astype(np.uint8), cv2.DIST_L2, 5)
    skel = dp.thin(ink)
    line_r = float(np.percentile(dist[skel > 0], 90)) if np.any(skel) else 1.0
    thick_r = max(THICK_MIN_PX, THICK_FACTOR * line_r)
    core = dist > thick_r
    if not core.any():
        return dp.build_strokes(ink, min_len_px, eps_px)

    # la mancha completa = todo lo que esta a un radio grueso del nucleo
    k = int(np.ceil(thick_r)) + 1
    grown = cv2.dilate(core.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1)))
    thick = (grown > 0) & ink
    strokes = dp.build_strokes(ink & ~thick, min_len_px, eps_px)
    contours, _ = cv2.findContours(thick.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for c in contours:
        if cv2.arcLength(c, True) < max(min_len_px, 4 * thick_r):
            continue
        pts = cv2.approxPolyDP(c, eps_px, True).reshape(-1, 2).astype(np.float64)
        if len(pts) >= 3:
            strokes.append(np.vstack([pts, pts[:1]]))  # cerrado
    return strokes

# Objetivo en mm REALES para el paso entre puntos consecutivos de un mismo
# trazo, una vez escalado al area de dibujo calibrada. Se aprendio en esta
# conversacion (ver notas en gcode_draw_lib.py) que por debajo de ~2-3mm el
# esfero se pega al papel por friccion -- SIMPLIFY_TARGET_MM apunta bien por
# encima de eso. MIN_STROKE_TARGET_MM descarta trazos/hatching demasiado
# cortos para dibujarse de forma confiable a este tamano de hoja.
SIMPLIFY_TARGET_MM = 6.0
MIN_STROKE_TARGET_MM = 3.0

# Niveles de detalle expuestos en la interfaz (ver DETAIL_LEVELS mas abajo):
# mas simplificacion = trazos mas largos y rectos = pasos mas grandes = mas
# seguro contra la friccion, pero menos fiel al boceto. Los valores de
# "medio" son los que se afinaron a mano en esta conversacion.
DETAIL_LEVELS = {
    'bajo': dict(simplify_target_mm=10.0, min_stroke_target_mm=6.0,
                 sketch_threshold=None, sketch_min_len_px=SKETCH_MIN_LEN_PX),
    'medio': dict(simplify_target_mm=SIMPLIFY_TARGET_MM, min_stroke_target_mm=MIN_STROKE_TARGET_MM,
                  sketch_threshold=None, sketch_min_len_px=SKETCH_MIN_LEN_PX),
    # "alto" bajo bastante mas que antes (era 3.0/1.5): con la priorizacion de
    # cara/ojos-nariz-boca de abajo, la zona de rasgos finos termina bien por
    # debajo de esto igual (ver CORE_FEATURE_DETAIL_BOOST), asi que "alto" ya
    # no necesita cargar solo todo el peso del detalle.
    #
    # sketch_threshold (vs el Otsu automatico, que en un boceto real de
    # Gemini eligio 168): el detalle fino de ojos/nariz/boca en un boceto de
    # Gemini suele venir dibujado con trazos MUY claros/finos, que el umbral
    # automatico descarta directamente al binarizar -- ANTES de que llegue a
    # simplificarse. Ningun ajuste de simplificacion puede recuperar tinta
    # que nunca entro a la mascara. Confirmado con un boceto real: subir el
    # umbral recupera cejas/parpados/dientes que "alto" antes perdia. Subido
    # de 225 a 235 (+ sketch_min_len_px mas chico) para que un retrato tipico
    # ronde ~700-800 puntos en vez de ~650.
    'alto': dict(simplify_target_mm=1.8, min_stroke_target_mm=0.9,
                 sketch_threshold=235, sketch_min_len_px=2),
}
DEFAULT_DETAIL = 'medio'

# Piso de seguridad: sin importar cuanto multipliquen los factores de detalle
# de abajo, nunca se pide una tolerancia por debajo de esto. Es el umbral
# aprendido en esta conversacion por debajo del cual el esfero se pega al
# papel por friccion en vez de deslizar -- por eso "alto" + la zona de ojos/
# nariz/boca no puede irse a 0, por mas detalle que se pida.
MIN_SAFE_STEP_MM = 1.2

# Si dos trazos separados quedan a esta distancia (mm reales) o menos, se
# conectan con la pluma abajo en vez de levantarla para viajar (ver
# _flat_strokes_to_moves): cada levantada es un aterrizaje nuevo, y en el
# brazo real la punta del esfero patina un poco al aterrizar por la presion
# del resorte, dejando una "V"/rasguno visible en vez de una linea limpia.
# Con un boceto muy fragmentado (pelo, hatching) eso se ve como decenas de
# rasgunos en vez de trazos limpios. Chico a proposito para no dibujar lineas
# largas que no estaban en el boceto -- solo conecta lo que ya estaba cerca.
JOIN_GAP_MM = 3.0

# Tope de cuantos trazos seguidos se pueden encadenar sin una levantada real
# (ver _flat_strokes_to_moves). gcode_draw_lib.draw_moves() NO confirma
# posicion en cada punto (is_in_position resulto muy poco confiable, ver su
# propio comentario junto a ASSUMED_SPEED_MM_S): se manda cada comando y se
# espera una pausa calculada por distancia/velocidad asumida, confiando en
# que el brazo ya llego. Esa aproximacion ya habia dado zigzags/cruces con
# retratos de cientos de puntos muy seguidos (por eso existe MIN_PACING) --
# encadenar trazos sin ninguna levantada real empeora exactamente ese riesgo
# (mas comandos seguidos sin una pausa de viaje de por medio que ayude al
# brazo a "asentarse"). Cortar la cadena cada tantos trazos fuerza una
# levantada/aterrizaje real (con su propia pausa de viaje) que actua como
# punto de resincronizacion, aunque el trazo siguiente este cerca.
JOIN_MAX_CHAIN = 2


def _flat_strokes_to_moves(strokes, join_gap_px=0.0, join_max_chain=JOIN_MAX_CHAIN):
    """[(x,y), ...] por trazo, YA en el sistema de coordenadas final (sin
    invertir Y) -> [(x, y, pluma_abajo), ...] plano.

    join_gap_px: si el siguiente trazo empieza a esta distancia o menos del
    final del trazo anterior, NO se levanta la pluma para viajar -- se sigue
    con la pluma abajo directo al siguiente trazo (un solo trazo continuo).
    Sin esto, cada trazo separado es un aterrizaje de pluma nuevo, y en el
    brazo real cada aterrizaje hace que la punta patine un poco hacia arriba
    por la presion del resorte del esfero, dejando una "V" o rasguno visible
    -- con boceto muy fragmentado (muchos trazos cortos y cercanos, tipico
    del pelo) eso se nota mucho mas que con pocos trazos largos. Conectar los
    que ya estan cerca evita esos aterrizajes de mas sin dibujar lineas
    largas que no correspondian al boceto (el umbral es chico a proposito).

    join_max_chain: como mucho esta cantidad de trazos seguidos sin una
    levantada real (ver JOIN_MAX_CHAIN)."""
    moves = []
    prev_point = None
    chain_len = 0
    for s in strokes:
        x0, y0 = float(s[0][0]), float(s[0][1])
        can_join = (prev_point is not None and join_gap_px > 0 and chain_len < join_max_chain and
                    ((x0 - prev_point[0]) ** 2 + (y0 - prev_point[1]) ** 2) ** 0.5 <= join_gap_px)
        if can_join:
            moves.append((x0, y0, True))   # seguir con la pluma abajo, sin levantar
            chain_len += 1
        else:
            moves.append((x0, y0, False))  # viaje hasta el inicio del trazo
            moves.append((x0, y0, True))   # bajar la pluma (mismo punto)
            chain_len = 0
        for x, y in s[1:]:
            moves.append((float(x), float(y), True))
        prev_point = moves[-1][:2]
    return moves


# Optimizacion del recorrido con vpype (https://github.com/abey79/vpype), la
# herramienta estandar de los plotters: une los trazos cuyos extremos quedan
# cerca (linemerge: una bajada de pluma en vez de varias), simplifica
# (linesimplify) y los ordena para viajar lo menos posible con la pluma arriba
# (linesort, dando vuelta trazos si conviene; --two-opt: mas fino, tarda un poco
# mas, se hace una vez por retrato).
VPYPE_MERGE_MM = 0.6
VPYPE_SIMPLIFY_MM = 0.08
VPYPE_TWO_OPT = True


def optimize_moves(moves, mm_per_px, merge_mm=VPYPE_MERGE_MM, simplify_mm=VPYPE_SIMPLIFY_MM):
    """moves (en pixeles) -> moves optimizados con vpype. Sin vpype, los mismos."""
    try:
        import vpype as vp
        import vpype_cli
    except ImportError:
        return moves
    strokes, cur = [], []
    for x, y, pen_down in moves:
        if pen_down:
            cur.append(complex(x, y))
        else:
            if cur:
                strokes.append(cur)
            cur = []
    if cur:
        strokes.append(cur)
    if len(strokes) < 2 or mm_per_px <= 0:
        return moves
    doc = vp.Document()
    doc.add(vp.LineCollection([np.array(st) for st in strokes]), 1)
    cmd = 'linemerge -t %.6f linesimplify -t %.6f linesort%s' % (
        merge_mm / mm_per_px, simplify_mm / mm_per_px, ' --two-opt' if VPYPE_TWO_OPT else '')
    out = vpype_cli.execute(cmd, doc)
    lines = [line for layer in out.layers.values() for line in layer]
    if not lines:
        return moves
    return _flat_strokes_to_moves([[(float(p.real), float(p.imag)) for p in line] for line in lines])


def _join_nearby_strokes(moves, join_gap_px):
    """Re-agrupa `moves` (ya planos, con pluma arriba/abajo) en trazos y los
    vuelve a aplanar con _flat_strokes_to_moves(join_gap_px=...), para que
    los trazos que ya estaban cerca queden conectados con la pluma abajo en
    vez de levantar y volver a aterrizar (ver _flat_strokes_to_moves)."""
    if join_gap_px <= 0:
        return moves
    strokes, cur = [], []
    for x, y, pen_down in moves:
        if pen_down:
            cur.append((x, y))
        elif cur:
            strokes.append(cur)
            cur = []
    if cur:
        strokes.append(cur)
    return _flat_strokes_to_moves(strokes, join_gap_px=join_gap_px)


def strokes_to_moves(strokes, img_height):
    """
    [(x,y), ...] por trazo (pixeles) -> [(x, y, pluma_abajo), ...] plano.

    Invierte el eje Y (y' = img_height - y): en la imagen, y crece hacia
    ABAJO (fila 0 = arriba), pero gcode_draw_lib.bilinear() interpreta
    "y_local mas grande" como v->1, o sea "arriba de la hoja". Sin este
    ajuste, la persona queda cabeza abajo en el papel (mismo problema que
    draw_person.strokes_to_mm ya resuelve con "v = cy - s[:,1]" en su propio
    pipeline; aca se hace explicito porque no pasamos por esa funcion).
    """
    flipped = [[(x, img_height - y) for x, y in s] for s in strokes]
    return _flat_strokes_to_moves(flipped)


# Si se aparta mas que esto de la proporcion ancho/alto del boceto, se asume
# que Gemini reencuadro/recorto la foto (no solo la escalo) y escalar los
# landmarks linealmente los ubicaria mal -- mejor no agregar nada. Que la
# proporcion coincida NO alcanza para confiar (Gemini puede reencuadrar con
# la misma proporcion): ver LANDMARK_MIN_ON_INK.
FACE_LANDMARK_ASPECT_TOLERANCE = 0.05


def _face_landmark_strokes_scaled(original_photo_path, sketch_shape):
    """
    Ojos/cejas/nariz/labios via MediaPipe Face Landmarker
    (dp.face_landmark_strokes) corrido sobre la FOTO ORIGINAL, escalado al
    tamano del boceto. Es el respaldo cuando no se detecta la cara en el
    propio boceto (ver _face_landmarks_on_sketch): solo sirve si Gemini
    mantuvo el encuadre de la foto, cosa que no siempre pasa -- quien lo usa
    debe validar que coincida (ver _aligned_face_landmarks).

    Devuelve (trazos, puntos) -- los 478 landmarks sueltos, ver
    dp.face_landmark_points --, o ([], None) sin lanzar excepcion si mediapipe
    no esta instalado, no detecta cara, no se puede leer la foto, o las
    proporciones no coinciden -- este refuerzo es opcional, nunca debe romper
    el pipeline principal.
    """
    try:
        orig_img = dp.load_image(original_photo_path)
    except Exception:
        return [], None
    oh, ow = orig_img.shape[:2]
    sh, sw = sketch_shape[:2]
    if not oh or not ow or abs((ow / oh) - (sw / sh)) > FACE_LANDMARK_ASPECT_TOLERANCE:
        return [], None
    try:
        pts = dp.face_landmark_points(orig_img)
        landmark_strokes = dp.face_landmark_strokes(orig_img, pts)
    except Exception:
        return [], None
    if not landmark_strokes:
        return [], None
    scale_x, scale_y = sw / ow, sh / oh
    return [s * [scale_x, scale_y] for s in landmark_strokes], [(x * scale_x, y * scale_y) for x, y in pts]


# Distancia (fraccion del lado mayor del boceto, ~10px en 800) hasta la que
# un punto de landmark cuenta como "sobre la tinta" del boceto.
LANDMARK_INK_TOL_FRACTION = 0.012
# Los landmarks escalados solo se usan si coinciden con el boceto: medido con
# trabajos reales, un boceto con el mismo encuadre que la foto dio 96% de
# puntos sobre tinta (mediana 0px), y uno que Gemini reencuadro (cara
# centrada y agrandada, misma proporcion 3:4) dio 63% (mediana 8px) -- en
# ese caso los ojos/boca de la foto caian sobre mejilla y menton, dibujando
# una segunda cara. Se probo buscar la mejor alineacion (escala + posicion)
# y no sirve: siempre "encuentra" 100% encogiendo los rasgos sobre cualquier
# zona con mucha tinta.
LANDMARK_MIN_ON_INK = 0.85
LANDMARK_MAX_MEDIAN_FRACTION = 0.004   # ~3px en 800
LANDMARK_SIDECAR_MIN_ON_INK = 0.5      # landmarks que trae el boceto local (ver _sidecar_landmarks)


def _landmarks_vs_sketch(landmark_strokes, mask):
    """Distancia de cada punto de cada trazo de landmarks a la tinta mas
    cercana del boceto. Devuelve (lista de arrays de distancias, tolerancia)."""
    import cv2
    h, w = mask.shape[:2]
    dist = cv2.distanceTransform(np.where(mask > 0, 0, 255).astype(np.uint8), cv2.DIST_L2, 5)
    out = []
    for s in landmark_strokes:
        xi = np.round(s[:, 0]).astype(int)
        yi = np.round(s[:, 1]).astype(int)
        inside = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
        d = np.full(len(s), np.inf)
        d[inside] = dist[yi[inside], xi[inside]]
        out.append(d)
    return out, LANDMARK_INK_TOL_FRACTION * max(h, w)


def _face_landmarks_on_sketch(img):
    """MediaPipe sobre el propio boceto, apenas desenfocado. Sobre line-art
    nitido no suele detectar cara (1 de 3 bocetos reales), pero con un
    desenfoque leve (sigma ~1.7px a 800px) la detecto en los 3, con los
    rasgos cayendo sobre la tinta (93-95%). Al salir del mismo boceto quedan
    alineados sin importar como Gemini haya encuadrado la foto."""
    import cv2
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.cvtColor(cv2.GaussianBlur(gray, (0, 0), max(gray.shape) / 480.0), cv2.COLOR_GRAY2BGR)
        pts = dp.face_landmark_points(blurred)
        return (dp.face_landmark_strokes(blurred, pts) or []), pts
    except Exception:
        return [], None


def _sidecar_landmarks(sketch_path, img):
    """Landmarks que dejo el que genero el boceto (local_sketch: los de la foto,
    llevados al boceto) en <boceto>.landmarks.json, escalados a `img`. Solo
    para ese mismo archivo (un recorte hecho despues es otro archivo)."""
    import json
    path = (sketch_path or '') + '.landmarks.json'
    if not sketch_path or not os.path.exists(path):
        return None
    try:
        with open(path, encoding='utf-8') as fh:
            data = json.load(fh)
        sw, sh = data['size']
        kx, ky = img.shape[1] / float(sw), img.shape[0] / float(sh)
        return [(x * kx, y * ky) for x, y in data['points']]
    except Exception:
        return None


def _aligned_face_landmarks(img, mask, original_photo_path=None, sidecar_points=None):
    """
    Landmarks de ojos/cejas/nariz/labios en coordenadas del boceto, solo si
    coinciden con lo que el boceto dibujo; ([], None) si no hay forma
    confiable. Primero se buscan en el propio boceto; si no, en la foto
    original escalada al boceto (solo sirve si Gemini mantuvo el encuadre).
    Devuelve (trazos, puntos sueltos de MediaPipe).
    """
    # (nombre, como obtenerlos, minimo sobre la tinta)
    sources = [('boceto', lambda: _face_landmarks_on_sketch(img), LANDMARK_MIN_ON_INK)]
    if sidecar_points:
        # los de la foto, ya llevados al boceto por quien lo genero (boceto local): alineados por
        # construccion, se piden menos sobre la tinta (si el dibujo quedo flojo en la cara, justo ahi
        # sirven para dibujar ojos y rasgos)
        sources.append(('foto, llevados al boceto', lambda: (dp.face_landmark_strokes(img, sidecar_points) or [],
                                                             sidecar_points), LANDMARK_SIDECAR_MIN_ON_INK))
    if original_photo_path:
        sources.append(('foto original', lambda: _face_landmark_strokes_scaled(original_photo_path, img.shape),
                        LANDMARK_MIN_ON_INK))
    for name, find, min_on_ink in sources:
        landmarks, points = find()
        if not landmarks:
            continue
        dists, tol = _landmarks_vs_sketch(landmarks, mask)
        all_d = np.concatenate(dists)
        on_ink = float(np.mean(all_d <= tol))
        median = float(np.median(all_d))
        if on_ink >= min_on_ink and median <= LANDMARK_MAX_MEDIAN_FRACTION * max(img.shape[:2]):
            print('rasgos de la cara ubicados desde el %s (%.0f%% sobre la tinta)' % (name, on_ink * 100), flush=True)
            return landmarks, points
        print('rasgos desde el %s descartados: no coinciden con el boceto (%.0f%% sobre la tinta, '
              'mediana %.0fpx)' % (name, on_ink * 100, median), flush=True)
    return [], None


def _landmark_bbox_flipped(landmark_strokes, img_height):
    """Cuadro delimitador (x_lo, x_hi, y_lo, y_hi) de los landmarks, en el
    MISMO sistema de coordenadas que terminan teniendo los `moves` finales
    (eje Y invertido, ver strokes_to_moves) -- para poder anclar ahi las
    zonas de cara/ojos-nariz-boca de simplify_for_robot en vez de adivinar
    su posicion por una fraccion fija del alto de la imagen."""
    xs = np.concatenate([s[:, 0] for s in landmark_strokes])
    ys = np.concatenate([s[:, 1] for s in landmark_strokes])
    ys_flipped = img_height - ys
    return float(xs.min()), float(xs.max()), float(ys_flipped.min()), float(ys_flipped.max())


# Indices de MediaPipe Face Landmarker (478 puntos: malla + iris) de cada ojo:
# lagrimal, esquina de afuera, centro del parpado de arriba y de abajo, centro
# del iris y su contorno, y el borde de abajo de la ceja (para no borrarla al
# sacar el ojo del boceto).
_EYE_LANDMARKS = [
    dict(inner=133, outer=33, upper=159, lower=145, iris=468, ring=(469, 470, 471, 472),
         brow=(46, 53, 52, 65, 55),
         # borde del ojo abierto, en orden (parpado de arriba y despues el de abajo)
         contour=(33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7)),
    dict(inner=362, outer=263, upper=386, lower=374, iris=473, ring=(474, 475, 476, 477),
         brow=(276, 283, 282, 295, 285),
         contour=(263, 466, 388, 387, 386, 385, 384, 398, 362, 382, 381, 380, 374, 373, 390, 249)),
]
_NOSE_TIP = 1


def eyes_from_landmarks(points, img_height):
    """
    Los dos ojos a partir de los landmarks, en el mismo sistema de ejes que
    los moves (Y invertido, ver strokes_to_moves): esquinas, parametros de la
    persona para eye_styles (apertura, mirada, iris), hacia donde queda la
    frente, y la zona a borrar del boceto. None si faltan puntos.
    """
    if not points or len(points) < 468:
        return None

    def P(i):
        x, y = points[i]
        return (float(x), float(img_height - y))

    nose = P(_NOSE_TIP)
    eyes = []
    for idx in _EYE_LANDMARKS:
        inner, outer = P(idx['inner']), P(idx['outer'])
        width = math.dist(inner, outer)
        if width <= 1:
            return None
        center = ((inner[0] + outer[0]) / 2, (inner[1] + outer[1]) / 2)
        iris_c = iris_r = None
        if len(points) >= 478:
            iris_c = P(idx['iris'])
            iris_r = sum(math.dist(iris_c, P(i)) for i in idx['ring']) / len(idx['ring'])
        params = eye_styles.eye_params_from_landmarks(inner, outer, P(idx['upper']), P(idx['lower']),
                                                      iris_c, iris_r)
        brow = [P(i) for i in idx['brow']]
        brow_c = (sum(p[0] for p in brow) / len(brow), sum(p[1] for p in brow) / len(brow))
        eyes.append(dict(inner=inner, outer=outer, center=center, width=width, params=params,
                         up_hint=(center[0] - nose[0], center[1] - nose[1]),
                         brow_dist=math.dist(center, brow_c), iris_c=iris_c, iris_r=iris_r,
                         contour=[P(i) for i in idx['contour']]))
    return eyes


# Zona que se borra del boceto alrededor de cada ojo antes de dibujar el
# estilo elegido (elipse alineada con el ojo, en medios anchos del ojo): lo
# bastante grande para llevarse las pestanas y sombras que Gemini dibuja
# alrededor, sin llegar a la ceja (nunca pasa de EYE_ERASE_BROW_FRACTION de
# la distancia hasta la ceja).
EYE_ERASE_HALF_WIDTH = 1.35
EYE_ERASE_HALF_HEIGHT = 0.8
EYE_ERASE_BROW_FRACTION = 0.7


def _eye_erase_ellipse(eye):
    hw = eye['width'] / 2
    ex = ((eye['outer'][0] - eye['inner'][0]) / eye['width'], (eye['outer'][1] - eye['inner'][1]) / eye['width'])
    b = min(EYE_ERASE_HALF_HEIGHT * hw, EYE_ERASE_BROW_FRACTION * eye['brow_dist'])
    return eye['center'], ex, EYE_ERASE_HALF_WIDTH * hw, b


def remove_eye_regions(moves, eyes):
    """Saca de los moves (pixeles, Y invertido) lo que el boceto dibujo dentro
    de la zona de cada ojo: los trazos que cruzan la zona se cortan ahi."""
    ellipses = [_eye_erase_ellipse(e) for e in eyes]

    def inside(x, y):
        for (cx, cy), ex, a, b in ellipses:
            dx, dy = x - cx, y - cy
            u = dx * ex[0] + dy * ex[1]
            v = -dx * ex[1] + dy * ex[0]
            if (u / a) ** 2 + (v / b) ** 2 <= 1:
                return True
        return False

    runs, cur = [], []
    for x, y, pen_down in moves:
        if not pen_down:
            if len(cur) >= 2:
                runs.append(cur)
            cur = []
            continue
        if inside(x, y):
            if len(cur) >= 2:
                runs.append(cur)
            cur = []
        else:
            cur.append((x, y))
    if len(cur) >= 2:
        runs.append(cur)
    return _flat_strokes_to_moves(runs)


def eye_style_moves(style, eyes, mm_per_px):
    """Los dos ojos dibujados con un estilo de eye_styles, ubicados y
    deformados segun los landmarks de la persona. Cada curva es un trazo
    aparte (nunca se unen: ver eye_styles)."""
    strokes = []
    for eye in eyes:
        strokes += eye_styles.place_eye(style, eye['inner'], eye['outer'], eye['up_hint'], eye['params'], mm_per_px)
    return _flat_strokes_to_moves(strokes)


def locate_face(sketch_path, threshold=None, original_photo_path=None, with_points=False):
    """(core_bbox, eyes) del boceto, igual que image_to_raw_moves pero sin
    extraer trazos (lo usa el estilo "trazo simple", que saca los trazos del SVG).
    with_points: tambien los 478 landmarks, en las coordenadas de los moves
    (Y invertido): (core_bbox, eyes, points)."""
    img = dp.load_image(sketch_path)
    mask = dp.sketch_to_mask(img, thresh=threshold)
    landmarks, points = _aligned_face_landmarks(img, mask, original_photo_path,
                                                _sidecar_landmarks(sketch_path, img))
    if not landmarks:
        return (None, None, None) if with_points else (None, None)
    h = img.shape[0]
    core, eyes = _landmark_bbox_flipped(landmarks, h), eyes_from_landmarks(points, h)
    if with_points:
        return core, eyes, [(float(x), float(h - y)) for x, y in points]
    return core, eyes


# Estilo "trazo simple" (boceto de lineas largas, ver gemini_sketch.STYLES):
# niveles de detalle propios, porque no pasa por simplify_for_robot.
#   smooth_px: suavizado a lo largo de cada linea (pixeles del boceto a ~800)
#   min_len_mm: lineas mas cortas que esto en el papel no se dibujan
LINE_ART_LEVELS = {
    'bajo': dict(smooth_px=3.0, min_len_mm=8.0),
    'medio': dict(smooth_px=2.0, min_len_mm=5.0),
    'alto': dict(smooth_px=1.2, min_len_mm=2.5),
    # dibujo a lapiz de lineas para la tableta (retrato_lapiz.py): mas detalle
    'lapiz': dict(smooth_px=1.0, min_len_mm=2.0),
}
LINE_ART_STEP_MM = 1.5   # paso entre puntos en el papel (ver MIN_SAFE_STEP_MM)


def line_art_svg_moves(sketch_path, svg_path, detail=DEFAULT_DETAIL, threshold=None):
    """
    Boceto -> SVG (lib/sketch_svg.py: lineas largas empalmadas en los cruces,
    suavizadas, como curvas Bezier) -> moves finos en pixeles, Y invertido
    como strokes_to_moves, leyendo el SVG recien escrito. El SVG queda en
    svg_path para verlo. Los moves todavia no estan en la escala del papel:
    ver finish_line_art_moves.
    """
    import sketch_svg
    level = LINE_ART_LEVELS.get(detail, LINE_ART_LEVELS[DEFAULT_DETAIL])
    img = dp.load_image(sketch_path)
    mask = dp.sketch_to_mask(img, thresh=threshold)
    h, w = img.shape[:2]
    paths = [sketch_svg.smooth_path(p, level['smooth_px'], 0.6) for p in sketch_svg.skeleton_paths(mask)]
    sketch_svg.write_svg([p for p in paths if len(p) >= 2], w, h, svg_path)
    polylines, (_sw, sh) = sketch_svg.svg_to_polylines(svg_path, step_px=1.0)
    if not polylines:
        raise RuntimeError('No se encontraron lineas en el boceto (revisa que tenga suficiente contraste).')
    return _flat_strokes_to_moves([[(x, sh - y) for x, y in p] for p in polylines])


def finish_line_art_moves(moves, detail=DEFAULT_DETAIL, rotation_deg=0):
    """Moves finos del SVG (pixeles) -> los que se dibujan: sin las lineas mas
    cortas que min_len_mm en el papel, con un punto cada LINE_ART_STEP_MM, y
    ordenadas para viajar poco con la pluma arriba."""
    import sketch_svg
    level = LINE_ART_LEVELS.get(detail, LINE_ART_LEVELS[DEFAULT_DETAIL])
    scale = _px_to_mm_scale(moves, rotation_deg)   # mm por pixel
    strokes, cur = [], []
    for x, y, pen_down in moves:
        if not pen_down:
            if len(cur) >= 2:
                strokes.append(np.array(cur))
            cur = []
        else:
            cur.append((x, y))
    if len(cur) >= 2:
        strokes.append(np.array(cur))
    strokes = [sketch_svg.resample(s, LINE_ART_STEP_MM / scale) for s in strokes
               if sketch_svg._length(s) * scale >= level['min_len_mm']]
    return _flat_strokes_to_moves(dp.order_strokes(_split_small_loops(strokes, scale)))


# Figuras cerradas chicas (un ojo, una fosa nasal) se parten en dos
# medialunas: al cerrar un lazo de un solo trazo, la pluma da la vuelta
# apoyada y la friccion la arrastra, y queda una raya en vez de la punta.
SMALL_LOOP_MAX_MM = 45.0
LOOP_CLOSE_MM = 1.0


def _split_small_loops(strokes, mm_per_px):
    import sketch_svg
    out = []
    for s in strokes:
        closed = len(s) >= 6 and math.dist(s[0], s[-1]) * mm_per_px <= LOOP_CLOSE_MM
        if not closed or sketch_svg._length(s) * mm_per_px > SMALL_LOOP_MAX_MM:
            out.append(s)
            continue
        loop = s[:-1]
        start = int(np.argmin(loop[:, 0]))              # arrancar en una punta
        loop = np.vstack([loop[start:], loop[:start], loop[start:start + 1]])
        far = int(np.argmax(np.hypot(*(loop - loop[0]).T)))   # la punta opuesta
        out += [loop[:far + 1], loop[far:]]
    return out


# Ningun trazo mas corto que esto (mm en el papel) llega al robot, en ningun
# estilo ni nivel de detalle. simplify_for_robot afloja su minimo en la zona de
# la cara para no perder rasgos, y dejaba rayitas de 0.3-2 mm (5 a 16 por
# retrato debajo de 4 mm): cada una es una bajada de pluma entera que en el
# papel sale como un punto o un gancho. Los ojos de eye_styles se agregan
# despues y no pasan por aca (la pupila es un punto a proposito).
MIN_DRAWN_STROKE_MM = 4.0


def drop_short_strokes(moves, rotation_deg=0, min_mm=MIN_DRAWN_STROKE_MM):
    """Saca los trazos (tramos seguidos con la pluma abajo) mas cortos que
    min_mm en el papel. Devuelve (moves, cuantos se sacaron)."""
    if not moves or min_mm <= 0:
        return moves, 0
    scale = _px_to_mm_scale(moves, rotation_deg)
    strokes, cur = [], []
    for x, y, pen_down in moves:
        if not pen_down:
            if cur:
                strokes.append(cur)
            cur = []
        else:
            cur.append((x, y))
    if cur:
        strokes.append(cur)
    keep = [s for s in strokes if sum(math.dist(a, b) for a, b in zip(s, s[1:])) * scale >= min_mm]
    return _flat_strokes_to_moves(keep), len(strokes) - len(keep)


def image_to_raw_moves(sketch_path, invert=False, threshold=None, min_len_px=SKETCH_MIN_LEN_PX,
                        original_photo_path=None):
    """
    Boceto (imagen) -> moves en pixeles, sin simplificar todavia.

    Si se pasa original_photo_path, se refuerzan ojos/cejas/nariz/labios con
    landmarks de MediaPipe detectados sobre esa foto (ver
    _face_landmark_strokes_scaled) -- el mismo refuerzo que hace
    draw_person.py en su comando `process`, adaptado para cuando el trazado
    principal viene de un boceto (Gemini) y no de la foto directamente.

    Los landmarks de MediaPipe se usan solo para UBICAR ojos/nariz/boca
    (core_bbox), no se agregan como trazos: el boceto ya los tiene dibujados,
    y con 3 bocetos reales el unico trazo de landmarks que "faltaba" era la
    linea recta del puente de la nariz -- una raya que ningun boceto dibuja.
    Ver _aligned_face_landmarks para de donde salen y como se valida.

    Devuelve (moves, shape, core_bbox, eyes). core_bbox es la posicion REAL
    de ojos/nariz/boca (ver _landmark_bbox_flipped), o None si no se pudo
    ubicar la cara -- en ese caso simplify_for_robot cae de vuelta a su
    heuristica de posicion. eyes: los dos ojos (ver eyes_from_landmarks), o
    None.
    """
    img = dp.load_image(sketch_path)
    mask = dp.sketch_to_mask(img, invert=invert, thresh=threshold)
    strokes = _extract_strokes(mask, min_len_px, SKETCH_SIMPLIFY_PX)
    img_height, img_width = img.shape[:2]

    landmarks, points = _aligned_face_landmarks(img, mask, original_photo_path,
                                                _sidecar_landmarks(sketch_path, img))
    core_bbox = _landmark_bbox_flipped(landmarks, img_height) if landmarks else None
    eyes = eyes_from_landmarks(points, img_height) if landmarks else None

    strokes = dp.order_strokes(strokes)
    if not strokes:
        raise RuntimeError('No se encontraron trazos en el boceto (prueba invert=True '
                            'si tiene fondo oscuro, o revisa que tenga suficiente contraste).')
    return strokes_to_moves(strokes, img_height), img.shape[:2], core_bbox, eyes


def rotate_moves(moves, degrees):
    """Gira el dibujo (antihorario, multiplo de 90) sobre si mismo. El mapeo
    a la hoja recalcula el encuadre desde los propios puntos, asi que no
    importa donde quede el origen."""
    turns = (int(degrees) // 90) % 4
    if turns == 0:
        return list(moves)
    rot = {1: lambda x, y: (-y, x), 2: lambda x, y: (-x, -y), 3: lambda x, y: (y, -x)}[turns]
    return [rot(x, y) + (pen_down,) for x, y, pen_down in moves]


def _px_to_mm_scale(moves, rotation_deg=0):
    """mm reales por pixel que resultaria de dibujar `moves` con preserve_aspect=True
    en el area calibrada actual (misma cuenta que hace draw_moves()). Con una
    rotacion de 90/270 (ver rotate_moves) el ancho y alto del dibujo se
    intercambian al llegar a la hoja."""
    xs = [m[0] for m in moves]
    ys = [m[1] for m in moves]
    design_w = max(xs) - min(xs) or 1.0
    design_h = max(ys) - min(ys) or 1.0
    if (int(rotation_deg) // 90) % 2:
        design_w, design_h = design_h, design_w
    quad_w, quad_h = lib._quad_edge_lengths()
    return min(quad_w / design_w, quad_h / design_h)


def resolve_detail(name):
    """Nombre de nivel de detalle (ver DETAIL_LEVELS) -> dict de parametros
    para simplify_for_robot(). Si el nombre no existe, usa DEFAULT_DETAIL."""
    return DETAIL_LEVELS.get(name, DETAIL_LEVELS[DEFAULT_DETAIL])


# La cara es lo que mas importa que quede reconocible. Se aproxima su zona
# igual que draw_person.face_region_mask(): la parte de ARRIBA del dibujo,
# que es donde cae la cara en un retrato tipico (cabeza y hombros, como pide
# GEMINI_SKETCH_PROMPT). Esos trazos se simplifican bastante menos (mas
# detalle) que el resto (pelo/ropa/fondo), que se simplifica un poco mas de
# lo normal para compensar y mantener el total de puntos parecido.
FACE_TOP_FRACTION = 0.6   # fraccion superior de la imagen tratada como "cara" (respaldo sin core_bbox)
FACE_DETAIL_BOOST = 0.55  # tolerancia de la cara = esta fraccion de la del resto (mas chico = mas detalle)

# La franja superior sola no alcanza para aislar la cara: a la altura de la
# cara tambien cae el pelo de los costados (mechones sueltos, muy fragmentados
# en un boceto tipico), y sin un limite de ancho ese pelo heredaba el mismo
# detalle "cara" que la piel, quedando igual de discontinuo. FACE_X_FRACTION
# recorta la zona de cara a una franja central angosta (ovalo cara), dejando
# el pelo de los costados -- aunque este arriba -- en la zona "resto".
FACE_X_FRACTION = 0.62   # ancho central de la imagen tratado como "cara" (no solo altura)

# Dentro de la zona de cara, una sub-zona mas chica y centrada para ojos,
# nariz y boca (no hay deteccion real de landmarks aca -- ver draw_person.py
# face_landmark_strokes/MediaPipe para eso -- es la misma heuristica de
# "banda central" aplicada dos veces). CORE_FEATURE_DETAIL_BOOST es relativo
# a la tolerancia de la CARA (no a la del fondo), asi que el detalle real de
# ojos/nariz/boca es face_detail_boost * core_feature_detail_boost del total.
CORE_FEATURE_X_FRACTION = 0.5   # ancho central de la imagen (debe ser < FACE_X_FRACTION, para quedar anidado)
CORE_FEATURE_Y_FRACTION = 0.6   # banda vertical central DENTRO de la zona de cara
CORE_FEATURE_DETAIL_BOOST = 0.45

# Todo lo que NO es cara (pelo, ropa, fondo/decoracion del boceto) se
# simplifica bastante MAS que el objetivo base pedido por el nivel de
# detalle, no menos: un retrato real (ej. de Gemini) suele traer de regalo
# pelo con mechones sueltos y garabatos/decoracion de fondo, cada uno un
# trazo corto y separado -- eso es lo que hace que el brazo suba y baje la
# pluma todo el tiempo fuera de la cara. REST_MIN_LEN_FACTOR sobre todo
# descarta esos trazos sueltos chicos (menos bajadas de pluma); REST_EPSILON_
# FACTOR ademas alisa lo que queda (menos zigzag, trazos mas rectos/largos).
REST_EPSILON_FACTOR = 3.5
REST_MIN_LEN_FACTOR = 8.0

# Con True, cada trazo se suaviza con un B-spline y se re-muestrea a paso
# uniforme (ver _smooth_and_resample) en vez de simplificarse con
# cv2.approxPolyDP: approxPolyDP da segmentos RECTOS entre esquinas
# detectadas (aspecto facetado/poligonal, como se ve en los preview
# anteriores), mientras que un spline sigue la curva real del trazo -- mas
# parecido a lo que daria vectorizar el boceto a SVG a mano, pero sin
# depender de potrace/autotrace (que trazan el CONTORNO de cada linea, no su
# centro -- para line art duplicarian cada trazo). Requiere scipy (ya es
# dependencia de draw_person.py); si no esta disponible cae con gracia al
# approxPolyDP de siempre.
SMOOTH_STROKES_ENABLED = True

# "s" de scipy.interpolate.splprep = suma de residuos^2 tolerada, por punto
# del trazo (se multiplica por la cantidad de puntos). El ruido tipico de un
# trazo extraido por skeletonizacion es de aprox 1-2px de desvio por punto
# -- este valor le da al spline margen para "limar" ese ruido sin alejarse
# de la forma real. Mas alto = curvas mas lisas pero con mas riesgo de
# redondear de mas esquinas que si eran filosas (marco de lentes, etc.).
SPLINE_SMOOTH_PER_PT = 2.0


def _smooth_and_resample(pts, step_px, smooth_per_pt=SPLINE_SMOOTH_PER_PT):
    """Ajusta un B-spline suave a `pts` (centerline de un trazo, con el
    ruido tipico de la skeletonizacion por pixeles) y lo re-muestrea a paso
    UNIFORME EN ARCO (ver mas abajo el porque) de `step_px` -- ver
    SMOOTH_STROKES_ENABLED. `step_px` es el mismo epsilon de zona que se
    usaria para approxPolyDP (ya viene respetando MIN_SAFE_STEP_MM). Si
    scipy no esta disponible, el trazo es muy corto, o el ajuste falla por
    cualquier motivo, devuelve los puntos originales tal cual (mismo
    fallback que ya tenia approxPolyDP)."""
    arr = np.asarray(pts, dtype=np.float64)
    keep = [0]
    for i in range(1, len(arr)):
        if np.hypot(*(arr[i] - arr[keep[-1]])) > 1e-6:
            keep.append(i)
    arr = arr[keep]
    if len(arr) < 4:
        return arr
    try:
        from scipy.interpolate import splprep, splev
    except ImportError:
        return arr
    try:
        tck, _u = splprep([arr[:, 0], arr[:, 1]], s=smooth_per_pt * len(arr), k=3)
    except Exception:
        return arr

    # splev a pasos iguales de "u" NO da pasos iguales de distancia real: u
    # es solo un parametro interno de la curva, y su "velocidad" (mm de
    # curva por unidad de u) varia a lo largo del trazo. Muestreando parejo
    # en u aparecian segmentos de 0.0mm mezclados con otros de mas de 10mm
    # en el mismo trazo -- justamente el riesgo de friccion (pasos por
    # debajo de MIN_SAFE_STEP_MM) que toda esta simplificacion existe para
    # evitar. Por eso se muestrea denso, se mide el largo de arco real
    # acumulado, y se re-parametriza por longitud de arco (estandar para
    # "muestreo parejo" de una spline) antes de bajar a step_px puntos.
    dense_u = np.linspace(0.0, 1.0, max(50, len(arr) * 4))
    try:
        dense_x, dense_y = splev(dense_u, tck)
    except Exception:
        return arr
    seg_len = np.hypot(np.diff(dense_x), np.diff(dense_y))
    cum_len = np.concatenate([[0.0], np.cumsum(seg_len)])
    arc_len = float(cum_len[-1])
    if arc_len < 1e-6:
        return arr

    # floor (no round): con round, la mitad de los pasos quedan por DEBAJO de
    # step_px en vez de alrededor de el -- exactamente el riesgo de friccion
    # que step_px (ya calculado respetando MIN_SAFE_STEP_MM) existe para
    # evitar. Con floor, arc_len/(n_samples-1) siempre da >= step_px.
    n_samples = max(2, int(arc_len // max(step_px, 1e-6)) + 1)
    target_lens = np.linspace(0.0, arc_len, n_samples)
    u_at_target = np.interp(target_lens, cum_len, dense_u)
    try:
        x_new, y_new = splev(u_at_target, tck)
    except Exception:
        return arr
    return np.column_stack([x_new, y_new])


def simplify_for_robot(moves, simplify_target_mm=SIMPLIFY_TARGET_MM,
                        min_stroke_target_mm=MIN_STROKE_TARGET_MM, prioritize_face=None,
                        face_top_fraction=FACE_TOP_FRACTION, face_detail_boost=FACE_DETAIL_BOOST,
                        face_x_fraction=FACE_X_FRACTION,
                        core_feature_x_fraction=CORE_FEATURE_X_FRACTION,
                        core_feature_y_fraction=CORE_FEATURE_Y_FRACTION,
                        core_feature_detail_boost=CORE_FEATURE_DETAIL_BOOST,
                        rest_epsilon_factor=REST_EPSILON_FACTOR,
                        rest_min_len_factor=REST_MIN_LEN_FACTOR,
                        join_gap_mm=JOIN_GAP_MM, core_bbox=None, rotation_deg=0):
    """
    Simplifica/filtra `moves` (pixeles) apuntando a distancias reales en mm
    (no un numero fijo de pixeles): calcula la escala mm/px que va a resultar
    al mapear al area calibrada, y convierte los objetivos en mm a la
    tolerancia en pixeles que corresponda -- asi el resultado es friccion-
    seguro sin importar la resolucion de la imagen de entrada.

    Si prioritize_face=True (default), prioriza detalle en 3 zonas (de menos
    a mas detalle): resto -> cara -> ojos/nariz/boca.

    Si se pasa core_bbox (ver image_to_raw_moves/_landmark_bbox_flipped: la
    posicion REAL de ojos/nariz/boca detectada por MediaPipe sobre la foto
    original), las zonas de cara/ojos-nariz-boca se anclan ahi (con margen),
    en vez de adivinar su posicion. Sin esto, la heuristica de respaldo
    asume que la cara cae en la franja superior-central del CUADRO
    DELIMITADOR REAL de `moves` (no del lienzo/imagen completo: un boceto de
    Gemini puede dejar un margen en blanco arriba de la cabeza -- bug real
    visto con un boceto real, contenido en Y [1,482] de una imagen de 800px
    de alto). Esa heuristica de respaldo asume ademas que la cara esta en el
    top de ese cuadro, lo cual falla en retratos donde algo (una gorra, pelo
    alto) ocupa buena parte de arriba y la cara/ojos terminan mas abajo del
    top_fraction -- caso real visto donde ojos/nariz/boca quedaban fuera de
    la zona de cara por completo y se perdian del todo. Por eso siempre que
    haya foto original conviene mandar core_bbox: ancla las zonas en datos
    reales en vez de en una posicion asumida.

    Ninguna tolerancia baja de MIN_SAFE_STEP_MM, el umbral de friccion
    aprendido en esta conversacion.
    """
    if not SIMPLIFICATION_ENABLED:
        return list(moves)
    if prioritize_face is None:
        prioritize_face = PRIORITIZE_FACE_ENABLED

    scale = _px_to_mm_scale(moves, rotation_deg)  # mm por pixel, con el giro que se va a aplicar despues
    safe_px = MIN_SAFE_STEP_MM / scale
    epsilon_px = max(simplify_target_mm / scale, safe_px)
    min_len_px = max(min_stroke_target_mm / scale, safe_px)
    join_gap_px = join_gap_mm / scale

    if not prioritize_face:
        moves = lib.simplify_moves(moves, epsilon=epsilon_px)
        moves = lib.filter_short_strokes(moves, min_length=min_len_px)
        return _join_nearby_strokes(moves, join_gap_px)

    import cv2

    face_epsilon_px = max(epsilon_px * face_detail_boost, safe_px)
    face_min_len_px = max(min_len_px * face_detail_boost, safe_px)
    core_epsilon_px = max(face_epsilon_px * core_feature_detail_boost, safe_px)
    core_min_len_px = max(face_min_len_px * core_feature_detail_boost, safe_px)
    rest_epsilon_px = epsilon_px * rest_epsilon_factor
    rest_min_len_px = min_len_px * rest_min_len_factor

    if core_bbox is not None:
        # Zonas ancladas en la posicion REAL de ojos/nariz/boca (ver
        # docstring). "core" = el cuadro de landmarks con un margen chico;
        # "face" = un cuadro bastante mas grande alrededor (cejas/frente
        # arriba, mejillas/mandibula a los costados y abajo).
        cx_lo, cx_hi, cy_lo, cy_hi = core_bbox
        core_w, core_h = cx_hi - cx_lo, cy_hi - cy_lo
        core_x_lo, core_x_hi = cx_lo - core_w * 0.25, cx_hi + core_w * 0.25
        core_y_lo, core_y_hi = cy_lo - core_h * 0.25, cy_hi + core_h * 0.25
        cx_mid, cy_mid = (cx_lo + cx_hi) / 2, (cy_lo + cy_hi) / 2
        face_x_lo, face_x_hi = cx_mid - core_w * 1.3, cx_mid + core_w * 1.3
        face_y_lo, face_y_hi = cy_lo - core_h * 1.6, cy_hi + core_h * 1.3
    else:
        # Heuristica de respaldo por posicion (ver docstring): sobre el
        # cuadro delimitador real del contenido. Los moves ya vienen con el
        # eje Y invertido (ver strokes_to_moves): la parte de "arriba" del
        # dibujo corresponde aca a y_local ALTO.
        content_xs = [m[0] for m in moves]
        content_ys = [m[1] for m in moves]
        content_y_min, content_y_max = min(content_ys), max(content_ys)
        content_x_min, content_x_max = min(content_xs), max(content_xs)
        content_height = content_y_max - content_y_min

        face_y_lo = content_y_max - content_height * face_top_fraction
        face_y_hi = content_y_max + 1.0  # sin techo: hasta arriba del todo
        core_y_center = (face_y_lo + content_y_max) / 2
        core_y_half = (content_y_max - face_y_lo) * core_feature_y_fraction / 2
        core_y_lo, core_y_hi = core_y_center - core_y_half, core_y_center + core_y_half
        content_x_center = (content_x_min + content_x_max) / 2
        face_x_half = (content_x_max - content_x_min) * face_x_fraction / 2
        face_x_lo, face_x_hi = content_x_center - face_x_half, content_x_center + face_x_half
        core_x_half = (content_x_max - content_x_min) * core_feature_x_fraction / 2
        core_x_lo, core_x_hi = content_x_center - core_x_half, content_x_center + core_x_half

    strokes, cur = [], []
    for x, y, pen_down in moves:
        if pen_down:
            cur.append((x, y))
        elif cur:
            strokes.append(cur)
            cur = []
    if cur:
        strokes.append(cur)

    def zone_of(pts):
        n = len(pts)
        in_face_band = sum(1 for x, y in pts
                            if face_y_lo <= y <= face_y_hi and face_x_lo <= x <= face_x_hi) / n >= 0.5
        if not in_face_band:
            return 'rest'
        in_core = sum(1 for x, y in pts
                      if core_x_lo <= x <= core_x_hi and core_y_lo <= y <= core_y_hi) / n >= 0.5
        return 'core' if in_core else 'face'

    tolerances = {
        'rest': (rest_epsilon_px, rest_min_len_px),
        'face': (face_epsilon_px, face_min_len_px),
        'core': (core_epsilon_px, core_min_len_px),
    }

    out_strokes = []
    for pts in strokes:
        eps, min_len = tolerances[zone_of(pts)]
        arr = np.array(pts, dtype=np.float32).reshape(-1, 1, 2)
        if cv2.arcLength(arr, False) < min_len:
            continue
        if SMOOTH_STROKES_ENABLED and len(pts) >= 4:
            simp = _smooth_and_resample(pts, step_px=eps)
        else:
            simp = cv2.approxPolyDP(arr, eps, False).reshape(-1, 2) if len(pts) >= 3 else arr.reshape(-1, 2)
        if len(simp) >= 2:
            out_strokes.append([(float(px), float(py)) for px, py in simp])

    return _flat_strokes_to_moves(out_strokes, join_gap_px=join_gap_px)


# ---------------------------------------------------------------------------
# 2. Preview (sin conectar al robot)
# ---------------------------------------------------------------------------
def render_preview(moves, out_path, preserve_aspect=True, bounds=None):
    """
    Dibuja un preview (PNG) del trazo como va a quedar SOBRE LA HOJA, con
    "arriba" = el lado de las esquinas "arriba" del perfil de calibracion
    (como se ve la hoja al medirla). Antes se dibujaba en coordenadas del
    robot (X hacia arriba), y con la hoja al costado del robot el dibujo se
    veia acostado aunque en el papel saliera derecho.

    Usa el mismo mapeo que draw_moves (lib.iter_sheet_uv para la posicion en
    la hoja, lib.iter_target_coords para las distancias reales del brazo),
    sin conectar el robot. Devuelve estadisticas (puntos, mm de dibujo/viaje,
    tiempo estimado).
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.patches import Rectangle

    quad_w, quad_h = lib._quad_edge_lengths()
    segs_draw, segs_travel = [], []
    draw_len = travel_len = 0.0
    n_strokes = 0
    prev_xyz = prev_sheet = None
    prev_pen_down = None

    # mismo plan que draw(): el preview muestra los tramos cortados y las entradas por el aire
    if j1u.ENABLED:
        moves, bounds = j1u.plan(moves, preserve_aspect, bounds)
    robot_pts = lib.iter_target_coords(moves, preserve_aspect=preserve_aspect, bounds=bounds)
    sheet_pts = lib.iter_sheet_uv(moves, preserve_aspect=preserve_aspect, bounds=bounds)
    for (_i, _t, _x, _y, pen_down, coords), (_i2, _t2, _x2, _y2, _pd, u, v) in zip(robot_pts, sheet_pts):
        xyz = tuple(coords[:3])
        sheet = (u * quad_w, v * quad_h)
        if prev_xyz is not None:
            d = ((xyz[0] - prev_xyz[0]) ** 2 + (xyz[1] - prev_xyz[1]) ** 2) ** 0.5
            if pen_down:
                segs_draw.append((prev_sheet, sheet))
                draw_len += d
            else:
                segs_travel.append((prev_sheet, sheet))
                travel_len += d
        if pen_down and not prev_pen_down:
            n_strokes += 1
        prev_pen_down = pen_down
        prev_xyz, prev_sheet = xyz, sheet

    fig, ax = plt.subplots(figsize=(6, 6.5))
    ax.add_patch(Rectangle((0, 0), quad_w, quad_h, fill=False, edgecolor='tab:blue',
                           linestyle='--', linewidth=0.8, label='area calibrada'))
    ax.add_collection(LineCollection(segs_travel, colors='tab:red', linewidths=0.4,
                                      linestyles=':', alpha=0.5, label='viaje (pluma arriba)'))
    ax.add_collection(LineCollection(segs_draw, colors='black', linewidths=1.1,
                                      label='dibujo (pluma abajo)'))
    ax.set_xlim(-3, quad_w + 3)
    ax.set_ylim(-3, quad_h + 3)
    ax.set_aspect('equal')
    ax.set_xlabel('ancho de la hoja (mm)')
    ax.set_ylabel('alto de la hoja (mm)  [arriba = esquinas "arriba"]')

    est_min = (draw_len + travel_len) / (lib.ASSUMED_SPEED_MM_S * 60.0)
    total_points = len(moves)
    ax.set_title('Preview en la hoja: %d trazos, %d puntos, %.0f mm dibujo, %.0f mm viaje\n'
                 '(~%.1f min estimado)' % (n_strokes, total_points, draw_len, travel_len, est_min),
                 fontsize=9)
    ax.legend(loc='lower right', fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)

    return dict(strokes=n_strokes, points=total_points, draw_mm=round(draw_len, 1),
                travel_mm=round(travel_len, 1), estimated_minutes=round(est_min, 1),
                area_mm=[round(quad_w, 1), round(quad_h, 1)])


# ---------------------------------------------------------------------------
# 3. Dibujar de verdad
# ---------------------------------------------------------------------------
def draw(mc, moves, progress_cb=None, preserve_aspect=True, bounds=None, check_every=25, unidirectional=None,
         pen_lift_mm=None, live=None, continuous=False):
    """Dibuja en el robot. Devuelve {'robot_errors': [(punto, codigo, texto)]}
    (ver lib.draw_moves). unidirectional: dibujar con la base girando en un
    solo sentido (ver j1_unidireccional); None = lo que diga el perfil activo."""
    if j1u.ENABLED if unidirectional is None else unidirectional:
        moves, bounds = j1u.plan(moves, preserve_aspect, bounds)
    # Si la calibracion dejo los servos liberados, se traban antes de moverse.
    mc.focus_all_servos()
    time.sleep(0.5)
    try:
        return lib.draw_moves(mc, moves, preserve_aspect=preserve_aspect, bounds=bounds,
                              progress_cb=progress_cb, check_every=check_every, pen_lift_mm=pen_lift_mm,
                              live=live, continuous=continuous)
    finally:
        try:
            mc.set_color(0, 255, 0)
        except Exception:
            pass
