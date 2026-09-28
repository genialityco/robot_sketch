#!/usr/bin/python
# -*- coding:utf-8 -*-
# Estilos de ojo para los retratos.
#
# Los ojos son lo que peor sale del boceto: rasgos chicos, llenos de trazos
# cortos que el brazo deforma (ver el juego de J1 en gcode_draw_lib). En vez
# de dibujar los ojos del boceto, se dibuja una plantilla elegida por el
# usuario (se prueban todas con la "prueba de ojos" de la calibracion), y la
# plantilla se ubica y se deforma con los landmarks de MediaPipe de cada
# persona: ancho, inclinacion, apertura y hacia donde mira. Asi todos los
# retratos comparten un trazo limpio y probado, pero cada ojo se parece al
# de la persona.
#
# Cada ojo se describe en un marco propio: x de -1 (lagrimal) a +1 (esquina
# de afuera), y hacia la frente, unidad = medio ancho del ojo. Las curvas
# cerradas van en tramos separados (medialunas), nunca en un solo trazo: al
# dar la vuelta con la pluma apoyada, la friccion la arrastra y queda una raya.

import math

STYLES = [
    ('almendra', 'Almendra con iris'),
    ('almendra_punto', 'Almendra con pupila (punto)'),
    ('parpado', 'Con pliegue del parpado'),
    ('simple', 'Simple: solo el parpado y la pupila'),
    ('pestanas', 'Con pestanas'),
    ('redondo', 'Redondo (caricatura)'),
]
STYLE_IDS = [s for s, _ in STYLES]

# Rangos en los que se deja variar cada ojo segun la persona: fuera de esto
# suele ser una deteccion mala (ojo cerrado en la foto, cara de perfil) y el
# ojo saldria raro o imposible de dibujar.
OPENNESS_RANGE = (0.22, 0.42)     # alto abierto / ancho del ojo
GAZE_RANGE = (-0.35, 0.35)        # corrimiento del iris, en medios anchos
IRIS_RANGE = (0.34, 0.5)          # radio del iris, en medios anchos
DEFAULTS = dict(openness=0.32, gaze=0.0, iris=0.42)

# Paso aproximado entre puntos de una curva (mm reales en el papel): mas fino
# no mejora nada en el brazo (ver MIN_SAFE_STEP_MM en pipeline.py).
STEP_MM = 1.5


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _n(length_units, units_per_mm):
    return max(4, int(math.ceil(length_units / units_per_mm / STEP_MM)))


def _arc(x0, x1, y, bulge, units_per_mm):
    """Curva de (x0, y) a (x1, y) que se aparta `bulge` en el medio (seno:
    dos curvas con las mismas puntas cierran en punta)."""
    n = _n(abs(x1 - x0) + abs(bulge), units_per_mm)
    return [(x0 + (x1 - x0) * k / n, y + bulge * math.sin(math.pi * k / n)) for k in range(n + 1)]


def _half_ellipse(cx, cy, rx, ry, upper, units_per_mm):
    """Media elipse (de arriba o de abajo), de izquierda a derecha."""
    n = _n(math.pi * (rx + ry) / 2, units_per_mm)
    out = []
    for k in range(n + 1):
        a = math.pi * (1 - k / n) if upper else math.pi * (1 + k / n)
        out.append((cx + rx * math.cos(a), cy + ry * math.sin(a)))
    return out


def _lids(p, units_per_mm, lower_from=-1.0, lower_to=1.0):
    """Parpado de arriba y de abajo (dos medialunas que cierran en las puntas)."""
    up, down = p['up'], p['down']
    upper = _arc(-1.0, 1.0, 0.0, up, units_per_mm)
    if lower_from == -1.0 and lower_to == 1.0:
        lower = _arc(-1.0, 1.0, 0.0, -down, units_per_mm)
    else:   # parpado de abajo mas corto: el mismo seno, recortado
        lower = [(x, y) for x, y in _arc(-1.0, 1.0, 0.0, -down, units_per_mm) if lower_from <= x <= lower_to]
    return [upper, lower]


def _inside_lids(x, y, p):
    if abs(x) >= 1:
        return False
    s = math.sin(math.pi * (x + 1) / 2)
    return -p['down'] * s <= y <= p['up'] * s


def _iris(p, units_per_mm):
    """Circulo del iris, en dos mitades para no cerrar un circulo de un solo
    trazo. En un ojo real el iris es mas alto que la abertura y los parpados
    lo tapan arriba y abajo, pero a este tamano eso deja solo dos rayitas a
    los costados: se achica hasta entrar casi entero entre los parpados (lo
    que igual queda afuera se recorta)."""
    gx = p['gaze']
    r = min(p['iris'], 0.95 * (p['up'] + p['down']) / 2)
    cy = (p['up'] - p['down']) / 4
    strokes = []
    for upper in (True, False):
        cur = []
        for x, y in _half_ellipse(gx, cy, r, r, upper, units_per_mm):
            if _inside_lids(x, y, p):
                cur.append((x, y))
            elif len(cur) >= 2:
                strokes.append(cur)
                cur = []
            else:
                cur = []
        if len(cur) >= 2:
            strokes.append(cur)
    return strokes


def _pupil_dot(p, y=0.0):
    return [[(p['gaze'], y)]]   # un solo punto: bajar y subir la pluma


def eye_local_strokes(style, openness=None, gaze=None, iris=None, units_per_mm=1.0):
    """
    Trazos de un ojo en su marco propio (ver arriba). units_per_mm = cuantas
    unidades del marco (medios anchos) hay en un mm del papel: define cuantos
    puntos lleva cada curva.
    """
    openness = _clamp(DEFAULTS['openness'] if openness is None else openness, *OPENNESS_RANGE)
    p = dict(
        up=openness * 2 * 0.6,       # el parpado de arriba se abre mas que el de abajo
        down=openness * 2 * 0.4,
        gaze=_clamp(DEFAULTS['gaze'] if gaze is None else gaze, *GAZE_RANGE),
        iris=_clamp(DEFAULTS['iris'] if iris is None else iris, *IRIS_RANGE),
    )
    u = units_per_mm
    if style == 'almendra':
        return _lids(p, u) + _iris(p, u)
    if style == 'almendra_punto':
        return _lids(p, u) + _pupil_dot(p)
    if style == 'parpado':
        crease = _arc(-0.6, 0.95, p['up'] + 0.3, 0.25, u)
        return _lids(p, u, -0.75, 0.85) + [crease] + _iris(p, u)
    if style == 'simple':
        return [_arc(-1.0, 1.0, 0.0, p['up'] * 1.2, u)] + _pupil_dot(p, p['up'] * 0.35)
    if style == 'pestanas':
        lashes = []
        for x in (0.35, 0.62, 0.88):
            y = p['up'] * math.sin(math.pi * (x + 1) / 2)
            ang = math.radians(70 - 35 * (x - 0.35) / 0.53)   # se abren hacia afuera
            lashes.append([(x, y), (x + 0.4 * math.cos(ang), y + 0.4 * math.sin(ang))])
        return _lids(p, u) + _iris(p, u) + lashes
    if style == 'redondo':
        ry = max(0.55, openness * 2 * 0.9)
        return [_half_ellipse(0.0, 0.0, 0.8, ry, True, u), _half_ellipse(0.0, 0.0, 0.8, ry, False, u)] + \
            _pupil_dot(p, -ry * 0.15)
    raise ValueError('estilo de ojo desconocido: %s' % style)


def eye_params_from_landmarks(inner, outer, upper, lower, iris_center=None, iris_radius=None):
    """Apertura, mirada y tamano del iris de un ojo real, a partir de los
    landmarks (en pixeles, cualquier sistema de ejes)."""
    width = math.dist(inner, outer)
    if width <= 0:
        return dict(DEFAULTS)
    ex = ((outer[0] - inner[0]) / width, (outer[1] - inner[1]) / width)
    center = ((inner[0] + outer[0]) / 2, (inner[1] + outer[1]) / 2)
    params = dict(openness=math.dist(upper, lower) / width)
    if iris_center is not None:
        params['gaze'] = ((iris_center[0] - center[0]) * ex[0] + (iris_center[1] - center[1]) * ex[1]) / (width / 2)
    if iris_radius:
        params['iris'] = iris_radius / (width / 2)
    return params


def place_eye(style, inner, outer, up_hint, params, mm_per_px):
    """
    Trazos de un ojo en pixeles del boceto: el marco del ojo se apoya en las
    esquinas reales (inner = lagrimal, outer = esquina de afuera), y "arriba"
    es el lado de la perpendicular que apunta hacia up_hint (un vector que va
    hacia la frente). mm_per_px: escala con la que se va a dibujar.
    """
    width = math.dist(inner, outer)
    hw = width / 2
    ex = ((outer[0] - inner[0]) / width, (outer[1] - inner[1]) / width)
    ey = (-ex[1], ex[0])
    if ey[0] * up_hint[0] + ey[1] * up_hint[1] < 0:
        ey = (-ey[0], -ey[1])
    cx, cy = (inner[0] + outer[0]) / 2, (inner[1] + outer[1]) / 2
    units_per_mm = 1.0 / (hw * mm_per_px)
    local = eye_local_strokes(style, units_per_mm=units_per_mm, **params)
    return [[(cx + hw * (x * ex[0] + y * ey[0]), cy + hw * (x * ex[1] + y * ey[1])) for x, y in s] for s in local]


# --- prueba de ojos -----------------------------------------------------------
# Numeros de 7 segmentos para rotular cada estilo en la hoja (alto 1, ancho 0.5).
_SEGMENTS = {'a': [(0, 1), (0.5, 1)], 'b': [(0.5, 1), (0.5, 0.5)], 'c': [(0.5, 0.5), (0.5, 0)],
             'd': [(0.5, 0), (0, 0)], 'e': [(0, 0), (0, 0.5)], 'f': [(0, 0.5), (0, 1)], 'g': [(0, 0.5), (0.5, 0.5)]}
_DIGITS = {1: 'bc', 2: 'abged', 3: 'abgcd', 4: 'fgbc', 5: 'afgcd', 6: 'afgedc', 7: 'abc', 8: 'abcdefg', 9: 'abcdfg'}


def _digit(n, x0, y0, size):
    return [[(x0 + x * size, y0 + y * size) for x, y in _SEGMENTS[s]] for s in _DIGITS[n]]


TEST_EYE_WIDTH_MM = 12.0   # tamano parecido al de un ojo en un retrato real
TEST_EYE_GAP_MM = 10.0     # entre los dos ojos de cada par
TEST_LABEL_MM = 6.0


def test_design():
    """
    Todos los estilos, cada uno como un par de ojos (izquierdo y derecho, en
    espejo, como en una cara) con su numero al costado, en una grilla de 2
    columnas. Devuelve (trazos en mm con y hacia arriba, ancho, alto).
    """
    w = TEST_EYE_WIDTH_MM
    col_w = TEST_LABEL_MM + 4 + 2 * w + TEST_EYE_GAP_MM + 8
    row_h = w * 1.3 + 8
    rows = int(math.ceil(len(STYLES) / 2))
    strokes = []
    for i, (style, _name) in enumerate(STYLES):
        col, row = i % 2, i // 2
        x0 = col * col_w
        y_mid = (rows - 1 - row) * row_h + row_h / 2
        strokes += _digit(i + 1, x0, y_mid - TEST_LABEL_MM / 2, TEST_LABEL_MM)
        left_outer = x0 + TEST_LABEL_MM + 4
        right_inner = left_outer + w + TEST_EYE_GAP_MM
        for inner, outer in (((left_outer + w, y_mid), (left_outer, y_mid)),
                             ((right_inner, y_mid), (right_inner + w, y_mid))):
            strokes += place_eye(style, inner, outer, (0, 1), {}, mm_per_px=1.0)
    return strokes, 2 * col_w - 8, rows * row_h


# --- prueba de ojo con puntos -------------------------------------------------
DOTS_EYE_WIDTH_MM = 70.0   # grande, para ver bien como queda un dibujo de puntos
DOTS_SPACING_MM = 3.5      # entre puntos de una linea
DOTS_FILL_MM = 2.5         # entre puntos del relleno de la pupila


def _dots_along(points, spacing):
    """Puntos cada `spacing` a lo largo de una polilinea (incluye las puntas)."""
    seg = [math.dist(a, b) for a, b in zip(points, points[1:])]
    total = sum(seg)
    n = max(1, int(round(total / spacing)))
    out, acc, k = [], 0.0, 0
    for j in range(n + 1):
        target = total * j / n
        while k < len(seg) - 1 and acc + seg[k] < target:
            acc += seg[k]
            k += 1
        t = (target - acc) / seg[k] if seg[k] else 0.0
        a, b = points[k], points[k + 1]
        out.append((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t))
    return out


def dots_eye_design():
    """
    Un ojo grande dibujado solo con puntos (la pluma baja y sube en el mismo
    lugar): parpados e iris como lineas de puntos, pupila rellena con un
    brillo sin puntos. Devuelve trazos de un solo punto en mm, y hacia arriba.
    """
    hw = DOTS_EYE_WIDTH_MM / 2
    up, down = hw * 0.45, hw * 0.3
    n = 400   # curvas muestreadas bien fino antes de repartir los puntos
    upper = [(-hw + 2 * hw * k / n, up * math.sin(math.pi * k / n)) for k in range(n + 1)]
    lower = [(-hw + 2 * hw * k / n, -down * math.sin(math.pi * k / n)) for k in range(n + 1)]
    dots = _dots_along(upper, DOTS_SPACING_MM) + _dots_along(lower, DOTS_SPACING_MM)[1:-1]  # puntas una vez

    r_iris, r_pupil = hw * 0.36, hw * 0.15
    cy = (up - down) / 4
    p = dict(up=up / hw, down=down / hw)
    circle = [(r_iris * math.cos(2 * math.pi * k / 360), cy + r_iris * math.sin(2 * math.pi * k / 360))
              for k in range(361)]
    for x, y in _dots_along(circle, DOTS_SPACING_MM)[:-1]:
        if _inside_lids(x / hw, y / hw, p):
            dots.append((x, y))

    glint = (r_pupil * 0.45, cy + r_pupil * 0.45)   # brillo: arriba a la derecha de la pupila
    n = int(r_pupil // DOTS_FILL_MM) + 1
    for i in range(-n, n + 1):
        for j in range(-n, n + 1):
            x = i * DOTS_FILL_MM + (DOTS_FILL_MM / 2 if j % 2 else 0)   # tresbolillo
            y = cy + j * DOTS_FILL_MM * 0.87
            if math.hypot(x, y - cy) <= r_pupil and math.dist((x, y), glint) > r_pupil * 0.35:
                dots.append((x, y))
    # donde el iris cruza los parpados quedan puntos casi encimados: uno solo
    kept = []
    for d in dots:
        if all(math.dist(d, k) >= 0.6 * DOTS_SPACING_MM for k in kept):
            kept.append(d)
    return [[d] for d in kept]
