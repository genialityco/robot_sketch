#!/usr/bin/python
# -*- coding:utf-8 -*-
# Logica compartida para dibujar un archivo gcode remapeando sus coordenadas
# X/Y al area de trabajo real del robot, definida por las 4 esquinas de la
# hoja medidas a mano con la pluma tocando el papel (interpolacion bilineal).
# La usan 280_draw_gcode_mapped.py y sketch_to_gcode.py.

import math
import re
import time

# ---------------------------------------------------------------------------
# Configuracion
# ---------------------------------------------------------------------------
PORT = 'COM5'
BAUD = 115200

# Lo que respondio get_fresh_mode() en el ultimo connect() (None = todavia no
# se conecto). 0 = modo cola; cualquier otra cosa = el robot puede saltearse
# puntos (ver connect()).
LAST_FRESH_MODE = None

DRAW_SPEED = 80       # velocidad con la pluma tocando el papel (dibujando)
TRAVEL_SPEED = 100       # velocidad con la pluma levantada (desplazamientos)
PEN_LIFT_MM = 35.0       # cuanto se levanta la pluma (mm). Subido de 15.6: el modelo de
                         # altura de contacto solo tiene 4 puntos medidos, y el error del
                         # modelo en el interior de la hoja puede ser de varios mm -- si el
                         # margen de levantado es muy justo, la pluma arrastra en los viajes
                         # entre trazos y conecta todo (posible causa de los dibujos erraticos
                         # que salieron identicos aun cambiando pacing y modo de movimiento).

# Encoge el area de dibujo hacia el centro este porcentaje por cada lado, para
# no pedirle al brazo las 4 esquinas literales (que suelen estar al limite de
# su alcance comodo y son las que mas fallan con "sin solucion adyacente").
MARGIN = 0

# NOTA sobre el enfoque de pacing: probamos verificar cada punto con
# is_in_position() (bloqueando hasta confirmar que el brazo llego exacto), y
# resulto muy poco confiable especificamente para los cambios de Z (subir o
# bajar la pluma): en una corrida real, el 100% de esas transiciones agotaron
# el timeout aunque el robot no reportaba ningun error real (get_error_information
# devolvia 0) -- es decir, probablemente si llegaba, pero is_in_position no lo
# confirmaba. Eso hacia perder ~4s por transicion (3s + reintento angular de 1s).
# Por eso ya no se verifica ningun punto: se manda el comando y se espera una
# pausa, confiando en que el brazo ejecuta el movimiento.
#
# Esa pausa NO puede ser un numero fijo: si el siguiente comando se manda
# antes de que el brazo termine de llegar al punto anterior, el brazo
# redirige en linea recta desde donde este en ese momento (no desde el punto
# esperado) hacia el nuevo destino -- eso es lo que genera zigzags y cruces
# en el dibujo. La pausa se calcula segun la distancia real de cada
# movimiento y una velocidad estimada, para no mandar el siguiente punto
# antes de tiempo.
ASSUMED_SPEED_MM_S = 45  # estimacion conservadora de mm/s reales del brazo en modo lineal
# El triangulo (8 puntos, distancias grandes) salio limpio y recto en modo
# lineal, lo que descarta un problema mecanico. La diferencia real con el
# retrato (526 puntos, casi todos a 1-5mm de distancia) es que ahi se mandan
# cientos de comandos muy seguidos y muy chicos: el piso de pausa (0.08s) es
# probablemente insuficiente para que el controlador registre cada
# micro-movimiento antes del siguiente. Se sube el piso para dar mas margen.
MIN_PACING = 0.3
MAX_PACING = 2.0
PACING_OVERHEAD = 0.05   # margen extra fijo por comando (procesamiento/serial)

# 1 = lineal (cartesiano, linea recta entre puntos) ; 0 = angular (interpola
# los joints directo, NO garantiza linea recta).
# Se probo 0 pensando en evitar saltos por singularidad de muneca en el
# retrato (puntos muy juntos, la curvatura de una interpolacion angular ahi
# es imperceptible). Pero la prueba fisica del triangulo (lados largos,
# 90-140mm) salio con los lados claramente curvos en vez de rectos -- eso
# confirma que modo angular NO traza recto a esta escala. Se vuelve a modo
# lineal para tener una prueba limpia (linea recta garantizada) y aislar si
# el problema de fondo es mecanico o de otra cosa.
MOVE_MODE = 1

# Las 4 esquinas del area de dibujo, medidas con la pluma tocando la hoja:
# [x, y, z, rx, ry, rz]
TOP_LEFT = [266.9, 62.8, 81.9, -179.11, 0.8, 57.13]       # arriba izquierda
TOP_RIGHT = [255.3, -51.8, 86.5, 176.03, -1.1, 37.75]       # arriba derecha
BOTTOM_RIGHT = [155.6, -46.9, 84.5, 178.78, 4.5, 37.92]   # abajo derecha
BOTTOM_LEFT = [164.5, 58.6, 85.6, -179.33, -3.25, 81.15]    # abajo izquierda

COORD_RE = re.compile(r'([XYZ])(-?\d+\.?\d*)')


# Las 4 esquinas medidas a mano no forman un paralelogramo perfecto (ruido de
# medicion, agravado por haberse recalibrado en sesiones distintas): el lado
# de arriba y el de abajo no son paralelos/iguales entre si, ni tampoco los
# lados izquierdo y derecho. Eso introduce una inclinacion sistematica --
# comprobado: dos puntos simetricos (mismo v, u espejado) terminaban ~5mm
# mas abajo en la pagina del lado derecho que del izquierdo, en toda la hoja.
# Se reemplazan las 4 esquinas por un paralelogramo "promedio": mismo centro,
# con cada lado igual al promedio de los dos lados opuestos medidos. Esto
# conserva el tamano/posicion/rotacion general implicita en las 4 medidas,
# pero elimina la inconsistencia entre lados opuestos.
def _regularized_corners():
    raw = [TOP_LEFT, TOP_RIGHT, BOTTOM_RIGHT, BOTTOM_LEFT]
    center = [sum(c[i] for c in raw) / 4 for i in range(3)]
    # vector "hacia la derecha": promedio de (TR-TL) y (BR-BL)
    w = [((TOP_RIGHT[i] - TOP_LEFT[i]) + (BOTTOM_RIGHT[i] - BOTTOM_LEFT[i])) / 2 for i in range(3)]
    # vector "hacia arriba": promedio de (TL-BL) y (TR-BR)
    h = [((TOP_LEFT[i] - BOTTOM_LEFT[i]) + (TOP_RIGHT[i] - BOTTOM_RIGHT[i])) / 2 for i in range(3)]

    # Una hoja de papel real es un rectangulo (esquinas a 90 grados). w y h
    # (en x,y) resultaron NO perpendiculares entre si por 5+ grados -- eso ya
    # no es solo "la hoja esta rotada en la mesa" (lo cual es normal), sino
    # un error real de medicion. Se fuerza la perpendicularidad repartiendo
    # la correccion entre ambos vectores (se promedian los dos angulos que
    # cada uno implica para la rotacion de la hoja), preservando el largo
    # medido de cada lado.
    w_len = math.hypot(w[0], w[1])
    h_len = math.hypot(h[0], h[1])
    angle_w = math.atan2(w[1], w[0])
    angle_h = math.atan2(h[1], h[0]) - math.pi / 2
    avg_angle = (angle_w + angle_h) / 2
    w[0], w[1] = w_len * math.cos(avg_angle), w_len * math.sin(avg_angle)
    h[0], h[1] = h_len * math.cos(avg_angle + math.pi / 2), h_len * math.sin(avg_angle + math.pi / 2)

    def corner(sw, sh):
        return [center[i] + sw * w[i] / 2 + sh * h[i] / 2 for i in range(3)]

    return {
        'TOP_LEFT': corner(-1, 1),
        'TOP_RIGHT': corner(1, 1),
        'BOTTOM_RIGHT': corner(1, -1),
        'BOTTOM_LEFT': corner(-1, -1),
    }


# Altura (Z) medida con la pluma tocando el CENTRO de la hoja, o None. Con
# solo las 4 esquinas, la altura del interior sale de un plano entre ellas, y
# el papel/mesa/brazo real no siempre es plano: visto en la practica, subir la
# pluma 2 mm se notaba en el contorno pero no en la cara, que esta en el
# medio. Solo corrige la altura: no cambia la forma ni el tamano del area.
CENTER_Z = None


def _recompute_regularized():
    global _REG_TOP_LEFT, _REG_TOP_RIGHT, _REG_BOTTOM_RIGHT, _REG_BOTTOM_LEFT, _PLANE_CENTER_Z
    reg = _regularized_corners()
    _REG_TOP_LEFT = reg['TOP_LEFT']
    _REG_TOP_RIGHT = reg['TOP_RIGHT']
    _REG_BOTTOM_RIGHT = reg['BOTTOM_RIGHT']
    _REG_BOTTOM_LEFT = reg['BOTTOM_LEFT']
    _PLANE_CENTER_Z = sum(c[2] for c in reg.values()) / 4


_recompute_regularized()


def _center_z_correction(u, v):
    """Cuanto subir/bajar Z en (u, v) para que en el centro de la hoja la
    altura sea CENTER_Z: la diferencia con el plano de las esquinas, repartida
    como una loma (16 u(1-u) v(1-v)) que vale 1 en el centro y 0 en todo el
    borde -- las esquinas y el contorno quedan igual que sin centro."""
    if CENTER_Z is None:
        return 0.0
    return (CENTER_Z - _PLANE_CENTER_Z) * 16 * u * (1 - u) * v * (1 - v)


def set_calibration(top_left, top_right, bottom_right, bottom_left,
                    manual_skew_mm=None, pen_lift_mm=None, center_z=None, j1_backlash_deg=None):
    """Reemplaza en caliente las 4 esquinas medidas (y opcionalmente el ajuste
    fino, el levantado de pluma y el juego de la base), sin editar este
    archivo. Lo usa la app web para aplicar perfiles de calibracion. center_z
    (ver CENTER_Z) se aplica siempre: None = sin punto de centro."""
    global TOP_LEFT, TOP_RIGHT, BOTTOM_RIGHT, BOTTOM_LEFT, MANUAL_SKEW_MM, PEN_LIFT_MM, CENTER_Z
    global J1_BACKLASH_DEG
    TOP_LEFT, TOP_RIGHT = list(top_left), list(top_right)
    BOTTOM_RIGHT, BOTTOM_LEFT = list(bottom_right), list(bottom_left)
    if manual_skew_mm is not None:
        MANUAL_SKEW_MM = manual_skew_mm
    if pen_lift_mm is not None:
        PEN_LIFT_MM = pen_lift_mm
    if j1_backlash_deg is not None:
        J1_BACKLASH_DEG = j1_backlash_deg
    CENTER_Z = center_z
    _recompute_regularized()


def get_calibration():
    return dict(top_left=list(TOP_LEFT), top_right=list(TOP_RIGHT),
                bottom_right=list(BOTTOM_RIGHT), bottom_left=list(BOTTOM_LEFT),
                manual_skew_mm=MANUAL_SKEW_MM, pen_lift_mm=PEN_LIFT_MM, center_z=CENTER_Z,
                j1_backlash_deg=J1_BACKLASH_DEG)


# Juego (holgura) de la articulacion de la base, J1, en grados. Medido en papel
# (2026-09-25, hoja a ~204 mm de la base): cada vez que la base cambia de
# sentido de giro, la pluma no se mueve durante los primeros ~5-6 mm (~1.5
# grados), asi que los movimientos cortos de lado (eje Y del robot) salen
# achicados o desaparecen -- la base de un triangulo de 8 mm quedaba en un
# punto, los ojos salian 4-5 mm mas angostos -- mientras que en X (alejarse /
# acercarse a la base) el dibujo sale bien. get_coords no lo muestra (lee
# mal: ver la prueba de precision). 0 = sin compensar.
J1_BACKLASH_DEG = 0.0
# Movimientos de la base mas chicos que esto (mm de arco) no cambian el
# sentido de giro que se esta compensando: evita saltos de todo el juego por
# el ruido de giro de un trazo que va casi derecho en X.
J1_BACKLASH_MIN_MOVE_MM = 0.3


class J1Backlash:
    """
    Compensa el juego de J1 como una fresadora CNC: guarda hacia que lado esta
    apoyado el engranaje y, cuando el giro de la base cambia de sentido, manda
    el punto medio juego mas alla en el sentido nuevo (y medio juego mas alla
    en todo el giro siguiente en ese sentido). Asi la pluma, que queda medio
    juego atrasada respecto de lo que se manda, cae donde se pidio. Solo gira
    x,y alrededor del eje de la base; radio, z y orientacion no cambian.
    """

    def __init__(self, backlash_deg):
        self.half = math.radians(backlash_deg) / 2
        # giro de la base hasta donde se llego en el sentido de apoyo actual (o,
        # sin sentido todavia, el ultimo): los cambios de sentido se miden desde
        # aca y no paso a paso. Cerca de la punta de una curva cada paso gira la
        # base menos que J1_BACKLASH_MIN_MOVE_MM; comparando solo con el paso
        # anterior el cambio de sentido no se detectaba nunca, la pluma quedaba
        # quieta en Y y la media curva siguiente salia recta (visto: la
        # medialuna de abajo de los ojos, la mas chata, salia como una raya).
        self.peak = None
        self.engaged = 0    # +1 / -1: sentido de giro en que esta apoyado el engranaje
        self.last = None    # ultimo punto pedido (donde deberia estar la pluma)

    def direction(self, a, b):
        """Sentido de giro de la base para ir de a a b (0 si es casi nada)."""
        ta, tb = math.atan2(a[1], a[0]), math.atan2(b[1], b[0])
        if abs(tb - ta) * math.hypot(b[0], b[1]) < J1_BACKLASH_MIN_MOVE_MM:
            return 0
        return 1 if tb > ta else -1

    def _new_direction(self, coords):
        """Sentido de apoyo que corresponde al ir a `coords` (el actual si no cambia)."""
        if self.peak is None:
            return self.engaged
        theta = math.atan2(coords[1], coords[0])
        moved_mm = (theta - self.peak) * math.hypot(coords[0], coords[1])
        if self.engaged and moved_mm * self.engaged >= 0:
            return self.engaged
        if abs(moved_mm) < J1_BACKLASH_MIN_MOVE_MM:
            return self.engaged
        return 1 if moved_mm > 0 else -1

    def reversal(self, coords):
        """Sentido nuevo si ir a `coords` invierte el giro de la base, si no 0."""
        if self.half <= 0 or self.last is None or not self.engaged:
            return 0
        d = self._new_direction(coords)
        return d if d != self.engaged else 0

    def command(self, coords, engage=None):
        """coords donde tiene que quedar la pluma -> coords a mandar. engage
        (+1/-1) fuerza el sentido de apoyo (precarga antes de bajar la pluma)."""
        if self.half <= 0:
            return list(coords)
        x, y = coords[0], coords[1]
        r, theta = math.hypot(x, y), math.atan2(y, x)
        d = engage or self._new_direction(coords)
        if d != self.engaged or self.peak is None or (theta - self.peak) * d > 0:
            self.peak = theta   # sentido nuevo, o se avanzo mas en el mismo
        self.engaged = d
        self.last = list(coords)
        cmd = theta + self.engaged * self.half
        return [r * math.cos(cmd), r * math.sin(cmd)] + list(coords[2:])


def calibration_diagnostics(top_left, top_right, bottom_right, bottom_left):
    """Largo de cada lado medido y cuanto se apartan las esquinas de un
    rectangulo, para ver de un vistazo si una medicion salio mal."""
    def dist(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def corner_angle(prev, at, nxt):
        a1 = math.atan2(prev[1] - at[1], prev[0] - at[0])
        a2 = math.atan2(nxt[1] - at[1], nxt[0] - at[0])
        ang = abs(math.degrees(a1 - a2)) % 360
        return min(ang, 360 - ang)

    tl, tr, br, bl = top_left, top_right, bottom_right, bottom_left
    angles = [corner_angle(bl, tl, tr), corner_angle(tl, tr, br),
              corner_angle(tr, br, bl), corner_angle(br, bl, tl)]
    zs = [c[2] for c in (tl, tr, br, bl)]
    return dict(
        top_mm=round(dist(tl, tr), 1), bottom_mm=round(dist(bl, br), 1),
        left_mm=round(dist(tl, bl), 1), right_mm=round(dist(tr, br), 1),
        max_angle_error_deg=round(max(abs(a - 90) for a in angles), 1),
        z_range_mm=round(max(zs) - min(zs), 1),
    )

# Ajuste manual fino, para corregir a ojo despues de una prueba impresa sin
# tener que volver a medir esquinas. Mm que se suman al lado derecho (u=1) y
# se restan al izquierdo (u=0) en el eje "arriba/abajo de la pagina" (X del
# robot). Si el lado derecho del dibujo sale mas ABAJO en la hoja que el
# izquierdo, sube este valor (positivo); si sale mas ARRIBA, bajalo (negativo).
MANUAL_SKEW_MM = 5


def _quad_lerp(values, u, v):
    bottom = values[0] * (1 - u) + values[1] * u  # abajo-izq, abajo-der
    top = values[2] * (1 - u) + values[3] * u      # arriba-izq, arriba-der
    return bottom * (1 - v) + top * v


# Las 4 esquinas se midieron a mano, moviendo el brazo libremente: la MISMA
# posicion x,y se puede "tocar" con la muneca en muchos angulos validos
# distintos, asi que rx,ry,rz de las 4 esquinas no son necesariamente
# consistentes entre si (ejemplo real: rz llego a diferir 42 grados entre
# las dos esquinas de abajo). Interpolar esa orientacion "ruidosa" punto por
# punto le pide al brazo una muneca en un angulo distinto en casi cada punto
# del trazo, sin necesidad real -- eso genera zigzags independientemente del
# pacing o el modo de movimiento (ya se probaron ambos sin cambios).
# Por eso la orientacion se fija en un solo valor (el promedio de las 4
# esquinas, calculado una vez) para toda la hoja, y solo x,y,z varian con la
# posicion real.
FIXED_ORIENTATION = True


def _average_orientation():
    avg = []
    for i in range(3, 6):
        angles = [BOTTOM_LEFT[i], BOTTOM_RIGHT[i], TOP_LEFT[i], TOP_RIGHT[i]]
        cos_vals = [math.cos(math.radians(a)) for a in angles]
        sin_vals = [math.sin(math.radians(a)) for a in angles]
        c = sum(cos_vals) / 4
        s = sum(sin_vals) / 4
        avg.append(math.degrees(math.atan2(s, c)))
    return avg


def bilinear(u, v):
    """
    Interpola las 6 coordenadas (x,y,z,rx,ry,rz) dentro del cuadrilatero
    definido por las 4 esquinas, para un punto normalizado u,v en [0,1].
    u=0 -> izquierda, u=1 -> derecha ; v=0 -> abajo, v=1 -> arriba
    """
    coords = []
    for i in range(3):  # x, y, z: interpolacion lineal normal (esquinas regularizadas)
        coords.append(_quad_lerp(
            [_REG_BOTTOM_LEFT[i], _REG_BOTTOM_RIGHT[i], _REG_TOP_LEFT[i], _REG_TOP_RIGHT[i]], u, v))

    if MANUAL_SKEW_MM:
        coords[0] += (u - 0.5) * MANUAL_SKEW_MM
    coords[2] += _center_z_correction(u, v)

    if FIXED_ORIENTATION:
        coords.extend(_average_orientation())
        return coords

    for i in range(3, 6):  # rx, ry, rz: interpolar como vector para evitar
        # saltos falsos al cruzar el limite +-180 grados (ej. -178 vs 179.6)
        angles = [BOTTOM_LEFT[i], BOTTOM_RIGHT[i], TOP_LEFT[i], TOP_RIGHT[i]]
        cos_vals = [math.cos(math.radians(a)) for a in angles]
        sin_vals = [math.sin(math.radians(a)) for a in angles]
        c = _quad_lerp(cos_vals, u, v)
        s = _quad_lerp(sin_vals, u, v)
        coords.append(math.degrees(math.atan2(s, c)))
    return coords


def parse_gcode(file_path):
    """
    Recorre el gcode y devuelve una lista de movimientos [(x_local, y_local, pluma_abajo), ...].

    Admite G0/G1 y G00/G01, en mayusculas o minusculas. La altura de dibujo
    (pluma abajo) se detecta automaticamente como el valor Z minimo usado en
    todo el archivo, en vez de fiarse de si el eje Z viene en una linea G0 o
    G1 (distintos generadores de gcode usan esa convencion de forma distinta).
    """
    points = []  # (x, y, z) acumulando el estado tras cada linea de movimiento
    x = y = z = None

    with open(file_path, 'r') as f:
        for raw_line in f:
            line = raw_line.split(';')[0].strip()  # quitar comentarios
            tokens = line.split()
            if not tokens:
                continue
            cmd = tokens[0].upper()
            if cmd[0] == 'G' and cmd[1:].isdigit():
                cmd = 'G' + str(int(cmd[1:]))  # G01 -> G1, G00 -> G0
            if cmd not in ('G0', 'G1'):
                continue

            params = dict(COORD_RE.findall(line.upper()))
            if 'X' in params:
                x = float(params['X'])
            if 'Y' in params:
                y = float(params['Y'])
            if 'Z' in params:
                z = float(params['Z'])

            if x is None or y is None or z is None:
                continue  # todavia no se conocen los 3 ejes

            points.append((x, y, z))

    if not points:
        return []

    z_draw = min(p[2] for p in points)
    return [(px, py, abs(pz - z_draw) < 0.5) for px, py, pz in points]


def write_gcode(moves, output_path):
    """
    Escribe una lista de movimientos [(x, y, pluma_abajo), ...] como un
    archivo gcode real (mismo formato/convencion que entiende parse_gcode:
    G0 Z alto = pluma arriba, G1 Z bajo = pluma abajo). Sirve para guardar en
    disco el gcode FINAL (ya recortado/simplificado/filtrado), que es
    distinto del gcode crudo que a veces se escribe antes de procesar.
    """
    UP, DOWN = 10.0, 0.0
    lines = ['; Generado por gcode_draw_lib.write_gcode', 'G90']
    prev_pen_down = None

    for x, y, pen_down in moves:
        if pen_down != prev_pen_down:
            lines.append(f'G0 Z{UP:.2f}' if not pen_down else f'G1 Z{DOWN:.2f} F1000')
            prev_pen_down = pen_down
        cmd = 'G1' if pen_down else 'G0'
        lines.append(f'{cmd} X{x:.2f} Y{y:.2f} F1000')

    lines.append(f'G0 Z{UP:.2f}')
    lines.append('M2 ; fin')

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')

    print(f'{len(moves)} puntos escritos en {output_path}', flush=True)


def simplify_moves(moves, epsilon):
    """
    Reduce la cantidad de puntos de cada trazo (Douglas-Peucker via
    cv2.approxPolyDP), preservando los tramos con la pluma arriba tal cual.
    Util para archivos gcode externos (p.ej. exportados por Luban) que traen
    miles de puntos muy juntos entre si.
    """
    import cv2
    import numpy as np

    if not moves:
        return moves

    result = []
    stroke = []

    def flush_stroke():
        if len(stroke) < 3:
            result.extend((x, y, True) for x, y in stroke)
            return
        pts = np.array(stroke, dtype=np.float32).reshape(-1, 1, 2)
        approx = cv2.approxPolyDP(pts, epsilon, closed=False)
        result.extend((float(x), float(y), True) for x, y in approx.reshape(-1, 2))

    for x, y, pen_down in moves:
        if pen_down:
            stroke.append((x, y))
        else:
            if stroke:
                flush_stroke()
                stroke = []
            result.append((x, y, False))
    if stroke:
        flush_stroke()
    return result


def filter_short_strokes(moves, min_length):
    """Descarta trazos (tramos seguidos con pluma abajo) mas cortos que min_length."""
    if not moves:
        return moves

    result = []
    stroke = []

    def stroke_length(pts):
        total = 0.0
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            total += math.hypot(x1 - x0, y1 - y0)
        return total

    def flush_stroke():
        pts = [(x, y) for x, y, _ in stroke]
        if stroke_length(pts) >= min_length:
            result.extend(stroke)

    for x, y, pen_down in moves:
        if pen_down:
            stroke.append((x, y, True))
        else:
            if stroke:
                flush_stroke()
                stroke = []
            result.append((x, y, False))
    if stroke:
        flush_stroke()
    return result


def _quad_edge_lengths():
    def dist(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])
    width = (dist(_REG_BOTTOM_LEFT, _REG_BOTTOM_RIGHT) + dist(_REG_TOP_LEFT, _REG_TOP_RIGHT)) / 2
    height = (dist(_REG_BOTTOM_LEFT, _REG_TOP_LEFT) + dist(_REG_BOTTOM_RIGHT, _REG_TOP_RIGHT)) / 2
    return width, height


def connect(port=PORT, baud=BAUD):
    from pymycobot.mycobot280 import MyCobot280  # pip install pymycobot si falta
    mc = MyCobot280(port, baud)
    time.sleep(0.5)

    # Si el brazo quedo apagado/sin torque (p.ej. tras un release_all_servos()
    # de una calibracion anterior), los send_coords no producen ningun error
    # pero tampoco mueven nada. Nos aseguramos de que este encendido.
    if mc.is_power_on() != 1:
        print('El brazo no tenia torque/energia activa, encendiendo servos...', flush=True)
        mc.power_on()
        time.sleep(2)
        print('is_power_on ahora:', mc.is_power_on(), flush=True)

    mc.set_fresh_mode(0)
    time.sleep(0.5)
    # Con fresh_mode=1 el robot descarta los comandos intermedios y va
    # directo al ultimo recibido: corta esquinas, se salta puntos y arranca el
    # trazo antes de terminar de bajar. Hay firmwares del Atom que ignoran
    # set_fresh_mode sin dar error, asi que se confirma leyendolo.
    try:
        fresh = mc.get_fresh_mode()
    except Exception as exc:
        fresh = f'error al leer: {exc!r}'
    global LAST_FRESH_MODE
    LAST_FRESH_MODE = fresh
    print(f'fresh_mode = {fresh} (0 = cola, lo que se necesita para dibujar)', flush=True)
    if fresh != 0:
        print('  !! ATENCION: el robot NO confirmo el modo cola. Con el modo "ultimo comando" '
              'se saltan puntos y se cortan esquinas. Si get_fresh_mode no responde, '
              'actualizar el firmware del Atom con myStudio.', flush=True)
    return mc


_ERROR_MEANINGS = {0: 'sin error'}
_ERROR_MEANINGS.update({j: f'joint {j} excede su limite' for j in range(1, 7)})
_ERROR_MEANINGS.update({c: 'proteccion por colision' for c in range(16, 20)})
_ERROR_MEANINGS[32] = 'la cinematica inversa no tiene solucion (posicion/orientacion inalcanzable)'
_ERROR_MEANINGS[33] = 'movimiento lineal sin solucion adyacente'
_ERROR_MEANINGS[34] = 'movimiento lineal sin solucion adyacente'


def _describe_error(code):
    try:
        return _ERROR_MEANINGS.get(int(code), f'codigo desconocido ({code})')
    except (TypeError, ValueError):
        return f'codigo desconocido ({code})'


def _check_and_clear_error(mc):
    """Consulta el error real del robot y lo limpia para poder seguir moviendo.
    Devuelve el codigo de error (0/None si no habia)."""
    try:
        err = mc.get_error_information()
    except Exception as exc:
        print(f'    (no se pudo leer get_error_information: {exc!r})', flush=True)
        return None
    if err:
        print(f'    error del robot: {err} -> {_describe_error(err)}', flush=True)
        try:
            mc.clear_error_information()
            time.sleep(0.2)
        except Exception as exc:
            print(f'    (no se pudo limpiar el error: {exc!r})', flush=True)
    return err


# Esperar a que el brazo se asiente en cada subida/bajada de pluma antes de
# mandar el siguiente movimiento (ver _wait_until_settled). Solo en esos
# puntos: en el resto del trazo sigue la pausa por distancia de siempre.
WAIT_AT_PEN_TRANSITIONS = True
ARRIVAL_TOL_MM = 1.5       # "llego" si queda a esta distancia del destino
STILL_TOL_MM = 0.3         # "quieto" si entre dos lecturas se movio menos que esto
STILL_TIME_S = 0.25        # ...durante este tiempo seguido
SETTLE_TIMEOUT_S = 4.0
SETTLE_PAUSE_S = 0.1       # margen extra tras asentarse (el esfero termina de apoyar)

# Cada cuantos comandos seguidos se espera a que el brazo llegue de verdad
# (ver _wait_until_settled) antes de mandar mas. Sin esto se manda un punto
# cada MIN_PACING sin confirmar nada: si el brazo tarda mas que eso en
# ejecutar cada micro-movimiento, los comandos se acumulan en la cola del
# controlador, y al llenarse se descartan -- el brazo sigue en linea recta al
# siguiente que acepto, y las curvas salen como rectas (visto en papel con el
# firmware actualizado y sin errores del robot: lentes redondos que salian
# aplanados y con esquinas, aunque lo enviado coincidia con el preview). Con
# esto la cola nunca pasa de SYNC_EVERY comandos. 0 = desactivado.
SYNC_EVERY = 4
SYNC_STILL_S = 1.0


def _read_xyz(mc):
    try:
        cur = mc.get_coords()
    except Exception:
        return None
    if isinstance(cur, (list, tuple)) and len(cur) >= 3:
        return cur[:3]
    return None


def _wait_until_settled(mc, target, expected_s, xy_only=False, still_time=None):
    """
    Espera hasta que el brazo llegue a `target` o deje de moverse.

    No usa is_in_position(): en la fase de calibracion nunca confirmo las
    bajadas de pluma (ver nota junto a ASSUMED_SPEED_MM_S), probablemente
    porque con el esfero apoyado el papel no deja llegar a la Z exacta pedida.
    Por eso tambien vale "quieto": lecturas de get_coords que ya no cambian,
    pero solo despues del tiempo de viaje esperado, para no confundir el
    instante previo a que arranque con haber llegado. Devuelve 'llego',
    'quieto' o 'timeout'.
    """
    t0 = time.time()
    last = None
    still_since = None
    timeout = max(SETTLE_TIMEOUT_S, expected_s * 2)
    while time.time() - t0 < timeout:
        cur = _read_xyz(mc)
        now = time.time()
        if cur is not None:
            n = 2 if xy_only else 3
            if math.dist(cur[:n], target[:n]) <= ARRIVAL_TOL_MM:
                return 'llego'
            if last is not None and math.dist(cur, last) < STILL_TOL_MM:
                still_since = still_since or now
                if now - t0 >= expected_s * 0.8 and now - still_since >= (still_time or STILL_TIME_S):
                    return 'quieto'
            else:
                still_since = None
            last = cur
        time.sleep(0.05)
    return 'timeout'


def iter_target_coords(moves, preserve_aspect=False, bounds=None):
    """
    Calcula, para cada punto de `moves`, las coordenadas reales del robot
    (mismo mapeo que usa draw_moves: normalizar a u,v y pasar por bilinear(),
    con el ajuste de PEN_LIFT_MM cuando la pluma esta arriba). No manda nada
    al robot -- sirve tanto para draw_moves() como para generar un preview
    fiel sin conectar hardware. Genera tuplas (i, total, x_local, y_local,
    pen_down, coords), 1-indexado.
    """
    for i, total, x_local, y_local, pen_down, u, v in iter_sheet_uv(moves, preserve_aspect, bounds):
        coords = bilinear(u, v)
        if not pen_down:
            coords[2] += PEN_LIFT_MM
        yield i, total, x_local, y_local, pen_down, coords


def iter_sheet_uv(moves, preserve_aspect=False, bounds=None):
    """
    Posicion de cada punto de `moves` sobre la hoja, normalizada: u de 0
    (izquierda) a 1 (derecha), v de 0 (abajo) a 1 (arriba), segun como se
    nombraron las esquinas. Es el paso previo a bilinear() en
    iter_target_coords; sirve para dibujar un preview de la hoja tal como se
    ve, sin importar donde este puesta respecto del robot. Genera tuplas
    (i, total, x_local, y_local, pen_down, u, v).
    """
    if not moves:
        return

    if bounds is not None:
        x_min, x_max, y_min, y_max = bounds
    else:
        xs = [m[0] for m in moves]
        ys = [m[1] for m in moves]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
    total = len(moves)

    if preserve_aspect:
        quad_w, quad_h = _quad_edge_lengths()
        design_w = x_max - x_min or 1.0
        design_h = y_max - y_min or 1.0
        scale = min(quad_w / design_w, quad_h / design_h)
        cx = (x_min + x_max) / 2
        cy = (y_min + y_max) / 2

    for i, (x_local, y_local, pen_down) in enumerate(moves, 1):
        if preserve_aspect:
            u = 0.5 + (x_local - cx) * scale / quad_w
            v = 0.5 + (y_local - cy) * scale / quad_h
        else:
            u = (x_local - x_min) / (x_max - x_min) if x_max != x_min else 0.5
            v = (y_local - y_min) / (y_max - y_min) if y_max != y_min else 0.5
        u = MARGIN + u * (1 - 2 * MARGIN)
        v = MARGIN + v * (1 - 2 * MARGIN)
        yield i, total, x_local, y_local, pen_down, u, v


# Largo maximo de un movimiento con la pluma abajo; los tramos mas largos se
# parten en pasos iguales. En una recta larga el firmware del myCobot a veces
# no encuentra camino lineal (error 33/34, "sin solucion adyacente") y
# descarta el movimiento entero: el brazo sigue al punto siguiente y se saltea
# una esquina (visto: el contorno de una hoja horizontal, lados de ~145 mm
# cruzando por detras de la base, salia como un triangulo). Los retratos casi
# no cambian: sus tramos ya suelen ser mas cortos que esto.
MAX_DRAW_SEGMENT_MM = 10.0


def draw_moves(mc, moves, preserve_aspect=False, bounds=None, progress_cb=None, check_every=25):
    """
    Envia al robot una lista de movimientos [(x_local, y_local, pluma_abajo), ...].

    preserve_aspect=False (default): estira X e Y de forma independiente para
    llenar todo el cuadrilatero (bien para disenos ya pensados para esa area).
    preserve_aspect=True: usa una sola escala para X e Y (centrado), para no
    deformar un diseno cuya proporcion ancho/alto no coincide con la de la hoja.

    bounds=(x_min, x_max, y_min, y_max): si se da, se usa en vez de calcular
    el rango a partir de los propios puntos. Util para que una figura mas
    chica que su "lienzo" (p.ej. un circulo chico pensado dentro de un area
    de referencia mas grande) no se estire siempre a pantalla completa -- el
    tamano relativo al lienzo se respeta.

    progress_cb(i, total, message): opcional, se llama despues de cada punto
    (ademas de lo que ya se imprime por consola). Pensado para que una
    interfaz externa (p.ej. una app web) pueda mostrar progreso sin tener que
    parsear stdout.

    check_every: cada cuantos puntos se consulta el error del robot (ademas
    de al final de cada trazo). 1 = en cada punto: mas lento, pero dice
    exactamente donde fallo; sirve para pruebas de calibracion.

    Devuelve {'robot_errors': [(punto, codigo, descripcion), ...]}.
    """
    if not moves:
        print('No hay movimientos para dibujar.')
        return dict(robot_errors=[])

    total = len(moves)
    print(f'{total} puntos.', flush=True)

    start_time = time.time()
    errors = 0
    robot_errors = []
    prev_pen_down = None
    prev_coords = None
    # since_sync: comandos mandados desde la ultima vez que se confirmo la
    # llegada; sync_waits: cuanto tuvo que esperar cada sincronizacion (si
    # son largas, el brazo iba atrasado respecto de lo que se le mandaba)
    state = {'xyz': None, 'since_sync': 0, 'sync_waits': [], 'sync_results': []}
    backlash = J1Backlash(J1_BACKLASH_DEG)
    if J1_BACKLASH_DEG > 0:
        print(f'Compensando juego de la base (J1): {J1_BACKLASH_DEG:g} grados.', flush=True)

    def note_error(point, code):
        if code:
            robot_errors.append((point, code, _describe_error(code)))

    def send(coords, speed, settle, label, engage=None):
        """Manda un punto y espera: a que se asiente si `settle` (subidas y
        bajadas de pluma), o la pausa por distancia en el resto. `coords` es
        donde tiene que quedar la pluma; lo que se manda pasa por la
        compensacion del juego de J1 (engage: ver J1Backlash.command)."""
        nonlocal errors
        if engage is None:
            d = backlash.reversal(coords)
            if d:
                # la base cambia de sentido: tomar el juego quieto sobre el punto
                # anterior y recien despues avanzar. Si se toma en el mismo
                # movimiento que avanza, la pluma recorre ese tramo solo en el
                # otro eje y queda un escalon.
                send(backlash.last, speed, settle=False, label=f'toma del juego antes de {label}', engage=d)
        coords = backlash.command(coords, engage)
        try:
            mc.send_coords(coords, speed, MOVE_MODE)
        except Exception as exc:
            errors += 1
            print(f'  !! error en {label}: {exc!r} (se continua con el siguiente)', flush=True)
        xyz = tuple(coords[:3])
        dist = math.dist(state['xyz'], xyz) if state['xyz'] is not None else 0.0
        state['xyz'] = xyz
        pacing = min(MAX_PACING, max(MIN_PACING, dist / ASSUMED_SPEED_MM_S + PACING_OVERHEAD))
        if settle and WAIT_AT_PEN_TRANSITIONS:
            result = _wait_until_settled(mc, coords, expected_s=pacing)
            if result == 'timeout':
                print(f'    (no se confirmo la llegada en {label}; se sigue igual)', flush=True)
            time.sleep(SETTLE_PAUSE_S)
            state['since_sync'] = 0
        else:
            time.sleep(pacing)
            state['since_sync'] += 1
            if SYNC_EVERY and state['since_sync'] >= SYNC_EVERY:
                t0 = time.time()
                # solo x,y: con la pluma apoyada la Z real no siempre llega a la
                # pedida. "Quieto" solo si lleva SYNC_STILL_S sin moverse: entre
                # dos micro-movimientos encolados el brazo puede parar un instante
                # sin haber terminado la cola.
                result = _wait_until_settled(mc, coords, expected_s=0, xy_only=True, still_time=SYNC_STILL_S)
                waited = time.time() - t0
                state['sync_waits'].append(waited)
                # diagnostico: donde quedo realmente el brazo respecto del punto pedido
                cur = _read_xyz(mc)
                off = math.dist(cur[:2], coords[:2]) if cur is not None else None
                state['sync_results'].append((result, off))
                print(f'    sync: {result} tras {waited:.2f}s; el brazo quedo a '
                      f'{"?" if off is None else f"{off:.1f}"} mm (x,y) del punto pedido', flush=True)
                state['since_sync'] = 0

    targets = list(iter_target_coords(moves, preserve_aspect, bounds))

    def first_stroke_direction(k):
        """Sentido de giro de la base del primer tramo que mueve J1 en el
        trazo que empieza (pluma abajo) en targets[k]; 0 si el trazo no gira la base."""
        start = targets[k][5]
        for t in targets[k + 1:]:
            if not t[4]:
                return 0
            d = backlash.direction(start, t[5])
            if d:
                return d
        return 0

    for idx, (i, total, x_local, y_local, pen_down, coords) in enumerate(targets):

        speed = DRAW_SPEED if pen_down else TRAVEL_SPEED
        estado = 'ABAJO' if pen_down else 'ARRIBA'
        is_transition = pen_down != prev_pen_down
        elapsed = time.time() - start_time
        rounded = [round(c, 1) for c in coords]
        tag = 'transicion' if is_transition else 'continuo'
        print(f'[{i:04d}/{total}] t={elapsed:6.1f}s pluma={estado:6s} ({tag}) '
              f'-> coords={rounded} vel={speed}', flush=True)
        if progress_cb is not None:
            progress_cb(i, total, f'pluma {estado.lower()} ({tag})')

        if prev_pen_down and not pen_down and prev_coords is not None:
            # Fin de trazo: subir en VERTICAL sobre el ultimo punto antes de
            # viajar. Sin esto, el viaje al proximo trazo salia en diagonal
            # desde el papel (sube y se corre de lado a la vez), y el esfero
            # arrastraba unos mm al despegarse: un gancho al final de cada trazo.
            lift = list(prev_coords)
            lift[2] += PEN_LIFT_MM
            send(lift, TRAVEL_SPEED, settle=True, label=f'levantar pluma antes del punto {i}')
            note_error(i - 1, _check_and_clear_error(mc))

        # Tramo largo con la pluma abajo: se parte en pasos de hasta
        # MAX_DRAW_SEGMENT_MM (ver ahi el porque).
        if pen_down and prev_pen_down and prev_coords is not None:
            seg = math.dist(prev_coords[:3], coords[:3])
            n_steps = int(math.ceil(seg / MAX_DRAW_SEGMENT_MM))
            for k in range(1, n_steps):
                t = k / n_steps
                mid = [a + (b - a) * t for a, b in zip(prev_coords, coords)]
                send(mid, speed, settle=False, label=f'tramo intermedio {k}/{n_steps} hacia el punto {i}')
                if check_every == 1:
                    note_error(i, _check_and_clear_error(mc))

        # La bajada (pluma abajo justo despues de un viaje) ya es vertical: el
        # punto de viaje esta sobre el mismo x,y. Se espera a que el esfero
        # este apoyado antes de mandar el primer movimiento de lado.
        landing = pen_down and prev_pen_down is False
        send(coords, speed, settle=landing, label=f'punto {i}')

        # Entrada con la pluma arriba: si el trazo siguiente arranca girando la
        # base al reves de como llego el brazo, se toma el juego de J1 en el aire
        # (la pluma no se mueve, solo el engranaje cambia de lado) en vez de al
        # empezar a dibujar, donde se perdian los primeros mm del trazo.
        if (not pen_down and backlash.half > 0 and idx + 1 < len(targets) and targets[idx + 1][4]):
            d = first_stroke_direction(idx + 1)
            if d and d != backlash.engaged:
                send(coords, speed, settle=False, label=f'precarga del juego antes del punto {i + 1}', engage=d)

        prev_pen_down = pen_down
        prev_coords = coords

        if i % check_every == 0:
            # Chequeo liviano (no bloquea esperando posicion, solo pregunta si
            # el firmware quedo con algun error real y lo limpia para que no
            # bloquee los comandos siguientes).
            note_error(i, _check_and_clear_error(mc))

    # Al terminar: primero subir en vertical sobre el ultimo punto (mismo
    # motivo que al final de cada trazo), despues ir al centro.
    if prev_pen_down and prev_coords is not None:
        lift = list(prev_coords)
        lift[2] += PEN_LIFT_MM
        send(lift, TRAVEL_SPEED, settle=True, label='levantar pluma al final')
    end_coords = bilinear(0.5, 0.5)
    end_coords[2] += PEN_LIFT_MM
    send(end_coords, TRAVEL_SPEED, settle=False, label='volver al centro')
    note_error(total, _check_and_clear_error(mc))

    total_time = time.time() - start_time
    print(f'Dibujo terminado en {total_time:.1f}s. Errores: {errors}/{total}. '
          f'Errores del robot: {len(robot_errors)}.', flush=True)
    waits = state['sync_waits']
    sync = None
    if waits:
        waits_sorted = sorted(waits)
        sync = dict(count=len(waits), mean_s=round(sum(waits) / len(waits), 2),
                    p90_s=round(waits_sorted[int(0.9 * (len(waits) - 1))], 2), max_s=round(waits_sorted[-1], 2))
        results = state['sync_results']
        offs = sorted(o for _, o in results if o is not None)
        sync['reached'] = sum(1 for r, _ in results if r == 'llego')
        sync['still'] = sum(1 for r, _ in results if r == 'quieto')
        if offs:
            sync['off_mean_mm'] = round(sum(offs) / len(offs), 1)
            sync['off_p90_mm'] = round(offs[int(0.9 * (len(offs) - 1))], 1)
            sync['off_max_mm'] = round(offs[-1], 1)
        print(f'Sincronizaciones: {sync["count"]}, espera media {sync["mean_s"]}s, '
              f'p90 {sync["p90_s"]}s, max {sync["max_s"]}s (esperas largas = el brazo iba atrasado)', flush=True)
        print(f'  llego al punto: {sync["reached"]}, se detuvo sin llegar: {sync["still"]}; '
              f'distancia al punto pedido: media {sync.get("off_mean_mm")} mm, p90 {sync.get("off_p90_mm")} mm, '
              f'max {sync.get("off_max_mm")} mm', flush=True)
    return dict(robot_errors=robot_errors, sync=sync)


def _stroke(points):
    x0, y0 = points[0]
    return [(x0, y0, False), (x0, y0, True)] + [(x, y, True) for x, y in points[1:]]


def area_bounds():
    """bounds para draw_moves/iter_target_coords con preserve_aspect=False en
    los que 1 unidad = 1 mm real sobre la hoja (origen abajo a la izquierda)."""
    quad_w, quad_h = _quad_edge_lengths()
    return (0, quad_w, 0, quad_h)


def outline_moves(inset_mm=2.0):
    """Rectangulo del area calibrada completa, apenas hacia adentro de las
    esquinas: la prueba mas directa de que las 4 esquinas estan bien."""
    _, w, _, h = area_bounds()
    a, b = inset_mm, inset_mm
    return _stroke([(a, b), (w - a, b), (w - a, h - b), (a, h - b), (a, b)])


TEST_PATTERN_SIZE_MM = 100      # tamano ideal (mm)
TEST_PATTERN_MIN_MM = 40        # por debajo de esto ya no sirve para medir nada
TEST_PATTERN_MARGIN_MM = 1.0    # separacion minima del borde del area


def test_pattern_size():
    """Lado (mm) con que se va a dibujar el patron: 100, o menos si el area
    calibrada no da (ej. una hoja horizontal de 140x88 mm -> 86 mm)."""
    _, w, _, h = area_bounds()
    return min(TEST_PATTERN_SIZE_MM, min(w, h) - 2 * TEST_PATTERN_MARGIN_MM)


def test_pattern_moves():
    """
    Patron de depuracion cuadrado, centrado en el area calibrada, en mm reales
    (usar con area_bounds()). Mide 100x100 mm si el area lo permite; si no,
    se achica en proporcion hasta entrar (ver test_pattern_size), y todas las
    medidas de abajo se escalan igual:
      - marco del lado completo: la escala medida sobre la distancia mas larga.
      - cuadrado de la mitad del lado (50 mm a tamano completo): esquinas
        redondeadas o lados que no cierran indican puntos salteados o
        movimiento no lineal.
      - circulo de 36 mm (72 pasos, ~1.6 mm cada uno), como las curvas del
        retrato: muestra escalones o temblor.
      - 10 rayitas verticales y 10 horizontales de 5 mm: cada una es un ciclo
        completo de bajar/dibujar/subir, y muestran si hay ganchos al empezar
        o al terminar el trazo, y si dependen de la direccion.
    Lanza ValueError si el area no da ni para TEST_PATTERN_MIN_MM.
    """
    _, w, _, h = area_bounds()
    size = test_pattern_size()
    if size < TEST_PATTERN_MIN_MM:
        raise ValueError(f'el area calibrada ({w:.0f}x{h:.0f} mm) es demasiado chica para el patron '
                         f'(minimo {TEST_PATTERN_MIN_MM}x{TEST_PATTERN_MIN_MM} mm).')
    s = size / TEST_PATTERN_SIZE_MM
    ox, oy = (w - size) / 2, (h - size) / 2

    def at(points):
        # el diseno esta en un cuadrado de 100x100; se escala a `size`
        return _stroke([(ox + x * s, oy + y * s) for x, y in points])

    moves = at([(0, 0), (100, 0), (100, 100), (0, 100), (0, 0)])
    moves += at([(6, 44), (56, 44), (56, 94), (6, 94), (6, 44)])
    cx, cy, r, n = 77, 69, 18, 72
    moves += at([(cx + r * math.cos(2 * math.pi * k / n), cy + r * math.sin(2 * math.pi * k / n))
                 for k in range(n + 1)])
    for k in range(10):
        x = 5 + 10 * k
        moves += at([(x, 24), (x, 29)])
    for k in range(10):
        x = 5 + 9.5 * k
        moves += at([(x, 10), (x + 5, 10)])
    return moves


FACE_MAX_HEIGHT_MM = 100        # alto de la cara (frente a menton) si el area lo permite
FACE_MIN_HEIGHT_MM = 40
FACE_MARGIN_MM = 5.0


def test_face_moves(rotation_deg=0):
    """
    Rostro generico (contorno ovalado, ojos en forma de almendra con pupila,
    nariz triangular y boca de dos curvas con la linea del medio), centrado
    en el area calibrada, en mm reales (usar con area_bounds()). Con
    rotation_deg=0 la frente apunta a las esquinas "arriba" del perfil; se
    gira igual que pipeline.rotate_moves
    (antihorario, multiplo de 90), asi que sale orientado como saldria un
    retrato con ese perfil. Lanza ValueError si el area no da ni para
    FACE_MIN_HEIGHT_MM.
    """
    def ellipse(cx, cy, rx, ry, n, a0=0.0, a1=2 * math.pi):
        return [(cx + rx * math.cos(a0 + (a1 - a0) * k / n), cy + ry * math.sin(a0 + (a1 - a0) * k / n))
                for k in range(n + 1)]

    def arc(x0, x1, y, bulge, n=24):
        # curva de (x0, y) a (x1, y) que se aparta `bulge` en el medio (+ arriba, - abajo);
        # con forma de seno, asi dos curvas con las mismas puntas cierran en punta
        return [(x0 + (x1 - x0) * k / n, y + bulge * math.sin(math.pi * k / n)) for k in range(n + 1)]

    def almond(cx, cy, half_w, up, down, middle=False):
        # dos medialunas separadas, las dos de izquierda a derecha, que se juntan en las
        # puntas (+ la linea recta del medio si middle). No va en un solo trazo: al dar la
        # vuelta en la punta con la pluma apoyada, la friccion la arrastra por el mismo
        # lado y la punta sale como una raya en vez de cerrar.
        x0, x1 = cx - half_w, cx + half_w
        strokes = [arc(x0, x1, cy, up), arc(x0, x1, cy, -down)]
        if middle:
            strokes.append([(x0, cy), (x1, cy)])
        return strokes

    # diseno con la cara de 100 de alto centrada en (0, 0), y hacia arriba = frente
    strokes = [
        ellipse(0, 0, 38, 50, 96),                                       # cara
        *almond(-14, 10, 9, 5, 3.5), *almond(14, 10, 9, 5, 3.5),         # ojos
        ellipse(-14, 10.5, 2.2, 2.2, 16), ellipse(14, 10.5, 2.2, 2.2, 16),  # pupilas
        [(0, 4), (-5, -9), (5, -9), (0, 4)],                             # nariz
        *almond(0, -22, 17, 4, 6, middle=True),                          # boca
    ]
    turns = (int(rotation_deg) // 90) % 4
    rot = {0: lambda x, y: (x, y), 1: lambda x, y: (-y, x),
           2: lambda x, y: (-x, -y), 3: lambda x, y: (y, -x)}[turns]
    strokes = [[rot(x, y) for x, y in s] for s in strokes]
    design_w, design_h = (100, 76) if turns % 2 else (76, 100)

    _, w, _, h = area_bounds()
    s = min((w - 2 * FACE_MARGIN_MM) / design_w, (h - 2 * FACE_MARGIN_MM) / design_h,
            FACE_MAX_HEIGHT_MM / 100)
    if 100 * s < FACE_MIN_HEIGHT_MM:
        raise ValueError(f'el area calibrada ({w:.0f}x{h:.0f} mm) es demasiado chica para el rostro '
                         f'(minimo {FACE_MIN_HEIGHT_MM} mm de alto).')
    moves = []
    for points in strokes:
        moves += _stroke([(w / 2 + x * s, h / 2 + y * s) for x, y in points])
    return moves


def draw_gcode_file(file_path, port=PORT, baud=BAUD):
    moves = parse_gcode(file_path)
    mc = connect(port, baud)
    draw_moves(mc, moves)
