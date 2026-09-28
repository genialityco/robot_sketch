#!/usr/bin/python
# -*- coding:utf-8 -*-
# Dibujar con la base (J1) girando siempre en el mismo sentido mientras la
# pluma esta apoyada.
#
# Por que: J1 tiene ~1.5 grados de juego (medido el 2026-09-25: ~5-6 mm con la
# hoja a ~204 mm de la base) y get_coords no lo ve. En X (acercarse/alejarse
# de la base) el dibujo sale bien porque J2/J3 cargan el peso del brazo y
# quedan siempre apoyados del mismo lado del engranaje. J1 gira en horizontal,
# nada lo precarga, y la friccion de la pluma lo pasa de un flanco al otro
# cada vez que la base invierte el giro. Por eso los rasgos chicos de lado a
# lado (ojos, pupilas, boca, base de la nariz) salen achicados, abiertos o
# desaparecen.
#
# En vez de compensar el juego (J1Backlash en gcode_draw_lib, que necesita
# saber exactamente cuanto es, y eso cambia con la friccion y la presion),
# aca se evita:
#   1. cada trazo se parte en los puntos donde la base invertiria el giro,
#   2. los tramos que irian "al reves" se dibujan desde el otro extremo,
#   3. a cada tramo se llega por el aire desde el mismo lado (APPROACH_MM
#      antes), para que el engranaje ya este apoyado en el flanco correcto
#      cuando baja la pluma.
# Asi, mientras se dibuja, el engranaje queda siempre del mismo lado y el
# tamano del juego deja de importar: solo corre el dibujo entero una
# distancia fija, que no se nota.
#
# Costo: mas levantadas de pluma (una figura cerrada, como una pupila, pasa a
# ser dos medias lunas) y algo mas de tiempo. Si el juego se elimina de raiz
# (precarga mecanica de J1), esto se apaga desde el perfil.
#
# Se conecta en pipeline.draw y pipeline.render_preview: plan() se llama
# justo antes de mandar los moves al robot o de dibujar el preview.

import math

import gcode_draw_lib as lib

# Lo prende/apaga el perfil de calibracion activo (casilla "Dibujar con la base
# en un solo sentido", ver calibration._apply). Apagado hasta que se elija.
ENABLED = False

# +1: con la pluma abajo la base solo gira en sentido antihorario visto desde
# arriba (theta = atan2(y, x) creciente, Y del robot creciente); -1: horario.
# Lo elige el perfil (j1_direction). NO da igual: en el brazo real J1 se traba
# girando hacia Y creciente (visto 2026-09-26 con "Ir encima": de abajo a la
# derecha al centro va a tirones, al reves fluye), y con +1 cada tramo
# arrancaba con una rayita (J1 pegado mientras X ya se movia) y terminaba
# sin cerrar (J1 atrasado).
DIRECTION = 1

# Retrocesos de la base mas chicos que esto (mm de arco) no parten el trazo:
# evita cortar una linea casi derecha en X por el temblor del esqueleto. Ese
# retroceso chico se pierde en el juego, pero es menos de medio milimetro.
HYSTERESIS_MM = 0.5

# Cuanto antes (mm de arco, en sentido contrario a DIRECTION) se pasa por el
# aire antes de ir al inicio de cada tramo. Tiene que ser mas que el juego
# (~5-6 mm a 204 mm de la base, algo mas en el borde lejano de la hoja).
APPROACH_MM = 9.0


def _robot_xy(x, y, preserve_aspect, bounds):
    """(x, y) del robot para un punto del dibujo, con el mismo mapeo que
    draw_moves. Con bounds fijos un punto suelto se mapea igual que dentro
    del dibujo completo."""
    for *_, coords in lib.iter_target_coords([(x, y, True)], preserve_aspect, bounds):
        return coords[0], coords[1]


def _angle_diff(a, b):
    """b - a en radianes, en (-pi, pi]."""
    return (b - a + math.pi) % (2 * math.pi) - math.pi


def _unwrap(angles):
    out = [angles[0]]
    for a in angles[1:]:
        out.append(out[-1] + _angle_diff(out[-1], a))
    return out


def _split(th, r):
    """Tramos [(i0, i1, sentido)] de un trazo en los que la base gira en un
    solo sentido, con histeresis de HYSTERESIS_MM. Los tramos consecutivos
    comparten el punto de corte. sentido 0 = el tramo casi no gira la base
    (una linea en X)."""
    n = len(th)
    runs = []
    start, d, peak = 0, 0, 0
    for k in range(1, n):
        if d == 0:
            moved = (th[k] - th[start]) * r[k]
            if abs(moved) >= HYSTERESIS_MM:
                d = 1 if moved > 0 else -1
                peak = max(range(start, k + 1), key=lambda j: th[j] * d)
        elif (th[k] - th[peak]) * d > 0:
            peak = k
        elif (th[peak] - th[k]) * d * r[k] >= HYSTERESIS_MM:
            runs.append((start, peak, d))
            start, d = peak, -d
            peak = max(range(start, k + 1), key=lambda j: th[j] * d)
    runs.append((start, n - 1, d))
    return runs


def _approach_point(x, y, preserve_aspect, bounds, span):
    """Punto del dibujo (pluma arriba) desde el cual, yendo en linea recta
    hasta (x, y), la base gira APPROACH_MM de arco en el sentido DIRECTION."""
    eps = span * 1e-3
    p0 = _robot_xy(x, y, preserve_aspect, bounds)
    px = _robot_xy(x + eps, y, preserve_aspect, bounds)
    py = _robot_xy(x, y + eps, preserve_aspect, bounds)
    th0 = math.atan2(p0[1], p0[0])
    gx = _angle_diff(th0, math.atan2(px[1], px[0])) / eps   # radianes de base por unidad del dibujo
    gy = _angle_diff(th0, math.atan2(py[1], py[0])) / eps
    g = math.hypot(gx, gy)
    if g < 1e-12:
        return x, y
    step = APPROACH_MM / (g * math.hypot(*p0))
    return x - DIRECTION * step * gx / g, y - DIRECTION * step * gy / g


def plan(moves, preserve_aspect=False, bounds=None):
    """
    moves [(x, y, pluma_abajo), ...] -> (moves nuevos, bounds).

    Devuelve los bounds FIJOS del dibujo original: los puntos de llegada
    quedan un poco afuera del dibujo y, si no se fijan, cambiarian la escala
    y el encuadre. Quien llame tiene que usar esos bounds al dibujar (y en el
    preview).
    """
    if not moves:
        return moves, bounds
    if bounds is None:
        xs = [m[0] for m in moves]
        ys = [m[1] for m in moves]
        bounds = (min(xs), max(xs), min(ys), max(ys))
    span = max(bounds[1] - bounds[0], bounds[3] - bounds[2]) or 1.0

    # trazos = corridas de puntos con la pluma abajo, con su posicion en el robot
    strokes, cur = [], []
    for _i, _t, x, y, pen_down, coords in lib.iter_target_coords(moves, preserve_aspect, bounds):
        if pen_down:
            cur.append((x, y, coords[0], coords[1]))
        elif cur:
            strokes.append(cur)
            cur = []
    if cur:
        strokes.append(cur)
    if not strokes:
        return moves, bounds

    pieces = []
    for s in strokes:
        if len(s) < 2:
            pieces.append(s)
            continue
        th = _unwrap([math.atan2(p[3], p[2]) for p in s])
        r = [math.hypot(p[2], p[3]) for p in s]
        for i0, i1, d in _split(th, r):
            piece = s[i0:i1 + 1]
            pieces.append(piece[::-1] if d == -DIRECTION else piece)

    # orden: siempre el tramo cuyo inicio queda mas cerca de donde termino el anterior
    out = []
    remaining = list(range(len(pieces)))
    pos = pieces[0][0][2:]
    while remaining:
        j = min(remaining, key=lambda k: math.dist(pieces[k][0][2:], pos))
        remaining.remove(j)
        piece = pieces[j]
        x0, y0 = piece[0][0], piece[0][1]
        ax, ay = _approach_point(x0, y0, preserve_aspect, bounds, span)
        out.append((ax, ay, False))
        out.append((x0, y0, False))
        out.extend((p[0], p[1], True) for p in piece)
        pos = piece[-1][2:]

    print('J1 en un solo sentido: %d trazos -> %d tramos' % (len(strokes), len(pieces)), flush=True)
    return out, bounds
