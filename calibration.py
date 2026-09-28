#!/usr/bin/python
# -*- coding:utf-8 -*-
# Perfiles de calibracion del area de dibujo.
#
# Un perfil = las 4 esquinas de la hoja medidas con la pluma tocando el papel
# ([x, y, z, rx, ry, rz] cada una, como las devuelve get_coords), mas el
# ajuste fino MANUAL_SKEW_MM y cuanto se levanta la pluma. Sirven para tener
# varias mesas/hojas/soportes calibrados y cambiar entre ellos sin editar
# gcode_draw_lib.py. Se guardan como JSON en calibration_profiles/, y el
# activo se aplica a gcode_draw_lib con set_calibration() -- desde ese momento
# lo usan el preview, el dibujo y las pruebas.

import json
import math
import os
import re
import time

import pipeline as pl

HERE = os.path.dirname(os.path.abspath(__file__))
PROFILES_DIR = os.path.join(HERE, 'calibration_profiles')
ACTIVE_FILE = os.path.join(PROFILES_DIR, '_activo.txt')

CORNERS = ('top_left', 'top_right', 'bottom_right', 'bottom_left')
CORNER_LABELS = {'top_left': 'arriba izquierda', 'top_right': 'arriba derecha',
                 'bottom_right': 'abajo derecha', 'bottom_left': 'abajo izquierda'}

_ID_RE = re.compile(r'^[a-z0-9_-]{1,40}$')


def _slug(name):
    import unicodedata
    ascii_name = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode()
    slug = re.sub(r'[^a-z0-9]+', '-', ascii_name.lower()).strip('-')[:40]
    return slug or 'perfil'


def _path(profile_id):
    if not _ID_RE.match(profile_id or ''):
        raise ValueError('id de perfil invalido')
    return os.path.join(PROFILES_DIR, profile_id + '.json')


def _validate(profile):
    name = str(profile.get('name', '')).strip()
    if not name or len(name) > 60:
        raise ValueError('el nombre del perfil es obligatorio (hasta 60 caracteres)')
    corners = {}
    for key in CORNERS:
        values = (profile.get('corners') or {}).get(key)
        if not isinstance(values, (list, tuple)) or len(values) != 6:
            raise ValueError('la esquina %s necesita 6 valores (x, y, z, rx, ry, rz)' % CORNER_LABELS[key])
        try:
            values = [float(v) for v in values]
        except (TypeError, ValueError):
            raise ValueError('la esquina %s tiene valores no numericos' % CORNER_LABELS[key])
        if not all(math.isfinite(v) for v in values):
            raise ValueError('la esquina %s tiene valores invalidos' % CORNER_LABELS[key])
        corners[key] = [round(v, 2) for v in values]
    z_adjust = {}
    for key in CORNERS:
        try:
            adj = float((profile.get('z_adjust_mm') or {}).get(key, 0))
        except (TypeError, ValueError):
            raise ValueError('el ajuste de altura de %s no es un numero' % CORNER_LABELS[key])
        if not math.isfinite(adj) or not -Z_ADJUST_LIMIT_MM <= adj <= Z_ADJUST_LIMIT_MM:
            raise ValueError('el ajuste de altura de %s debe estar entre -%g y %g mm'
                             % (CORNER_LABELS[key], Z_ADJUST_LIMIT_MM, Z_ADJUST_LIMIT_MM))
        z_adjust[key] = round(adj, 1)
    skew = float(profile.get('manual_skew_mm', 0))
    lift = float(profile.get('pen_lift_mm', pl.lib.PEN_LIFT_MM))
    if not -30 <= skew <= 30:
        raise ValueError('el ajuste fino debe estar entre -30 y 30 mm')
    if not 5 <= lift <= 80:
        raise ValueError('el levantado de pluma debe estar entre 5 y 80 mm')
    center_z = profile.get('center_z')
    if center_z in (None, ''):
        center_z = None
    else:
        try:
            center_z = float(center_z)
        except (TypeError, ValueError):
            raise ValueError('la altura del centro no es un numero')
        corners_z = sum(corners[k][2] for k in CORNERS) / 4
        if not math.isfinite(center_z) or abs(center_z - corners_z) > CENTER_MAX_OFFSET_MM:
            raise ValueError('la altura del centro (Z %.1f) difiere mas de %d mm del promedio de las esquinas '
                             '(Z %.1f): probablemente se capturo sin la pluma apoyada en el papel'
                             % (center_z, CENTER_MAX_OFFSET_MM, corners_z))
        center_z = round(center_z, 2)
    try:
        center_adjust = round(float(profile.get('center_z_adjust_mm', 0) or 0), 1)
    except (TypeError, ValueError):
        raise ValueError('el ajuste de altura del centro no es un numero')
    if not -Z_ADJUST_LIMIT_MM <= center_adjust <= Z_ADJUST_LIMIT_MM:
        raise ValueError('el ajuste de altura del centro debe estar entre -%g y %g mm'
                         % (Z_ADJUST_LIMIT_MM, Z_ADJUST_LIMIT_MM))
    rotation = profile.get('drawing_rotation', 0)
    if rotation == 'auto':  # modo que existio brevemente; giraba mal las hojas al costado
        rotation = 0
    try:
        rotation = int(rotation)
    except (TypeError, ValueError):
        rotation = None
    if rotation not in (0, 90, 180, 270):
        raise ValueError('la orientacion del dibujo debe ser 0, 90, 180 o 270')
    try:
        backlash = round(float(profile.get('j1_backlash_deg', 0) or 0), 2)
    except (TypeError, ValueError):
        raise ValueError('el juego de la base no es un numero')
    if not 0 <= backlash <= J1_BACKLASH_MAX_DEG:
        raise ValueError('el juego de la base debe estar entre 0 y %g grados' % J1_BACKLASH_MAX_DEG)
    center_xy = profile.get('center_xy')
    if center_xy in (None, '', []):
        center_xy = None
    else:
        try:
            center_xy = [round(float(v), 2) for v in center_xy]
        except (TypeError, ValueError):
            raise ValueError('la posicion del centro debe ser dos numeros (x, y)')
        if len(center_xy) != 2 or not all(math.isfinite(v) for v in center_xy):
            raise ValueError('la posicion del centro debe ser dos numeros (x, y)')
        ccx, ccy = _corners_center_xy(corners)
        off = math.dist(center_xy, (ccx, ccy))
        if off > CENTER_XY_MAX_OFFSET_MM:
            raise ValueError('la punta esta en x %.1f, y %.1f, a %.0f mm del centro que dan las esquinas '
                             '(x %.1f, y %.1f; se acepta hasta %d mm). O la punta no estaba en el centro de la '
                             'hoja, o alguna esquina esta mal capturada: con "Ir encima" de cada esquina se ve '
                             'cual no cae en su lugar.'
                             % (center_xy[0], center_xy[1], off, ccx, ccy, CENTER_XY_MAX_OFFSET_MM))
    return dict(name=name, corners=corners, z_adjust_mm=z_adjust, center_z=center_z,
                center_z_adjust_mm=center_adjust, center_xy=center_xy, manual_skew_mm=skew, pen_lift_mm=lift,
                drawing_rotation=rotation, j1_backlash_deg=backlash,
                j1_unidirectional=bool(profile.get('j1_unidirectional', False)),
                j1_direction=-1 if str(profile.get('j1_direction', 1)) == '-1' else 1,
                tablet_mode=bool(profile.get('tablet_mode', False)),
                tablet_orient=_tablet_orient(profile.get('tablet_orient')))


# Donde queda, en la pantalla de la tableta, el lado de la hoja lejos del robot
# ("arriba" de las esquinas). Es del perfil (depende de como se puso la tableta
# al calibrarlo): /tablet lo toma del perfil activo. None = el ajuste de la tableta.
TABLET_ORIENTS = ('arriba', 'derecha', 'abajo', 'izquierda')


def _tablet_orient(value):
    if value in (None, ''):
        return None
    if value not in TABLET_ORIENTS:
        raise ValueError('orientacion de la tableta invalida: %s' % value)
    return value


def active_tablet_orient():
    profile_id = active_id()
    return load(profile_id).get('tablet_orient') if profile_id else None


# Un centro capturado mas lejos que esto del que dan las esquinas no es un
# error de medicion: la punta no estaba en el centro de la hoja.
CENTER_XY_MAX_OFFSET_MM = 20


def _corners_center_xy(corners):
    return (sum(corners[k][0] for k in CORNERS) / 4, sum(corners[k][1] for k in CORNERS) / 4)


# Compensacion del juego de J1 (ver gcode_draw_lib.J1Backlash). Medido ~1.5
# grados; mucho mas que esto seria un error de tipeo, y cada cambio de sentido
# de la base haria un salto grande.
J1_BACKLASH_MAX_DEG = 4


# Una Z de centro mas lejos que esto del promedio de las esquinas no es una
# hoja real: seguro se capturo con la pluma en el aire.
CENTER_MAX_OFFSET_MM = 15


def effective_center_z(profile):
    """Z del centro con su ajuste aplicado (mismo signo que las esquinas:
    negativo = mas arriba), o None si el perfil no tiene centro."""
    if profile.get('center_z') is None:
        return None
    return round(profile['center_z'] - profile.get('center_z_adjust_mm', 0), 2)


def drawing_orientation(profile):
    """rx, ry, rz con los que se dibuja todo con este perfil: el promedio de
    las 4 esquinas (misma cuenta que gcode_draw_lib._average_orientation, que
    se fija asi para toda la hoja)."""
    pts = [profile['corners'][k] for k in CORNERS]
    out = []
    for i in range(3, 6):
        s = sum(math.sin(math.radians(p[i])) for p in pts)
        co = sum(math.cos(math.radians(p[i])) for p in pts)
        out.append(math.degrees(math.atan2(s, co)))
    return out


def corner_target(profile, corner):
    """[x, y, z, rx, ry, rz] de una esquina para "Ir encima"/"Tocar", con la
    orientacion del DIBUJO (no la de la captura): si la esquina es alcanzable
    con la orientacion que se capturo pero no con la fija del dibujo, la
    prueba tiene que fallar igual que va a fallar el dibujo."""
    return effective_corners(profile)[corner][:3] + drawing_orientation(profile)


def center_target(profile):
    """[x, y, z, rx, ry, rz] del centro de la hoja para "Ir encima"/"Tocar":
    x, y y la orientacion salen de las esquinas (el centro solo aporta Z); Z
    es la del centro si esta medida, o la del plano de las esquinas si no."""
    c = effective_corners(profile)
    xyz = [sum(c[k][i] for k in CORNERS) / 4 for i in range(3)]
    if profile.get('center_xy') is not None:
        xyz[0], xyz[1] = profile['center_xy']
    center_z = effective_center_z(profile)
    if center_z is not None:
        xyz[2] = center_z
    return xyz + drawing_orientation(profile)


def drawing_rotation_deg(profile):
    """
    Cuantos grados (antihorario, multiplo de 90) se gira el dibujo dentro de
    la hoja. 0 = la cabeza hacia las esquinas "arriba" del perfil, que es
    como se ve la hoja al medirla. Hubo un modo 'auto' que giraba el dibujo
    para que la cabeza apuntara lejos del robot en coordenadas del robot: con
    la hoja al costado eso la dejaba acostada en el papel. El preview ahora
    se dibuja en el marco de la hoja (ver pipeline.render_preview), asi que
    ya no hace falta girar nada para que se vea vertical.
    """
    rotation = profile.get('drawing_rotation', 0)
    return 0 if rotation == 'auto' else int(rotation)


def active_rotation_deg():
    profile_id = active_id()
    return drawing_rotation_deg(load(profile_id)) if profile_id else 0


# Ajuste de altura por esquina, en decimas de mm: positivo = la pluma baja
# mas (mas presion), negativo = sube (-1 = 1 mm mas arriba). Por eso se
# RESTA de la Z medida (en el robot, Z mas alta = mas arriba). Se guarda aparte de la Z capturada para poder volver a
# capturar una esquina sin perder el ajuste, y para ver siempre cuanto se
# corrigio a mano.
Z_ADJUST_LIMIT_MM = 10


def effective_corners(profile):
    """Esquinas con el ajuste de altura aplicado: las que se usan para dibujar."""
    adjust = profile.get('z_adjust_mm') or {}
    out = {}
    for key in CORNERS:
        c = list(profile['corners'][key])
        c[2] = round(c[2] - adjust.get(key, 0), 2)
        out[key] = c
    return out


def _write(profile_id, profile):
    os.makedirs(PROFILES_DIR, exist_ok=True)
    data = dict(profile, id=profile_id, updated_at=time.strftime('%Y-%m-%d %H:%M'))
    tmp = _path(profile_id) + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
    os.replace(tmp, _path(profile_id))
    return data


def load(profile_id):
    try:
        with open(_path(profile_id), encoding='utf-8') as fh:
            data = json.load(fh)
    except FileNotFoundError:
        raise ValueError('no existe el perfil "%s"' % profile_id)
    data['id'] = profile_id
    data.setdefault('drawing_rotation', 0)
    data.setdefault('center_z', None)
    data.setdefault('center_xy', None)
    data.setdefault('center_z_adjust_mm', 0.0)
    data.setdefault('j1_backlash_deg', 0.0)
    data.setdefault('j1_unidirectional', False)
    data.setdefault('j1_direction', 1)
    data.setdefault('tablet_mode', False)
    data.setdefault('tablet_orient', None)
    data.setdefault('z_adjust_mm', {})  # perfiles guardados antes de existir el ajuste
    for key in CORNERS:
        data['z_adjust_mm'].setdefault(key, 0.0)
    return data


def list_profiles():
    if not os.path.isdir(PROFILES_DIR):
        return []
    out = []
    for fname in sorted(os.listdir(PROFILES_DIR)):
        if fname.endswith('.json'):
            try:
                p = load(fname[:-5])
            except (ValueError, json.JSONDecodeError):
                continue
            out.append(dict(id=p['id'], name=p.get('name', p['id']), updated_at=p.get('updated_at')))
    return out


def active_id():
    try:
        with open(ACTIVE_FILE, encoding='utf-8') as fh:
            return fh.read().strip() or None
    except FileNotFoundError:
        return None


def _apply(profile):
    c = effective_corners(profile)
    pl.lib.set_calibration(c['top_left'], c['top_right'], c['bottom_right'], c['bottom_left'],
                           manual_skew_mm=profile['manual_skew_mm'], pen_lift_mm=profile['pen_lift_mm'],
                           center_z=effective_center_z(profile), j1_backlash_deg=profile['j1_backlash_deg'],
                           center_xy=profile['center_xy'])
    pl.j1u.ENABLED = profile['j1_unidirectional']
    pl.j1u.DIRECTION = profile['j1_direction']
    pl.lib.TABLET_MODE = profile['tablet_mode']


def activate(profile_id):
    profile = load(profile_id)
    _apply(profile)
    with open(ACTIVE_FILE, 'w', encoding='utf-8') as fh:
        fh.write(profile_id)
    return profile


def create(name, copy_from=None):
    source = load(copy_from) if copy_from else current_as_profile(name)
    profile = _validate(dict(source, name=name))
    profile_id = base = _slug(name)
    n = 2
    while os.path.exists(_path(profile_id)):
        profile_id = '%s-%d' % (base[:36], n)
        n += 1
    return _write(profile_id, profile)


def update(profile_id, changes):
    current = load(profile_id)
    merged = dict(current)
    for key in ('name', 'manual_skew_mm', 'pen_lift_mm', 'drawing_rotation', 'center_z', 'center_z_adjust_mm', 'center_xy',
                'j1_backlash_deg', 'j1_unidirectional', 'j1_direction', 'tablet_mode', 'tablet_orient'):
        if key in changes:
            merged[key] = changes[key]
    if 'corners' in changes:
        merged['corners'] = dict(current['corners'], **changes['corners'])
    if 'z_adjust_mm' in changes:
        merged['z_adjust_mm'] = dict(current['z_adjust_mm'], **changes['z_adjust_mm'])
    data = _write(profile_id, _validate(merged))
    if profile_id == active_id():
        _apply(data)
    return data


def delete(profile_id):
    if profile_id == active_id():
        raise ValueError('no se puede borrar el perfil activo: activa otro primero')
    os.remove(_path(profile_id))


def current_as_profile(name):
    cal = pl.lib.get_calibration()
    return dict(name=name, corners={k: cal[k] for k in CORNERS},
                manual_skew_mm=cal['manual_skew_mm'], pen_lift_mm=cal['pen_lift_mm'])


def bootstrap():
    """Al arrancar: si no hay perfiles, guarda la calibracion que ya estaba
    escrita en gcode_draw_lib.py como primer perfil (no se pierde nada); y
    aplica el perfil activo."""
    if not list_profiles():
        created = create('Calibracion original')
        activate(created['id'])
        return
    profile_id = active_id()
    ids = [p['id'] for p in list_profiles()]
    activate(profile_id if profile_id in ids else ids[0])


def describe(profile):
    """Perfil + diagnostico de las esquinas + tamano util del area."""
    c = effective_corners(profile)
    diag = pl.lib.calibration_diagnostics(c['top_left'], c['top_right'], c['bottom_right'], c['bottom_left'])
    plane_center_z = sum(c[k][2] for k in CORNERS) / 4
    center_z = effective_center_z(profile)
    diag['plane_center_z'] = round(plane_center_z, 2)
    diag['center_offset_mm'] = None if center_z is None else round(center_z - plane_center_z, 1)
    cx, cy = _corners_center_xy(c)
    diag['corners_center_xy'] = [round(cx, 2), round(cy, 2)]
    diag['center_xy_offset_mm'] = (None if profile.get('center_xy') is None else
                                   round(math.dist(profile['center_xy'], (cx, cy)), 1))
    return dict(profile, effective_corners=c, diagnostics=diag, active=(profile['id'] == active_id()),
                drawing_rotation_resolved=drawing_rotation_deg(profile))


def area_mm():
    w, h = pl.lib._quad_edge_lengths()
    return [round(w, 1), round(h, 1)]


# ---------------------------------------------------------------------------
# Acciones sobre el brazo (siempre dentro de robot.use(...))
# ---------------------------------------------------------------------------
# Parametros internos de los servos (direcciones de pymycobot/demo/change_pid.py,
# la herramienta oficial de Elephant). Valores de referencia de esa demo:
# "alta precision" P10 D0 I1 arranque0 zm3/3; "estabilizar" J1-J3 P5 D15 I0.
# P baja + I=0 = el servo cede al empujarlo y se queda corto antes del punto
# pedido, del lado del que venia (se ve como juego aunque no lo sea).
SERVO_PARAMS = [(21, 'P'), (22, 'D'), (23, 'I'), (24, 'arranque'), (26, 'zona_muerta_1'), (27, 'zona_muerta_2')]


def read_servo_params(mc):
    """Solo LEE los parametros de los 6 servos (no escribe nada en el brazo).
    Devuelve {1: {'P': 10, ...}, ...}; None donde el servo no respondio."""
    out = {}
    for servo in range(1, 7):
        row = {}
        for address, name in SERVO_PARAMS:
            try:
                value = mc.get_servo_data(servo, address)
            except Exception:
                value = None
            row[name] = value if isinstance(value, int) and value >= 0 else None
            time.sleep(0.05)
        out[servo] = row
    return out


# Ajuste de J1 (base). Leidos 2026-09-26: J1 P=15 D=20 I=0, contra P=32 en J2
# y J3 -- la base tenia la mitad de rigidez que los ejes que dibujan bien, y
# cedia ~1 cm al empujarla con el brazo trabado. Solo se deja tocar P, D e I
# de J1, en rangos cerca de los presets oficiales (change_pid.py y la GUI de
# fabrica usan P 5-32, D 0-25, I 0-1): valores mucho mas altos pueden hacer
# vibrar el eje. Lo que se escribe queda guardado en el servo (sobrevive al
# apagado); J1_ORIGINAL permite volver atras.
J1_TUNABLE = {'P': (21, 5, 40), 'D': (22, 0, 40), 'I': (23, 0, 3)}   # nombre: (direccion, min, max)
J1_ORIGINAL = {'P': 15, 'D': 20, 'I': 0}


def write_j1_param(mc, name, value):
    """Escribe un parametro de J1 y lo verifica leyendolo de vuelta."""
    if name not in J1_TUNABLE:
        raise ValueError('parametro invalido: solo se puede cambiar %s de J1' % ', '.join(J1_TUNABLE))
    address, lo, hi = J1_TUNABLE[name]
    try:
        value = int(value)
    except (TypeError, ValueError):
        raise ValueError('el valor de %s debe ser un numero entero' % name)
    if not lo <= value <= hi:
        raise ValueError('%s de J1 debe estar entre %d y %d' % (name, lo, hi))
    mc.set_servo_data(1, address, value)
    time.sleep(0.2)
    read = mc.get_servo_data(1, address)
    if read != value:
        raise ValueError('se escribio %s=%d en J1 pero el servo devolvio %r: el cambio no quedo aplicado'
                         % (name, value, read))
    return value


def restore_j1(mc):
    for name, value in J1_ORIGINAL.items():
        write_j1_param(mc, name, value)


def read_coords(mc, attempts=4):
    """get_coords a veces devuelve -1 o una lista vacia/en cero mientras el
    brazo se asienta o con los servos liberados; se reintenta un poco."""
    for _ in range(attempts):
        try:
            coords = mc.get_coords()
        except Exception:
            coords = None
        if isinstance(coords, (list, tuple)) and len(coords) == 6 and any(abs(v) > 1e-6 for v in coords):
            return [round(float(v), 2) for v in coords]
        time.sleep(0.2)
    raise RuntimeError('el brazo no devolvio su posicion (get_coords). Revisa la conexion y reintenta.')


PRECISION_GRID = [(u, v) for v in (0.8, 0.5, 0.2) for u in (0.2, 0.5, 0.8)]
PRECISION_STEP_MM = 5.0
PRECISION_HOVER_MM = 10.0


def precision_test(mc, progress_cb=None):
    """
    Mide cuanto se aparta el brazo de los puntos que se le piden, para saber
    por que los rasgos chicos (ojos, boca) salen deformados. En 9 puntos de la
    hoja: posicion con la pluma en el aire, apoyada, y despues de movimientos
    cortos (PRECISION_STEP_MM) en las 4 direcciones con la pluma apoyada, ida y
    vuelta. En cada caso espera a que el brazo quede quieto y lee donde quedo
    (get_coords, en x,y). Deja unas crucecitas en el papel.

    Como leer el resultado:
      - en el aire bien pero apoyada mal: la friccion del esfero contra el
        papel no deja terminar los movimientos (menos presion / otro esfero).
      - en los movimientos cortos se queda corto en la direccion del
        movimiento: juego/zona muerta de los servos (no depende de la presion).
      - error parecido en todos lados y en la misma direccion: corrimiento fijo
        (se puede compensar corriendo la calibracion).
    Devuelve un dict con las estadisticas y un texto de resumen.
    """
    lib = pl.lib
    quad_w, quad_h = lib._quad_edge_lengths()
    mc.focus_all_servos()
    time.sleep(0.5)
    air, down, moves, errors = [], [], [], 0

    def wait_offset(target):
        lib._wait_until_settled(mc, target, expected_s=1.0, xy_only=True, still_time=1.0)
        cur = lib._read_xyz(mc)
        if cur is None:
            return None
        return (cur[0] - target[0], cur[1] - target[1])

    def check():
        nonlocal errors
        if lib._check_and_clear_error(mc):
            errors += 1

    first = lib.bilinear(*PRECISION_GRID[0])
    first[2] += 30
    mc.send_coords(first, 40, 0)  # acercamiento desde donde este: angular y alto
    lib._wait_until_settled(mc, first, expected_s=3.0)
    for n, (u, v) in enumerate(PRECISION_GRID, 1):
        base = lib.bilinear(u, v)
        hover = list(base)
        hover[2] += PRECISION_HOVER_MM
        mc.send_coords(hover, lib.TRAVEL_SPEED, 1)
        air.append(wait_offset(hover))
        mc.send_coords(base, lib.DRAW_SPEED, 1)
        down.append(wait_offset(base))
        for du, dv in ((1, 0), (0, 1), (-1, 0), (0, -1)):
            target = lib.bilinear(u + du * PRECISION_STEP_MM / quad_w, v + dv * PRECISION_STEP_MM / quad_h)
            for frm, to in ((base, target), (target, base)):
                mc.send_coords(to, lib.DRAW_SPEED, 1)
                off = wait_offset(to)
                d = (to[0] - frm[0], to[1] - frm[1])
                norm = math.hypot(*d) or 1.0
                moves.append((off, (d[0] / norm, d[1] / norm)))
        mc.send_coords(hover, lib.TRAVEL_SPEED, 1)
        lib._wait_until_settled(mc, hover, expected_s=1.0)
        check()
        fmt_off = lambda o: '?' if o is None else '%.1f mm (dx %+.1f, dy %+.1f)' % (math.hypot(*o), o[0], o[1])
        print('  punto %d (u=%.1f v=%.1f): aire %s | apoyada %s | movimientos: %s' % (
            n, u, v, fmt_off(air[-1]), fmt_off(down[-1]),
            ', '.join('%.1f' % math.hypot(*o) if o else '?' for o, _ in moves[-8:])), flush=True)
        if progress_cb:
            progress_cb(n, len(PRECISION_GRID), 'punto %d de %d' % (n, len(PRECISION_GRID)))

    def mags(offs):
        vals = sorted(math.hypot(*o) for o in offs if o is not None)
        if not vals:
            return None
        return dict(mean=round(sum(vals) / len(vals), 1), max=round(vals[-1], 1))

    valid_moves = [(o, d) for o, d in moves if o is not None]
    along = [o[0] * d[0] + o[1] * d[1] for o, d in valid_moves]  # <0 = se quedo corto
    perp = [abs(o[0] * d[1] - o[1] * d[0]) for o, d in valid_moves]
    down_valid = [o for o in down if o is not None]
    bias = (sum(o[0] for o in down_valid) / len(down_valid), sum(o[1] for o in down_valid) / len(down_valid)) \
        if down_valid else (0.0, 0.0)
    stats = dict(
        aire=mags(air), apoyada=mags(down), movimientos=mags([o for o, _ in valid_moves]),
        corto_mm=round(-sum(along) / len(along), 1) if along else None,
        lateral_mm=round(sum(perp) / len(perp), 1) if perp else None,
        corrimiento_fijo_mm=round(math.hypot(*bias), 1),
        errores_robot=errors,
    )
    fmt = lambda m: 'sin datos' if m is None else 'promedio %s mm (max %s)' % (m['mean'], m['max'])
    stats['resumen'] = (
        'Pluma en el aire: %s. Pluma apoyada: %s. Tras movimientos cortos de %g mm: %s; en promedio se queda '
        '%s mm corto en la direccion del movimiento y %s mm de costado. Corrimiento fijo (igual en todos '
        'lados): %s mm. Errores del robot: %d.' % (
            fmt(stats['aire']), fmt(stats['apoyada']), PRECISION_STEP_MM, fmt(stats['movimientos']),
            stats['corto_mm'], stats['lateral_mm'], stats['corrimiento_fijo_mm'], errors))
    print('PRUEBA DE PRECISION:', stats['resumen'], flush=True)
    try:
        mc.set_color(0, 255, 0)
    except Exception:
        pass
    return stats


def goto_corner(mc, corner_coords, hover_mm):
    """Lleva la pluma sobre una esquina: primero a hover_mm por encima (modo
    angular, rapido y robusto desde cualquier posicion), y si hover_mm es 0,
    baja en vertical (modo lineal, lento) hasta tocar."""
    lib = pl.lib

    def move(coords, speed, mode, expected_s):
        mc.send_coords(coords, speed, mode)
        lib._wait_until_settled(mc, coords, expected_s=expected_s)
        code = lib._check_and_clear_error(mc)
        if code:
            raise ValueError('el robot no pudo llegar (error %s: %s). Con la orientacion fija del dibujo '
                             'este punto no es alcanzable: al dibujar tambien lo va a saltear.'
                             % (code, lib._describe_error(code)))

    above = list(corner_coords)
    above[2] += max(hover_mm, 10.0)
    move(above, 40, 0, 3.0)
    if hover_mm < 10.0:
        target = list(corner_coords)
        target[2] += hover_mm
        move(target, 15, 1, 1.5)
