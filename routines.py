#!/usr/bin/python
# -*- coding:utf-8 -*-
# Rutinas de movimiento del brazo (saludo, invitacion, ...): "ensenar y repetir".
#
# En la calibracion se libera el brazo, se lo lleva con la mano a cada pose y se
# la agrega a la rutina; cada paso guarda los ANGULOS de las 6 articulaciones
# (no coordenadas: se repiten exactos, sin depender de la calibracion de la hoja
# ni de la cinematica inversa), mas la velocidad para llegar y una pausa. Al
# ejecutar, el brazo va de pose en pose esperando a llegar a cada una.
#
# Se guardan en routines.json, junto a este archivo.

import json
import math
import os
import re
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(HERE, 'routines.json')
DEFAULT_ROUTINES = [('saludo', 'Saludo'), ('invitacion', 'Invitacion')]   # vacias: se ensenan en la calibracion

SPEED_RANGE = (1, 100)
DEFAULT_SPEED = 50
PAUSE_RANGE = (0.0, 30.0)
MAX_STEPS = 60
MAX_REPEAT = 20
ARRIVE_TOL_DEG = 2.0        # "llego" a la pose si ninguna articulacion esta a mas de esto
STILL_TOL_DEG = 0.3         # "quieto" si entre dos lecturas se movio menos que esto...
STILL_TIME_S = 0.5          # ...durante este tiempo
MOVE_TIMEOUT_S = 12.0

_lock = threading.Lock()


# -------------------------------------------------------------------------------
# Guardado
# -------------------------------------------------------------------------------
def _slug(name):
    s = re.sub(r'[^a-z0-9]+', '-', name.lower()).strip('-')
    return (s or 'rutina')[:40]


def _validate_step(step):
    try:
        angles = [round(float(a), 2) for a in step['angles']]
    except (KeyError, TypeError, ValueError):
        raise ValueError('cada paso necesita 6 angulos numericos')
    if len(angles) != 6 or not all(math.isfinite(a) and -200 <= a <= 200 for a in angles):
        raise ValueError('cada paso necesita 6 angulos validos')
    try:
        speed = int(step.get('speed', DEFAULT_SPEED))
        pause = round(float(step.get('pause_s', 0)), 2)
    except (TypeError, ValueError):
        raise ValueError('la velocidad y la pausa deben ser numeros')
    if not SPEED_RANGE[0] <= speed <= SPEED_RANGE[1]:
        raise ValueError('la velocidad debe estar entre %d y %d' % SPEED_RANGE)
    if not PAUSE_RANGE[0] <= pause <= PAUSE_RANGE[1]:
        raise ValueError('la pausa debe estar entre %g y %g s' % PAUSE_RANGE)
    return dict(angles=angles, speed=speed, pause_s=pause)


def load_all():
    with _lock:
        try:
            with open(PATH, encoding='utf-8') as fh:
                data = json.load(fh)
        except (FileNotFoundError, ValueError):
            data = {}
        if not data:   # primera vez: las rutinas de ejemplo, vacias
            data = {rid: dict(name=name, steps=[]) for rid, name in DEFAULT_ROUTINES}
            _write(data)
        return data


def _write(data):
    tmp = PATH + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
    os.replace(tmp, PATH)


def listing():
    return [dict(id=rid, name=r['name'], steps=r['steps']) for rid, r in load_all().items()]


def get(rid):
    data = load_all()
    if rid not in data:
        raise ValueError('no existe la rutina "%s"' % rid)
    return data[rid]


def create(name):
    name = str(name or '').strip()
    if not name or len(name) > 40:
        raise ValueError('el nombre de la rutina es obligatorio (hasta 40 caracteres)')
    data = load_all()
    rid, n = _slug(name), 2
    base = rid
    while rid in data:
        rid = '%s-%d' % (base, n)
        n += 1
    data[rid] = dict(name=name, steps=[])
    with _lock:
        _write(data)
    return rid


def update(rid, name=None, steps=None):
    data = load_all()
    if rid not in data:
        raise ValueError('no existe la rutina "%s"' % rid)
    if name is not None:
        name = str(name).strip()
        if not name or len(name) > 40:
            raise ValueError('el nombre de la rutina es obligatorio (hasta 40 caracteres)')
        data[rid]['name'] = name
    if steps is not None:
        if len(steps) > MAX_STEPS:
            raise ValueError('una rutina puede tener hasta %d pasos' % MAX_STEPS)
        data[rid]['steps'] = [_validate_step(s) for s in steps]
    with _lock:
        _write(data)
    return data[rid]


def delete(rid):
    data = load_all()
    if rid not in data:
        raise ValueError('no existe la rutina "%s"' % rid)
    del data[rid]
    with _lock:
        _write(data)


# -------------------------------------------------------------------------------
# Brazo
# -------------------------------------------------------------------------------
def read_angles(mc, attempts=4):
    """get_angles con reintentos (a veces devuelve -1 o una lista vacia)."""
    for _ in range(attempts):
        try:
            a = mc.get_angles()
        except Exception:
            a = None
        if isinstance(a, (list, tuple)) and len(a) == 6:
            return [round(float(v), 2) for v in a]
        time.sleep(0.2)
    raise RuntimeError('el brazo no devolvio sus angulos (get_angles). Revisa la conexion y reintenta.')


def move_to(mc, angles, speed, stop_event=None):
    """Manda la pose y espera a que llegue (o a que quede quieto, o timeout)."""
    mc.send_angles(list(angles), int(speed))
    t0, last, still_since = time.time(), None, None
    time.sleep(0.15)
    while time.time() - t0 < MOVE_TIMEOUT_S:
        if stop_event is not None and stop_event.is_set():
            mc.stop()   # frena donde este
            return 'detenido'
        try:
            cur = mc.get_angles()
        except Exception:
            cur = None
        now = time.time()
        if isinstance(cur, (list, tuple)) and len(cur) == 6:
            if max(abs(a - b) for a, b in zip(cur, angles)) <= ARRIVE_TOL_DEG:
                return 'llego'
            if last is not None and max(abs(a - b) for a, b in zip(cur, last)) < STILL_TOL_DEG:
                still_since = still_since or now
                if now - still_since >= STILL_TIME_S and now - t0 > 0.8:
                    return 'quieto'
            else:
                still_since = None
            last = cur
        time.sleep(0.08)
    return 'timeout'


def play(mc, routine, repeat=1, progress_cb=None, stop_event=None):
    """Ejecuta la rutina `repeat` veces. progress_cb(vuelta, paso, pasos).
    Devuelve 'terminado' o 'detenido' (stop_event: el brazo frena donde esta)."""
    result = _play(mc, routine, repeat, progress_cb, stop_event)
    if result == 'detenido':
        mc.stop()
    return result


def _play(mc, routine, repeat, progress_cb, stop_event):
    steps = routine['steps']
    if not steps:
        raise ValueError('la rutina no tiene pasos: agrega posiciones antes de ejecutarla')
    mc.focus_all_servos()
    time.sleep(0.5)
    for lap in range(1, repeat + 1):
        for k, step in enumerate(steps, 1):
            if stop_event is not None and stop_event.is_set():
                return 'detenido'
            if progress_cb:
                progress_cb(lap, k, len(steps))
            if move_to(mc, step['angles'], step['speed'], stop_event) == 'detenido':
                return 'detenido'
            end = time.time() + step['pause_s']
            while time.time() < end:
                if stop_event is not None and stop_event.is_set():
                    return 'detenido'
                time.sleep(0.05)
    return 'terminado'
