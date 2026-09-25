#!/usr/bin/python
# -*- coding:utf-8 -*-
# Backend Flask: foto -> boceto (Gemini) -> aceptar/rechazar -> gcode + preview
# -> aceptar/rechazar -> dibujar en el myCobot 280 real.
#
# Estado en memoria (un solo operador local, sin base de datos): JOBS[job_id].
# Cada job pasa por etapas (job['stage']):
#   uploaded -> sketching -> sketch_ready (o sketch_failed) -> building_gcode
#   -> gcode_ready -> drawing -> done (o error)
#
# Los pasos que tardan (Gemini, dibujar) corren en threads de fondo; el
# frontend consulta GET /api/jobs/<id> para ver el progreso.

import json
import os
import threading
import time
import traceback
import uuid

from dotenv import load_dotenv
load_dotenv()

from flask import Flask, jsonify, request, send_from_directory, abort

import calibration as cal
import gemini_sketch
import pipeline as pl
import robot

HERE = os.path.dirname(os.path.abspath(__file__))
RUNTIME_DIR = os.path.join(HERE, 'runtime')
os.makedirs(RUNTIME_DIR, exist_ok=True)

app = Flask(__name__, static_folder='static', static_url_path='')
# Sin esto el navegador puede seguir usando una copia vieja cacheada de
# app.js/index.html despues de editarlos, dando bugs fantasma (el boton
# de aceptar no aparece, etc. con codigo que en el server ya esta arreglado).
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

# Aplica el perfil de calibracion activo antes de generar cualquier preview.
cal.bootstrap()

JOBS = {}
JOBS_LOCK = threading.Lock()
DRAWING_JOB_ID = {'id': None}

ALLOWED_EXT = {'.png', '.jpg', '.jpeg', '.webp'}


def _job_dir(job_id):
    d = os.path.join(RUNTIME_DIR, job_id)
    os.makedirs(d, exist_ok=True)
    return d


def _new_job():
    job_id = uuid.uuid4().hex[:12]
    job = dict(
        id=job_id, stage='new', created_at=time.time(),
        original_path=None, sketch_path=None, sketch_error=None,
        moves=None, gcode_final_path=None, preview_path=None, preview_stats=None,
        draw_progress=None, draw_error=None, draw_done_at=None,
    )
    with JOBS_LOCK:
        JOBS[job_id] = job
    return job


def _get_job(job_id):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if job is None:
        abort(404, description='job no encontrado')
    return job


def _public(job):
    """Version del job segura para mandar al frontend (sin paths absolutos del server)."""
    job_id = job['id']

    def url(path_field):
        p = job.get(path_field)
        if not p:
            return None
        # ?v= cambia cuando cambia el archivo (ej. un boceto regenerado se sigue
        # llamando sketch.png): sin esto el navegador podia mostrar el viejo
        try:
            version = int(os.path.getmtime(p) * 1000)
        except OSError:
            version = 0
        return '/runtime/%s/%s?v=%d' % (job_id, os.path.basename(p), version)

    return dict(
        id=job_id, stage=job['stage'],
        original_url=url('original_path'),
        sketch_url=url('sketch_path'),
        crop=job.get('crop'),
        sketch_error=job.get('sketch_error'),
        preview_url=url('preview_path'),
        preview_stats=job.get('preview_stats'),
        draw_progress=job.get('draw_progress'),
        draw_error=job.get('draw_error'),
        draw_done_at=job.get('draw_done_at'),
        robot_errors=job.get('robot_errors') or [],
        sync=job.get('sync'),
        is_drawing=(DRAWING_JOB_ID['id'] == job_id),
        detail=job.get('detail', pl.DEFAULT_DETAIL),
    )


# ---------------------------------------------------------------------------
# Etapa 1: subir foto + generar boceto con Gemini
# ---------------------------------------------------------------------------
@app.route('/api/jobs', methods=['POST'])
def create_job():
    file = request.files.get('image')
    if file is None or file.filename == '':
        return jsonify(error='falta el archivo "image"'), 400
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXT:
        return jsonify(error='formato no soportado (%s). Usa png/jpg/webp' % ext), 400

    job = _new_job()
    path = os.path.join(_job_dir(job['id']), 'original' + ext)
    file.save(path)
    job['original_path'] = path
    job['stage'] = 'uploaded'
    return jsonify(_public(job))


def _run_sketch(job_id):
    job = _get_job(job_id)
    job['stage'] = 'sketching'
    job['sketch_error'] = None
    try:
        out_path = os.path.join(_job_dir(job_id), 'sketch.png')
        gemini_sketch.generate_sketch(job['original_path'], out_path)
        job['sketch_path'] = out_path
        job['crop'] = None  # boceto nuevo: el recorte anterior ya no aplica
        job['stage'] = 'sketch_ready'
    except Exception as exc:
        traceback.print_exc()
        job['sketch_error'] = str(exc)
        job['stage'] = 'sketch_failed'


@app.route('/api/jobs/<job_id>/sketch', methods=['POST'])
def start_sketch(job_id):
    job = _get_job(job_id)
    if job['stage'] not in ('uploaded', 'sketch_failed', 'sketch_ready'):
        return jsonify(error='etapa invalida para generar boceto: %s' % job['stage']), 409
    threading.Thread(target=_run_sketch, args=(job_id,), daemon=True).start()
    job['stage'] = 'sketching'
    return jsonify(_public(job))


@app.route('/api/jobs/<job_id>/sketch/manual', methods=['POST'])
def upload_manual_sketch(job_id):
    """Alternativa a Gemini: subir un boceto ya hecho a mano (ver advertencia
    de cuota en gemini_sketch.py)."""
    job = _get_job(job_id)
    file = request.files.get('sketch')
    if file is None or file.filename == '':
        return jsonify(error='falta el archivo "sketch"'), 400
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXT:
        return jsonify(error='formato no soportado (%s)' % ext), 400
    path = os.path.join(_job_dir(job_id), 'sketch.png')
    file.save(path)
    job['sketch_path'] = path
    job['crop'] = None
    job['sketch_error'] = None
    job['stage'] = 'sketch_ready'
    return jsonify(_public(job))


# ---------------------------------------------------------------------------
# Etapa 2: aceptar/rechazar boceto -> generar gcode + preview
# ---------------------------------------------------------------------------
@app.route('/api/jobs/<job_id>/sketch/reject', methods=['POST'])
def reject_sketch(job_id):
    """Rechazar el boceto: se vuelve a pedir a Gemini (mismo prompt)."""
    job = _get_job(job_id)
    if job['stage'] != 'sketch_ready':
        return jsonify(error='etapa invalida: %s' % job['stage']), 409
    threading.Thread(target=_run_sketch, args=(job_id,), daemon=True).start()
    job['stage'] = 'sketching'
    return jsonify(_public(job))


MIN_CROP_FRACTION = 0.05  # un recorte mas chico que esto (en ancho o alto) es un click sin querer
CROP_WORK_SIDE_PX = 800   # lado al que se lleva el recorte antes de extraer trazos


def _parse_crop(raw):
    """Recorte enviado por la interfaz: '[x0, y0, x1, y1]' en fracciones de
    la imagen del boceto (0..1, origen arriba a la izquierda), o vacio = sin
    recorte. Devuelve la lista o None; ValueError si no sirve."""
    if not raw:
        return None
    try:
        vals = [min(1.0, max(0.0, float(v))) for v in json.loads(raw)]
    except (TypeError, ValueError):
        raise ValueError('recorte invalido')
    if len(vals) != 4:
        raise ValueError('recorte invalido')
    x0, y0, x1, y1 = min(vals[0], vals[2]), min(vals[1], vals[3]), max(vals[0], vals[2]), max(vals[1], vals[3])
    if x1 - x0 < MIN_CROP_FRACTION or y1 - y0 < MIN_CROP_FRACTION:
        raise ValueError('el recorte es demasiado chico')
    return [round(v, 4) for v in (x0, y0, x1, y1)]


def _sketch_for_gcode(job):
    """Imagen de la que se sacan los trazos: el boceto completo, o solo la zona
    recortada en la interfaz (guardada como sketch_crop.png)."""
    crop = job.get('crop')
    if not crop:
        return job['sketch_path']
    import cv2
    img = cv2.imread(job['sketch_path'])
    if img is None:
        raise RuntimeError('no se pudo leer el boceto para recortarlo')
    h, w = img.shape[:2]
    x0, y0, x1, y1 = (int(round(crop[0] * w)), int(round(crop[1] * h)),
                      int(round(crop[2] * w)), int(round(crop[3] * h)))
    region = img[y0:y1, x0:x1]
    # los trazos se extraen a ~800 px de lado (draw_person.load_image achica,
    # pero no agranda): sin esto un recorte chico perdia resolucion justo en
    # la zona que se quiere ampliar
    side = max(region.shape[:2])
    if side < CROP_WORK_SIDE_PX:
        f = CROP_WORK_SIDE_PX / side
        region = cv2.resize(region, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
    path = os.path.join(_job_dir(job['id']), 'sketch_crop.png')
    cv2.imwrite(path, region)
    return path


def _build_gcode(job, detail):
    """
    Extrae los trazos del boceto (o de su recorte, ver _sketch_for_gcode) y los
    simplifica segun `detail`, y regenera gcode + preview. Vuelve a extraer del
    boceto (no solo re-simplifica lo ya extraido): el umbral de binarizado
    tambien depende del nivel de detalle (ver sketch_threshold en
    DETAIL_LEVELS), asi que cambiar de nivel puede cambiar que tinta del
    boceto entra en juego, no solo cuanto se simplifica.
    """
    job['detail'] = detail
    params = dict(pl.resolve_detail(detail))
    threshold = params.pop('sketch_threshold', None)
    min_len_px = params.pop('sketch_min_len_px', pl.SKETCH_MIN_LEN_PX)
    source = _sketch_for_gcode(job)
    # la foto original solo sirve para ubicar la cara si el boceto no se recorto
    # (los rasgos de la foto se escalan al boceto completo)
    photo = None if job.get('crop') else job.get('original_path')
    raw_moves, _shape, core_bbox = pl.image_to_raw_moves(source, threshold=threshold, min_len_px=min_len_px,
                                                          original_photo_path=photo)
    job['raw_moves'] = raw_moves
    # orientacion segun el perfil activo (hoja de frente o al costado del robot)
    rotation = cal.active_rotation_deg()
    moves = pl.simplify_for_robot(raw_moves, core_bbox=core_bbox, rotation_deg=rotation, **params)
    if not moves:
        raise RuntimeError('No quedaron trazos utilizables con este nivel de detalle '
                            '(probá un nivel mas alto).')
    job['rotation'] = rotation
    moves = pl.rotate_moves(moves, rotation)
    job['moves'] = moves

    gcode_path = os.path.join(_job_dir(job['id']), 'final.gcode')
    pl.lib.write_gcode(moves, gcode_path)
    job['gcode_final_path'] = gcode_path

    preview_path = os.path.join(_job_dir(job['id']), 'preview.png')
    stats = pl.render_preview(moves, preview_path)
    job['preview_path'] = preview_path
    job['preview_stats'] = stats


@app.route('/api/jobs/<job_id>/sketch/accept', methods=['POST'])
def accept_sketch(job_id):
    """Boceto aceptado: extraer trazos, simplificar (friccion-seguro) y
    generar el preview del gcode. Es rapido (segundos, sin hardware), se
    hace sincronico."""
    job = _get_job(job_id)
    if job['stage'] != 'sketch_ready':
        return jsonify(error='etapa invalida: %s' % job['stage']), 409
    detail = request.form.get('detail')
    if not detail and request.is_json:
        detail = (request.get_json(silent=True) or {}).get('detail')
    detail = detail or pl.DEFAULT_DETAIL
    try:
        job['crop'] = _parse_crop(request.form.get('crop'))
    except ValueError as exc:
        return jsonify(error=str(exc)), 400
    job['stage'] = 'building_gcode'
    try:
        _build_gcode(job, detail)
        job['stage'] = 'gcode_ready'
    except Exception as exc:
        traceback.print_exc()
        job['stage'] = 'sketch_ready'  # vuelve atras, el boceto sigue disponible
        return jsonify(error=str(exc)), 500
    return jsonify(_public(job))


@app.route('/api/jobs/<job_id>/gcode/rebuild', methods=['POST'])
def rebuild_gcode(job_id):
    """Recalcula gcode + preview con otro nivel de detalle, reusando los
    trazos ya extraidos del boceto aceptado (no vuelve a llamar a Gemini)."""
    job = _get_job(job_id)
    if job['stage'] != 'gcode_ready':
        return jsonify(error='etapa invalida: %s' % job['stage']), 409
    detail = request.form.get('detail') or pl.DEFAULT_DETAIL
    try:
        _build_gcode(job, detail)
    except Exception as exc:
        traceback.print_exc()
        return jsonify(error=str(exc)), 500
    return jsonify(_public(job))


# ---------------------------------------------------------------------------
# Etapa 3: aceptar/rechazar gcode -> dibujar de verdad
# ---------------------------------------------------------------------------
@app.route('/api/jobs/<job_id>/gcode/reject', methods=['POST'])
def reject_gcode(job_id):
    """Rechazar el preview: vuelve a la revision del boceto (se puede
    rechazar el boceto tambien desde ahi para regenerarlo)."""
    job = _get_job(job_id)
    if job['stage'] != 'gcode_ready':
        return jsonify(error='etapa invalida: %s' % job['stage']), 409
    job['stage'] = 'sketch_ready'
    return jsonify(_public(job))


class _tee_stdout:
    """Todo lo que se imprima mientras dura el bloque va a la consola Y al
    archivo: gcode_draw_lib imprime cada comando, los errores del robot y las
    esperas, y sin esto se perdia al cerrar la consola (ej. para investigar un
    movimiento brusco a mitad de un dibujo). Solo hay un dibujo a la vez (ver
    robot.LOCK), asi que reemplazar sys.stdout es seguro."""

    def __init__(self, fh):
        self.fh = fh

    def __enter__(self):
        import sys
        self.orig = sys.stdout
        sys.stdout = self
        return self

    def write(self, s):
        self.orig.write(s)
        self.fh.write(s)

    def flush(self):
        self.orig.flush()
        self.fh.flush()

    def __exit__(self, *exc):
        import sys
        sys.stdout = self.orig
        return False


def _run_draw(job_id):
    try:
        job = _get_job(job_id)
        job['stage'] = 'drawing'
        job['draw_error'] = None
        job['robot_errors'] = []
        job['sync'] = None
        job['draw_progress'] = dict(i=0, total=len(job['moves']), message='conectando...')

        def progress_cb(i, total, message):
            job['draw_progress'] = dict(i=i, total=total, message=message)

        log_path = os.path.join(_job_dir(job_id), 'draw.log')
        with open(log_path, 'a', encoding='utf-8') as log, _tee_stdout(log):
            print('=== dibujo iniciado %s ===' % time.strftime('%Y-%m-%d %H:%M:%S'), flush=True)
            result = pl.draw(robot.get(), job['moves'], progress_cb=progress_cb)
        job['robot_errors'] = _describe_robot_errors(result, job['moves'])
        job['sync'] = (result or {}).get('sync')
        job['stage'] = 'done'
        job['draw_done_at'] = time.time()
    except Exception as exc:
        traceback.print_exc()
        robot.disconnect()
        job['draw_error'] = str(exc)
        job['stage'] = 'error'
    finally:
        DRAWING_JOB_ID['id'] = None
        robot.release()


@app.route('/api/jobs/<job_id>/gcode/accept', methods=['POST'])
def accept_gcode(job_id):
    job = _get_job(job_id)
    if job['stage'] != 'gcode_ready':
        return jsonify(error='etapa invalida: %s' % job['stage']), 409
    return _start_draw(job)


@app.route('/api/jobs/<job_id>/redraw', methods=['POST'])
def redraw(job_id):
    """Dibujar otra vez el mismo trazo, al terminar (o tras un error: arranca
    de nuevo desde el principio). Usa la calibracion activa en este momento."""
    job = _get_job(job_id)
    if job['stage'] not in ('done', 'error') or not job.get('moves'):
        return jsonify(error='etapa invalida: %s' % job['stage']), 409
    return _start_draw(job)


def _start_draw(job):
    try:
        robot.acquire('dibujando')
    except robot.RobotBusy as exc:
        return jsonify(error=str(exc)), 409
    DRAWING_JOB_ID['id'] = job['id']
    job['draw_done_at'] = None
    threading.Thread(target=_run_draw, args=(job['id'],), daemon=True).start()
    job['stage'] = 'drawing'
    return jsonify(_public(job))


# ---------------------------------------------------------------------------
# Calibracion: perfiles del area de dibujo + acciones sobre el brazo
# ---------------------------------------------------------------------------
CAL_TASK = dict(kind=None, running=False, progress=None, message=None, error=None, robot_errors=[],
                sync=None, log=None, precision=None)
TEST_KINDS = {'outline': 'prueba del contorno del area', 'pattern': 'prueba del patron',
              'precision': 'prueba de precision', 'face': 'prueba del rostro'}


def _cal_state(selected_id=None):
    profiles = cal.list_profiles()
    ids = [p['id'] for p in profiles]
    active = cal.active_id()
    sel = selected_id if selected_id in ids else active
    return dict(
        profiles=profiles, active_id=active,
        profile=cal.describe(cal.load(sel)) if sel else None,
        area_mm=cal.area_mm(),
        pattern_mm=round(pl.lib.test_pattern_size(), 1),
        active_rotation=cal.active_rotation_deg(),
        robot=dict(connected=robot.is_connected(), busy=robot.BUSY['what'],
                   fresh_mode=str(pl.lib.LAST_FRESH_MODE) if pl.lib.LAST_FRESH_MODE is not None else None),
        task=dict(CAL_TASK),
        corner_labels=cal.CORNER_LABELS,
    )


def _cal_error(exc):
    if isinstance(exc, robot.RobotBusy):
        return jsonify(error=str(exc)), 409
    if isinstance(exc, ValueError):
        return jsonify(error=str(exc)), 400
    traceback.print_exc()
    return jsonify(error=str(exc)), 500


def _without_robot_moving(what, fn):
    """Cambiar la calibracion activa mientras el brazo dibuja cambiaria el
    area a mitad del dibujo: se toma el brazo mientras dura el cambio."""
    robot.acquire(what)
    try:
        return fn()
    finally:
        robot.release()


@app.route('/api/calibration', methods=['GET'])
def calibration_state():
    return jsonify(_cal_state(request.args.get('profile')))


@app.route('/api/calibration/profiles', methods=['POST'])
def calibration_create():
    body = request.get_json(silent=True) or {}
    try:
        profile = cal.create(str(body.get('name', '')).strip(), copy_from=body.get('copy_from'))
    except Exception as exc:
        return _cal_error(exc)
    return jsonify(_cal_state(profile['id']))


@app.route('/api/calibration/profiles/<profile_id>', methods=['PUT'])
def calibration_update(profile_id):
    body = request.get_json(silent=True) or {}
    try:
        if profile_id == cal.active_id():
            _without_robot_moving('guardando calibracion', lambda: cal.update(profile_id, body))
        else:
            cal.update(profile_id, body)
    except Exception as exc:
        return _cal_error(exc)
    return jsonify(_cal_state(profile_id))


@app.route('/api/calibration/profiles/<profile_id>', methods=['DELETE'])
def calibration_delete(profile_id):
    try:
        cal.delete(profile_id)
    except Exception as exc:
        return _cal_error(exc)
    return jsonify(_cal_state())


@app.route('/api/calibration/profiles/<profile_id>/activate', methods=['POST'])
def calibration_activate(profile_id):
    try:
        _without_robot_moving('cambiando de perfil', lambda: cal.activate(profile_id))
    except Exception as exc:
        return _cal_error(exc)
    return jsonify(_cal_state(profile_id))


@app.route('/api/calibration/profiles/<profile_id>/capture/<corner>', methods=['POST'])
def calibration_capture(profile_id, corner):
    """Lee la posicion actual del brazo y la guarda como esa esquina. Para
    'center' solo se guarda la altura (Z): el centro no cambia el area."""
    if corner not in cal.CORNERS + ('center',):
        return jsonify(error='esquina invalida'), 400
    try:
        cal.load(profile_id)
        with robot.use('leyendo posicion') as mc:
            coords = cal.read_coords(mc)
            if corner == 'center':
                cal.update(profile_id, {'center_z': coords[2]})
            else:
                cal.update(profile_id, {'corners': {corner: coords}})
    except Exception as exc:
        return _cal_error(exc)
    return jsonify(dict(_cal_state(profile_id), captured=dict(corner=corner, coords=coords)))


@app.route('/api/calibration/profiles/<profile_id>/goto/<corner>', methods=['POST'])
def calibration_goto(profile_id, corner):
    """Lleva la pluma a una esquina guardada, o al centro de la hoja (por
    encima, o tocando si hover_mm=0) para verificar la medicion."""
    if corner not in cal.CORNERS + ('center',):
        return jsonify(error='esquina invalida'), 400
    body = request.get_json(silent=True) or {}
    try:
        hover = float(body.get('hover_mm', 10))
        if not 0 <= hover <= 50:
            raise ValueError('la altura debe estar entre 0 y 50 mm')
        profile = cal.load(profile_id)
        target = cal.center_target(profile) if corner == 'center' else cal.corner_target(profile, corner)
        with robot.use('moviendo a una esquina') as mc:
            mc.focus_all_servos()
            time.sleep(0.3)
            cal.goto_corner(mc, target, hover)
    except Exception as exc:
        return _cal_error(exc)
    return jsonify(_cal_state(profile_id))


@app.route('/api/calibration/robot/<action>', methods=['POST'])
def calibration_robot(action):
    """release = servos sueltos para mover el brazo a mano; lock = trabarlos;
    read = leer posicion actual; disconnect = cerrar el puerto (el proximo
    uso reconecta de cero)."""
    try:
        if action == 'disconnect':
            _without_robot_moving('desconectando', robot.disconnect)
            return jsonify(_cal_state(request.args.get('profile')))
        if action not in ('release', 'lock', 'read'):
            return jsonify(error='accion invalida'), 400
        coords = None
        with robot.use({'release': 'liberando servos', 'lock': 'trabando servos',
                        'read': 'leyendo posicion'}[action]) as mc:
            if action == 'release':
                mc.release_all_servos()
            elif action == 'lock':
                mc.focus_all_servos()
            else:
                coords = cal.read_coords(mc)
    except Exception as exc:
        return _cal_error(exc)
    return jsonify(dict(_cal_state(request.args.get('profile')), coords=coords))


# En el contorno (ver lib.outline_moves) cada punto es una esquina de la hoja.
OUTLINE_POINT_LABELS = {2: 'abajo izquierda', 3: 'abajo derecha', 4: 'arriba derecha',
                        5: 'arriba izquierda', 6: 'abajo izquierda'}


def _describe_robot_errors(result, moves, point_labels=None):
    """Errores que reporto el robot al dibujar, agrupados por punto, como
    texto para la interfaz. En esos puntos el brazo no fue adonde se le pidio
    (ej. se salteo una esquina)."""
    grouped = {}
    for point, code, text in (result or {}).get('robot_errors', []):
        key = (point, code, text)
        grouped[key] = grouped.get(key, 0) + 1
    out = []
    for (point, code, text), n in grouped.items():
        label = (point_labels or {}).get(point)
        where = ('yendo a la esquina %s' % label) if label else ('cerca del punto %d de %d' % (point, len(moves)))
        out.append('%s: error %s, %s%s' % (where, code, text, ' (%d veces)' % n if n > 1 else ''))
    return out


def _run_cal_task(kind, moves):
    try:
        def progress_cb(i, total, message):
            CAL_TASK['progress'] = dict(i=i, total=total)
            CAL_TASK['message'] = message

        log_dir = os.path.join(RUNTIME_DIR, 'calibracion')
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, '%s_%s.log' % (time.strftime('%Y%m%d_%H%M%S'), kind))
        with open(log_path, 'w', encoding='utf-8') as log, _tee_stdout(log):
            print('=== prueba %s, perfil %s ===' % (kind, cal.active_id()), flush=True)
            if kind == 'precision':
                CAL_TASK['precision'] = cal.precision_test(robot.get(), progress_cb=progress_cb)
            else:
                result = pl.draw(robot.get(), moves, progress_cb=progress_cb, preserve_aspect=False,
                                 bounds=pl.lib.area_bounds(), check_every=1)
                CAL_TASK['sync'] = (result or {}).get('sync')
                CAL_TASK['robot_errors'] = _describe_robot_errors(
                    result, moves, OUTLINE_POINT_LABELS if kind == 'outline' else None)
        CAL_TASK['log'] = os.path.relpath(log_path, HERE)
        CAL_TASK['message'] = 'terminado'
    except Exception as exc:
        traceback.print_exc()
        robot.disconnect()
        CAL_TASK['error'] = str(exc)
    finally:
        CAL_TASK['running'] = False
        robot.release()


@app.route('/api/calibration/test/<kind>', methods=['POST'])
def calibration_test(kind):
    """Dibuja el contorno del area, el patron de prueba o un rostro generico
    (girado segun la orientacion del perfil) con el perfil activo."""
    if kind not in TEST_KINDS:
        return jsonify(error='prueba invalida'), 400
    try:
        if kind == 'precision':
            moves = []
        elif kind == 'face':
            moves = pl.lib.test_face_moves(cal.active_rotation_deg())
        else:
            moves = pl.lib.outline_moves() if kind == 'outline' else pl.lib.test_pattern_moves()
    except ValueError as exc:
        return jsonify(error='%s Usa el contorno.' % exc), 400
    try:
        robot.acquire('haciendo la %s' % TEST_KINDS[kind])
    except robot.RobotBusy as exc:
        return jsonify(error=str(exc)), 409
    total = len(cal.PRECISION_GRID) if kind == 'precision' else len(moves)
    CAL_TASK.update(kind=kind, running=True, progress=dict(i=0, total=total),
                    message='conectando...', error=None, robot_errors=[], sync=None, log=None, precision=None)
    threading.Thread(target=_run_cal_task, args=(kind, moves), daemon=True).start()
    return jsonify(_cal_state())


# ---------------------------------------------------------------------------
# Consulta de estado + archivos
# ---------------------------------------------------------------------------
@app.route('/api/jobs/<job_id>', methods=['GET'])
def get_job(job_id):
    return jsonify(_public(_get_job(job_id)))


@app.route('/api/status', methods=['GET'])
def api_status():
    return jsonify(drawing_job_id=DRAWING_JOB_ID['id'])


@app.route('/runtime/<job_id>/<path:filename>')
def serve_runtime_file(job_id, filename):
    return send_from_directory(_job_dir(job_id), filename)


@app.route('/')
def index():
    return app.send_static_file('index.html')


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5050))
    app.run(host='0.0.0.0', port=port, debug=True, threaded=True, use_reloader=False)
