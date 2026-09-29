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
import routines

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
        sketch_note=job.get('sketch_note'),
        preview_url=url('preview_path'),
        svg_url=url('svg_path'),
        style=job.get('style', gemini_sketch.DEFAULT_STYLE),
        engine=_job_engine(job),
        engine_name=SKETCH_ENGINES[_job_engine(job)],
        sketch_engine=job.get('sketch_engine'),
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
def _style_from(value):
    return value if value in gemini_sketch.STYLES else gemini_sketch.DEFAULT_STYLE


@app.route('/api/styles', methods=['GET'])
def list_styles():
    import local_sketch
    return jsonify(styles=[dict(id=k, name=v[0]) for k, v in gemini_sketch.STYLES.items()],
                   default=gemini_sketch.DEFAULT_STYLE,
                   engines=[dict(id=k, name=v) for k, v in SKETCH_ENGINES.items()],
                   engine_default=_drawing_settings()['sketch_engine'], local_ready=local_sketch.available())


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
    job['style'] = _style_from(request.form.get('style'))
    try:
        job['engine'] = _engine_from(request.form.get('engine'))
    except ValueError as exc:
        return jsonify(error=str(exc)), 400
    job['stage'] = 'uploaded'
    return jsonify(_public(job))


# Motores del boceto (foto -> dibujo de lineas). El elegido en la calibracion
# (drawing_settings: sketch_engine) va primero; si falla, el otro; y para el
# estilo lapiz, al final el lapiz de OpenCV (retrato_lapiz.opencv_pencil).
SKETCH_ENGINES = {'local': 'Local (U²-Net, sin internet, ~1 s)', 'gemini': 'Gemini (en la nube)',
                  'opencv': 'Lápiz de OpenCV (sin IA, instantáneo)'}
SKETCH_ENGINE_DEFAULT = 'local'
AI_ENGINES = ('local', 'gemini')   # los que se reemplazan entre si si uno falla


def _job_engine(job):
    """El motor elegido para este trabajo (pestana Dibujar) o, si no eligio,
    el de la calibracion (el que usa la pantalla del publico)."""
    return job.get('engine') if job.get('engine') in SKETCH_ENGINES else _drawing_settings()['sketch_engine']


def _engine_from(value):
    """Motor pedido en un formulario: None si no vino (se usa el de la calibracion)."""
    if value in (None, ''):
        return None
    if value not in SKETCH_ENGINES:
        raise ValueError('motor de boceto desconocido: %s' % value)
    return value


def _make_sketch(job, out_path):
    """Genera el boceto con el motor elegido (ver _job_engine) y, si falla, con
    el otro; en el estilo lapiz, al final el lapiz de OpenCV. Devuelve una nota
    si se uso un respaldo (o None); RuntimeError si no salio ninguno."""
    import local_sketch
    import retrato_lapiz
    engine = _job_engine(job)
    if engine == 'opencv':   # elegido a proposito: sin IA
        retrato_lapiz.opencv_pencil(job['original_path'], out_path)
        job['sketch_engine'] = 'opencv'
        return None
    order = [engine] + [e for e in AI_ENGINES if e != engine]
    errors = []
    for eng in order:
        try:
            if eng == 'local':
                local_sketch.generate(job['original_path'], out_path)
            else:
                gemini_sketch.generate_sketch(job['original_path'], out_path,
                                              job.get('style', gemini_sketch.DEFAULT_STYLE))
            job['sketch_engine'] = eng
            if eng == engine:
                return None
            return '%s no funciono (%s): se uso %s.' % (SKETCH_ENGINES[engine], errors[0][:120], SKETCH_ENGINES[eng])
        except Exception as exc:
            traceback.print_exc()
            errors.append(str(exc))
    if job.get('style') == 'lapiz':   # respaldo sin red ni modelo (ver retrato_lapiz.opencv_pencil)
        retrato_lapiz.opencv_pencil(job['original_path'], out_path)
        job['sketch_engine'] = 'opencv'
        return 'No salio el boceto (%s): se uso el lapiz de OpenCV.' % ' / '.join(e[:80] for e in errors)
    raise RuntimeError(' / '.join(errors))


def _run_sketch(job_id):
    job = _get_job(job_id)
    job['stage'] = 'sketching'
    job['sketch_error'] = None
    job['sketch_note'] = None
    try:
        out_path = os.path.join(_job_dir(job_id), 'sketch.png')
        job['sketch_note'] = _make_sketch(job, out_path)
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
    """Rechazar el boceto (o reintentar si fallo): se genera otro, con el estilo
    y el motor que vengan en el form (o los mismos de antes)."""
    job = _get_job(job_id)
    if job['stage'] not in ('sketch_ready', 'sketch_failed'):
        return jsonify(error='etapa invalida: %s' % job['stage']), 409
    if request.form.get('style'):
        job['style'] = _style_from(request.form.get('style'))
    try:
        if request.form.get('engine'):
            job['engine'] = _engine_from(request.form.get('engine'))
    except ValueError as exc:
        return jsonify(error=str(exc)), 400
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
    job['lapiz'] = None
    if job.get('style') == 'lapiz':
        _build_pencil(job, source)
        return
    line_art = job.get('style') == 'trazo'
    if line_art:
        # estilo "trazo simple": boceto -> SVG de lineas largas -> moves (ver lib/sketch_svg.py)
        svg_path = os.path.join(_job_dir(job['id']), 'sketch.svg')
        raw_moves = pl.line_art_svg_moves(source, svg_path, detail)
        job['svg_path'] = svg_path
        core_bbox, eyes = pl.locate_face(source, original_photo_path=photo)
    else:
        job['svg_path'] = None
        raw_moves, _shape, core_bbox, eyes = pl.image_to_raw_moves(source, threshold=threshold, min_len_px=min_len_px,
                                                                    original_photo_path=photo)
    job['raw_moves'] = raw_moves
    # estilo de ojos elegido en la calibracion (ver eye_styles): se borran los
    # ojos del boceto y se dibuja la plantilla, ajustada a los de la persona
    eye_style = _drawing_settings().get('eye_style')
    use_eye_style = bool(eye_style and eyes)
    if use_eye_style:
        raw_moves = pl.remove_eye_regions(raw_moves, eyes)
    # orientacion segun el perfil activo (hoja de frente o al costado del robot)
    rotation = cal.active_rotation_deg()
    if line_art:
        moves = pl.finish_line_art_moves(raw_moves, detail, rotation)
    else:
        moves = pl.simplify_for_robot(raw_moves, core_bbox=core_bbox, rotation_deg=rotation, **params)
    moves, job['short_dropped'] = pl.drop_short_strokes(moves, rotation)
    if not moves:
        raise RuntimeError('No quedaron trazos utilizables con este nivel de detalle '
                            '(probá un nivel mas alto).')
    # recorrido optimizado (vpype): menos bajadas de pluma y menos viaje. Antes de
    # los ojos de estilo: sus curvas no se unen nunca (ver eye_styles)
    moves = pl.optimize_moves(moves, pl._px_to_mm_scale(moves, rotation))
    if use_eye_style:
        moves += pl.eye_style_moves(eye_style, eyes, pl._px_to_mm_scale(moves, rotation))
    job['eyes_note'] = (('ojos: estilo %d, %s' % (pl.eye_styles.STYLE_IDS.index(eye_style) + 1,
                                                   dict(pl.eye_styles.STYLES)[eye_style]))
                        if use_eye_style else
                        'ojos: no se ubicaron en el boceto, quedan los del boceto' if eye_style else None)
    job['rotation'] = rotation
    moves = pl.rotate_moves(moves, rotation)
    job['moves'] = moves

    gcode_path = os.path.join(_job_dir(job['id']), 'final.gcode')
    pl.lib.write_gcode(moves, gcode_path)
    job['gcode_final_path'] = gcode_path

    preview_path = os.path.join(_job_dir(job['id']), 'preview.png')
    stats = pl.render_preview(moves, preview_path)
    stats['eyes'] = job['eyes_note']
    stats['short_dropped'] = job['short_dropped']
    stats['min_stroke_mm'] = pl.MIN_DRAWN_STROKE_MM
    stats['style'] = gemini_sketch.STYLES.get(job.get('style'), gemini_sketch.STYLES[gemini_sketch.DEFAULT_STYLE])[0]
    job['preview_path'] = preview_path
    job['preview_stats'] = stats


def _split_textures(job, source, rotation):
    """Zonas de textura del boceto (ver texturas.py): devuelve (boceto sin ellas,
    trazos del rayado en pixeles de trabajo con Y hacia arriba, como los moves).
    Sin texturas: (el mismo boceto, [])."""
    import shutil
    import numpy as np
    import sketch_svg
    import texturas
    try:
        img = pl.dp.load_image(source)
        h = img.shape[0]
        ys, xs = np.nonzero(pl.dp.sketch_to_mask(img))
        if not len(xs):
            return source, []
        # mm por pixel de trabajo, como va a quedar el dibujo (encuadre de la tinta)
        mm = pl._px_to_mm_scale([(float(xs.min()), float(h - ys.max()), True),
                                 (float(xs.max()), float(h - ys.min()), True)], rotation)
        _core, _eyes, pts = pl.locate_face(source, with_points=True)
        clean = os.path.join(_job_dir(job['id']), 'sketch_limpio.png')
        strokes, _region, info = texturas.split_textures(source, mm, clean, texturas.feature_zones(pts, h))
        job['textures'] = info
        if not info['zones']:
            return source, []
        if os.path.exists(source + '.landmarks.json'):   # los landmarks del boceto local valen igual
            shutil.copy(source + '.landmarks.json', clean + '.landmarks.json')
        step = 0.8 / mm
        out = []
        for st in strokes:
            st = sketch_svg.resample(np.asarray(st, float), step)
            if len(st) >= 2:
                out.append(np.column_stack([st[:, 0], h - st[:, 1]]))
        return clean, out
    except Exception:
        traceback.print_exc()   # si algo falla, el boceto como estaba
        return source, []


def _build_pencil(job, source):
    """Dibujo a lapiz de lineas (ver retrato_lapiz.py), en dos versiones: la
    tableta muestra el dibujo DETALLADO (display_moves) y el brazo recorre uno
    SIMPLE y mas rapido (moves); la tableta va dibujando lo detallado al ritmo y
    por la zona de la punta. Las dos usan el encuadre del detallado (bounds),
    para que coincidan sobre la hoja. El preview muestra las dos."""
    import retrato_lapiz
    rotation = cal.active_rotation_deg()
    svg_path = os.path.join(_job_dir(job['id']), 'sketch.svg')
    # pelo, barba y otros rellenos: se sacan del boceto y se redibujan con un
    # rayado que sigue la direccion del pelo (ver texturas.py): vectorizados tal
    # cual quedaban como una red de lineas
    source, texture = _split_textures(job, source, rotation)
    raw = pl.line_art_svg_moves(source, svg_path, 'lapiz')
    job['raw_moves'] = raw
    # ojos, cejas, nariz y boca vectorizados a mas resolucion (ver retrato_lapiz)
    detailed, _face, feature_box = retrato_lapiz.with_detailed_features(
        source, raw, rotation, line_mm=_drawing_settings()['tablet_line_mm'],
        # el boceto local deja los ojos flojos: se dibujan desde los landmarks (mirada igualada)
        eyes_from='landmarks' if job.get('sketch_engine') == 'local' else 'sketch')
    lines_only = detailed   # sin el rayado: el brazo lo recorre aparte, en zigzag
    if texture:
        import numpy as np
        detailed = pl._flat_strokes_to_moves(pl.dp.order_strokes(
            [np.asarray(st) for st in retrato_lapiz._strokes(detailed)] + texture))
    if not detailed:
        raise RuntimeError('No se encontraron lineas en el dibujo.')
    mm_per_px = pl._px_to_mm_scale(detailed, rotation)
    robot_moves, seconds, n_lines, n_total = retrato_lapiz.robot_version(
        lines_only, mm_per_px, feature_box=feature_box, hatch=texture)
    # la tableta no muestra lo que el brazo no recorre cerca (ver prune_display)
    detailed, job['display_pruned'] = retrato_lapiz.prune_display(detailed, robot_moves, mm_per_px)
    display = pl.rotate_moves(detailed, rotation)
    moves = pl.rotate_moves(robot_moves, rotation)
    # marco del retrato: PORTRAIT_ASPECT vertical, centrado en el dibujo, con un
    # poco de aire. Es el encuadre (bounds) con el que el brazo lo ubica en la
    # hoja, girado segun el perfil (con 90 grados, horizontal); lo que se ve y se
    # descarga es este marco, derecho (ver _frame_uv).
    xs, ys = [m[0] for m in detailed], [m[1] for m in detailed]
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    aw, ah = PORTRAIT_ASPECT
    fh = max(max(ys) - min(ys), (max(xs) - min(xs)) * ah / aw) * (1 + 2 * PORTRAIT_AIR)
    fw = fh * aw / ah
    corners = pl.rotate_moves([(cx - fw / 2, cy - fh / 2, True), (cx + fw / 2, cy + fh / 2, True)], rotation)
    job['bounds'] = (min(c[0] for c in corners), max(c[0] for c in corners),
                     min(c[1] for c in corners), max(c[1] for c in corners))
    job['portrait_frame'] = True
    job['rotation'] = rotation
    job['moves'] = moves
    job['display_moves'] = display
    job['feature_box'] = None
    if feature_box:   # la zona de los rasgos, girada como el dibujo
        fx0, fx1, fy0, fy1 = feature_box
        corners = pl.rotate_moves([(fx0, fy0, True), (fx1, fy1, True), (fx0, fy1, True), (fx1, fy0, True)], rotation)
        job['feature_box'] = (min(c[0] for c in corners), max(c[0] for c in corners),
                              min(c[1] for c in corners), max(c[1] for c in corners))
    job['eyes_note'] = None
    job['lapiz'] = dict(seconds=seconds)
    job['svg_path'] = svg_path
    gcode_path = os.path.join(_job_dir(job['id']), 'final.gcode')          # lo que recorre el brazo
    pl.lib.write_gcode(moves, gcode_path)
    pl.lib.write_gcode(display, os.path.join(_job_dir(job['id']), 'tableta_detallado.gcode'))
    job['gcode_final_path'] = gcode_path
    preview_path = os.path.join(_job_dir(job['id']), 'preview.png')
    stats = _pencil_preview(job, preview_path)
    n_display = len(retrato_lapiz._strokes(display))
    stats['style'] = gemini_sketch.STYLES['lapiz'][0]
    stats['estimated_minutes'] = round(seconds / 60.0, 1)   # en modo tableta
    stats['pencil'] = ('la tableta muestra el dibujo detallado (%d lineas); el brazo recorre una version simple '
                       '(%d lineas%s), ~%d s en modo tableta'
                       % (n_display, n_lines, (' de %d largas, por el tope de tiempo' % n_total)
                          if n_total > n_lines else '', seconds))
    job['preview_path'] = preview_path
    job['preview_stats'] = stats


# Proporcion del retrato (ancho, alto): la imagen que se ve y se descarga.
PORTRAIT_ASPECT = (3, 4)
PORTRAIT_AIR = 0.05      # aire alrededor del dibujo dentro del marco (fraccion del alto)


def _frame_uv(job):
    """El marco del retrato sobre la hoja: (u0, u1, v0, v1) como los puntos
    que publica draw_moves (lib.iter_sheet_uv), o None (trabajos viejos)."""
    b = job.get('bounds')
    if not b or not job.get('portrait_frame'):
        return None
    uv = [(u, v) for *_x, u, v in pl.lib.iter_sheet_uv([(b[0], b[2], True), (b[1], b[3], True)], True, b)]
    return (min(p[0] for p in uv), max(p[0] for p in uv), min(p[1] for p in uv), max(p[1] for p in uv))


def _display_points(job):
    """Lo que muestra la tableta del dibujo a lapiz: [(u, v, pluma, grosor)]
    sobre la hoja; los rasgos con el trazo mas fino: FEATURE_LINE_FACTOR en el
    centro de la zona (ojos, cejas, nariz, boca) y de a poco hasta el grosor
    normal en su margen (FEATURE_MARGIN), para que un contorno que la cruza no
    cambie de grosor de golpe."""
    import retrato_lapiz
    shown = job.get('display_moves') or job['moves']
    box = job.get('feature_box')
    thin = retrato_lapiz.FEATURE_LINE_FACTOR
    if box:
        m = retrato_lapiz.FEATURE_MARGIN / (1 + 2 * retrato_lapiz.FEATURE_MARGIN)
        rx, ry = m * (box[1] - box[0]), m * (box[3] - box[2])   # ancho del margen
        core = (box[0] + rx, box[1] - rx, box[2] + ry, box[3] - ry)
    out = []
    for (x, y, _pd), (*_x, pd, u, v) in zip(shown, pl.lib.iter_sheet_uv(shown, True, job.get('bounds'))):
        w = 1
        if box:
            # 0 dentro del centro, 1 en el borde de la zona o afuera
            t = max((core[0] - x) / rx, (x - core[1]) / rx, (core[2] - y) / ry, (y - core[3]) / ry, 0.0)
            w = thin + (1 - thin) * min(1.0, t)
        out.append((u, v, pd, round(w, 2)))
    return out


def _pencil_preview(job, out_path):
    """Preview del lapiz: a la izquierda el retrato como se ve y se descarga
    (derecho, PORTRAIT_ASPECT), a la derecha lo que recorre el brazo sobre la
    hoja (girado segun el perfil). Devuelve las estadisticas del recorrido del brazo."""
    import cv2
    import numpy as np
    import share
    base, _ext = os.path.splitext(out_path)
    stats = pl.render_preview(job['moves'], base + '_brazo.png', bounds=job['bounds'])
    arm = cv2.imread(base + '_brazo.png', cv2.IMREAD_COLOR)
    share.render_portrait(_display_points(job), cal.area_mm(), job.get('rotation', 0),
                          _drawing_settings()['tablet_line_mm'], base + '_retrato.png', side_px=1000, pencil=True,
                          color=job.get('color'), frame=_frame_uv(job))
    portrait = cv2.imread(base + '_retrato.png', cv2.IMREAD_COLOR)
    k = 0.86 * arm.shape[0] / portrait.shape[0]        # del alto del grafico del brazo
    portrait = cv2.resize(portrait, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    portrait = cv2.copyMakeBorder(portrait, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=(200, 200, 200))
    panels = []
    for img, label in ((portrait, 'Retrato (asi se ve)'), (arm, 'Brazo (simple, sobre la hoja)')):
        img = cv2.copyMakeBorder(img, 0, 0, 16, 16, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        head = np.full((44, img.shape[1], 3), 255, np.uint8)
        cv2.putText(head, label, (16, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (40, 40, 40), 2, cv2.LINE_AA)
        panels.append(np.vstack([head, img]))
    h = max(p.shape[0] for p in panels)
    panels = [np.vstack([p, np.full((h - p.shape[0], p.shape[1], 3), 255, np.uint8)]) for p in panels]
    cv2.imwrite(out_path, np.hstack([panels[0], np.full((h, 12, 3), 255, np.uint8), panels[1]]))
    return stats


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
    if request.form.get('style'):   # como convertir a gcode (un boceto normal tambien se puede pasar por SVG)
        job['style'] = _style_from(request.form.get('style'))
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
            lapiz = job.get('lapiz')
            if lapiz:
                # el brazo recorre la version simple; la tableta dibuja como grafito la
                # detallada, siguiendo a la punta, y se descarga la detallada
                bounds = job.get('bounds')
                display = _display_points(job) if job.get('display_moves') else None
                result = pl.draw(robot.get(), job['moves'], progress_cb=progress_cb, unidirectional=False,
                                 bounds=bounds, live=LIVE.named('retrato', share=True, render='pencil', display=display,
                                                                color=job.get('color'), frame=_frame_uv(job)))
            else:
                result = pl.draw(robot.get(), job['moves'], progress_cb=progress_cb, live=LIVE.named('retrato', share=True))
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
              'precision': 'prueba de precision', 'face': 'prueba del rostro', 'eyes': 'prueba de ojos',
              'dots': 'prueba del ojo de puntos', 'stipple': 'rostro con puntos'}

# Rostro real con puntos (ver puntillismo.py): se genera primero una vista
# previa (tarda un segundo) y se dibuja despues (tarda minutos). El plan depende
# de la calibracion (tamano del area, orientacion): si el perfil cambio entre la
# vista previa y el dibujo, se vuelve a planificar con los mismos parametros.
STIPPLE = dict(plan=None, params=None, profile=None, source=None, preview=None, stats=None)
STIPPLE_DOTS_RANGE = (100, 1200)
STIPPLE_METHODS = ('voronoi', 'atkinson', 'floyd')


def _profile_signature():
    pid = cal.active_id()
    return (pid, cal.load(pid).get('updated_at') if pid else None)


def _stipple_plan(path, dots, method):
    import puntillismo
    return puntillismo.plan_from_image(path, n_dots=dots, method=method, rotation_deg=cal.active_rotation_deg())


# Preferencias de dibujo que no dependen de la hoja (a diferencia de los
# perfiles de calibracion): por ahora, el estilo de ojos de los retratos.
DRAWING_SETTINGS_PATH = os.path.join(HERE, 'drawing_settings.json')


def _drawing_settings():
    try:
        with open(DRAWING_SETTINGS_PATH, encoding='utf-8') as fh:
            data = json.load(fh)
    except (FileNotFoundError, ValueError):
        data = {}
    if data.get('eye_style') not in pl.eye_styles.STYLE_IDS:
        data['eye_style'] = None
    try:
        data['tablet_line_mm'] = float(data.get('tablet_line_mm', TABLET_LINE_MM_DEFAULT))
    except (TypeError, ValueError):
        data['tablet_line_mm'] = TABLET_LINE_MM_DEFAULT
    if data.get('sketch_engine') not in SKETCH_ENGINES:
        data['sketch_engine'] = SKETCH_ENGINE_DEFAULT
    # marco del retrato (archivo en marcos/; '' = sin marco; si nunca se eligio, marcos/marco.jpg)
    import marco
    if 'frame' not in data:
        data['frame'] = FRAME_DEFAULT if os.path.exists(marco.frame_path(FRAME_DEFAULT)) else ''
    if data['frame'] and not os.path.exists(marco.frame_path(data['frame'])):
        data['frame'] = ''
    return data


FRAME_DEFAULT = 'marco.jpg'


def _save_drawing_settings(data):
    with open(DRAWING_SETTINGS_PATH, 'w', encoding='utf-8') as fh:
        json.dump(data, fh, indent=2)


# Grosor del trazo en la tableta (mm reales sobre la pantalla), tanto de las
# lineas lisas como del lapiz (y de la imagen que se descarga). Grueso alcanza
# para que los errores chicos de posicion del brazo no se noten; para el lapiz
# se ve mas natural fino (0.4-1 mm).
TABLET_LINE_MM_DEFAULT = 2.5
TABLET_LINE_MM_RANGE = (0.2, 8.0)


@app.route('/api/drawing-settings', methods=['POST'])
def update_drawing_settings():
    """Cambia solo los ajustes que vengan: eye_style (id de eye_styles.STYLES,
    o null = dejar los ojos del boceto; vale para los retratos que se generen
    desde ahora) y tablet_line_mm (grosor del trazo en la tableta, al instante)."""
    body = request.get_json(silent=True) or {}
    data = _drawing_settings()
    if 'eye_style' in body:
        style = body.get('eye_style') or None
        if style is not None and style not in pl.eye_styles.STYLE_IDS:
            return jsonify(error='estilo de ojos desconocido'), 400
        data['eye_style'] = style
    if 'tablet_line_mm' in body:
        try:
            line = round(float(body['tablet_line_mm']), 2)
        except (TypeError, ValueError):
            return jsonify(error='el grosor debe ser un numero'), 400
        if not TABLET_LINE_MM_RANGE[0] <= line <= TABLET_LINE_MM_RANGE[1]:
            return jsonify(error='el grosor debe estar entre %g y %g mm' % TABLET_LINE_MM_RANGE), 400
        data['tablet_line_mm'] = line
    if 'sketch_engine' in body:
        if body['sketch_engine'] not in SKETCH_ENGINES:
            return jsonify(error='motor de boceto desconocido'), 400
        data['sketch_engine'] = body['sketch_engine']
    _save_drawing_settings(data)
    return jsonify(_cal_state(request.args.get('profile')))


def _local_sketch_ready():
    import local_sketch
    return local_sketch.available()


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
        eye_styles=[dict(id=s, name=n) for s, n in pl.eye_styles.STYLES],
        eye_style=_drawing_settings().get('eye_style'),
        tablet_line_mm=_drawing_settings()['tablet_line_mm'],
        sketch_engine=_drawing_settings()['sketch_engine'],
        sketch_engines=[dict(id=k, name=v) for k, v in SKETCH_ENGINES.items()],
        local_sketch_ready=_local_sketch_ready(),
        stipple=_stipple_state(),
        routines=routines.listing(),
        routine_task=dict(ROUTINE_TASK),
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


@app.route('/api/calibration/profiles/<profile_id>/from-center', methods=['POST'])
def calibration_from_center(profile_id):
    """Calibracion desde el centro (ver cal.corners_from_center): {center,
    width_cm, height_cm, angle_deg, sag_mm_per_10cm}. Calcula las 4 esquinas y
    las guarda en el perfil, con el centro medido como centro de la hoja (los
    ajustes de altura por esquina vuelven a 0: son de la calibracion anterior).
    Con preview=true solo devuelve el calculo, sin guardar."""
    body = request.get_json(silent=True) or {}
    try:
        corners, info = cal.corners_from_center(body)
        if body.get('preview'):
            return jsonify(dict(_cal_state(profile_id), from_center=info))
        center = info['params']['center']
        changes = dict(corners=corners, z_adjust_mm={k: 0.0 for k in cal.CORNERS}, center_z=center[2],
                       center_xy=center[:2], center_z_adjust_mm=0.0, center_calib=info['params'])
        if profile_id == cal.active_id():
            _without_robot_moving('guardando calibracion', lambda: cal.update(profile_id, changes))
        else:
            cal.update(profile_id, changes)
    except Exception as exc:
        return _cal_error(exc)
    return jsonify(dict(_cal_state(profile_id), from_center=info))


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
    'center' se guarda la altura (Z) y la posicion (x, y) del centro real de
    la hoja: corrige el medio del dibujo, pero no cambia el area."""
    if corner not in cal.CORNERS + ('center',):
        return jsonify(error='esquina invalida'), 400
    try:
        cal.load(profile_id)
        with robot.use('leyendo posicion') as mc:
            coords = cal.read_coords(mc)
            if corner == 'center':
                cal.update(profile_id, {'center_z': coords[2], 'center_xy': coords[:2]})
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
        if action not in ('release', 'lock', 'read', 'servos'):
            return jsonify(error='accion invalida'), 400
        coords = servos = None
        with robot.use({'release': 'liberando servos', 'lock': 'trabando servos',
                        'read': 'leyendo posicion', 'servos': 'leyendo parametros de servos'}[action]) as mc:
            if action == 'release':
                mc.release_all_servos()
            elif action == 'lock':
                mc.focus_all_servos()
            elif action == 'servos':
                servos = cal.read_servo_params(mc)
            else:
                coords = cal.read_coords(mc)
        if servos is not None:
            _log_servo_params(servos)
    except Exception as exc:
        return _cal_error(exc)
    return jsonify(dict(_cal_state(request.args.get('profile')), coords=coords, servos=servos))


def _log_servo_params(servos):
    log_dir = os.path.join(RUNTIME_DIR, 'calibracion')
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, '%s_servos.log' % time.strftime('%Y%m%d_%H%M%S')), 'w', encoding='utf-8') as log:
        for servo, params in servos.items():
            log.write('J%d %s\n' % (servo, ' '.join('%s=%s' % kv for kv in params.items())))


@app.route('/api/calibration/servo/j1', methods=['POST'])
def calibration_servo_j1():
    """Cambia P, D o I del servo de J1 ({param, value}) o vuelve a los valores
    originales ({restore: true}). Devuelve los parametros leidos despues."""
    body = request.get_json(silent=True) or {}
    try:
        with robot.use('cambiando parametros de J1') as mc:
            if body.get('restore'):
                cal.restore_j1(mc)
                change = 'J1 restaurado a %s' % cal.J1_ORIGINAL
            else:
                value = cal.write_j1_param(mc, str(body.get('param', '')), body.get('value'))
                change = 'J1 %s=%d' % (body.get('param'), value)
            servos = cal.read_servo_params(mc)
        # historial de cambios, para saber siempre que se toco y como volver
        log_dir = os.path.join(RUNTIME_DIR, 'calibracion')
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, 'servo_cambios.log'), 'a', encoding='utf-8') as log:
            log.write('%s %s -> J1 ahora %s\n' % (time.strftime('%Y-%m-%d %H:%M:%S'), change, servos[1]))
        _log_servo_params(servos)
    except Exception as exc:
        return _cal_error(exc)
    return jsonify(dict(_cal_state(request.args.get('profile')), servos=servos, servo_change=change))


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


def _run_cal_task(kind, moves, draw_kw=None):
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
                # el contorno queda igual: sus etiquetas de esquina dependen del orden de los puntos
                kw = dict(preserve_aspect=False, bounds=pl.lib.area_bounds(),
                          # el contorno queda igual: sus etiquetas de esquina dependen del orden de los puntos
                          unidirectional=None if kind != 'outline' else False,
                          # el contorno, seguido: sin pausas de sincronizacion (ver lib.CONTINUOUS_SPEED_MM_S)
                          continuous=(kind == 'outline'))
                kw.update(draw_kw or {})
                result = pl.draw(robot.get(), moves, progress_cb=progress_cb, check_every=1,
                                 # los puntos viajan apenas por encima del papel (ver DOTS_PEN_LIFT_MM)
                                 pen_lift_mm=pl.lib.DOTS_PEN_LIFT_MM if kind in ('dots', 'stipple') else None,
                                 live=LIVE.named(TEST_KINDS[kind], share=(kind == 'stipple')), **kw)
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


def _stipple_state():
    if not STIPPLE['plan'] or not os.path.exists(STIPPLE['preview']):
        return None
    st = STIPPLE['stats']
    return dict(dots=st['points'], bands=st['bands'], method=STIPPLE['params']['method'],
                found=STIPPLE['plan']['face'], source=STIPPLE['source'],
                preview_url='/runtime/calibracion/stipple_preview.png?v=%d' % int(os.path.getmtime(STIPPLE['preview'])),
                estimated_minutes=st['estimated_minutes'])


def _latest_job_photo():
    """La ultima foto subida en "Dibujar" (original.* del trabajo mas nuevo)."""
    best = None
    for name in os.listdir(RUNTIME_DIR):
        d = os.path.join(RUNTIME_DIR, name)
        if name == 'calibracion' or not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            if f.startswith('original.'):
                p = os.path.join(d, f)
                if best is None or os.path.getmtime(p) > os.path.getmtime(best):
                    best = p
    return best


@app.route('/api/calibration/stipple', methods=['POST'])
def calibration_stipple():
    """Genera la vista previa del rostro con puntos: con la foto que venga en
    el form (image) o, si no viene, con la ultima foto subida en "Dibujar"."""
    try:
        dots = int(request.form.get('dots', 400))
    except ValueError:
        return jsonify(error='cantidad de puntos invalida'), 400
    if not STIPPLE_DOTS_RANGE[0] <= dots <= STIPPLE_DOTS_RANGE[1]:
        return jsonify(error='la cantidad de puntos debe estar entre %d y %d' % STIPPLE_DOTS_RANGE), 400
    method = request.form.get('method', 'voronoi')
    if method not in STIPPLE_METHODS:
        return jsonify(error='metodo desconocido: %s' % method), 400
    out_dir = _job_dir('calibracion')
    file = request.files.get('image')
    if file is not None and file.filename:
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in ALLOWED_EXT:
            return jsonify(error='formato no soportado (%s)' % ext), 400
        source = os.path.join(out_dir, 'stipple_source' + ext)
        file.save(source)
        source_name = file.filename
    else:
        source = _latest_job_photo()
        if not source:
            return jsonify(error='Elegí una foto (o subí una antes en "Dibujar").'), 400
        source_name = 'la ultima foto de "Dibujar"'
    try:
        import puntillismo
        plan = _stipple_plan(source, dots, method)
        preview = os.path.join(out_dir, 'stipple_preview.png')
        stats = puntillismo.render_preview(plan, preview)
    except Exception as exc:
        return _cal_error(exc)
    STIPPLE.update(plan=plan, params=dict(path=source, dots=dots, method=method), profile=_profile_signature(),
                   source=source_name, preview=preview, stats=stats)
    return jsonify(_cal_state(request.args.get('profile')))


@app.route('/api/calibration/test/<kind>', methods=['POST'])
def calibration_test(kind):
    """Dibuja el contorno del area, el patron de prueba o un rostro generico
    (girado segun la orientacion del perfil) con el perfil activo."""
    if kind not in TEST_KINDS:
        return jsonify(error='prueba invalida'), 400
    draw_kw = None   # parametros de dibujo propios de la prueba (si no, los de las pruebas en mm del area)
    try:
        if kind == 'precision':
            moves = []
        elif kind == 'face':
            moves = pl.lib.test_face_moves(cal.active_rotation_deg())
        elif kind == 'eyes':
            moves = pl.lib.test_eyes_moves(cal.active_rotation_deg())
        elif kind == 'dots':
            moves = pl.lib.test_dots_moves(cal.active_rotation_deg())
        elif kind == 'stipple':
            if not STIPPLE['plan']:
                return jsonify(error='Primero generá la vista previa del rostro con puntos.'), 400
            if STIPPLE['profile'] != _profile_signature():   # la calibracion cambio: replanificar
                p = STIPPLE['params']
                STIPPLE.update(plan=_stipple_plan(p['path'], p['dots'], p['method']), profile=_profile_signature())
            import puntillismo
            moves, bounds = puntillismo.plan_to_moves(STIPPLE['plan'])
            draw_kw = dict(preserve_aspect=True, bounds=bounds, unidirectional=False)
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
    threading.Thread(target=_run_cal_task, args=(kind, moves, draw_kw), daemon=True).start()
    return jsonify(_cal_state())


# ---------------------------------------------------------------------------
# Rutinas de movimiento (saludo, invitacion, ...): ver routines.py
# ---------------------------------------------------------------------------
ROUTINE_TASK = dict(id=None, name=None, running=False, lap=None, repeat=None, step=None, steps=None,
                    message=None, error=None)
ROUTINE_STOP = threading.Event()


def _routine_state(**extra):
    return jsonify(dict(routines=routines.listing(), routine_task=dict(ROUTINE_TASK), **extra))


@app.route('/api/routines', methods=['GET'])
def routines_list():
    return _routine_state()


@app.route('/api/routines', methods=['POST'])
def routines_create():
    body = request.get_json(silent=True) or {}
    try:
        rid = routines.create(body.get('name'))
    except ValueError as exc:
        return jsonify(error=str(exc)), 400
    return _routine_state(created=rid)


@app.route('/api/routines/<rid>', methods=['PUT'])
def routines_update(rid):
    """{name} y/o {steps: [{angles, speed, pause_s}, ...]} (la lista completa:
    asi se editan velocidades y pausas, se reordenan y se borran pasos)."""
    body = request.get_json(silent=True) or {}
    try:
        routines.update(rid, name=body.get('name'), steps=body.get('steps'))
    except ValueError as exc:
        return jsonify(error=str(exc)), 400
    return _routine_state()


@app.route('/api/routines/<rid>', methods=['DELETE'])
def routines_delete(rid):
    if ROUTINE_TASK['running'] and ROUTINE_TASK['id'] == rid:
        return jsonify(error='la rutina se esta ejecutando: detenela primero'), 409
    try:
        routines.delete(rid)
    except ValueError as exc:
        return jsonify(error=str(exc)), 400
    return _routine_state()


@app.route('/api/routines/<rid>/capture', methods=['POST'])
def routines_capture(rid):
    """Lee los angulos actuales del brazo (movido a mano con los servos
    sueltos) y los agrega como paso nuevo, o reemplazan el paso {index}."""
    body = request.get_json(silent=True) or {}
    try:
        steps = list(routines.get(rid)['steps'])
        index = body.get('index')
        with robot.use('leyendo la pose') as mc:
            angles = routines.read_angles(mc)
        if index is None:
            steps.append(dict(angles=angles, speed=body.get('speed', routines.DEFAULT_SPEED),
                              pause_s=body.get('pause_s', 0.5)))
        else:
            index = int(index)
            if not 0 <= index < len(steps):
                raise ValueError('paso invalido')
            steps[index] = dict(steps[index], angles=angles)
        routines.update(rid, steps=steps)
    except Exception as exc:
        return _cal_error(exc)
    return _routine_state(captured=angles)


@app.route('/api/routines/<rid>/goto/<int:index>', methods=['POST'])
def routines_goto(rid, index):
    """Lleva el brazo a un paso (para revisarlo o retocarlo): traba los servos."""
    try:
        steps = routines.get(rid)['steps']
        if not 0 <= index < len(steps):
            raise ValueError('paso invalido')
        with robot.use('yendo a una pose de la rutina') as mc:
            mc.focus_all_servos()
            time.sleep(0.3)
            ROUTINE_STOP.clear()
            routines.move_to(mc, steps[index]['angles'], steps[index]['speed'], ROUTINE_STOP)
    except Exception as exc:
        return _cal_error(exc)
    return _routine_state()


def _run_routine(routine, repeat):
    def progress_cb(lap, step, steps):
        ROUTINE_TASK.update(lap=lap, step=step, steps=steps,
                            message='vuelta %d de %d, paso %d de %d' % (lap, repeat, step, steps))
    try:
        ROUTINE_TASK['message'] = routines.play(robot.get(), routine, repeat, progress_cb, ROUTINE_STOP)
    except Exception as exc:
        traceback.print_exc()
        if not isinstance(exc, ValueError):
            robot.disconnect()
        ROUTINE_TASK['error'] = str(exc)
    finally:
        ROUTINE_TASK['running'] = False
        robot.release()


def start_routine(rid, repeat=1):
    """Ejecuta la rutina en segundo plano. Lanza ValueError / RobotBusy."""
    routine = routines.get(rid)
    if not routine['steps']:
        raise ValueError('la rutina "%s" no tiene pasos: agrega posiciones antes de ejecutarla' % routine['name'])
    repeat = int(repeat)
    if not 1 <= repeat <= routines.MAX_REPEAT:
        raise ValueError('las repeticiones deben estar entre 1 y %d' % routines.MAX_REPEAT)
    robot.acquire('ejecutando la rutina %s' % routine['name'])
    ROUTINE_STOP.clear()
    ROUTINE_TASK.update(id=rid, name=routine['name'], running=True, lap=1, repeat=repeat, step=0,
                        steps=len(routine['steps']), message='conectando...', error=None)
    threading.Thread(target=_run_routine, args=(routine, repeat), daemon=True).start()


@app.route('/api/routines/<rid>/play', methods=['POST'])
def routines_play(rid):
    body = request.get_json(silent=True) or {}
    try:
        start_routine(rid, body.get('repeat', 1))
    except (TypeError, ValueError) as exc:
        return jsonify(error=str(exc)), 400
    except robot.RobotBusy as exc:
        return jsonify(error=str(exc)), 409
    return _routine_state()


@app.route('/api/routines/stop', methods=['POST'])
def routines_stop():
    """Frena la rutina en curso (o un "Ir" a una pose): el brazo queda donde esta."""
    ROUTINE_STOP.set()
    return _routine_state()


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


# ---------------------------------------------------------------------------
# Tableta: la pagina /tablet muestra el trazo planeado a medida que el brazo
# avanza (con un lapiz tactil o sin tocar la pantalla), en vez de dibujar en papel
# ---------------------------------------------------------------------------
class LivePublisher:
    """Lo que se esta dibujando y hasta donde llego el brazo (ver
    lib.draw_moves: `live`). Lo lee la tableta con /api/live."""

    def __init__(self):
        self.lock = threading.Lock()
        self.data = dict(id=None, title=None, points=[], reveal=0.0, done=True, area_mm=None, started_at=None,
                         result=None)
        self.next_title, self.next_share = None, False
        self.next_render = self.next_display = self.mapper = self.next_color = self.next_frame = None

    def named(self, title, share=False, render=None, display=None, color=None, frame=None):
        """share: al terminar, generar la imagen y subirla a Firebase para
        descargarla con QR (ver share.py). render='pencil': la tableta dibuja
        el recorrido como trazos de grafito (y la descarga sale igual); si no,
        como lineas lisas. display: [(u, v, pluma), ...] si la tableta tiene que
        mostrar otro dibujo que el recorrido del brazo (el lapiz detallado): se
        dibuja al ritmo del brazo (ver retrato_lapiz.display_sync). color: tono
        del grafito (id de share.PORTRAIT_COLORS; None = grafito)."""
        self.next_title, self.next_share = title, share
        self.next_render = render
        self.next_display = display
        self.next_color = color
        self.next_frame = frame   # (u0, u1, v0, v1): el marco del retrato (ver _frame_uv)
        return self

    def start(self, points):
        mapper = None
        if self.next_display:
            import retrato_lapiz
            points, mapper = retrato_lapiz.display_sync(points, self.next_display, cal.area_mm())
        with self.lock:
            self.mapper = mapper
            self.data = dict(id=uuid.uuid4().hex[:10], title=self.next_title, share=self.next_share,
                             points=[[round(p[0], 5), round(p[1], 5), 1 if p[2] else 0] +
                                     ([round(p[3], 2)] if len(p) > 3 and p[3] != 1 else []) for p in points],
                             reveal=0.0, done=False, area_mm=cal.area_mm(), started_at=time.time(), finished_at=None,
                             rotation=cal.active_rotation_deg(), result=None,
                             render=self.next_render, color=self.next_color, color_hex=_color_hex(self.next_color),
                             frame=self.next_frame)

    def reveal(self, value):
        """value: hasta donde llego el brazo (indice fraccionario en su recorrido)."""
        with self.lock:
            if self.mapper is not None:
                value = self.mapper(value)
            self.data['reveal'] = max(self.data['reveal'], round(float(value), 3))

    def finish(self):
        with self.lock:
            self.data['done'] = True
            self.data['finished_at'] = time.time()   # el tiempo del dibujo deja de correr
            if not self.data.get('share'):
                return
            self.data['result'] = dict(status='preparing')
            snapshot = dict(self.data)
        threading.Thread(target=self._share, args=(snapshot,), daemon=True).start()

    def _share(self, d):
        """Imagen del retrato -> Firebase Storage -> enlace para el QR."""
        import share
        try:
            image = os.path.join(_job_dir('share'), '%s.png' % d['id'])
            line_mm = _drawing_settings()['tablet_line_mm']   # el mismo grosor que dibujo la tableta
            if d.get('render') == 'pencil':
                share.render_portrait(d['points'], d['area_mm'], d['rotation'], line_mm, image, pencil=True,
                                      color=d.get('color'), frame=d.get('frame'))
            else:
                share.render_portrait(d['points'], d['area_mm'], d['rotation'], line_mm, image, frame=d.get('frame'))
            # con marco (si hay), del tamano del papel 10x15 (ver marco.py): es la que se
            # muestra y se descarga (y se imprime desde la app de la impresora)
            frame = _drawing_settings()['frame']
            shown = image
            if frame:
                import marco
                shown = os.path.join(_job_dir('share'), '%s_marco.jpg' % d['id'])
                marco.compose(image, shown, marco.frame_path(frame))
            image_url = '/runtime/share/%s' % os.path.basename(shown)
            with self.lock:
                if self.data['id'] == d['id']:
                    self.data['result'] = dict(status='uploading', image_url=image_url, framed=shown != image)
            portrait_id, url = share.upload(shown, d['title'])
            result = dict(status='ready', url=url, portrait_id=portrait_id, framed=shown != image,
                          image_url=image_url, qr_url='/api/live/qr.svg?id=%s' % d['id'])
        except Exception as exc:
            traceback.print_exc()
            result = dict(status='error', error=str(exc)[:300])
        with self.lock:
            if self.data['id'] == d['id']:   # si ya empezo otro dibujo, este resultado no se muestra
                self.data['result'] = result

    def public(self, have=None):
        with self.lock:
            d = dict(self.data)
        d['total'] = len(d['points'])
        # segundos desde que empezo, calculado aca (el reloj de cada pantalla puede no coincidir con el del server)
        # hasta que termino (si termino), no hasta ahora
        end = d.get('finished_at') or time.time()
        d['elapsed'] = round(end - d['started_at'], 1) if d.get('started_at') else None
        d['line_mm'] = _drawing_settings()['tablet_line_mm']   # se cambia en la calibracion
        try:   # como esta puesta la tableta segun el perfil activo (None = el ajuste de la tableta)
            d['tablet_orient'] = cal.active_tablet_orient()
        except Exception:
            d['tablet_orient'] = None
        if have and have == d['id']:
            del d['points']   # la tableta ya los tiene: solo el avance
        return d


LIVE = LivePublisher()


def _color_hex(color_id):
    import share
    return share.color_hex(color_id)


@app.route('/api/live', methods=['GET'])
def api_live():
    return jsonify(LIVE.public(request.args.get('have')))


@app.route('/api/live/qr.svg', methods=['GET'])
def api_live_qr():
    """QR del enlace de descarga del ultimo retrato (lo genera el servidor:
    la tableta no necesita internet para mostrarlo)."""
    import share
    d = LIVE.public()
    result = d.get('result') or {}
    if d.get('id') != request.args.get('id') or result.get('status') != 'ready':
        return jsonify(error='no hay enlace listo para ese dibujo'), 404
    return app.response_class(share.qr_svg(result['url']), mimetype='image/svg+xml')


# ---------------------------------------------------------------------------
# Marco del retrato (ver marco.py): la imagen que se descarga, del tamano del
# papel 10x15 de la Canon SELPHY CP1500 (se imprime desde la app de Canon)
# ---------------------------------------------------------------------------
def _frame_state(**extra):
    import marco
    frame = _drawing_settings()['frame']
    frame_url = None
    if frame:
        frame_url = '/marcos/%s?v=%d' % (frame, int(os.path.getmtime(marco.frame_path(frame))))
    return jsonify(dict(frame=frame, frame_url=frame_url, page_mm=marco.PAGE_MM, page_px=marco.PAGE_PX,
                        dpi=marco.DPI, **extra))


@app.route('/marcos/<path:name>')
def serve_frame(name):
    import marco
    return send_from_directory(marco.FRAMES_DIR, name)


@app.route('/api/frame', methods=['GET'])
def frame_state():
    return _frame_state()


@app.route('/api/frame', methods=['POST'])
def frame_select():
    """{frame}: nombre de un marco de marcos/, o '' = sin marco."""
    import marco
    body = request.get_json(silent=True) or {}
    name = os.path.basename(body.get('frame') or '')
    if name and not os.path.exists(marco.frame_path(name)):
        return jsonify(error='no existe ese marco'), 400
    data = _drawing_settings()
    data['frame'] = name
    _save_drawing_settings(data)
    return _frame_state()


@app.route('/api/frame/upload', methods=['POST'])
def frame_upload():
    """Subir un marco nuevo (PNG/JPG) y usarlo: el retrato va en su hueco del medio."""
    import marco
    file = request.files.get('frame')
    if file is None or file.filename == '':
        return jsonify(error='falta la imagen del marco'), 400
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ('.png', '.jpg', '.jpeg'):
        return jsonify(error='el marco tiene que ser PNG o JPG'), 400
    os.makedirs(marco.FRAMES_DIR, exist_ok=True)
    name = 'marco_%s%s' % (time.strftime('%Y%m%d_%H%M%S'), '.jpg' if ext == '.jpeg' else ext)
    file.save(marco.frame_path(name))
    data = _drawing_settings()
    data['frame'] = name
    _save_drawing_settings(data)
    return _frame_state()


@app.route('/api/frame/sample', methods=['POST'])
def frame_sample():
    """Vista previa del marco con el ultimo retrato terminado."""
    import glob
    import marco
    frame = _drawing_settings()['frame']
    if not frame:
        return jsonify(error='no hay marco elegido'), 400
    portraits = [p for p in glob.glob(os.path.join(_job_dir('share'), '*.png'))]
    if not portraits:
        return jsonify(error='todavia no hay ningun retrato terminado para probar el marco'), 400
    out = os.path.join(_job_dir('share'), '_muestra_marco.jpg')
    marco.compose(max(portraits, key=os.path.getmtime), out, marco.frame_path(frame))
    return _frame_state(sample_url='/runtime/share/_muestra_marco.jpg?v=%d' % int(time.time() * 1000))


@app.route('/tablet')
def tablet():
    return app.send_static_file('tablet.html')


@app.route('/en-vivo')
def en_vivo():
    """Para mirar el dibujo mientras se hace (una pantalla grande, un celular):
    lo mismo que /tablet, pero derecho, encuadrado sobre el dibujo (sin los
    margenes de la tableta), con el avance y el QR al terminar."""
    resp = app.send_static_file('en_vivo.html')
    resp.headers['Cache-Control'] = 'no-store'
    return resp


@app.route('/tablet.webmanifest')
def tablet_manifest():
    """Agregada a la pantalla de inicio, /tablet abre en pantalla completa
    (en Android, como app instalada solo con https; si no, como acceso directo)."""
    return app.response_class(json.dumps(dict(
        name='Lienzo del robot', short_name='Lienzo', start_url='/tablet', scope='/tablet',
        display='fullscreen', background_color='#ffffff', theme_color='#ffffff')), mimetype='application/manifest+json')


# ---------------------------------------------------------------------------
# Direcciones para conectar la tableta y la pantalla del publico (cambian con
# la red: Wi-Fi del lugar, hotspot de un celular o de esta misma PC)
# ---------------------------------------------------------------------------
WINDOWS_HOTSPOT_IP = '192.168.137.1'   # "Zona con cobertura inalambrica movil" de Windows: siempre esta
SCREEN_PAGES = {'tablet': 'Tableta (lienzo del robot)', 'resultado': 'Pantalla del publico (retrato y QR)',
                'en-vivo': 'Dibujo en vivo (para mirar el avance)'}


def _lan_ips():
    """IPv4 de esta PC en la red local: primero la de la conexion principal,
    despues las demas (sin loopback ni autoconfiguradas 169.254.x.x)."""
    import socket
    ips = []
    try:   # la interfaz por la que saldria el trafico (no manda nada: UDP sin conectar de verdad)
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('10.255.255.255', 1))
        ips.append(s.getsockname()[0])
        s.close()
    except OSError:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ips.append(info[4][0])
    except OSError:
        pass
    out = []
    for ip in ips:
        if ip not in out and not ip.startswith(('127.', '169.254.', '0.')):
            out.append(ip)
    return out


def _screen_urls(port):
    return [dict(ip=ip, hotspot=(ip == WINDOWS_HOTSPOT_IP),
                 pages=[dict(page=p, name=n, url='http://%s:%s/%s' % (ip, port, p)) for p, n in SCREEN_PAGES.items()])
            for ip in _lan_ips()]


def _request_port():
    """Puerto con el que se abrio esta pagina (el mismo que va a usar la tableta)."""
    host = request.host or ''
    if host.rsplit(':', 1)[-1].isdigit():
        return host.rsplit(':', 1)[-1]
    return str(os.environ.get('PORT', 5050))


@app.route('/api/connection', methods=['GET'])
def api_connection():
    port = _request_port()
    return jsonify(port=port, addresses=_screen_urls(port))


@app.route('/api/connection/qr.svg', methods=['GET'])
def api_connection_qr():
    """QR de una de las direcciones de /api/connection (solo esas: no es un
    generador de QR para cualquier texto)."""
    import share
    port = _request_port()
    url = request.args.get('url', '')
    valid = {p['url'] for a in _screen_urls(port) for p in a['pages']}
    if url not in valid:
        return jsonify(error='direccion desconocida'), 404
    return app.response_class(share.qr_svg(url), mimetype='image/svg+xml')


# ---------------------------------------------------------------------------
# Pantalla del publico (/resultado): la persona se toma la foto con la camara,
# elige el tono del retrato y la manda; el servidor sigue el mismo flujo que la
# interfaz de "Dibujar" (boceto con Gemini -> gcode) con el estilo de la
# tableta, muestra como va a quedar y, cuando la persona lo confirma, dibuja.
# Hay un solo retrato del publico a la vez (KIOSK).
# ---------------------------------------------------------------------------
KIOSK = {'job_id': None, 'dismissed_live': None}
KIOSK_STYLE = 'lapiz'
KIOSK_BUSY_STAGES = ('uploaded', 'sketching', 'building_gcode')


def _kiosk_job():
    with JOBS_LOCK:
        return JOBS.get(KIOSK['job_id']) if KIOSK['job_id'] else None


def _kiosk_state():
    import share
    job = _kiosk_job()
    out = dict(colors=[dict(id=c, name=n, hex=h) for c, n, h in share.PORTRAIT_COLORS],
               default_color=share.DEFAULT_COLOR, dismissed_live=KIOSK['dismissed_live'],
               robot_busy=robot.BUSY['what'], job=None)
    if job:
        pub = _public(job)
        out['job'] = dict(id=job['id'], stage=job['stage'], color=job.get('color'), original_url=pub['original_url'],
                          preview_url=('/runtime/%s/%s?v=%d' % (job['id'], os.path.basename(job['kiosk_preview']),
                                                                 int(job.get('kiosk_preview_at', 0) * 1000))
                                       if job.get('kiosk_preview') else None),
                          error=job.get('kiosk_error') or job.get('sketch_error') or job.get('draw_error'))
    return out


def _run_kiosk(job_id):
    """Boceto -> gcode -> imagen de como va a quedar, sin intervencion."""
    _run_sketch(job_id)
    job = _get_job(job_id)
    if job['stage'] != 'sketch_ready':
        return   # sketch_failed: el error queda en sketch_error
    job['stage'] = 'building_gcode'
    try:
        _build_gcode(job, pl.DEFAULT_DETAIL)
        _kiosk_preview(job)
        job['stage'] = 'gcode_ready'
    except Exception as exc:
        traceback.print_exc()
        job['kiosk_error'] = str(exc)
        job['stage'] = 'kiosk_failed'


def _kiosk_preview(job):
    """El retrato como va a quedar en la tableta (y en la descarga): lo
    detallado, del tono elegido."""
    import share
    points = _display_points(job)
    path = os.path.join(_job_dir(job['id']), 'kiosk_preview.png')
    share.render_portrait(points, cal.area_mm(), job.get('rotation', 0), _drawing_settings()['tablet_line_mm'], path,
                          side_px=1000, pencil=True, color=job.get('color'), frame=_frame_uv(job))
    frame = _drawing_settings()['frame']
    if frame:   # como se va a descargar e imprimir
        import marco
        framed = os.path.join(_job_dir(job['id']), 'kiosk_preview_marco.jpg')
        marco.compose(path, framed, marco.frame_path(frame))
        path = framed
    job['kiosk_preview'] = path
    job['kiosk_preview_at'] = time.time()


@app.route('/api/kiosk', methods=['GET'])
def kiosk_state():
    return jsonify(_kiosk_state())


@app.route('/api/kiosk', methods=['POST'])
def kiosk_create():
    """Foto de la camara ('image') + tono ('color') -> retrato nuevo del publico."""
    import share
    current = _kiosk_job()
    if current and (current['stage'] in KIOSK_BUSY_STAGES or DRAWING_JOB_ID['id'] == current['id']):
        return jsonify(error='Ya se esta preparando un retrato. Espera a que termine.'), 409
    file = request.files.get('image')
    if file is None or file.filename == '':
        return jsonify(error='falta la foto'), 400
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXT:
        return jsonify(error='formato no soportado (%s). Usa png/jpg/webp' % ext), 400
    color = request.form.get('color')
    if color not in {c for c, _n, _h in share.PORTRAIT_COLORS}:
        color = share.DEFAULT_COLOR
    job = _new_job()
    path = os.path.join(_job_dir(job['id']), 'original' + ext)
    file.save(path)
    job.update(original_path=path, style=KIOSK_STYLE, color=color, kiosk=True, stage='sketching')
    KIOSK['job_id'] = job['id']
    threading.Thread(target=_run_kiosk, args=(job['id'],), daemon=True).start()
    return jsonify(_kiosk_state())


@app.route('/api/kiosk/draw', methods=['POST'])
def kiosk_draw():
    """La persona confirmo: dibujar su retrato."""
    job = _kiosk_job()
    if not job or job['stage'] != 'gcode_ready':
        return jsonify(error='No hay un retrato listo para dibujar.'), 409
    resp = _start_draw(job)
    if isinstance(resp, tuple):   # el brazo esta ocupado
        return jsonify(error='El robot esta terminando otro dibujo. Proba de nuevo en un momento.'), 409
    return jsonify(_kiosk_state())


@app.route('/api/kiosk/reset', methods=['POST'])
def kiosk_reset():
    """Volver al inicio (otra foto, u otra persona). No corta un dibujo en curso."""
    job = _kiosk_job()
    if job and DRAWING_JOB_ID['id'] == job['id']:
        return jsonify(error='El robot esta dibujando este retrato.'), 409
    KIOSK['job_id'] = None
    KIOSK['dismissed_live'] = LIVE.public().get('id')   # el ultimo resultado ya se vio
    return jsonify(_kiosk_state())


@app.route('/resultado')
def resultado():
    """Pantalla para el publico: tomarse la foto y elegir el tono, ver como va
    a quedar, el avance del dibujo y, al terminar, la imagen con el QR para
    descargarla (ver share.py)."""
    resp = app.send_static_file('resultado.html')
    resp.headers['Cache-Control'] = 'no-store'   # el celular siempre carga la version actual
    return resp


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5050))
    app.run(host='0.0.0.0', port=port, debug=True, threaded=True, use_reloader=False)
