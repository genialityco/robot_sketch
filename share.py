#!/usr/bin/python
# -*- coding:utf-8 -*-
# Retrato terminado -> PNG -> Firebase Storage -> enlace de descarga (QR).
#
# Lo usa la pantalla de la tableta (/tablet): cuando termina un dibujo se genera
# la imagen del trazo planeado (lo mismo que mostro la tableta, derecho y en
# alta resolucion), se sube a Storage con un token de descarga (enlace que no
# vence y no depende de las reglas del bucket) y la tableta muestra un QR con
# ese enlace para que la persona se lleve su retrato.
#
# Credenciales: la cuenta de servicio de Firebase (JSON) en la carpeta del
# proyecto, o la ruta en FIREBASE_CREDENTIALS. Esta en .gitignore: nunca al
# repositorio. Bucket: FIREBASE_STORAGE_BUCKET o <proyecto>.firebasestorage.app.

import glob
import json
import os
import threading
import time
import uuid

HERE = os.path.dirname(os.path.abspath(__file__))
STORAGE_FOLDER = 'retratos'          # carpeta dentro del bucket
FIRESTORE_COLLECTION = 'retratos'    # registro de cada retrato (si el proyecto tiene Firestore)
IMAGE_SIDE_PX = 1600                 # lado mayor del PNG
IMAGE_MARGIN = 0.06                  # margen blanco alrededor del dibujo (fraccion)

# Tonos del retrato (los elige el publico en /resultado): la tableta dibuja el
# grafito de ese color y la descarga sale igual. (id, nombre, color '#rrggbb')
PORTRAIT_COLORS = [
    ('grafito', 'Grafito', '#2d2d2d'),
    ('sepia', 'Sepia', '#6b4423'),
    ('sanguina', 'Sanguina', '#a23f25'),
    ('tinta', 'Tinta azul', '#1f3a7a'),
    ('bosque', 'Verde bosque', '#2f5a3a'),
]
DEFAULT_COLOR = 'grafito'


def color_hex(color_id):
    """Id de PORTRAIT_COLORS -> '#rrggbb' (el grafito si no existe)."""
    for cid, _name, hx in PORTRAIT_COLORS:
        if cid == color_id:
            return hx
    return PORTRAIT_COLORS[0][2]

_app_lock = threading.Lock()
_app = None


def _credentials_path():
    path = os.environ.get('FIREBASE_CREDENTIALS')
    if path:
        return path
    found = sorted(glob.glob(os.path.join(HERE, '*firebase-adminsdk*.json')))
    if not found:
        raise RuntimeError('no se encontro la cuenta de servicio de Firebase (*firebase-adminsdk*.json) '
                           'ni FIREBASE_CREDENTIALS')
    return found[0]


def _firebase():
    """App de firebase_admin, inicializada una sola vez."""
    global _app
    with _app_lock:
        if _app is None:
            import firebase_admin
            from firebase_admin import credentials
            path = _credentials_path()
            with open(path, encoding='utf-8') as fh:
                project = json.load(fh)['project_id']
            bucket = os.environ.get('FIREBASE_STORAGE_BUCKET') or '%s.firebasestorage.app' % project
            _app = firebase_admin.initialize_app(credentials.Certificate(path), {'storageBucket': bucket},
                                                 name='robot-sketch')
        return _app


def render_portrait(points, area_mm, rotation_deg, line_mm, out_path, side_px=IMAGE_SIDE_PX, pencil=False,
                    color=None, frame=None):
    """
    Trazo (puntos (u, v, pluma_abajo) sobre la hoja, como los publica
    draw_moves) -> PNG blanco con lineas negras. Se deshace el giro del perfil
    (rotation_deg, ver pipeline.rotate_moves) para que el retrato quede derecho.
    pencil=True: lineas de grafito finas y suavizadas, como las dibuja la
    tableta, del tono `color` (id de PORTRAIT_COLORS; None = grafito).
    frame: (u0, u1, v0, v1) el marco del retrato sobre la hoja (ver
    app._frame_uv): la imagen es exactamente ese marco (p. ej. 3:4 vertical),
    sin margen extra; sin frame, toda el area con un margen.
    """
    margin_frac = IMAGE_MARGIN
    if frame:
        u0, u1, v0, v1 = frame
        aw, ah = area_mm or (100.0, 100.0)
        points = [((p[0] - u0) / (u1 - u0), (p[1] - v0) / (v1 - v0)) + tuple(p[2:]) for p in points]
        area_mm = ((u1 - u0) * aw, (v1 - v0) * ah)
        margin_frac = 0.0   # el marco ya trae su aire
    if pencil:
        return _render_pencil(points, area_mm, rotation_deg, line_mm, out_path, side_px, color, margin_frac)
    from PIL import Image, ImageDraw
    aw, ah = area_mm or (100.0, 100.0)
    scale = side_px / max(aw, ah)                       # px por mm
    margin = margin_frac * side_px
    w, h = int(aw * scale + 2 * margin), int(ah * scale + 2 * margin)
    img = Image.new('L', (w, h), 255)
    d = ImageDraw.Draw(img)
    lw = max(2, int(round(line_mm * scale)))

    def px(p):
        return (margin + p[0] * aw * scale, margin + (1 - p[1]) * ah * scale)

    for i, p in enumerate(points):
        if not p[2]:
            continue
        prev = points[i - 1] if i else None
        nxt = points[i + 1] if i + 1 < len(points) else None
        if prev is not None and prev[2]:
            d.line([px(prev), px(p)], fill=0, width=lw)
            x, y = px(p)
            d.ellipse([x - lw / 2, y - lw / 2, x + lw / 2, y + lw / 2], fill=0)   # uniones redondas
        elif nxt is None or not nxt[2]:   # punto suelto
            x, y = px(p)
            d.ellipse([x - lw / 2, y - lw / 2, x + lw / 2, y + lw / 2], fill=0)
    if rotation_deg:
        img = img.rotate(-int(rotation_deg), expand=True, fillcolor=255)   # deshace el giro antihorario
    img.save(out_path, optimize=True)
    return out_path


def _render_pencil(points, area_mm, rotation_deg, line_mm, out_path, side_px, color=None, margin_frac=IMAGE_MARGIN):
    """Como la tableta: trazos del tono elegido, suavizados (se dibuja al doble y se reduce)."""
    import cv2
    import numpy as np
    ss = 2
    aw, ah = area_mm or (100.0, 100.0)
    scale = side_px * ss / max(aw, ah)                 # px por mm, al doble
    margin = margin_frac * side_px * ss
    w, h = int(aw * scale + 2 * margin), int(ah * scale + 2 * margin)
    ink = np.zeros((h, w), np.float32)
    lw = max(1, int(round(line_mm * scale)))

    def px(p):
        return (int(round(margin + p[0] * aw * scale)), int(round(margin + (1 - p[1]) * ah * scale)))

    for i in range(1, len(points)):
        a, b = points[i - 1], points[i]
        if a[2] and b[2]:   # cuarto valor: grosor relativo (los rasgos, mas finos)
            cv2.line(ink, px(a), px(b), 1.0, max(1, int(round(lw * (b[3] if len(b) > 3 else 1)))), cv2.LINE_AA)
        elif b[2] and (i + 1 >= len(points) or not points[i + 1][2]):   # punto suelto
            cv2.circle(ink, px(b), max(1, lw // 2), 1.0, -1, cv2.LINE_AA)
    hx = color_hex(color or DEFAULT_COLOR)
    bgr = np.array([int(hx[5:7], 16), int(hx[3:5], 16), int(hx[1:3], 16)], np.float32)
    ink = np.clip(ink, 0, 1)[:, :, None]
    img = 255 - ink * (255 - bgr)                                      # papel blanco, trazo del tono elegido
    img = cv2.resize(img, (w // ss, h // ss), interpolation=cv2.INTER_AREA).astype(np.uint8)
    turns = (int(rotation_deg) // 90) % 4
    if turns:   # deshace el giro antihorario del perfil
        img = np.ascontiguousarray(np.rot90(img, -turns))
    cv2.imwrite(out_path, img)
    return out_path


def upload(path, title=None):
    """Sube el PNG a Storage. Devuelve (id, url de descarga)."""
    from firebase_admin import storage
    app = _firebase()
    portrait_id = time.strftime('%Y%m%d-%H%M%S-') + uuid.uuid4().hex[:6]
    blob = storage.bucket(app=app).blob('%s/%s.png' % (STORAGE_FOLDER, portrait_id))
    token = uuid.uuid4().hex
    blob.metadata = {'firebaseStorageDownloadTokens': token}
    blob.content_disposition = 'inline; filename="retrato-%s.png"' % portrait_id
    blob.cache_control = 'public, max-age=31536000'
    blob.upload_from_filename(path, content_type='image/png')
    from urllib.parse import quote
    url = 'https://firebasestorage.googleapis.com/v0/b/%s/o/%s?alt=media&token=%s' % (
        blob.bucket.name, quote(blob.name, safe=''), token)
    _record(app, portrait_id, url, title)
    return portrait_id, url


def _record(app, portrait_id, url, title):
    """Registro en Firestore (para una galeria o estadisticas). Opcional: si el
    proyecto no tiene Firestore, el retrato igual queda en Storage."""
    try:
        from firebase_admin import firestore
        firestore.client(app=app).collection(FIRESTORE_COLLECTION).document(portrait_id).set(
            {'url': url, 'title': title, 'created_at': firestore.SERVER_TIMESTAMP})
    except Exception as exc:
        print('share: sin registro en Firestore (%s): el retrato igual quedo en Storage' % str(exc)[:160], flush=True)


def qr_svg(url):
    """SVG del codigo QR del enlace (lo muestra la tableta; no necesita internet)."""
    import io
    import segno
    buf = io.BytesIO()
    segno.make(url, error='m').save(buf, kind='svg', scale=10, border=2, dark='#111')
    return buf.getvalue()
