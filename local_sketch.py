#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Boceto local (sin internet): foto de una persona -> retrato de lineas, con
U²-Net Portrait (https://github.com/xuebinqin/U-2-Net, modelo u2net_portrait,
entrenado con el dataset APDrawing) exportado a ONNX (models/u2net_portrait.onnx)
y corrido con onnxruntime. ~1 s en CPU.

Igual que la demo oficial (u2net_portrait_demo.py): se recorta la cabeza, se
lleva a 512x512 y la red devuelve el dibujo (lineas oscuras sobre blanco). El
recorte de la demo sale de un detector Haar de OpenCV, que ya no viene con
OpenCV 5: aca el recuadro de la cara sale de MediaPipe (los mismos landmarks que
usa el resto del programa) con las mismas proporciones.

Si no hay modelo, onnxruntime o cara, lanza LocalSketchError: app.py pasa al
siguiente motor (Gemini, y despues el lapiz de OpenCV).
"""

import json
import os
import threading

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, 'models', 'u2net_portrait.onnx')
INPUT_SIDE = 512            # el modelo trabaja a 512x512
OUTPUT_SIDE = 1024          # el boceto se guarda mas grande (el vectorizador trabaja a ~800 px)
# Recorte (como la demo, en anchos de la cara): a los costados, arriba (pelo) y
# abajo (cuello) del recuadro de la cara.
CROP_SIDE, CROP_TOP, CROP_BOTTOM = 0.4, 0.6, 0.2
BACKGROUND_TO_WHITE = True  # fondo de la foto -> blanco antes de la red (la entrenaron con fondos lisos)
# Cara chica en la foto (camara web de baja resolucion, lejos): ampliada a 512
# queda borrosa y la red casi no encuentra bordes. Se le sube el contraste local
# (CLAHE) antes. Con caras grandes no hace falta (solo engrosaria las lineas).
# Niveles de la salida (0 = negro, 1 = blanco): lo mas claro que WHITE pasa a
# blanco (la textura gris de pelo, barba y piel, que vectorizada queda como
# una red) y lo mas oscuro que BLACK a negro (las lineas firmes).
LINE_LEVELS = (0.35, 0.8)   # (BLACK, WHITE); None = la salida tal cual
SMALL_FACE_PX = 320
SMALL_FACE_CLAHE = 2.0


class LocalSketchError(RuntimeError):
    pass


_session = {'s': None}
_lock = threading.Lock()


def available():
    return os.path.exists(MODEL_PATH)


def _get_session():
    with _lock:
        if _session['s'] is None:
            if not available():
                raise LocalSketchError('falta el modelo %s' % os.path.relpath(MODEL_PATH, HERE))
            try:
                import onnxruntime as ort
            except ImportError:
                raise LocalSketchError('falta onnxruntime (pip install onnxruntime)')
            _session['s'] = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
        return _session['s']


def _face_box(img):
    """Recuadro (x, y, w, h) de la cara como el del detector Haar de la demo:
    cuadrado, de mejilla a mejilla, desde la mitad de la frente hasta cerca del
    menton. None si no hay cara."""
    import pipeline as pl
    pts = pl.dp.face_landmark_points(img)
    if pts is None or len(pts) < 468:
        return None, None
    pts = np.asarray(pts, float)
    x0, x1 = pts[234][0], pts[454][0]      # mejillas (borde de la cara)
    w = abs(x1 - x0)
    cx = (x0 + x1) / 2
    chin = pts[152][1]
    brow = min(pts[105][1], pts[334][1])   # cejas
    # Haar: el cuadrado arranca un poco arriba de las cejas y termina cerca del menton
    top = brow - 0.25 * w
    h = max(w, chin - top)
    return (int(cx - w / 2), int(top), int(w), int(h)), pts


def _white_background(img, pts):
    """Fondo -> blanco: fuera de la persona (segmentacion de MediaPipe si esta
    el modelo; si no, se deja la foto como esta)."""
    try:
        import mediapipe as mp
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.core.base_options import BaseOptions
    except ImportError:
        return img
    model = _selfie_model()
    if model is None:
        return img
    seg = vision.ImageSegmenter.create_from_options(vision.ImageSegmenterOptions(
        base_options=BaseOptions(model_asset_path=model), output_confidence_masks=True))
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = seg.segment(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)))
    seg.close()
    if not res.confidence_masks:
        return img
    person = res.confidence_masks[-1].numpy_view().astype(np.float32)   # la ultima clase: la persona
    if person.shape[:2] != img.shape[:2]:
        person = cv2.resize(person, (img.shape[1], img.shape[0]))
    person = cv2.GaussianBlur(np.clip(person, 0, 1), (0, 0), 2)[:, :, None]
    return (img * person + 255 * (1 - person)).astype(np.uint8)


SELFIE_MODEL_URL = ('https://storage.googleapis.com/mediapipe-models/image_segmenter/'
                    'selfie_segmenter/float16/latest/selfie_segmenter.tflite')
SELFIE_MODEL_PATH = os.path.join(HERE, 'models', 'selfie_segmenter.tflite')


def _selfie_model():
    if not os.path.exists(SELFIE_MODEL_PATH):
        try:
            import urllib.request
            os.makedirs(os.path.dirname(SELFIE_MODEL_PATH), exist_ok=True)
            urllib.request.urlretrieve(SELFIE_MODEL_URL, SELFIE_MODEL_PATH)
        except Exception:
            return None   # sin internet la primera vez: sin quitar el fondo
    return SELFIE_MODEL_PATH


def _crop_face(img, face):
    """Como crop_face de la demo: recorte con margenes, relleno blanco hasta
    cuadrado y 512x512. Sin cara: la imagen entera, cuadrada. Devuelve
    (recorte, (ox, oy, lado)): un punto (x, y) de la foto queda en
    ((x - ox) / lado, (y - oy) / lado) del recorte, en fracciones."""
    ox = oy = 0
    if face is not None:
        x, y, w, h = face
        H, W = img.shape[:2]
        left, right = x - int(w * CROP_SIDE), x + w + int(w * CROP_SIDE)
        top, bottom = y - int(h * CROP_TOP), y + h + int(h * CROP_BOTTOM)
        pad = ((max(0, -top), max(0, bottom - H)), (max(0, -left), max(0, right - W)), (0, 0))
        img = np.pad(img[max(0, top):min(H, bottom), max(0, left):min(W, right)], pad,
                     mode='constant', constant_values=255)
        ox, oy = left, top
    hf, wf = img.shape[:2]
    if hf > wf:
        p = (hf - wf) // 2
        img = np.pad(img, ((0, 0), (p, hf - wf - p), (0, 0)), mode='constant', constant_values=255)
        ox -= p
    elif wf > hf:
        p = (wf - hf) // 2
        img = np.pad(img, ((p, wf - hf - p), (0, 0), (0, 0)), mode='constant', constant_values=255)
        oy -= p
    return cv2.resize(img, (INPUT_SIDE, INPUT_SIDE), interpolation=cv2.INTER_AREA), (ox, oy, img.shape[0])


def _inference(face_bgr):
    """Como inference() de la demo: normalizacion (con su mismo orden de
    canales), la salida principal invertida y llevada a 0..1 (1 = blanco)."""
    x = face_bgr.astype(np.float32) / max(1.0, float(face_bgr.max()))
    t = np.empty_like(x)
    t[:, :, 0] = (x[:, :, 2] - 0.406) / 0.225
    t[:, :, 1] = (x[:, :, 1] - 0.456) / 0.224
    t[:, :, 2] = (x[:, :, 0] - 0.485) / 0.229
    t = t.transpose(2, 0, 1)[None].astype(np.float32)
    d1 = _get_session().run(None, {'input': t})[0][0, 0]
    pred = 1.0 - d1
    return (pred - pred.min()) / max(1e-6, pred.max() - pred.min())


def generate(photo_path, out_path):
    """Foto -> boceto de lineas (PNG, lineas oscuras sobre blanco). Devuelve
    un texto corto de lo que se hizo (para mostrar)."""
    img = cv2.imread(photo_path, cv2.IMREAD_COLOR)
    if img is None:
        raise LocalSketchError('no se pudo leer la foto')
    # a un tamano razonable (fotos de celular de 12 MP: MediaPipe y el recorte no necesitan tanto)
    k = 1600.0 / max(img.shape[:2])
    if k < 1:
        img = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    face, pts = _face_box(img)
    if face is not None and face[2] < SMALL_FACE_PX:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = cv2.createCLAHE(clipLimit=SMALL_FACE_CLAHE, tileGridSize=(8, 8)).apply(lab[:, :, 0])
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    if BACKGROUND_TO_WHITE:
        img = _white_background(img, pts)
    crop, (ox, oy, side) = _crop_face(img, face)
    pred = _inference(crop)
    if LINE_LEVELS:
        lo, hi = LINE_LEVELS
        pred = np.clip((pred - lo) / (hi - lo), 0, 1)
    out = cv2.resize((pred * 255).astype(np.uint8), (OUTPUT_SIDE, OUTPUT_SIDE), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(out_path, cv2.cvtColor(out, cv2.COLOR_GRAY2BGR))
    # los landmarks de la cara (de la foto) en coordenadas del boceto: sobre este
    # dibujo MediaPipe no siempre encuentra la cara (ver pipeline.locate_face)
    landmarks_path = out_path + '.landmarks.json'
    if pts is not None:
        k = OUTPUT_SIDE / float(side)
        with open(landmarks_path, 'w', encoding='utf-8') as fh:
            json.dump(dict(size=[OUTPUT_SIDE, OUTPUT_SIDE],
                           points=[[round((x - ox) * k, 2), round((y - oy) * k, 2)] for x, y in pts]), fh)
    elif os.path.exists(landmarks_path):
        os.remove(landmarks_path)
    return 'boceto local (U2-Net Portrait)' + ('' if face is not None else ', sin cara detectada: foto entera')
