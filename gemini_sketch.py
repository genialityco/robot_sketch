#!/usr/bin/python
# -*- coding:utf-8 -*-
# Convierte una foto en un boceto de linea usando Gemini. Mismo patron que
# C:\Users\prosopon\Documents\myCobot280\firebase_listener.py
# (generate_sketch_with_gemini), sin la parte de Firebase.
#
# ADVERTENCIA conocida (ver el docstring de firebase_listener.py): el modelo
# gemini-3.1-flash-lite-image requiere billing habilitado en el proyecto de
# Google AI; con la cuota gratis da 429 RESOURCE_EXHAUSTED. Verificado
# 2026-09-22: el nombre del modelo existe y el request se acepta, pero la
# llamada real fallo por cuota. Si esto vuelve a pasar, la excepcion sube tal
# cual hasta la interfaz (ver app.py), que la muestra como error de la etapa
# "boceto" con opcion de reintentar o subir un boceto ya hecho a mano.

import os

GEMINI_IMAGE_MODEL = os.environ.get('GEMINI_IMAGE_MODEL', 'gemini-3.1-flash-lite-image')
GEMINI_SKETCH_PROMPT = (
    "de la imagen de esta persona crea un boceto de su rostro con pocos trazos, y con trazos definidos trazos negros y fondo blanco, sin relleno, lineas con la misma intensidad, ojos, nariz, y boca bien definidos que el rostro este centrado, sin accesorios como gafas"
)


def generate_sketch(image_path, out_path):
    """
    Manda la imagen a Gemini con GEMINI_SKETCH_PROMPT y guarda la primera
    imagen que devuelva en out_path. Lanza RuntimeError con un mensaje claro
    si falta la API key o si Gemini no devuelve ninguna imagen (p.ej. por
    cuota agotada, ver advertencia arriba).
    """
    api_key = os.environ.get('GEMINI_API_KEY')
    if not api_key:
        raise RuntimeError('Falta GEMINI_API_KEY (copia .env.example a .env y completala)')

    from google import genai
    from google.genai import types

    with open(image_path, 'rb') as fh:
        image_bytes = fh.read()
    ext = os.path.splitext(image_path)[1].lower()
    mime = 'image/png' if ext == '.png' else 'image/jpeg'

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=GEMINI_IMAGE_MODEL,
        contents=[GEMINI_SKETCH_PROMPT, types.Part.from_bytes(data=image_bytes, mime_type=mime)],
    )

    candidate = (response.candidates or [None])[0]
    parts = candidate.content.parts if candidate and candidate.content else []
    for part in parts:
        if part.inline_data is not None:
            with open(out_path, 'wb') as fh:
                fh.write(part.inline_data.data)
            return out_path

    text = ' '.join(p.text for p in parts if p.text) or '(sin texto)'
    raise RuntimeError('Gemini no devolvio una imagen (modelo %s). Respuesta: %s'
                        % (GEMINI_IMAGE_MODEL, text[:300]))
