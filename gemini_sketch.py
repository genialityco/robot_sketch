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

# Estilo "trazo simple": retrato de linea continua (tipo "one line art"), con
# pocas lineas largas y fluidas y rasgos minimos. Es mucho mas facil de
# dibujar para el brazo: menos bajadas de pluma, menos trazos cortos, curvas
# suaves. Se convierte a gcode pasando por SVG (ver lib/sketch_svg.py).
GEMINI_LINE_ART_PROMPT = (
    "convierte la foto de esta persona en un retrato de arte lineal minimalista, estilo dibujo de una sola "
    "linea continua: pocas lineas largas, fluidas y suaves, negras, finas y todas del mismo grosor, sobre fondo "
    "blanco. dibuja solo el contorno de la cara, la silueta del cabello con pocas lineas largas, el cuello y la "
    "linea de los hombros. ojos, cejas, nariz y boca con un trazo minimo y simple cada uno. sin sombras, sin "
    "relleno, sin texturas, sin sombreado ni rayado, sin trazos cortos, repetidos ni superpuestos, sin "
    "garabatos. el rostro centrado y de frente, sin accesorios como gafas"
)

# Estilo "dibujo a lapiz" (para la tableta, ver retrato_lapiz.py): un dibujo
# de lineas sin sombras, mas detallado que "trazo simple". De ahi sale el
# recorrido del brazo, y la tableta dibuja exactamente esas lineas.
GEMINI_PENCIL_PROMPT = (
    "convierte la foto de esta persona en un dibujo a lapiz de lineas, realista y detallado: el contorno "
    "de la cara, las orejas, los mechones del cabello, el cuello, los hombros y los pliegues principales de "
    "la ropa. los rasgos con mucho detalle: ojos con parpado superior e inferior, iris redondo con la pupila "
    "y un brillo, pestanas marcadas y cejas con la direccion del pelo; nariz con el puente, las aletas y las "
    "dos fosas nasales; boca con el contorno de ambos labios, el arco del labio superior, la linea entre los "
    "labios y las comisuras. solo lineas de grafito "
    "limpias y continuas, del mismo grosor, sin sombreado, sin rellenos, sin tramado ni degradados. fondo "
    "completamente blanco. solo cabeza y hombros, centrado y de frente. conserva fielmente los rasgos, la "
    "expresion, las gafas y el peinado de la persona. sin color, sin texto, sin marco ni firma"
)

# estilo -> (nombre para la interfaz, prompt)
STYLES = {
    'detallado': ('Detallado (boceto normal)', GEMINI_SKETCH_PROMPT),
    'trazo': ('Trazo simple (lineas largas, mas facil de dibujar)', GEMINI_LINE_ART_PROMPT),
    'lapiz': ('Dibujo a lapiz de lineas, detallado (para la tableta)', GEMINI_PENCIL_PROMPT),
}
DEFAULT_STYLE = 'detallado'


def generate_sketch(image_path, out_path, style=DEFAULT_STYLE):
    """
    Manda la imagen a Gemini con el prompt del estilo (ver STYLES) y guarda la primera
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
        contents=[STYLES.get(style, STYLES[DEFAULT_STYLE])[1], types.Part.from_bytes(data=image_bytes, mime_type=mime)],
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
