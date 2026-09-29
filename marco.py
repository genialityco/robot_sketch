#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Retrato con marco, del tamano justo para imprimir en papel 10x15 de la Canon
SELPHY CP1500 (postal: 100 x 148 mm) a 300 dpi: 1181 x 1748 px, con los 300 dpi
guardados en el JPG. Se imprime desde la app de Canon con la imagen que se
descarga (boton "Descargar imagen" / QR): asi entra exacta, sin recortes.

compose(): el retrato (la imagen que genera share.render_portrait) dentro del
hueco del marco. El hueco se detecta solo: la franja vacia mas alta entre lo
que el marco tiene arriba (logo) y abajo (lema), dentro de su borde. El retrato
se funde con el marco (multiplicar: sus lineas oscurecen el marco y su blanco
toma el color de fondo del marco), sin rectangulo blanco.

Los marcos se guardan en marcos/ (el elegido va en drawing_settings.json).
"""

import os

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FRAMES_DIR = os.path.join(HERE, 'marcos')
PAGE_MM = (100.0, 148.0)          # papel postal de la CP1500 (10x15)
DPI = 300
PAGE_PX = (int(round(PAGE_MM[0] / 25.4 * DPI)), int(round(PAGE_MM[1] / 25.4 * DPI)))   # 1181 x 1748
PORTRAIT_PAD = 0.04               # aire del retrato dentro del hueco (fraccion del hueco)
BG_TOL = 14                       # diferencia de color que ya cuenta como "algo dibujado" en el marco


def frame_path(name):
    return os.path.join(FRAMES_DIR, name) if name else None


def _load_frame_page(path):
    """El marco llevado a la pagina (PAGE_PX): entero (sin recortar su diseno),
    centrado, y lo que sobre relleno con su color de fondo."""
    frame = cv2.imread(path, cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError('no se pudo leer el marco %s' % os.path.basename(path))
    bg = np.median(np.concatenate([frame[:4].reshape(-1, 3), frame[-4:].reshape(-1, 3),
                                   frame[:, :4].reshape(-1, 3), frame[:, -4:].reshape(-1, 3)]), axis=0)
    pw, ph = PAGE_PX
    k = min(pw / frame.shape[1], ph / frame.shape[0])
    fw, fh = int(round(frame.shape[1] * k)), int(round(frame.shape[0] * k))
    frame = cv2.resize(frame, (fw, fh), interpolation=cv2.INTER_CUBIC if k > 1 else cv2.INTER_AREA)
    page = np.empty((ph, pw, 3), np.uint8)
    page[:] = bg.astype(np.uint8)
    x0, y0 = (pw - fw) // 2, (ph - fh) // 2
    page[y0:y0 + fh, x0:x0 + fw] = frame
    return page, bg


def free_box(page, bg):
    """Hueco del marco (x0, y0, x1, y1) en pixeles de la pagina: la franja
    vacia mas alta en el centro, entre el borde de los costados."""
    h, w = page.shape[:2]
    ink = (np.abs(page.astype(np.int16) - bg.astype(np.int16)).max(axis=2) > BG_TOL)
    # los costados: el borde del marco (la primera columna con algo, desde afuera hacia el centro)
    cols = ink[int(h * 0.3):int(h * 0.7)].mean(axis=0) > 0.3
    left = next((x for x in range(w // 2) if cols[x]), 0)
    right = next((x for x in range(w - 1, w // 2, -1) if cols[x]), w - 1)
    inner_l, inner_r = left + int(0.02 * w), right - int(0.02 * w)
    # filas vacias en la franja interior: la tira mas larga
    empty = ~ink[:, inner_l:inner_r].any(axis=1)
    best, start = (0, 0, 0), None
    for y in range(h + 1):
        if y < h and empty[y]:
            start = y if start is None else start
        elif start is not None:
            if y - start > best[0]:
                best = (y - start, start, y)
            start = None
    _, top, bottom = best
    if bottom - top < 0.3 * h:   # marco sin hueco claro: la pagina entera con margen
        return int(0.08 * w), int(0.08 * h), int(0.92 * w), int(0.92 * h)
    return inner_l, top, inner_r, bottom


def compose(portrait_path, out_path, frame_file):
    """Retrato (PNG) dentro del marco -> out_path: JPG de PAGE_PX con DPI
    guardado (10x15 a 300 dpi). Devuelve el hueco usado (x0, y0, x1, y1)."""
    from PIL import Image
    page, bg = _load_frame_page(frame_file)
    x0, y0, x1, y1 = free_box(page, bg)
    portrait = cv2.imread(portrait_path, cv2.IMREAD_COLOR)
    if portrait is None:
        raise RuntimeError('no se pudo leer el retrato')
    bw, bh = x1 - x0, y1 - y0
    pad = PORTRAIT_PAD * min(bw, bh)
    k = min((bw - 2 * pad) / portrait.shape[1], (bh - 2 * pad) / portrait.shape[0])
    pw, ph = max(1, int(portrait.shape[1] * k)), max(1, int(portrait.shape[0] * k))
    portrait = cv2.resize(portrait, (pw, ph), interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_CUBIC)
    px, py = x0 + (bw - pw) // 2, y0 + (bh - ph) // 2
    region = page[py:py + ph, px:px + pw].astype(np.float32)
    page[py:py + ph, px:px + pw] = (region * portrait.astype(np.float32) / 255.0).astype(np.uint8)   # multiplicar
    Image.fromarray(cv2.cvtColor(page, cv2.COLOR_BGR2RGB)).save(out_path, 'JPEG', quality=95, dpi=(DPI, DPI))
    return x0, y0, x1, y1
