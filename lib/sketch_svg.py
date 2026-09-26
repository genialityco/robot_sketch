#!/usr/bin/python
# -*- coding:utf-8 -*-
# Boceto de linea (imagen) -> SVG de trazos -> polilineas para el brazo.
#
# Pensado para el estilo "trazo simple" (ver gemini_sketch.STYLES): un retrato
# de pocas lineas largas y fluidas. El camino de siempre (esqueleto -> un trazo
# por cada tramo entre uniones -> Douglas-Peucker) corta cada linea en todos
# los cruces y deja esquinas: muchas bajadas de pluma y trazos cortos, que es
# justo lo que peor dibuja el brazo. Aca:
#   1. se limpia la mascara (se cierran huecos chicos, se tiran manchitas),
#   2. del esqueleto se recortan las ramitas cortas que deja el adelgazado,
#   3. en cada union se empalman los tramos que siguen mas derechos, asi cada
#      linea del dibujo sale como UN trazo largo,
#   4. se suaviza y se escribe como curvas Bezier en un SVG (se puede abrir en
#      el navegador o en Inkscape para ver exactamente que se va a dibujar),
#   5. el gcode sale de leer ese SVG y muestrearlo (svg_to_polylines).

import math
import re

import cv2
import numpy as np

import draw_person as dp

# Ramitas del esqueleto (de una punta suelta a una union) mas cortas que esto
# se consideran ruido del adelgazado, no parte del dibujo (fraccion del lado
# mayor de la imagen: ~6 px a 800).
SPUR_FRACTION = 0.008
# En una union se empalman dos tramos si el segundo sigue al primero girando
# menos que esto (grados). Mas: se cortaria una esquina real del dibujo.
JOIN_MAX_TURN_DEG = 65
# Huecos de la tinta de hasta este tamano se cierran antes de adelgazar
# (fraccion del lado mayor, ~3 px a 800): una linea de Gemini con un corte
# minimo no debe convertirse en dos trazos.
CLOSE_FRACTION = 0.004
# Manchas de tinta sueltas con menos area que esto se descartan (fraccion del
# area de la imagen).
MIN_BLOB_FRACTION = 0.00004


def _clean_mask(mask):
    h, w = mask.shape[:2]
    k = max(1, int(round(CLOSE_FRACTION * max(h, w))))
    ink = cv2.morphologyEx((mask > 0).astype(np.uint8), cv2.MORPH_CLOSE,
                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1)))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= MIN_BLOB_FRACTION * h * w
    return keep[labels]


def _length(path):
    p = np.asarray(path, np.float64)
    return float(np.sum(np.hypot(*np.diff(p, axis=0).T))) if len(p) > 1 else 0.0


def _direction(path, from_start, reach):
    """Hacia donde sale el tramo desde una punta (vector unitario), mirando
    `reach` pixeles hacia adentro para no depender del escalonado."""
    p = np.asarray(path if from_start else path[::-1], np.float64)
    d = np.hypot(*np.diff(p, axis=0).T)
    k = int(np.searchsorted(np.cumsum(d), reach)) + 1
    v = p[min(k, len(p) - 1)] - p[0]
    n = np.hypot(*v)
    return v / n if n else v


def skeleton_paths(mask):
    """Mascara de tinta -> polilineas largas en pixeles (x, y)."""
    ink = _clean_mask(mask)
    h, w = ink.shape[:2]
    edges = [list(e) for e in dp.trace_skeleton(dp.thin(ink)) if len(e) >= 2]

    # nodos = puntas de los tramos; pixeles de union pegados se toman como un solo nodo
    parent = {}

    def find(a):
        while parent.setdefault(a, a) != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    ends = {p for e in edges for p in (e[0], e[-1])}
    for p in ends:
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                q = (p[0] + dx, p[1] + dy)
                if q in ends:
                    parent[find(p)] = find(q)

    def incident():
        inc = {}
        for i, e in enumerate(edges):
            if e is None:
                continue
            inc.setdefault(find(e[0]), []).append((i, True))
            inc.setdefault(find(e[-1]), []).append((i, False))
        return inc

    # recortar ramitas: tramo corto con una punta suelta y la otra en una union
    spur = SPUR_FRACTION * max(h, w)
    for _ in range(3):
        inc = incident()
        removed = False
        for i, e in enumerate(edges):
            if e is None or _length(e) >= spur:
                continue
            da, db = len(inc[find(e[0])]), len(inc[find(e[-1])])
            if (da == 1 and db >= 3) or (db == 1 and da >= 3):
                edges[i] = None
                removed = True
        if not removed:
            break

    # empalmes: en cada union, los pares de tramos que siguen mas derechos
    inc = incident()
    reach = max(3.0, spur)
    link = {}   # (tramo, punta) -> (tramo, punta) con el que sigue
    cos_max = math.cos(math.radians(180 - JOIN_MAX_TURN_DEG))
    for node, items in inc.items():
        if len(items) < 2:
            continue
        dirs = {it: _direction(edges[it[0]], it[1], reach) for it in items}
        pairs = []
        for a in range(len(items)):
            for b in range(a + 1, len(items)):
                ia, ib = items[a], items[b]
                if ia[0] == ib[0]:
                    continue   # un lazo que vuelve al mismo nodo
                c = float(np.dot(dirs[ia], dirs[ib]))   # -1 = siguen derecho
                if c <= cos_max or len(items) == 2:
                    pairs.append((c, ia, ib))
        for _c, ia, ib in sorted(pairs):
            if ia not in link and ib not in link:
                link[ia] = ib
                link[ib] = ia

    # recorrer cadenas de tramos empalmados
    used = set()
    paths = []

    def chain_from(i, at_start):
        pts = []
        cur, start = i, at_start
        while cur not in used:
            used.add(cur)
            e = edges[cur]
            seg = e if start else e[::-1]
            pts.extend(seg if not pts else seg[1:])
            nxt = link.get((cur, not start))   # la otra punta del tramo actual
            if nxt is None:
                break
            cur, start = nxt[0], nxt[1]
        return pts

    for i, e in enumerate(edges):
        if e is None or i in used:
            continue
        # arrancar desde una punta sin empalme (si la hay), para no cortar la cadena al medio
        start = i, True
        seen = set()
        while True:
            back = link.get(start)
            if back is None or back[0] in seen:
                break
            seen.add(start[0])
            start = back[0], not back[1]
        paths.append(chain_from(*start))
    return [np.array(p, np.float64) for p in paths if len(p) >= 2]


def smooth_path(path, sigma_px, eps_px):
    """Suaviza (gaussiano a lo largo del trazo, las puntas quedan fijas) y
    simplifica. Un trazo que cierra sobre si mismo se suaviza como lazo."""
    p = np.asarray(path, np.float64)
    if len(p) < 3 or sigma_px <= 0:
        return p
    closed = len(p) > 8 and np.hypot(*(p[0] - p[-1])) <= 2.0
    r = int(math.ceil(3 * sigma_px))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma_px) ** 2)
    k /= k.sum()
    if closed:
        q = np.vstack([p[-r - 1:-1], p, p[1:r + 1]])
        s = np.column_stack([np.convolve(q[:, i], k, 'same') for i in (0, 1)])[r:r + len(p)]
        s[-1] = s[0]
    else:
        q = np.vstack([np.repeat(p[:1], r, 0), p, np.repeat(p[-1:], r, 0)])
        s = np.column_stack([np.convolve(q[:, i], k, 'same') for i in (0, 1)])[r:r + len(p)]
        s[0], s[-1] = p[0], p[-1]
    out = cv2.approxPolyDP(s.astype(np.float32).reshape(-1, 1, 2), eps_px, closed).reshape(-1, 2)
    if closed:
        out = np.vstack([out, out[:1]])
    return out.astype(np.float64)


def _bezier_segments(p):
    """Puntos -> curvas Bezier cubicas que pasan por todos (Catmull-Rom)."""
    n = len(p)
    segs = []
    for i in range(n - 1):
        p0 = p[i - 1] if i > 0 else p[i]
        p3 = p[i + 2] if i + 2 < n else p[i + 1]
        c1 = p[i] + (p[i + 1] - p0) / 6.0
        c2 = p[i + 1] - (p3 - p[i]) / 6.0
        segs.append((c1, c2, p[i + 1]))
    return segs


def write_svg(paths, width, height, out_path):
    """Polilineas (pixeles) -> SVG con un <path> de curvas Bezier por trazo."""
    body = []
    for p in paths:
        d = ['M %.2f %.2f' % tuple(p[0])]
        for c1, c2, end in _bezier_segments(p):
            d.append('C %.2f %.2f %.2f %.2f %.2f %.2f' % (*c1, *c2, *end))
        body.append('  <path d="%s"/>' % ' '.join(d))
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %d %d" width="%d" height="%d">\n'
           '  <rect width="100%%" height="100%%" fill="white"/>\n'
           '  <g fill="none" stroke="black" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">\n'
           '%s\n  </g>\n</svg>\n' % (width, height, width, height, '\n'.join(body)))
    with open(out_path, 'w', encoding='utf-8') as fh:
        fh.write(svg)
    return out_path


_TOKEN = re.compile(r'[MLCZmlcz]|-?\d*\.?\d+(?:[eE][-+]?\d+)?')


def svg_to_polylines(svg_path, step_px=1.0):
    """
    Lee los <path> de un SVG (comandos absolutos M, L, C, Z, que es lo que
    escribe write_svg) y los muestrea como polilineas en las unidades del SVG,
    con un punto cada ~step_px. Devuelve (polilineas, (ancho, alto)).
    """
    with open(svg_path, encoding='utf-8') as fh:
        text = fh.read()
    vb = re.search(r'viewBox="([^"]+)"', text)
    size = tuple(float(v) for v in vb.group(1).split()[2:4]) if vb else (0.0, 0.0)
    polylines = []
    for d in re.findall(r'<path[^>]*\sd="([^"]+)"', text):
        tokens = _TOKEN.findall(d)
        i, cmd, cur, start, pts = 0, None, None, None, []

        def flush():
            if len(pts) >= 2:
                polylines.append(np.array(pts, np.float64))

        while i < len(tokens):
            if tokens[i].isalpha():
                cmd = tokens[i].upper()
                i += 1
                if cmd == 'Z':
                    if start is not None and cur is not None:
                        pts.append(start)
                        cur = start
                    continue
            vals = tokens[i:]
            if cmd == 'M':
                flush()
                cur = start = (float(vals[0]), float(vals[1]))
                pts = [cur]
                i += 2
                cmd = 'L'   # pares siguientes a un M son lineas (norma SVG)
            elif cmd == 'L':
                cur = (float(vals[0]), float(vals[1]))
                pts.append(cur)
                i += 2
            elif cmd == 'C':
                c1 = np.array(vals[0:2], float)
                c2 = np.array(vals[2:4], float)
                end = np.array(vals[4:6], float)
                p0 = np.array(cur, float)
                approx = np.hypot(*(c1 - p0)) + np.hypot(*(c2 - c1)) + np.hypot(*(end - c2))
                n = max(1, int(math.ceil(approx / step_px)))
                for k in range(1, n + 1):
                    t = k / n
                    b = ((1 - t) ** 3) * p0 + 3 * ((1 - t) ** 2) * t * c1 + 3 * (1 - t) * t * t * c2 + (t ** 3) * end
                    pts.append((float(b[0]), float(b[1])))
                cur = (float(end[0]), float(end[1]))
                i += 6
            else:
                raise ValueError('comando SVG no soportado: %s' % cmd)
        flush()
    return polylines, size


def resample(poly, step):
    """Puntos equiespaciados cada `step` a lo largo de la polilinea (las dos
    puntas se conservan)."""
    p = np.asarray(poly, np.float64)
    d = np.hypot(*np.diff(p, axis=0).T)
    total = float(d.sum())
    if total <= step:
        return p[[0, -1]]
    n = int(math.ceil(total / step))
    s = np.concatenate([[0], np.cumsum(d)])
    t = np.linspace(0, total, n + 1)
    return np.column_stack([np.interp(t, s, p[:, 0]), np.interp(t, s, p[:, 1])])
