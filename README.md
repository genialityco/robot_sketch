# Foto → boceto → robot (interfaz web)

App web local (Flask) para el flujo: subir una foto de una persona → generar
un boceto con Gemini → revisar y aceptar/rechazar → convertir el boceto a
gcode y previsualizar el trazo → revisar y aceptar/rechazar → dibujar con el
myCobot 280 real.

## Módulos en `lib/`

El procesamiento de imagen y la calibración/ejecución del robot vienen de
dos proyectos anteriores, copiados en `lib/` para que el repositorio funcione
solo al clonarlo:

- **`lib/draw_person.py`** (originalmente de
  `C:\Users\prosopon\Documents\myCobot280\`): extracción de trazos desde una
  imagen (`sketch_to_mask`, `build_strokes`, `order_strokes`) y rasgos
  faciales con MediaPipe (`face_landmark_strokes`). También trae comandos de
  línea propios (quitar fondo con rembg, etc.) que esta app no usa.
- **`lib/gcode_draw_lib.py`** (originalmente de
  `pymycobot/demo/myCobot_280_demo/`): calibración (4 esquinas regularizadas
  a rectángulo, `MANUAL_SKEW_MM`), pacing, sincronización con el brazo,
  `MOVE_MODE`, `PEN_LIFT_MM`, patrón de prueba. Es el resultado de depurar
  extensamente el dibujo contra el robot físico real (fricción del lápiz en
  pasos chicos, esquinas no perpendiculares, comandos descartados, etc.).

Son copias: los cambios que se hagan acá no llegan a los proyectos originales
ni al revés. Para usar otra copia, `DRAW_PERSON_DIR` y `GCODE_LIB_DIR` (en
`.env`) apuntan a la carpeta de cada una.

`pipeline.py` es el único código nuevo de "negocio": convierte los trazos en
píxeles que devuelve `draw_person.py` a distancias en **milímetros reales**
según el área calibrada actual, y ahí aplica los umbrales de
simplificación/filtrado aprendidos en esta conversación para que el lápiz no
se trabe por fricción en pasos muy cortos (`SIMPLIFY_TARGET_MM`,
`MIN_STROKE_TARGET_MM`).

## Instalación

```
cd robot_sketch_app
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Editá `.env` y completá `GEMINI_API_KEY` (podés copiar el mismo valor que ya
está en `C:\Users\prosopon\Documents\myCobot280\.env`).

⚠️ **Advertencia conocida sobre Gemini**: el modelo `gemini-3.1-flash-lite-image`
requiere billing habilitado en el proyecto de Google AI. Con la cuota
gratuita da `429 RESOURCE_EXHAUSTED` (verificado 2026-09-22 en el proyecto
original). Si te pasa esto, la interfaz muestra el error en la etapa del
boceto y te deja subir un boceto hecho a mano como alternativa (sección
plegable "¿Gemini falló...?").

## Correr

```
python app.py
```

Abrí `http://localhost:5050`. El puerto serial del robot (COM5 por defecto)
se configura en `lib/gcode_draw_lib.py` (`PORT`); la calibración, en la
pestaña Calibración.

## Flujo

1. **Subir foto** → se manda a Gemini con un prompt fijo (ver
   `gemini_sketch.py`) que pide un boceto de línea limpio, preservando
   identidad facial. Opcionalmente, en el mismo paso se puede adjuntar un
   boceto ya hecho a mano (sección plegable "¿Ya tenés un boceto hecho?"):
   en ese caso se salta la llamada a Gemini por completo y el job pasa
   directo a la etapa de revisión del boceto.
2. **Revisar boceto** → Aceptar sigue al paso 3. Rechazar vuelve a pedirle a
   Gemini otro boceto (mismo prompt, nueva llamada). Opcionalmente se puede
   **recortar**: arrastrando un rectángulo sobre el boceto, solo esa zona se
   usa para el gcode (`sketch_crop.png`, llevado a 800 px de lado antes de
   extraer trazos), agrandada para ocupar la hoja. Sirve para dibujar solo la
   cara: los ojos y la boca salen mucho más grandes, y el error de posición
   del brazo (varios mm) pesa menos sobre ellos. El recorte se mantiene al
   cambiar el nivel de detalle y se descarta si el boceto se regenera o se
   reemplaza.
3. **Trazo generado** → se extraen los trazos del boceto, se simplifican a
   pasos friccion-seguros, y se muestra un preview (mismo estilo que
   `draw_person.py preview`: negro = dibujo, rojo punteado = viaje con la
   pluma arriba, más estadísticas de tiempo estimado).
4. **Aceptar y dibujar** → conecta con el robot real y empieza a dibujar
   (pide una confirmación extra en el navegador antes de mandar el comando,
   porque esto mueve hardware de verdad). Rechazar vuelve a la revisión del
   boceto.
5. Progreso en vivo mientras dibuja, y pantalla final de listo/error.

Solo se puede dibujar un trabajo a la vez (lock global en el backend); si
intentás aceptar un segundo mientras el robot está ocupado, la API devuelve
un error claro en vez de mandar comandos encimados.

## Calibración y perfiles (pestaña "Calibración")

Un **perfil** guarda las 4 esquinas de la hoja (`[x, y, z, rx, ry, rz]`, tal
como las devuelve `get_coords` con la pluma tocando el papel), el ajuste fino
de inclinación (`MANUAL_SKEW_MM`) y cuánto se levanta la pluma. Sirve para
tener varias mesas/hojas/soportes calibrados y cambiar entre ellos sin editar
`gcode_draw_lib.py`. El perfil activo es el que usan el preview, el dibujo y
las pruebas; se aplica en caliente con `gcode_draw_lib.set_calibration()`.

- Se guardan como JSON en `calibration_profiles/` (el activo, en
  `_activo.txt`). La primera vez que arranca la app, si no hay ninguno, crea
  "Calibracion original" con las esquinas que ya estaban escritas en
  `gcode_draw_lib.py`, así que no se pierde la calibración anterior.
- Para medir: **Liberar brazo** (sostenelo: sin torque cae), apoyar la punta
  en una esquina con el esfero vertical, **Capturar**; repetir en las 4 y
  **Trabar brazo**. "Arriba" es el lado de la hoja más lejos del robot.
- Para verificar: **Ir encima** (queda 10 mm sobre la esquina) o **Tocar**
  (baja en vertical hasta el papel), y después **Dibujar contorno** (el borde
  del área, 2 mm hacia adentro) o **Dibujar patrón de prueba** (ver
  `gcode_draw_lib.test_pattern_moves`).
- Cada esquina tiene un **ajuste de altura** en décimas de mm, aparte de la
  Z capturada: negativo sube la pluma (−1 = 1 mm más arriba), positivo la
  baja (más presión). Se aplica al dibujar, a "Tocar" y a las pruebas, y no
  se pierde al volver a capturar la esquina.
- **Sincronización con el brazo** (`SYNC_EVERY` en `gcode_draw_lib.py`,
  default 4): antes se mandaba un punto cada `MIN_PACING` sin confirmar nada.
  Si el brazo tarda más que eso en cada micro-movimiento, los comandos se
  acumulan en la cola del controlador y, al llenarse, se descartan: las
  curvas salen como rectas (visto en papel con el firmware actualizado y sin
  errores del robot, aunque lo enviado coincidía con el preview). Ahora cada
  `SYNC_EVERY` comandos se espera a que el brazo llegue de verdad (leyendo
  su posición en x,y), así la cola nunca crece. La pantalla final muestra
  cuánto se esperó en promedio y como máximo: esperas largas confirman que el
  brazo iba atrasado. Cuesta algo de tiempo (~100 s más en un retrato típico).
- **Prueba de precisión** (pestaña Calibración, `calibration.precision_test`):
  en 9 puntos de la hoja mide a cuántos mm del punto pedido queda realmente el
  brazo (leyendo `get_coords` ya quieto) con la pluma en el aire, apoyada y
  tras movimientos cortos de 5 mm en las 4 direcciones. Motivo: en un retrato
  real el brazo se detenía en promedio a ~4 mm del punto pedido con la pluma
  apoyada; eso no se nota en el contorno de la cabeza pero deshace ojos, boca
  y nariz (8–10 mm). El resumen dice si el error viene de la fricción
  (apoyada mucho peor que en el aire), del juego de los servos (se queda corto
  en la dirección del movimiento) o de un corrimiento fijo.
- **Puntos que el brazo no alcanza**: si el firmware no encuentra camino
  (errores 32/33/34), descarta ese movimiento y sigue al siguiente; en el
  contorno eso se ve como un triángulo (una esquina salteada). Para evitarlo,
  los tramos con pluma abajo se parten en pasos de `MAX_DRAW_SEGMENT_MM`
  (10 mm) en `gcode_draw_lib.draw_moves`, y las pruebas de calibración
  revisan el error del robot en cada punto y muestran en la página qué
  esquina falló. "Ir encima"/"Tocar" usan la misma orientación fija que el
  dibujo (no la de la captura), así que si una esquina falla ahí, también va a
  fallar al dibujar. Al terminar un dibujo, la pantalla final avisa si el
  robot reportó errores.
- **Centro de la hoja (solo altura, opcional)**: con solo las 4 esquinas, la
  altura del interior sale de un plano entre ellas. Si el papel/mesa/brazo no
  es plano, el dibujo del medio queda más apretado o más flojo que el borde
  (visto en la práctica: un ajuste de −2 mm se notaba en el contorno pero no
  en la cara). Capturando la Z con la punta apoyada en el centro, la altura se
  corrige con una "loma" que vale la diferencia completa en el centro y 0 en
  todo el borde (`_center_z_correction` en `gcode_draw_lib.py`): no cambia la
  forma ni el tamaño del área, ni las esquinas ni el contorno. Tiene su propio
  ajuste de altura y un control que rechaza una Z a más de 15 mm de las
  esquinas (captura con la pluma en el aire).
- **Orientación**: la cabeza de la persona va hacia las esquinas "arriba" del
  perfil, y el preview muestra la hoja así (en el marco de la hoja, no en
  coordenadas del robot). Antes el preview se dibujaba con X del robot hacia
  arriba: con la hoja al costado la persona se veía acostada aunque en el
  papel saliera derecha. Hubo un modo que giraba el dibujo para "arreglar" el
  preview, y eso la dejaba acostada en el papel; se quitó. Si querés la
  cabeza hacia otro lado de la hoja, el perfil permite girar el dibujo
  90/180/270° (`rotate_moves` en `pipeline.py`); si cambiás de perfil con un
  trazo ya generado, volvé a generarlo.
- Debajo de las esquinas se muestran los lados medidos, el desvío respecto de
  un rectángulo y la diferencia de altura entre esquinas, con un aviso si
  parece que una esquina quedó mal medida.

La conexión al brazo es una sola, compartida entre dibujo y calibración
(`robot.py`), con un candado: mientras dibuja no se puede capturar, mover ni
cambiar el perfil activo. Si la calibración deja los servos liberados, el
dibujo los traba antes de empezar.

Los scripts sueltos del proyecto original `pymycobot/demo/myCobot_280_demo/`
(ej. `280_draw_test_pattern.py`) usan su propia copia de `gcode_draw_lib.py`
con las esquinas escritas en el archivo, no los perfiles de la app.

## Manchas de tinta gruesas (iris, cejas rellenas)

Los trazos se sacan del boceto reduciendo la tinta a su línea central
(esqueleto). Eso sirve para líneas, pero no para manchas rellenas: el
esqueleto de un iris negro fundido con el párpado es un rombo con una cruz, y
los ojos salían como garabatos sin importar el nivel de detalle. Por eso
`_extract_strokes()` detecta las zonas de tinta mucho más gruesas que las
líneas normales de ese boceto y las dibuja por su **contorno**
(`THICK_FACTOR` × el percentil 90 del grosor de línea, mínimo
`THICK_MIN_PX`). Se calibró con 4 bocetos reales: comparar contra el grosor
típico en vez del p90 hacía que el sombreado denso de un boceto de trazo fino
se volviera decenas de manchitas contorneadas.

## Ubicación de la cara con MediaPipe (opcional)

`simplify_for_robot()` da más detalle a ojos/nariz/boca que al resto. Para
saber dónde están, `image_to_raw_moves()` busca los rasgos con
`draw_person.py::face_landmark_strokes()` (MediaPipe Face Landmarker) y
devuelve su cuadro delimitador (`core_bbox`). Ver `_aligned_face_landmarks`:

1. **Primero sobre el propio boceto**, apenas desenfocado. Sobre line-art
   nítido MediaPipe casi nunca encuentra la cara (1 de 3 bocetos reales), pero
   con un desenfoque leve la encontró en los 3. Así los rasgos salen del mismo
   boceto y quedan alineados sin importar cómo encuadró Gemini.
2. **Si no, sobre la foto original**, escalada al boceto. Solo sirve si Gemini
   mantuvo el encuadre de la foto.
3. En los dos casos se **valida** que al menos el 85 % de los puntos caiga
   sobre la tinta del boceto. Antes solo se comparaba la proporción de las
   imágenes, y eso no alcanza: con una foto 3:4 donde la cara estaba chica y
   corrida, Gemini devolvió un boceto 3:4 con la cara centrada y agrandada; los
   rasgos de la foto caían sobre mejilla y mentón y se dibujaba **una segunda
   cara**. Ese caso da 63 % y se descarta.

Los rasgos de MediaPipe **no se agregan como trazos**: el boceto ya los tiene
dibujados. Medido con 3 bocetos reales, el único trazo de MediaPipe que "le
faltaba" al boceto era la línea recta del puente de la nariz, una raya que
ningún boceto dibuja.

Es opcional y degrada con gracia: si `mediapipe` no está instalado o no se
puede ubicar la cara de forma confiable, se usa la heurística de posición
(ver "Priorización de detalle por zona" más abajo), sin error. `mediapipe` está en `requirements.txt`
como dependencia opcional; instalarlo descarga un modelo de ~4MB a
`lib/.cache/` la primera vez que se usa (ignorado por git).

## Estructura

```
app.py             Flask: rutas, estado de los jobs en memoria, threads de fondo
pipeline.py         Puente: imagen -> trazos -> simplificado friccion-seguro -> preview -> dibujar
calibration.py       Perfiles de calibracion (esquinas de la hoja) y acciones de calibracion
robot.py             Conexion unica al brazo + candado compartido por dibujo y calibracion
calibration_profiles/ Perfiles guardados (JSON) y cual esta activo
gemini_sketch.py     Llamada a Gemini (foto -> boceto)
static/              Frontend (HTML/CSS/JS vanilla, sin build step)
runtime/              Por job: original, boceto, preview.png, final.gcode (gitignored)
```

## Limitaciones conocidas / próximos pasos

- Sin base de datos: el estado de los jobs vive en memoria del proceso; si
  reiniciás `app.py` se pierden los jobs en curso (los archivos en
  `runtime/` quedan).
- Un solo operador/navegador a la vez tiene sentido real (es un solo brazo).
- Los umbrales de simplificación (`SIMPLIFY_TARGET_MM`,
  `MIN_STROKE_TARGET_MM` en `pipeline.py`) son un punto medio razonable
  encontrado en esta conversación para el tamaño de hoja actual; si cambia
  el área calibrada en `gcode_draw_lib.py` puede valer la pena revisarlos.
- Priorización de detalle por zona (`PRIORITIZE_FACE_ENABLED=True`): la cara
  y dentro de ella ojos/nariz/boca se simplifican mucho menos que el resto
  (pelo/ropa/fondo). El resto además se simplifica bastante MÁS que el
  objetivo base del nivel de detalle (`REST_EPSILON_FACTOR`,
  `REST_MIN_LEN_FACTOR`): un boceto real trae pelo con mechones sueltos y a
  veces decoración de fondo, cada uno un trazo corto separado, que hace que
  el brazo suba/baje la pluma todo el tiempo fuera de la cara.
  `REST_MIN_LEN_FACTOR` descarta esos trazos sueltos chicos en vez de
  dibujarlos todos. Si un boceto futuro sigue viniendo muy fragmentado fuera
  de la cara, subir más ese factor (o `REST_EPSILON_FACTOR`) es el primer
  lugar donde tocar.
  - **Dónde caen esas zonas**: si hay foto original con landmarks de
    MediaPipe disponibles (ver sección de refuerzo facial más arriba), las
    zonas de cara/ojos-nariz-boca se anclan en la posición REAL detectada
    (`core_bbox`, calculado en `image_to_raw_moves` y usado en
    `simplify_for_robot`), con margen alrededor. **Sin eso** (sin foto
    original, sin mediapipe instalado, o subiendo un boceto propio sin
    foto), cae a una heurística de posición: asume que la cara está en la
    franja superior-central del cuadro delimitador real del boceto
    (`FACE_TOP_FRACTION` x `FACE_X_FRACTION`). Esa heurística es aproximada
    y puede fallar en retratos donde algo (gorra, pelo alto) ocupa buena
    parte de arriba del encuadre y la cara queda más abajo del
    `FACE_TOP_FRACTION` asumido — bug real visto con un boceto real: al
    quedar los ojos/nariz/boca fuera de la zona de cara, se los trataba como
    "resto" y se filtraban casi por completo (ojos y nariz directamente
    desaparecían, boca reducida a dos rayas). `FACE_TOP_FRACTION` se subió
    de 0.4 a 0.6 como mitigación general, pero para encuadres inusuales la
    heurística de posición sigue siendo un punto débil sin foto+landmarks.
- Conexión de trazos cercanos (`JOIN_GAP_MM` en `pipeline.py`, default 3mm):
  cuando el boceto viene muy fragmentado (pelo, hatching) los trazos cortos
  suelen quedar muy cerca uno del otro, y cada uno significa una bajada de
  pluma nueva. En el brazo real, al bajar la pluma la punta patina un poco
  hacia arriba por la presión del resorte del esfero, dejando una "V"/rasguño
  visible en vez de una línea limpia — con muchas bajadas seguidas en la
  misma zona eso se ve como rasguños por todos lados (visto con un boceto
  real, ver captura del 2026-09-24). Por eso los trazos que ya quedan a
  `JOIN_GAP_MM` o menos entre sí se conectan con la pluma abajo en vez de
  levantarla (`_flat_strokes_to_moves`/`_join_nearby_strokes` en
  `pipeline.py`): en ese boceto real bajó las bajadas de pluma de 189 a 78
  sin agregar líneas largas visibles que no estuvieran ya insinuadas en el
  boceto (el umbral es chico a propósito).
  - **`JOIN_MAX_CHAIN`** (default 4): tope de cuántos trazos seguidos se
    encadenan sin una levantada real. `gcode_draw_lib.draw_moves()` no
    confirma posición en cada punto (`is_in_position()` resultó poco
    confiable, ver su propio comentario junto a `ASSUMED_SPEED_MM_S`) — manda
    cada comando y espera una pausa calculada por distancia/velocidad
    asumida, confiando en que el brazo ya llegó. Esa aproximación ya había
    dado zigzags/cruces con retratos de cientos de puntos muy seguidos (por
    eso existe `MIN_PACING`) — visto en la práctica: encadenar demasiados
    trazos sin ninguna levantada real (zonas muy densas, ej. patillas/oreja)
    reintrodujo ese mismo problema, con el dibujo físico bastante
    deformado/desplazado respecto al preview aunque preview y ejecución usan
    el mismo mapeo de coordenadas. Cortar la cadena cada `JOIN_MAX_CHAIN`
    trazos fuerza una levantada real (con su propia pausa de viaje) que actúa
    como punto de resincronización. Si el problema vuelve a aparecer en zonas
    muy densas, bajar este número (o `JOIN_GAP_MM`) es el primer lugar donde
    tocar; si en cambio los rasguños de aterrizaje vuelven a ser el problema
    dominante, se puede subir con más margen.
- Suavizado por spline (`SMOOTH_STROKES_ENABLED=True` en `pipeline.py`, con
  `scipy` como dependencia opcional): en vez de simplificar cada trazo con
  `cv2.approxPolyDP` (da segmentos RECTOS entre esquinas detectadas, aspecto
  poligonal/facetado — se nota sobre todo en curvas como la mandíbula, cejas
  o el marco de lentes), se ajusta un B-spline suave
  (`scipy.interpolate.splprep`) y se re-muestrea a paso uniforme **en
  longitud de arco real** (no en el parámetro `u` de la spline: muestrear
  parejo en `u` da pasos de distancia real muy desparejos — se probó y salían
  segmentos de hasta 0 mm mezclados con otros de 15+ mm en el mismo trazo,
  exactamente el riesgo de fricción de `MIN_SAFE_STEP_MM`). El paso objetivo
  es el mismo epsilon de zona que antes usaba `approxPolyDP` (`SPLINE_SMOOTH_
  PER_PT` controla cuánto se le permite al spline apartarse de los puntos
  originales para alisar ruido de la skeletonización, ~1-2px por punto).
  Aproximadamente duplica la cantidad de puntos (una curva necesita más
  segmentos que sus esquinas para verse lisa), pero el resultado se ve
  notablemente más natural/curvo en vez de "poligonal". Degrada con gracia a
  `approxPolyDP` si `scipy` no está instalado o el ajuste falla. Se
  descartó vectorizar el boceto a SVG con una herramienta externa (potrace/
  autotrace): esas herramientas trazan el CONTORNO de cada línea, no su
  centro — para line art duplicarían cada trazo del boceto en vez de dar una
  curva única por el medio.
