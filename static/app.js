(function () {
  'use strict';

  let jobId = null;
  let pollTimer = null;
  // recorte del boceto: [x0, y0, x1, y1] en fracciones de la imagen, o null
  let crop = null;
  let cropSyncedFor = null;  // sketch_url para el que ya se tomo el recorte del servidor

  const el = (id) => document.getElementById(id);
  const stages = ['upload', 'sketch', 'gcode', 'drawing', 'done'];

  function showStage(name) {
    stages.forEach((s) => el('stage-' + s).classList.toggle('hidden', s !== name));
  }

  async function api(path, options) {
    const res = await fetch(path, options);
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      throw new Error(data.error || ('Error HTTP ' + res.status));
    }
    return data;
  }

  function stopPolling() {
    if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }
  }

  function startPolling() {
    stopPolling();
    pollTimer = setInterval(refreshJob, 1200);
  }

  async function refreshJob() {
    if (!jobId) return;
    try {
      const job = await api('/api/jobs/' + jobId);
      render(job);
    } catch (err) {
      console.error(err);
    }
  }

  function render(job) {
    switch (job.stage) {
      case 'uploaded':
      case 'sketching':
        showStage('sketch');
        el('style-select-2').value = job.style || 'detallado';
        el('img-original').src = job.original_url || '';
        el('img-sketch').src = job.sketch_url || '';
        crop = null;
        cropSyncedFor = null;
        renderCrop();
        el('sketch-status').textContent = job.stage === 'sketching'
          ? 'Generando boceto con Gemini...' : '';
        el('sketch-error').textContent = '';
        el('btn-sketch-accept').disabled = job.stage !== 'sketch_ready';
        el('btn-sketch-reject').disabled = true;
        break;

      case 'sketch_ready':
        showStage('sketch');
        el('style-select-2').value = job.style || 'detallado';
        el('img-original').src = job.original_url || '';
        el('img-sketch').src = job.sketch_url || '';
        if (cropSyncedFor !== job.sketch_url) {
          // boceto nuevo (o recien cargado): tomar el recorte que tenga guardado
          crop = job.crop || null;
          cropSyncedFor = job.sketch_url;
        }
        renderCrop();
        el('sketch-status').textContent = job.sketch_note || '';
        el('sketch-error').textContent = '';
        el('btn-sketch-accept').disabled = false;
        el('btn-sketch-reject').disabled = false;
        stopPolling();
        break;

      case 'sketch_failed':
        showStage('sketch');
        el('style-select-2').value = job.style || 'detallado';
        el('img-original').src = job.original_url || '';
        el('sketch-status').textContent = '';
        el('sketch-error').textContent = 'No se pudo generar el boceto: ' + job.sketch_error;
        el('btn-sketch-accept').disabled = true;
        el('btn-sketch-reject').disabled = false;
        stopPolling();
        break;

      case 'building_gcode':
        showStage('gcode');
        el('preview-stats').textContent = 'Generando trazo...';
        break;

      case 'gcode_ready':
        showStage('gcode');
        // el timestamp evita que el navegador muestre el preview viejo cacheado al regenerar
        el('img-preview').src = job.preview_url || '';  // ya trae ?v= con la version del archivo
        el('preview-stats').textContent = describeStats(job.preview_stats);
        el('svg-link-row').classList.toggle('hidden', !job.svg_url);
        if (job.svg_url) el('svg-link').href = job.svg_url;
        el('gcode-error').textContent = '';
        el('detail-select-2').value = job.detail || 'medio';
        el('btn-gcode-rebuild').disabled = false;
        el('btn-gcode-accept').disabled = false;
        stopPolling();
        break;

      case 'drawing':
        showStage('drawing');
        if (job.preview_url) el('img-drawing').src = job.preview_url;
        renderDrawProgress(job);
        break;

      case 'done':
        showStage('done');
        el('done-title').textContent = 'Listo';
        el('done-message').textContent = 'El robot terminó de dibujar.' +
          (job.robot_errors && job.robot_errors.length ? ' Atención: el robot reportó ' + job.robot_errors.length +
            ' error(es) y en esos puntos no fue adonde se le pidió: ' + job.robot_errors.join(' · ') : '') +
          (job.sync ? ' ' + describeSync(job.sync) : '');
        el('btn-redraw').textContent = 'Volver a dibujar el mismo';
        el('btn-redraw').disabled = false;
        stopPolling();
        break;

      case 'error':
        showStage('done');
        el('done-title').textContent = 'Error al dibujar';
        el('done-message').textContent = job.draw_error || 'Error desconocido.';
        el('btn-redraw').textContent = 'Volver a dibujar el mismo (desde el principio)';
        el('btn-redraw').disabled = false;
        stopPolling();
        break;

      default:
        showStage('upload');
    }
  }

  function describeSync(s) {
    return 'Sincronización con el brazo: ' + s.count + ' esperas, media ' + s.mean_s + ' s, máxima ' + s.max_s +
      ' s. Llegó al punto pedido ' + s.reached + ' veces, se detuvo sin llegar ' + s.still + ' veces' +
      (s.off_mean_mm !== undefined ? ' (quedó en promedio a ' + s.off_mean_mm + ' mm del punto, p90 ' +
        s.off_p90_mm + ' mm).' : '.');
  }
  window.describeSync = describeSync;

  function describeStats(stats) {
    if (!stats) return '';
    return (stats.style ? stats.style + ': ' : '') + stats.strokes + ' trazos, ' + stats.points + ' puntos, ' +
      stats.draw_mm + ' mm de dibujo, ' + stats.travel_mm + ' mm de viaje ' +
      '(~' + stats.estimated_minutes + ' min estimado, área ' +
      stats.area_mm[0] + '×' + stats.area_mm[1] + ' mm)' +
      (stats.pencil ? '. ' + stats.pencil[0].toUpperCase() + stats.pencil.slice(1) : '') +
      (stats.short_dropped ? '. Se omitieron ' + stats.short_dropped + ' rayas de menos de ' + stats.min_stroke_mm + ' mm' : '') +
      (stats.eyes ? '. ' + stats.eyes[0].toUpperCase() +
        stats.eyes.slice(1) + '.' : '');
  }

  function renderDrawProgress(job) {
    const p = job.draw_progress || { i: 0, total: 1, message: '' };
    const pct = p.total ? Math.round((100 * p.i) / p.total) : 0;
    el('draw-progress').value = pct;
    el('draw-message').textContent = p.message + '  (' + p.i + '/' + p.total + ')';
  }

  // --- Etapa 1: subir foto -----------------------------------------------
  el('file-input').addEventListener('change', () => {
    el('btn-upload').disabled = !el('file-input').files.length;
  });

  el('btn-upload').addEventListener('click', async () => {
    const file = el('file-input').files[0];
    if (!file) return;
    const ownSketch = el('own-sketch-input').files[0];
    el('upload-error').textContent = '';
    el('btn-upload').disabled = true;
    try {
      const form = new FormData();
      form.append('image', file);
      form.append('style', el('style-select').value);
      const job = await api('/api/jobs', { method: 'POST', body: form });
      jobId = job.id;
      let updated;
      if (ownSketch) {
        const sketchForm = new FormData();
        sketchForm.append('sketch', ownSketch);
        updated = await api('/api/jobs/' + jobId + '/sketch/manual', { method: 'POST', body: sketchForm });
      } else {
        updated = await api('/api/jobs/' + jobId + '/sketch', { method: 'POST' });
      }
      render(updated);
      startPolling();
    } catch (err) {
      el('upload-error').textContent = err.message;
      el('btn-upload').disabled = false;
    }
  });

  // --- Etapa 2: recorte del boceto -----------------------------------------
  // Se arrastra un rectangulo sobre el boceto; al aceptar, solo esa zona se usa
  // para generar el gcode (ej. la cara sola, para que ojos y boca salgan mas
  // grandes en la hoja).
  const MIN_CROP = 0.05;  // igual que MIN_CROP_FRACTION en app.py

  function renderCrop() {
    const box = el('crop-box');
    if (!crop) {
      box.classList.add('hidden');
      el('btn-crop-clear').classList.add('hidden');
      el('crop-info').textContent = 'Sin recorte: se dibuja el boceto completo. ' +
        'Arrastrá sobre el boceto para elegir solo una parte (por ejemplo la cara).';
      return;
    }
    box.classList.remove('hidden');
    box.style.left = (crop[0] * 100) + '%';
    box.style.top = (crop[1] * 100) + '%';
    box.style.width = ((crop[2] - crop[0]) * 100) + '%';
    box.style.height = ((crop[3] - crop[1]) * 100) + '%';
    el('btn-crop-clear').classList.remove('hidden');
    el('crop-info').textContent = 'Recorte: se dibuja solo la zona marcada (' +
      Math.round((crop[2] - crop[0]) * 100) + '% × ' + Math.round((crop[3] - crop[1]) * 100) +
      '% del boceto), agrandada para ocupar la hoja.';
  }

  function pointerToFraction(ev) {
    const r = el('img-sketch').getBoundingClientRect();
    const fx = Math.min(1, Math.max(0, (ev.clientX - r.left) / r.width));
    const fy = Math.min(1, Math.max(0, (ev.clientY - r.top) / r.height));
    return [fx, fy];
  }

  let dragStart = null;
  const cropWrap = el('crop-wrap');
  cropWrap.addEventListener('pointerdown', (ev) => {
    if (!el('img-sketch').src || el('btn-sketch-accept').disabled) return;  // sin boceto listo no se recorta
    ev.preventDefault();
    cropWrap.setPointerCapture(ev.pointerId);
    dragStart = pointerToFraction(ev);
    crop = [dragStart[0], dragStart[1], dragStart[0], dragStart[1]];
    renderCrop();
  });
  cropWrap.addEventListener('pointermove', (ev) => {
    if (!dragStart) return;
    const p = pointerToFraction(ev);
    crop = [Math.min(dragStart[0], p[0]), Math.min(dragStart[1], p[1]),
            Math.max(dragStart[0], p[0]), Math.max(dragStart[1], p[1])];
    renderCrop();
  });
  function endDrag() {
    if (!dragStart) return;
    dragStart = null;
    if (crop && (crop[2] - crop[0] < MIN_CROP || crop[3] - crop[1] < MIN_CROP)) {
      crop = null;  // un click o un arrastre minimo no es un recorte
    }
    renderCrop();
  }
  cropWrap.addEventListener('pointerup', endDrag);
  cropWrap.addEventListener('pointercancel', endDrag);

  el('btn-crop-clear').addEventListener('click', () => {
    crop = null;
    renderCrop();
  });

  // --- Etapa 2: aceptar/rechazar boceto -----------------------------------
  el('btn-sketch-accept').addEventListener('click', async () => {
    el('btn-sketch-accept').disabled = true;
    el('gcode-error').textContent = '';
    try {
      const form = new FormData();
      form.append('detail', el('detail-select').value);
      form.append('style', el('style-select-2').value);
      form.append('crop', crop ? JSON.stringify(crop) : '');
      const job = await api('/api/jobs/' + jobId + '/sketch/accept', { method: 'POST', body: form });
      render(job);
    } catch (err) {
      el('sketch-error').textContent = err.message;
      el('btn-sketch-accept').disabled = false;
    }
  });

  el('btn-sketch-reject').addEventListener('click', async () => {
    try {
      const form = new FormData();
      form.append('style', el('style-select-2').value);
      const job = await api('/api/jobs/' + jobId + '/sketch/reject', { method: 'POST', body: form });
      render(job);
      startPolling();
    } catch (err) {
      el('sketch-error').textContent = err.message;
    }
  });

  el('btn-manual-sketch').addEventListener('click', async () => {
    const file = el('manual-sketch-input').files[0];
    if (!file || !jobId) return;
    try {
      const form = new FormData();
      form.append('sketch', file);
      const job = await api('/api/jobs/' + jobId + '/sketch/manual', { method: 'POST', body: form });
      render(job);
    } catch (err) {
      el('sketch-error').textContent = err.message;
    }
  });

  // --- Etapa 3: aceptar/rechazar gcode -------------------------------------
  el('btn-gcode-rebuild').addEventListener('click', async () => {
    el('btn-gcode-rebuild').disabled = true;
    el('gcode-error').textContent = '';
    try {
      const form = new FormData();
      form.append('detail', el('detail-select-2').value);
      const job = await api('/api/jobs/' + jobId + '/gcode/rebuild', { method: 'POST', body: form });
      render(job);
    } catch (err) {
      el('gcode-error').textContent = err.message;
    } finally {
      el('btn-gcode-rebuild').disabled = false;
    }
  });

  el('btn-gcode-accept').addEventListener('click', async () => {
    if (!confirm('El brazo real va a empezar a dibujar. ¿Confirmás?')) return;
    el('btn-gcode-accept').disabled = true;
    try {
      const job = await api('/api/jobs/' + jobId + '/gcode/accept', { method: 'POST' });
      render(job);
      startPolling();
    } catch (err) {
      el('gcode-error').textContent = err.message;
      el('btn-gcode-accept').disabled = false;
    }
  });

  el('btn-gcode-reject').addEventListener('click', async () => {
    try {
      const job = await api('/api/jobs/' + jobId + '/gcode/reject', { method: 'POST' });
      render(job);
    } catch (err) {
      el('gcode-error').textContent = err.message;
    }
  });

  // --- Etapa final ---------------------------------------------------------
  el('btn-redraw').addEventListener('click', async () => {
    if (!confirm('El brazo va a dibujar el mismo trazo otra vez. ¿Pusiste una hoja nueva?')) return;
    el('btn-redraw').disabled = true;
    el('done-error').textContent = '';
    try {
      const job = await api('/api/jobs/' + jobId + '/redraw', { method: 'POST' });
      render(job);
      startPolling();
    } catch (err) {
      el('done-error').textContent = err.message;
      el('btn-redraw').disabled = false;
    }
  });

  el('btn-restart').addEventListener('click', () => {
    jobId = null;
    crop = null;
    cropSyncedFor = null;
    stopPolling();
    el('file-input').value = '';
    el('own-sketch-input').value = '';
    el('btn-upload').disabled = true;
    el('upload-error').textContent = '';
    el('done-error').textContent = '';
    showStage('upload');
  });

  showStage('upload');
})();
