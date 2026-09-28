(function () {
  'use strict';

  const el = (id) => document.getElementById(id);
  // orden visual de la grilla 2x2 = como se ve la hoja (arriba = lejos del robot)
  const GRID_ORDER = ['top_left', 'top_right', 'bottom_left', 'bottom_right'];
  const AXES = ['x', 'y', 'z', 'rx', 'ry', 'rz'];

  let state = null;
  let selectedId = null;
  let dirty = false;
  let pollTimer = null;

  async function api(path, method, body) {
    const opts = { method: method || 'GET' };
    if (body !== undefined) {
      opts.headers = { 'Content-Type': 'application/json' };
      opts.body = JSON.stringify(body);
    }
    const res = await fetch(path, opts);
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.error || ('Error HTTP ' + res.status));
    return data;
  }

  function profileUrl(suffix) {
    return '/api/calibration/profiles/' + encodeURIComponent(selectedId) + (suffix || '');
  }

  // --- Pestañas ------------------------------------------------------------
  function showTab(name) {
    const cal = name === 'cal';
    el('app').classList.toggle('hidden', cal);
    el('cal-view').classList.toggle('hidden', !cal);
    el('tab-draw').classList.toggle('active', !cal);
    el('tab-cal').classList.toggle('active', cal);
    el('tab-draw').setAttribute('aria-selected', String(!cal));
    el('tab-cal').setAttribute('aria-selected', String(cal));
    if (cal) refresh(selectedId, true);
  }
  el('tab-draw').addEventListener('click', () => showTab('draw'));
  el('tab-cal').addEventListener('click', () => showTab('cal'));

  // --- Construccion de la grilla de esquinas (una sola vez) -----------------
  function buildCorners(labels) {
    const grid = el('cal-corners');
    if (grid.childElementCount) return;
    GRID_ORDER.forEach((corner) => {
      const card = document.createElement('div');
      card.className = 'corner';
      card.dataset.corner = corner;
      const title = document.createElement('h3');
      title.textContent = labels[corner].charAt(0).toUpperCase() + labels[corner].slice(1);
      card.appendChild(title);
      const axes = document.createElement('div');
      axes.className = 'axes';
      AXES.forEach((axis, idx) => {
        const label = document.createElement('label');
        label.textContent = axis + ' ';
        const input = document.createElement('input');
        input.type = 'number';
        input.step = '0.1';
        input.dataset.idx = String(idx);
        input.addEventListener('input', markDirty);
        label.appendChild(input);
        axes.appendChild(label);
      });
      card.appendChild(axes);
      const zrow = document.createElement('div');
      zrow.className = 'zadj';
      const zlabel = document.createElement('label');
      zlabel.textContent = 'Ajuste de altura (mm) ';
      const zinput = document.createElement('input');
      zinput.type = 'number';
      zinput.step = '0.1';
      zinput.min = '-10';
      zinput.max = '10';
      zinput.dataset.role = 'zadj';
      zinput.title = 'Negativo = sube (-1 = 1 mm más arriba). Positivo = baja (más presión).';
      zinput.addEventListener('input', () => { markDirty(); updateEffectiveZ(corner); });
      zlabel.appendChild(zinput);
      zrow.appendChild(zlabel);
      const zeff = document.createElement('span');
      zeff.className = 'status';
      zeff.dataset.role = 'zeff';
      zrow.appendChild(zeff);
      card.appendChild(zrow);
      card.querySelector('.axes input[data-idx="2"]').addEventListener('input', () => updateEffectiveZ(corner));
      const actions = document.createElement('div');
      actions.className = 'row';
      [['capture', 'Capturar', ''], ['hover', 'Ir encima', 'secondary'], ['touch', 'Tocar', 'secondary']]
        .forEach(([action, text, cls]) => {
          const b = document.createElement('button');
          b.textContent = text;
          b.dataset.action = action;
          if (cls) b.className = cls;
          b.addEventListener('click', () => cornerAction(corner, action));
          actions.appendChild(b);
        });
      card.appendChild(actions);
      grid.appendChild(card);
    });
  }

  function markDirty() {
    dirty = true;
    el('cal-save').textContent = 'Guardar cambios *';
  }

  function clearDirty() {
    dirty = false;
    el('cal-save').textContent = 'Guardar cambios';
  }

  // --- Render --------------------------------------------------------------
  function cornerCard(corner) {
    return el('cal-corners').querySelector('[data-corner="' + corner + '"]');
  }

  function fillCorner(corner, values) {
    cornerCard(corner).querySelectorAll('.axes input').forEach((input) => {
      input.value = values[Number(input.dataset.idx)];
    });
    updateEffectiveZ(corner);
  }

  function updateEffectiveZ(corner) {
    const card = cornerCard(corner);
    const z = Number(card.querySelector('.axes input[data-idx="2"]').value);
    const adj = Number(card.querySelector('[data-role="zadj"]').value || 0);
    const text = adj === 0 ? 'sin ajuste' :
      (adj < 0 ? Math.abs(adj).toFixed(1) + ' mm más arriba' : adj.toFixed(1) + ' mm más abajo');
    card.querySelector('[data-role="zeff"]').textContent = 'Z al dibujar: ' + (z - adj).toFixed(2) + ' (' + text + ')';
  }

  function centerXYText() {
    const d = state && state.profile ? state.profile.diagnostics : null;
    const x = el('cal-center-x').value, y = el('cal-center-y').value;
    if (x === '' || y === '') {
      return 'Posición sin medir: el centro se calcula con las esquinas' +
        (d ? ' (x ' + d.corners_center_xy[0].toFixed(1) + ', y ' + d.corners_center_xy[1].toFixed(1) + ')' : '') + '. ';
    }
    if (!d) return '';
    const dx = Number(x) - d.corners_center_xy[0], dy = Number(y) - d.corners_center_xy[1];
    return 'Posición: el centro real queda a ' + Math.hypot(dx, dy).toFixed(1) + ' mm del calculado con las ' +
      'esquinas (x ' + (dx >= 0 ? '+' : '') + dx.toFixed(1) + ', y ' + (dy >= 0 ? '+' : '') + dy.toFixed(1) +
      '); el medio del dibujo se corre eso. ';
  }

  function updateCenterInfo() {
    const raw = el('cal-center-z').value;
    const plane = state && state.profile ? state.profile.diagnostics.plane_center_z : null;
    if (raw === '') {
      el('cal-center-eff').textContent = centerXYText() + 'Altura sin medir: se calcula solo con las esquinas' +
        (plane !== null ? ' (Z ' + plane.toFixed(2) + ').' : '.');
      return;
    }
    const eff = Number(raw) - Number(el('cal-center-adj').value || 0);
    let text = 'Z al dibujar en el centro: ' + eff.toFixed(2);
    if (plane !== null) {
      const diff = eff - plane;
      text += ' (' + (Math.abs(diff) < 0.05 ? 'igual que' : Math.abs(diff).toFixed(1) + ' mm más ' +
        (diff > 0 ? 'arriba' : 'abajo') + ' que') + ' el plano de las esquinas)';
    }
    el('cal-center-eff').textContent = centerXYText() + text;
  }

  function fillForm(profile) {
    Object.keys(profile.corners).forEach((c) => {
      cornerCard(c).querySelector('[data-role="zadj"]').value = (profile.z_adjust_mm || {})[c] || 0;
      fillCorner(c, profile.corners[c]);
    });
    el('cal-center-z').value = profile.center_z === null || profile.center_z === undefined ? '' : profile.center_z;
    el('cal-center-x').value = profile.center_xy ? profile.center_xy[0] : '';
    el('cal-center-y').value = profile.center_xy ? profile.center_xy[1] : '';
    el('cal-center-adj').value = profile.center_z_adjust_mm || 0;
    updateCenterInfo();
    el('cal-name').value = profile.name;
    el('cal-skew').value = profile.manual_skew_mm;
    el('cal-lift').value = profile.pen_lift_mm;
    el('cal-backlash').value = profile.j1_backlash_deg || 0;
    el('cal-unidir').checked = !!profile.j1_unidirectional;
    el('cal-tablet-mode').checked = !!profile.tablet_mode;
    el('cal-tablet-orient').value = profile.tablet_orient || '';
    el('cal-j1-direction').value = String(profile.j1_direction || 1);
    el('cal-rotation').value = String(profile.drawing_rotation_resolved);
    el('cal-rotation-info').textContent = 'El preview muestra la hoja con las esquinas "arriba" hacia arriba, ' +
      'como la ves al medirla. Girá el dibujo solo si querés la cabeza hacia otro lado de la hoja.';
    clearDirty();
  }

  function renderDiagnostics(profile) {
    const d = profile.diagnostics;
    el('cal-diagnostics').textContent =
      'Lados medidos: arriba ' + d.top_mm + ' mm · abajo ' + d.bottom_mm + ' mm · izquierda ' +
      d.left_mm + ' mm · derecha ' + d.right_mm + ' mm · desvío máximo del ángulo recto ' +
      d.max_angle_error_deg + '° · diferencia de altura entre esquinas ' + d.z_range_mm + ' mm' +
      (d.center_offset_mm === null ? '.' : ' · centro ' + (d.center_offset_mm > 0 ? '+' : '') +
        d.center_offset_mm + ' mm respecto del plano de las esquinas.');
    const warnings = [];
    if (Math.abs(d.top_mm - d.bottom_mm) > 5 || Math.abs(d.left_mm - d.right_mm) > 5) {
      warnings.push('Lados opuestos difieren más de 5 mm: probablemente una esquina quedó mal medida.');
    }
    if (d.max_angle_error_deg > 4) {
      warnings.push('Las esquinas se apartan ' + d.max_angle_error_deg +
        '° de un rectángulo. Se corrige al dibujar, pero conviene re-medir la esquina sospechosa.');
    }
    if (d.z_range_mm > 6) {
      warnings.push('La altura del papel cambia ' + d.z_range_mm +
        ' mm entre esquinas: revisá que la mesa esté nivelada y que la pluma tocara en todas.');
    }
    el('cal-diag-warning').textContent = warnings.join(' ');
  }

  function renderStatus() {
    const r = state.robot;
    let fresh = '';
    if (r.fresh_mode !== null && r.fresh_mode !== undefined) {
      fresh = r.fresh_mode === '0' ? ' Modo de comandos: cola (correcto).' :
        ' ATENCIÓN: el robot no confirmó el modo cola (respondió ' + r.fresh_mode + '): puede saltearse ' +
        'puntos y convertir curvas en rectas. Si no responde, hay que actualizar el firmware del Atom con myStudio.';
    }
    el('cal-robot-status').textContent =
      (r.connected ? 'Conectado.' : 'Sin conectar (se conecta solo al usarlo).') +
      (r.busy ? ' Ocupado: ' + r.busy + '.' : '') + fresh;

    const t = state.task;
    const prog = el('cal-test-progress');
    if (t.running) {
      prog.classList.remove('hidden');
      const p = t.progress || { i: 0, total: 1 };
      prog.value = p.total ? Math.round((100 * p.i) / p.total) : 0;
      el('cal-test-message').textContent = ({ outline: 'Dibujando contorno', pattern: 'Dibujando patrón',
        face: 'Dibujando rostro', eyes: 'Dibujando prueba de ojos', dots: 'Dibujando ojo de puntos', stipple: 'Dibujando rostro con puntos', precision: 'Midiendo precisión' }[t.kind] || 'Probando') +
        '… ' + (t.message || '') + (t.progress ? ' (' + t.progress.i + '/' + t.progress.total + ')' : '');
    } else {
      prog.classList.add('hidden');
      const errs = t.robot_errors || [];
      el('cal-test-message').textContent = t.error ? 'La prueba falló: ' + t.error :
        (!t.kind ? '' : t.kind === 'precision' ? 'Prueba de precisión terminada. ' +
          (t.precision ? t.precision.resumen : '') + (t.log ? ' Registro: ' + t.log : '') :
          (errs.length === 0 ? 'Prueba terminada sin errores del robot.' :
          'Prueba terminada, pero el robot reportó ' + errs.length + ' error(es) y en esos puntos no fue ' +
          'adonde se le pidió: ' + errs.join(' · ')) +
          (t.sync && window.describeSync ? ' ' + window.describeSync(t.sync) : '') +
          (t.log ? ' Registro: ' + t.log : ''));
    }
    const robotBusy = !!r.busy;
    renderRoutines();
    document.querySelectorAll('#cal-view button').forEach((b) => {
      if (!['cal-new', 'cal-save', 'cal-delete', 'cal-center-clear', 'rt-new', 'rt-rename', 'rt-delete', 'rt-stop'].includes(b.id)) {
        b.disabled = robotBusy || !!b.dataset.fixedDisabled;
      }
    });
    el('rt-stop').disabled = !robotBusy && !(state.routine_task || {}).running;   // frena una rutina o un "Ir" en curso
    const isActive = state.profile && state.profile.id === state.active_id;
    el('cal-test-outline').disabled = robotBusy || !isActive;
    el('cal-test-pattern').disabled = robotBusy || !isActive;
    el('cal-test-face').disabled = robotBusy || !isActive;
    el('cal-test-eyes').disabled = robotBusy || !isActive;
    el('cal-test-dots').disabled = robotBusy || !isActive;
    el('cal-test-stipple').disabled = robotBusy || !isActive || !state.stipple;
    const sp = state.stipple;
    el('cal-stipple-img').classList.toggle('hidden', !sp);
    if (sp) {
      if (el('cal-stipple-img').getAttribute('src') !== sp.preview_url) el('cal-stipple-img').src = sp.preview_url;
      el('cal-stipple-info').textContent = sp.dots + ' puntos (' + sp.method + ', ' + sp.bands + ' bandas), de ' + sp.source +
        (sp.found ? '' : ' (no se detectó una cara: se usó el centro de la foto)') +
        '. Dibujarlo lleva unos ' + sp.estimated_minutes + ' min.';
    }
    el('cal-test-precision').disabled = robotBusy || !isActive;
    el('cal-activate').disabled = robotBusy || isActive;
    el('cal-delete').disabled = isActive;
    const activeName = (state.profiles.find((p) => p.id === state.active_id) || {}).name || '—';
    el('cal-test-inactive').textContent = isActive ? '' : 'Estás viendo otro perfil: las pruebas usan el ' +
      'activo (' + activeName + '). Tocá "Usar este perfil" para probar este.';
  }

  // direcciones de /tablet y /resultado con su QR (la IP cambia con la red)
  async function loadConnection() {
    const box = el('cal-conn');
    try {
      const res = await fetch('/api/connection', { cache: 'no-store' });
      const data = await res.json();
      box.innerHTML = '';
      if (!data.addresses.length) {
        box.textContent = 'No se encontró una red: conectá esta PC a un Wi-Fi o a un hotspot.';
        return;
      }
      data.addresses.forEach((a) => {
        const ip = document.createElement('div');
        ip.className = 'conn-ip';
        ip.textContent = 'Red ' + a.ip + (a.hotspot ? ' (zona con cobertura de esta PC)' : '');
        box.appendChild(ip);
        a.pages.forEach((p) => {
          const card = document.createElement('div');
          card.className = 'conn-card';
          const name = document.createElement('div');
          name.className = 'name';
          name.textContent = p.name;
          const img = document.createElement('img');
          img.alt = 'QR para abrir ' + p.url;
          img.src = '/api/connection/qr.svg?url=' + encodeURIComponent(p.url);
          const code = document.createElement('code');
          code.textContent = p.url;
          card.append(name, img, code);
          box.appendChild(card);
        });
      });
    } catch (err) {
      box.textContent = 'No se pudieron obtener las direcciones: ' + err.message;
    }
  }
  el('cal-conn-refresh').addEventListener('click', loadConnection);
  loadConnection();

  function renderTablet() {
    const input = el('cal-tablet-line');
    if (document.activeElement !== input) input.value = state.tablet_line_mm;
  }

  function renderEyeStyles() {
    const sel = el('cal-eye-style');
    if (!sel.options.length) {
      sel.appendChild(new Option('Los del boceto (sin cambiar)', ''));
      state.eye_styles.forEach((s, i) => sel.appendChild(new Option((i + 1) + '. ' + s.name, s.id)));
    }
    sel.value = state.eye_style || '';
  }

  function renderProfiles() {
    renderEyeStyles();
    renderTablet();
    const sel = el('cal-profile');
    sel.innerHTML = '';
    state.profiles.forEach((p) => {
      const opt = document.createElement('option');
      opt.value = p.id;
      opt.textContent = p.name + (p.id === state.active_id ? ' (activo)' : '') +
        (p.updated_at ? ' · ' + p.updated_at : '');
      sel.appendChild(opt);
    });
    sel.value = selectedId;
    const active = state.profiles.find((p) => p.id === state.active_id);
    el('active-profile-label').textContent = active ? active.name : 'ninguno';
    el('cal-area').textContent = state.area_mm[0] + ' × ' + state.area_mm[1] + ' mm';
    const s = state.pattern_mm;
    const mm = (f) => (Math.round(s * f * 10) / 10) + ' mm';
    el('cal-pattern-desc').textContent = s < 40 ? 'El área es demasiado chica para el patrón de prueba.' :
      'El patrón (' + mm(1) + ' × ' + mm(1) + ', centrado en el área' +
      (s < 100 ? ', achicado para entrar: a tamaño completo mide 100 mm' : '') + ') dibuja un marco de ' +
      mm(1) + ', un cuadrado de ' + mm(0.5) + ', un círculo de ' + mm(0.36) + ' y 20 rayitas de ' + mm(0.05) +
      ' para medir escala, esquinas y ganchos.';
    const rot = state.active_rotation;
    el('cal-face-desc').textContent = 'El rostro (cara, ojos, nariz y boca) sale como un retrato con la ' +
      'orientación guardada del perfil activo: ' + (rot ? 'girado ' + rot + '° antihorario' :
      'la frente hacia las esquinas de arriba') + '.';
  }

  function apply(data, refillForm) {
    state = data;
    buildCorners(state.corner_labels);
    const changedProfile = !state.profile || state.profile.id !== selectedId;
    selectedId = state.profile ? state.profile.id : null;
    renderProfiles();
    if (state.profile) {
      if (refillForm || changedProfile) fillForm(state.profile);
      renderDiagnostics(state.profile);
    }
    renderStatus();
    if (state.task.running || state.robot.busy || (state.routine_task || {}).running) startPolling(); else stopPolling();
  }

  async function refresh(profileId, refillForm) {
    try {
      const q = profileId ? '?profile=' + encodeURIComponent(profileId) : '';
      apply(await api('/api/calibration' + q), refillForm);
    } catch (err) {
      el('cal-error').textContent = err.message;
    }
  }

  function startPolling() {
    if (!pollTimer) pollTimer = setInterval(() => refresh(selectedId, false), 1200);
  }

  function stopPolling() {
    if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }
  }

  async function run(fn) {
    el('cal-error').textContent = '';
    try {
      await fn();
    } catch (err) {
      el('cal-error').textContent = err.message;
    }
  }

  function readForm() {
    const corners = {};
    const zAdjust = {};
    el('cal-corners').querySelectorAll('.corner').forEach((card) => {
      corners[card.dataset.corner] = Array.from(card.querySelectorAll('.axes input'))
        .sort((a, b) => a.dataset.idx - b.dataset.idx)
        .map((input) => Number(input.value));
      zAdjust[card.dataset.corner] = Number(card.querySelector('[data-role="zadj"]').value || 0);
    });
    return {
      name: el('cal-name').value.trim(),
      manual_skew_mm: Number(el('cal-skew').value),
      pen_lift_mm: Number(el('cal-lift').value),
      j1_backlash_deg: Number(el('cal-backlash').value || 0),
      j1_unidirectional: el('cal-unidir').checked,
      tablet_mode: el('cal-tablet-mode').checked,
      tablet_orient: el('cal-tablet-orient').value || null,
      j1_direction: Number(el('cal-j1-direction').value),
      drawing_rotation: el('cal-rotation').value,
      center_z: el('cal-center-z').value === '' ? null : Number(el('cal-center-z').value),
      center_xy: el('cal-center-x').value === '' || el('cal-center-y').value === '' ? null :
        [Number(el('cal-center-x').value), Number(el('cal-center-y').value)],
      center_z_adjust_mm: Number(el('cal-center-adj').value || 0),
      corners: corners,
      z_adjust_mm: zAdjust,
    };
  }

  // --- Perfiles ------------------------------------------------------------
  el('cal-profile').addEventListener('change', () => {
    const next = el('cal-profile').value;
    if (dirty && !confirm('Hay cambios sin guardar en este perfil. ¿Descartarlos?')) {
      el('cal-profile').value = selectedId;
      return;
    }
    refresh(next, true);
  });

  el('cal-activate').addEventListener('click', () => run(async () => {
    if (dirty && !confirm('Hay cambios sin guardar: se va a activar la versión guardada. ¿Seguir?')) return;
    apply(await api(profileUrl('/activate'), 'POST'), true);
    el('cal-test-message').textContent = 'Perfil activado. Si tenías un trazo ya generado, ' +
      'volvé a generarlo para que el preview use esta calibración.';
  }));

  el('cal-new').addEventListener('click', () => run(async () => {
    const name = prompt('Nombre del nuevo perfil (copia los valores del perfil seleccionado):');
    if (!name || !name.trim()) return;
    apply(await api('/api/calibration/profiles', 'POST', { name: name.trim(), copy_from: selectedId }), true);
  }));

  el('cal-delete').addEventListener('click', () => run(async () => {
    if (!confirm('¿Eliminar el perfil "' + state.profile.name + '"? No se puede deshacer.')) return;
    selectedId = null;
    apply(await api('/api/calibration/profiles/' + encodeURIComponent(state.profile.id), 'DELETE'), true);
  }));

  el('cal-save').addEventListener('click', () => run(async () => {
    apply(await api(profileUrl(), 'PUT', readForm()), true);
  }));

  ['cal-name', 'cal-skew', 'cal-lift', 'cal-backlash', 'cal-rotation'].forEach((id) => el(id).addEventListener('input', markDirty));
  el('cal-unidir').addEventListener('change', markDirty);
  el('cal-tablet-mode').addEventListener('change', markDirty);
  el('cal-tablet-orient').addEventListener('change', markDirty);
  el('cal-j1-direction').addEventListener('change', markDirty);

  // --- Esquinas ------------------------------------------------------------
  function labelOf(corner) {
    return corner === 'center' ? 'el centro' : state.corner_labels[corner];
  }

  async function cornerAction(corner, action) {
    await run(async () => {
      try {
        await cornerActionInner(corner, action);
      } catch (err) {
        // el error general sale al final de la pagina: el del centro tambien en su tarjeta
        if (corner === 'center') el('cal-center-eff').textContent = 'No se pudo: ' + err.message;
        throw err;
      }
    });
  }

  async function cornerActionInner(corner, action) {
    if (action === 'capture') {
      const data = await api(profileUrl('/capture/' + corner), 'POST');
      apply(data, false);
      if (corner === 'center') {
        el('cal-center-x').value = data.captured.coords[0];
        el('cal-center-y').value = data.captured.coords[1];
        el('cal-center-z').value = data.captured.coords[2];
        updateCenterInfo();
        el('cal-coords').textContent = 'Centro capturado y guardado: x ' + data.captured.coords[0] + ', y ' +
          data.captured.coords[1] + ', Z ' + data.captured.coords[2] + '.';
      } else {
        fillCorner(corner, data.captured.coords);
        el('cal-coords').textContent = labelOf(corner) + ' capturada: ' + data.captured.coords.join(', ');
      }
      return;
    }
    if (dirty) {
      throw new Error('Guardá los cambios primero: el brazo va a la posición guardada, no a la editada.');
    }
    if (action === 'touch' && !confirm('El brazo va a ir a la esquina y bajar hasta tocar el papel. ¿Seguir?')) {
      return;
    }
    el('cal-coords').textContent = 'Moviendo a ' + labelOf(corner) + '…';
    apply(await api(profileUrl('/goto/' + corner), 'POST', { hover_mm: action === 'touch' ? 0 : 10 }), false);
    el('cal-coords').textContent = 'En ' + labelOf(corner) +
      (action === 'touch' ? ' (tocando).' : ' (10 mm arriba).');
  }

  el('cal-center-capture').addEventListener('click', () => cornerAction('center', 'capture'));
  el('cal-center-hover').addEventListener('click', () => cornerAction('center', 'hover'));
  el('cal-center-touch').addEventListener('click', () => cornerAction('center', 'touch'));
  el('cal-center-clear').addEventListener('click', () => {
    el('cal-center-z').value = '';
    el('cal-center-x').value = '';
    el('cal-center-y').value = '';
    markDirty();
    updateCenterInfo();
  });
  ['cal-center-x', 'cal-center-y', 'cal-center-z', 'cal-center-adj'].forEach((id) => el(id).addEventListener('input', () => {
    markDirty();
    updateCenterInfo();
  }));

  // --- Brazo ---------------------------------------------------------------
  function robotAction(action, message) {
    return () => run(async () => {
      const data = await api('/api/calibration/robot/' + action + '?profile=' + encodeURIComponent(selectedId), 'POST');
      apply(data, false);
      el('cal-coords').textContent = data.coords ? 'Posición actual: ' + data.coords.join(', ') :
        data.servos ? Object.entries(data.servos).map(([servo, p]) => 'J' + servo + '  ' +
          Object.entries(p).map(([k, v]) => k + '=' + (v === null ? '?' : v)).join('  ')).join('\n') : message;
    });
  }
  el('cal-release').addEventListener('click', robotAction('release', 'Brazo liberado: movelo a mano.'));
  el('cal-lock').addEventListener('click', robotAction('lock', 'Brazo trabado.'));
  el('cal-read').addEventListener('click', robotAction('read', ''));
  el('cal-servos').addEventListener('click', robotAction('servos', ''));

  function showServos(data, prefix) {
    el('cal-coords').textContent = prefix + '\n' + Object.entries(data.servos).map(([servo, p]) => 'J' + servo +
      '  ' + Object.entries(p).map(([k, v]) => k + '=' + (v === null ? '?' : v)).join('  ')).join('\n');
  }
  function j1Action(body, question) {
    return () => run(async () => {
      if (!confirm(question())) return;
      const data = await api('/api/calibration/servo/j1?profile=' + encodeURIComponent(selectedId), 'POST', body());
      apply(data, false);
      showServos(data, 'Hecho: ' + data.servo_change + '.');
    });
  }
  el('cal-j1-write').addEventListener('click', j1Action(
    () => ({ param: el('cal-j1-param').value, value: Number(el('cal-j1-value').value) }),
    () => '¿Escribir ' + el('cal-j1-param').value + ' = ' + el('cal-j1-value').value +
      ' en el servo de J1? Queda guardado en el servo.'));
  el('cal-j1-restore').addEventListener('click', j1Action(
    () => ({ restore: true }), () => '¿Volver J1 a los valores originales (P 15, D 20, I 0)?'));
  el('cal-disconnect').addEventListener('click', robotAction('disconnect', 'Conexión cerrada.'));

  // --- Pruebas -------------------------------------------------------------
  function testAction(kind, label) {
    return () => run(async () => {
      if (!confirm('El brazo va a dibujar el ' + label + '. ¿Hay una hoja colocada?')) return;
      apply(await api('/api/calibration/test/' + kind, 'POST'), false);
    });
  }
  el('cal-test-outline').addEventListener('click', testAction('outline', 'contorno del área'));
  el('cal-test-pattern').addEventListener('click', testAction('pattern', 'patrón de prueba'));
  el('cal-test-face').addEventListener('click', testAction('face', 'rostro de prueba'));
  el('cal-test-eyes').addEventListener('click', testAction('eyes', 'prueba de ojos (todos los estilos, numerados)'));
  el('cal-test-dots').addEventListener('click', testAction('dots', 'ojo grande hecho solo de puntos (varios minutos)'));
  el('cal-test-stipple').addEventListener('click', testAction('stipple', 'rostro con puntos de la vista previa (varios minutos)'));
  el('cal-stipple-preview').addEventListener('click', () => run(async () => {
    const form = new FormData();
    const file = el('cal-stipple-file').files[0];
    if (file) form.append('image', file);
    form.append('dots', el('cal-stipple-dots').value);
    form.append('method', el('cal-stipple-method').value);
    el('cal-stipple-info').textContent = 'Generando…';
    const res = await fetch('/api/calibration/stipple?profile=' + encodeURIComponent(selectedId), { method: 'POST', body: form });
    const data = await res.json();
    if (!res.ok) {
      el('cal-stipple-info').textContent = '';
      throw new Error(data.error || ('error ' + res.status));
    }
    apply(data, false);
  }));
  el('cal-tablet-line').addEventListener('change', () => run(async () => {
    apply(await api('/api/drawing-settings?profile=' + encodeURIComponent(selectedId), 'POST',
      { tablet_line_mm: Number(el('cal-tablet-line').value) }), false);
    el('cal-test-message').textContent = 'Grosor de la tableta: ' + state.tablet_line_mm + ' mm.';
  }));
  el('cal-eye-style').addEventListener('change', () => run(async () => {
    apply(await api('/api/drawing-settings?profile=' + encodeURIComponent(selectedId), 'POST',
      { eye_style: el('cal-eye-style').value || null }), false);
    el('cal-test-message').textContent = 'Estilo de ojos guardado: se usa en los retratos que generes desde ahora.';
  }));
  el('cal-test-precision').addEventListener('click', testAction('precision', 'crucecitas de la prueba de precisión'));

  // --- Rutinas de movimiento -------------------------------------------------
  let routineId = null;
  let stepsDrawn = null;   // lo ultimo dibujado en la lista (para no pisar lo que se esta editando)

  function currentRoutine() {
    return (state.routines || []).find((r) => r.id === routineId) || null;
  }

  function routineUrl(suffix) {
    return '/api/routines/' + encodeURIComponent(routineId) + (suffix || '');
  }

  // las respuestas de /api/routines traen solo las rutinas: se mezclan con el estado
  function applyRoutines(data) {
    apply(Object.assign({}, state, { routines: data.routines, routine_task: data.routine_task }), false);
  }

  function renderRoutines() {
    const list = state.routines || [];
    if (!list.some((r) => r.id === routineId)) routineId = list.length ? list[0].id : null;
    const sel = el('rt-select');
    const opts = list.map((r) => r.id + '\u0000' + r.name + ' (' + r.steps.length + ')').join('\u0001');
    if (sel.dataset.opts !== opts) {
      sel.innerHTML = '';
      list.forEach((r) => sel.appendChild(new Option(r.name + ' (' + r.steps.length +
        (r.steps.length === 1 ? ' paso)' : ' pasos)'), r.id)));
      sel.dataset.opts = opts;
    }
    sel.value = routineId || '';

    const routine = currentRoutine();
    const t = state.routine_task || {};
    const playing = t.running && t.id === routineId;
    const box = el('rt-steps');
    const key = JSON.stringify([routineId, routine ? routine.steps : null]);
    if (key !== stepsDrawn && !box.contains(document.activeElement)) {
      stepsDrawn = key;
      box.innerHTML = '';
      (routine ? routine.steps : []).forEach((step, i) => box.appendChild(stepRow(step, i, routine.steps.length)));
    }
    box.querySelectorAll('.rt-step').forEach((li, i) =>
      li.classList.toggle('current', !!playing && t.step === i + 1));
    el('rt-empty').textContent = !routine ? 'No hay rutinas: creá una con "Nueva rutina".' :
      routine.steps.length ? '' : 'Esta rutina todavía no tiene pasos.';

    el('rt-message').textContent = t.running ? 'Ejecutando "' + t.name + '"… ' + (t.message || '') :
      t.error ? 'La rutina "' + t.name + '" falló: ' + t.error :
      t.message === 'detenido' ? 'Rutina "' + t.name + '" detenida: el brazo quedó donde estaba.' :
      t.message === 'terminado' ? 'Rutina "' + t.name + '" terminada.' : '';
  }

  function stepRow(step, i, total) {
    const li = document.createElement('li');
    li.className = 'rt-step';
    const num = document.createElement('span');
    num.className = 'num';
    num.textContent = 'Paso ' + (i + 1);
    const angles = document.createElement('span');
    angles.className = 'angles';
    angles.textContent = step.angles.map((a, j) => 'J' + (j + 1) + ' ' + a.toFixed(1) + '°').join('  ');
    const field = (label, key, attrs) => {
      const lab = document.createElement('label');
      lab.textContent = label + ' ';
      const input = document.createElement('input');
      input.type = 'number';
      Object.assign(input, attrs);
      input.value = step[key];
      input.addEventListener('change', () => editSteps((steps) => { steps[i][key] = Number(input.value); }));
      lab.appendChild(input);
      return lab;
    };
    const button = (text, title, fn, disabled) => {
      const b = document.createElement('button');
      b.className = 'secondary';
      b.textContent = text;
      b.title = title;
      b.disabled = !!disabled;
      b.dataset.fixedDisabled = disabled ? '1' : '';
      b.addEventListener('click', () => run(fn));
      return b;
    };
    li.append(num, angles,
      field('Velocidad', 'speed', { min: 1, max: 100, step: 1, title: 'Velocidad para llegar a esta pose (1 a 100)' }),
      field('Pausa (s)', 'pause_s', { min: 0, max: 30, step: 0.1, title: 'Cuánto se queda en esta pose antes de seguir' }),
      button('Ir', 'Trabar el brazo y llevarlo a esta pose', async () => {
        startPolling();   // mientras va, el estado muestra el brazo ocupado y habilita "Detener"
        applyRoutines(await api(routineUrl('/goto/' + i), 'POST'));
      }),
      button('Reemplazar', 'Guardar la posición actual del brazo en este paso', async () => {
        if (!confirm('¿Reemplazar el paso ' + (i + 1) + ' por la posición actual del brazo?')) return;
        applyRoutines(await api(routineUrl('/capture'), 'POST', { index: i }));
      }),
      button('↑', 'Subir', () => editSteps((s) => s.splice(i - 1, 0, s.splice(i, 1)[0])), i === 0),
      button('↓', 'Bajar', () => editSteps((s) => s.splice(i + 1, 0, s.splice(i, 1)[0])), i === total - 1),
      button('Borrar', 'Borrar este paso', async () => {
        if (!confirm('¿Borrar el paso ' + (i + 1) + '?')) return;
        await editSteps((s) => s.splice(i, 1));
      }));
    return li;
  }

  async function editSteps(change) {
    const steps = JSON.parse(JSON.stringify(currentRoutine().steps));
    change(steps);
    stepsDrawn = null;
    try {
      applyRoutines(await api(routineUrl(), 'PUT', { steps }));
    } catch (err) {
      renderRoutines();   // vuelve a mostrar lo guardado
      throw err;
    }
  }

  el('rt-select').addEventListener('change', () => { routineId = el('rt-select').value; renderRoutines(); });
  el('rt-new').addEventListener('click', () => run(async () => {
    const name = prompt('Nombre de la rutina nueva (por ejemplo: Despedida):');
    if (!name || !name.trim()) return;
    const data = await api('/api/routines', 'POST', { name: name.trim() });
    routineId = data.created;
    applyRoutines(data);
  }));
  el('rt-rename').addEventListener('click', () => run(async () => {
    const routine = currentRoutine();
    if (!routine) return;
    const name = prompt('Nuevo nombre de la rutina:', routine.name);
    if (!name || !name.trim()) return;
    applyRoutines(await api(routineUrl(), 'PUT', { name: name.trim() }));
  }));
  el('rt-delete').addEventListener('click', () => run(async () => {
    const routine = currentRoutine();
    if (!routine || !confirm('¿Borrar la rutina "' + routine.name + '" con sus ' + routine.steps.length + ' pasos?')) return;
    applyRoutines(await api(routineUrl(), 'DELETE'));
  }));
  el('rt-release').addEventListener('click', robotAction('release', 'Brazo liberado: llevalo a la pose y tocá "Agregar posición actual".'));
  el('rt-lock').addEventListener('click', robotAction('lock', 'Brazo trabado.'));
  el('rt-capture').addEventListener('click', () => run(async () => {
    if (!routineId) throw new Error('Primero creá o elegí una rutina.');
    const data = await api(routineUrl('/capture'), 'POST', {});
    applyRoutines(data);
    el('rt-message').textContent = 'Paso ' + currentRoutine().steps.length + ' agregado: ' +
      data.captured.map((a, j) => 'J' + (j + 1) + ' ' + a.toFixed(1) + '°').join('  ');
  }));
  el('rt-play').addEventListener('click', () => run(async () => {
    const routine = currentRoutine();
    if (!routine) return;
    if (!confirm('El brazo se va a trabar y a ejecutar "' + routine.name + '". ¿Está libre el espacio alrededor?')) return;
    applyRoutines(await api(routineUrl('/play'), 'POST', { repeat: Number(el('rt-repeat').value || 1) }));
  }));
  el('rt-stop').addEventListener('click', () => run(async () => {
    applyRoutines(await api('/api/routines/stop', 'POST'));
  }));

  // nombre del perfil activo en el encabezado, tambien en la pestaña Dibujar
  refresh(null, true);
})();
