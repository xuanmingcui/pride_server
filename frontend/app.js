/* ── Utilities ──────────────────────────────────────────────────── */

const $ = (sel, ctx = document) => ctx.querySelector(sel);
const $$ = (sel, ctx = document) => [...ctx.querySelectorAll(sel)];

let _toastTimer;
function toast(msg, type = 'info') {
  const el = $('#toast');
  el.textContent = msg;
  el.className = `show ${type}`;
  clearTimeout(_toastTimer);
  _toastTimer = setTimeout(() => { el.className = ''; }, 3500);
}

function fmt(secs) {
  if (secs == null) return '';
  const m = Math.floor(secs / 60), s = Math.floor(secs % 60);
  return `${m}:${String(s).padStart(2, '0')}`;
}

function fmtTimeRange(start, end) {
  if (start == null || end == null) return '';
  const fmtOne = v => (Number(v) < 60
    ? `${Number(v).toFixed(2)}s`
    : fmt(v) + `.${String(Math.floor((Number(v) % 1) * 100)).padStart(2, '0')}`);
  return `${fmtOne(start)} – ${fmtOne(end)}`;
}

function copyToClipboard(text) {
  const execFallback = () => {
    const ta = document.createElement('textarea');
    ta.value = text;
    ta.setAttribute('readonly', '');
    ta.style.cssText = 'position:absolute;left:-9999px;top:0';
    document.body.appendChild(ta);
    ta.select();
    try { document.execCommand('copy'); } catch (_) {}
    document.body.removeChild(ta);
  };
  if (navigator.clipboard && window.isSecureContext) {
    return navigator.clipboard.writeText(text).catch(() => { execFallback(); });
  }
  execFallback();
  return Promise.resolve();
}

function setStatus(idPrefix, msg, type = '') {
  const box = $(`#${idPrefix}-status`);
  const txt = $(`#${idPrefix}-status-text`);
  const spinner = $(`#${idPrefix}-spinner`);
  if (!box) return;
  if (msg === null) { box.className = 'status-box'; return; }
  box.className = `status-box visible${type ? ' ' + type : ''}`;
  if (txt) txt.textContent = msg;
  if (spinner) spinner.style.display = type ? 'none' : '';
}

/** Poll /api/tasks/:id until done or error. Returns the result dict. */
async function pollTask(taskId, onStatus) {
  while (true) {
    const r = await fetch(`/api/tasks/${taskId}`);
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const data = await r.json();
    if (data.status === 'done') return data.result;
    if (data.status === 'error') throw new Error(data.error || 'Task failed');
    if (onStatus) onStatus(data.status);
    await new Promise(res => setTimeout(res, 2000));
  }
}

/* ── Tab switching ──────────────────────────────────────────────── */
$$('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    $$('.tab-btn').forEach(b => b.classList.remove('active'));
    $$('.tab-pane').forEach(p => p.classList.remove('active'));
    btn.classList.add('active');
    $(`#tab-${btn.dataset.tab}`).classList.add('active');
    if (btn.dataset.tab === 'validate') loadDatabasesIntoSelect('#val-database');
    if (btn.dataset.tab === 'databases') refreshDatabaseSelector();
  });
});

/* ── Upload zone helper ─────────────────────────────────────────── */
function setupUploadZone(zoneId, fileInputId, fileNameId) {
  const zone = $(`#${zoneId}`);
  const input = $(`#${fileInputId}`);
  const label = $(`#${fileNameId}`);

  zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('drag-over'); });
  zone.addEventListener('dragleave', () => zone.classList.remove('drag-over'));
  zone.addEventListener('drop', e => {
    e.preventDefault();
    zone.classList.remove('drag-over');
    if (e.dataTransfer.files[0]) {
      input.files = e.dataTransfer.files;
      showFileName(label, e.dataTransfer.files[0].name);
      input.dispatchEvent(new Event('change'));
    }
  });
  input.addEventListener('change', () => {
    if (input.files[0]) showFileName(label, input.files[0].name);
    else { label.textContent = ''; label.classList.remove('visible'); }
  });
}

function showFileName(el, name) {
  el.textContent = '📎 ' + name;
  el.classList.add('visible');
}

setupUploadZone('sg-upload-zone', 'sg-file', 'sg-filename');
setupUploadZone('val-upload-zone', 'val-file', 'val-filename');
setupUploadZone('db-file-zone', 'db-facts-file', 'db-file-name');

/* ══════════════════════════════════════════════════════════════════
   SCENE GRAPH
══════════════════════════════════════════════════════════════════ */

// Most recent scene-graph result; kept here so the "Normalize entities"
// button can re-submit its segments to /api/scenegraph/normalize.
let _sgLastResult = null;
let _sgLastTaskId = null;

$('#sg-submit').addEventListener('click', async () => {
  const file = $('#sg-file').files[0];
  const text = $('#sg-text').value.trim();
  if (!file && !text) { toast('Provide a file or text.', 'error'); return; }

  const btn = $('#sg-submit');
  btn.disabled = true;
  setStatus('sg', 'Submitting…');
  $('#sg-copy-btn').style.display = 'none';
  $('#sg-download-btn').style.display = 'none';
  $('#sg-normalize-btn').style.display = 'none';
  $('#sg-result').innerHTML = '';

  try {
    // Handle prompt override / permanent save
    const detailsOpen = $('#sg-prompt-details').open;
    const sysTa  = $('#sg-prompt-system-ta');
    const userTa = $('#sg-prompt-user-ta');
    const sysModified  = detailsOpen && sysTa.value  !== _sgPrompt.serverSys;
    const userModified = detailsOpen && userTa.value !== _sgPrompt.serverUser;
    const isModified = sysModified || userModified;
    const saveMode = document.querySelector('input[name="sg-prompt-save"]:checked').value;

    if (isModified && saveMode === 'permanent') {
      const body = {};
      if (sysModified)  body.system = sysTa.value;
      if (userModified) body.user   = userTa.value;
      const pr = await fetch(`/api/prompts/${encodeURIComponent(_sgPrompt.name)}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      if (pr.ok) {
        const pd = await pr.json();
        _promptCache[_sgPrompt.name].template = pd.template;
        _promptCache[_sgPrompt.name].is_custom = true;
        _sgPrompt.serverSys  = pd.template.system;
        _sgPrompt.serverUser = pd.template.user;
        updateSgPromptModifiedBadge();
        $('#sg-prompt-reset').style.display = '';
        toast('Prompt saved as default.', 'success');
      }
    }

    const outputType = $('#sg-output-type').value;
    const fd = new FormData();
    if (file) fd.append('file', file);
    if (text) fd.append('text', text);
    fd.append('output_type', outputType === 'raw' ? 'json' : outputType);
    fd.append('raw_output', outputType === 'raw' ? 'true' : 'false');
    fd.append('mode', document.querySelector('input[name="sg-mode"]:checked').value);
    const normalizePre = $('#sg-normalize-pre').checked;
    fd.append('normalize', normalizePre ? 'true' : 'false');
    const fps = $('#sg-fps').value;
    if (fps) fd.append('fps', fps);
    const temp = $('#sg-temperature').value;
    if (temp) fd.append('temperature', temp);
    if (isModified && saveMode === 'once') {
      if (sysModified)  fd.append('system_prompt_override', sysTa.value);
      if (userModified) fd.append('user_prompt_override',   userTa.value);
    }

    const r = await fetch('/api/scenegraph', { method: 'POST', body: fd });
    const j = await r.json();
    if (!r.ok) throw new Error(j.error || 'Submission failed');

    setStatus('sg', 'Processing (this may take a minute)…');
    const result = await pollTask(j.task_id, s => setStatus('sg', `Status: ${s}…`));
    setStatus('sg', null);
    if (normalizePre) result.normalized = true;
    renderSceneGraph(result, j.task_id);
  } catch (err) {
    setStatus('sg', err.message, 'error');
  } finally {
    btn.disabled = false;
  }
});

function renderSceneGraph(result, taskId) {
  _sgLastResult = result;
  _sgLastTaskId = taskId;

  const segments   = result.segments || [];
  const isTemporal = result.is_temporal !== false;
  const isRaw      = result.raw_output === true;
  const total      = isRaw ? segments.length : segments.reduce((n, s) => n + s.triplets.length, 0);
  const hasOverlay = result.overlay_path;

  // The Normalize button only applies to parsed (non-raw) results that
  // actually carry triplets / quintuples to normalize.
  const normalizeBtn = $('#sg-normalize-btn');
  const canNormalize = !isRaw && total > 0;
  normalizeBtn.style.display = canNormalize ? '' : 'none';
  normalizeBtn.disabled = false;
  normalizeBtn.textContent = 'Refine results';

  const container = $('#sg-result');
  container.innerHTML = '';

  // Detect quintuple (video) vs triplet (image/text) output by inspecting first item.
  const firstItem = segments.find(s => (s.triplets || []).length)?.triplets?.[0];
  const hasTimes  = !!(firstItem && firstItem.start_sec != null);
  const unit = hasTimes ? 'quintuple' : 'triplet';

  // Summary chips
  const chips = document.createElement('div');
  chips.className = 'chips';
  if (isRaw) {
    chips.innerHTML =
      `<span class="chip">raw text</span>`;
  } else if (isTemporal) {
    chips.innerHTML =
      `<span class="chip"><strong>${segments.length}</strong> segment${segments.length !== 1 ? 's' : ''}</span>` +
      `<span class="chip"><strong>${total}</strong> ${unit}${total !== 1 ? 's' : ''}</span>`;
  } else {
    chips.innerHTML =
      `<span class="chip">image / text</span>` +
      `<span class="chip"><strong>${total}</strong> ${unit}${total !== 1 ? 's' : ''}</span>`;
  }
  if (result.normalized) {
    chips.innerHTML += `<span class="chip" style="color:var(--accent);border-color:var(--accent)">refined</span>`;
  }
  container.appendChild(chips);

  // Overlay video/image
  if (hasOverlay) {
    const wrap = document.createElement('div');
    wrap.className = 'overlay-wrap';
    const ext = result.overlay_path.split('.').pop().toLowerCase();
    const src = `/api/tasks/${taskId}/file`;
    if (ext === 'mp4' || ext === 'webm') {
      wrap.innerHTML = `<video controls src="${src}"></video>`;
    } else {
      wrap.innerHTML = `<img src="${src}" alt="Overlay">`;
    }
    container.appendChild(wrap);

    const dlBtn = $('#sg-download-btn');
    dlBtn.href = src;
    dlBtn.download = `pride_overlay.${ext}`;
    dlBtn.style.display = '';
  }

  if (result.overlay_error) {
    const warn = document.createElement('p');
    warn.style.cssText = 'color:var(--warning);font-size:.8rem;margin:8px 0';
    warn.textContent = '⚠ Overlay error: ' + result.overlay_error;
    container.appendChild(warn);
  }

  // Segment list. For JSON/quintuple output, skip segments with no rows
  // so the result panel only shows cards that have content. Raw cards are
  // always rendered because their `raw_text` is the content.
  const list = document.createElement('div');
  list.className = 'segment-list';
  list.style.marginTop = '12px';

  const visibleSegments = isRaw
    ? segments
    : segments.filter(s => (s.triplets || []).length > 0);

  visibleSegments.forEach((seg, i) => {
    const card = document.createElement('div');
    card.className = 'segment-card';

    const header = document.createElement('div');
    header.className = 'segment-header';
    const timeStr = isTemporal && seg.start != null
      ? `${fmt(seg.start)} – ${fmt(seg.end)} &nbsp;·&nbsp; ` : '';
    const segLabel = isRaw
      ? 'Raw output'
      : (isTemporal ? `Segment ${i + 1}` : (hasTimes ? 'Quintuples' : 'Triplets'));
    header.innerHTML =
      `<span>${segLabel}</span>` +
      `<span>${timeStr}${seg.triplets.length} ${unit}${seg.triplets.length !== 1 ? 's' : ''}</span>`;
    card.appendChild(header);

    const body = document.createElement('div');
    body.className = 'segment-body';

    if (isRaw && seg.raw_text != null) {
      const pre = document.createElement('pre');
      pre.className = 'raw-text-output';
      pre.textContent = seg.raw_text;
      body.appendChild(pre);
    } else {
      // Non-raw cards are only rendered when they have triplets (see filter
      // above), so we can skip the empty-state branch here.
      seg.triplets.forEach(t => {
        const row = document.createElement('div');
        const itemHasTime = t.start_sec != null && t.end_sec != null;
        row.className = itemHasTime ? 'triplet-row with-time' : 'triplet-row';
        const cells =
          `<span class="triplet-subj">${esc(t.subject)}</span>` +
          `<span class="triplet-rel">→ ${esc(t.relation)} →</span>` +
          `<span class="triplet-obj">${esc(t.object)}</span>`;
        const timeCell = itemHasTime
          ? `<span class="triplet-time">${esc(fmtTimeRange(t.start_sec, t.end_sec))}</span>`
          : '';
        row.innerHTML = cells + timeCell;
        body.appendChild(row);
      });
    }
    card.appendChild(body);
    list.appendChild(card);
  });
  container.appendChild(list);

  if (visibleSegments.length === 0) {
    container.innerHTML = `<div class="empty"><p>No ${unit}s extracted.</p></div>`;
  }

  // Copy button
  const copyBtn = $('#sg-copy-btn');
  copyBtn.style.display = '';
  if (isRaw) {
    copyBtn.textContent = 'Copy text';
    copyBtn.onclick = () => {
      const text = segments.map(s => s.raw_text || '').join('\n\n---\n\n');
      copyToClipboard(text).then(() => toast('Text copied!', 'success'));
    };
  } else {
    copyBtn.textContent = 'Copy JSON';
    copyBtn.onclick = () => {
      const json = isTemporal
        ? JSON.stringify({ segments: segments.map(s => ({start: s.start, end: s.end, triplets: s.triplets})) }, null, 2)
        : JSON.stringify({ triplets: segments[0]?.triplets || [] }, null, 2);
      copyToClipboard(json).then(() => toast('JSON copied!', 'success'));
    };
  }
}

$('#sg-normalize-btn').addEventListener('click', async () => {
  if (!_sgLastResult || !_sgLastResult.segments) return;
  const btn = $('#sg-normalize-btn');
  btn.disabled = true;
  btn.textContent = 'Refining…';
  setStatus('sg', 'Refining results…');
  try {
    const body = JSON.stringify({ segments: _sgLastResult.segments });
    const r = await fetch('/api/scenegraph/normalize', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body,
    });
    const j = await r.json();
    if (!r.ok) throw new Error(j.error || 'Submission failed');
    const result = await pollTask(j.task_id, s => setStatus('sg', `Status: ${s}…`));
    setStatus('sg', null);
    // Re-render with normalized segments, preserving overlay metadata so the
    // existing player and download links stay valid.
    const merged = {
      ..._sgLastResult,
      segments: result.segments || [],
      normalized: true,
    };
    renderSceneGraph(merged, _sgLastTaskId);
    toast('Results refined.', 'success');
  } catch (err) {
    setStatus('sg', err.message, 'error');
    btn.disabled = false;
    btn.textContent = 'Refine results';
  }
});

function esc(s) {
  return String(s)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;')
    .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

/* ══════════════════════════════════════════════════════════════════
   VALIDATE
══════════════════════════════════════════════════════════════════ */

async function loadDatabasesIntoSelect(selId) {
  try {
    const r = await fetch('/api/databases');
    const dbs = await r.json();
    const sel = $(selId);
    const cur = sel.value;
    sel.innerHTML = '<option value="">default</option>';
    dbs.forEach(db => {
      const opt = document.createElement('option');
      opt.value = db.name;
      opt.textContent = `${db.name} (${db.count})`;
      sel.appendChild(opt);
    });
    if (cur) sel.value = cur;
  } catch (_) {}
}

$('#val-submit').addEventListener('click', async () => {
  const file = $('#val-file').files[0];
  const text = $('#val-text').value.trim();
  if (!file && !text) { toast('Provide a file or text.', 'error'); return; }

  const btn = $('#val-submit');
  btn.disabled = true;
  setStatus('val', 'Submitting…');
  $('#val-result').innerHTML = '';

  try {
    // Handle prompt override / permanent save
    const detailsOpen = $('#val-prompt-details').open;
    const sysTa  = $('#val-prompt-system-ta');
    const userTa = $('#val-prompt-user-ta');
    const sysModified  = detailsOpen && sysTa.value  !== _valPrompt.serverSys;
    const userModified = detailsOpen && userTa.value !== _valPrompt.serverUser;
    const isModified = sysModified || userModified;
    const saveMode = document.querySelector('input[name="val-prompt-save"]:checked').value;

    if (isModified && saveMode === 'permanent') {
      const body = {};
      if (sysModified)  body.system = sysTa.value;
      if (userModified) body.user   = userTa.value;
      const pr = await fetch('/api/prompts/validation', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      if (pr.ok) {
        const pd = await pr.json();
        _promptCache['validation'].template = pd.template;
        _promptCache['validation'].is_custom = true;
        _valPrompt.serverSys  = pd.template.system;
        _valPrompt.serverUser = pd.template.user;
        updateValPromptModifiedBadge();
        $('#val-prompt-reset').style.display = '';
        toast('Prompt saved as default.', 'success');
      }
    }

    const fd = new FormData();
    if (file) fd.append('file', file);
    if (text) fd.append('text', text);
    const db = $('#val-database').value;
    if (db) fd.append('database', db);
    const topk = $('#val-topk').value;
    if (topk) fd.append('top_k', topk);
    if (isModified && saveMode === 'once') {
      if (sysModified)  fd.append('system_prompt_override', sysTa.value);
      if (userModified) fd.append('user_prompt_override',   userTa.value);
    }

    const r = await fetch('/api/validate', { method: 'POST', body: fd });
    const j = await r.json();
    if (!r.ok) throw new Error(j.error || 'Submission failed');

    setStatus('val', 'Analysing… (this may take a minute)');
    const result = await pollTask(j.task_id, s => setStatus('val', `Status: ${s}…`));
    setStatus('val', null);
    renderValidation(result);
  } catch (err) {
    setStatus('val', err.message, 'error');
  } finally {
    btn.disabled = false;
  }
});

function renderValidation(result) {
  const container = $('#val-result');
  container.innerHTML = '';

  // Summary chips
  const chips = document.createElement('div');
  chips.className = 'chips';
  chips.innerHTML =
    `<span class="chip">Database: <strong>${esc(result.database)}</strong></span>` +
    `<span class="chip"><strong>${result.num_facts_found}</strong> fact${result.num_facts_found !== 1 ? 's' : ''} retrieved</span>`;
  container.appendChild(chips);

  // Report
  const report = document.createElement('div');
  report.className = 'report-text';
  report.textContent = result.report || '(No report generated)';
  container.appendChild(report);

  // Retrieved facts (collapsible)
  if (result.retrieved_facts && result.retrieved_facts.length > 0) {
    const det = document.createElement('details');
    det.innerHTML = `<summary>Retrieved facts (${result.retrieved_facts.length})</summary>`;
    const inner = document.createElement('div');
    inner.className = 'inner facts-list';
    result.retrieved_facts.forEach((f, i) => {
      const item = document.createElement('div');
      item.className = 'fact-item';
      item.innerHTML = `<span class="fact-index">${i + 1}.</span><span>${esc(f)}</span>`;
      inner.appendChild(item);
    });
    det.appendChild(inner);
    container.appendChild(det);
  }

  // Transcript (collapsible)
  if (result.transcript) {
    const det = document.createElement('details');
    det.innerHTML = `<summary>Audio transcript</summary><div class="inner">${esc(result.transcript)}</div>`;
    container.appendChild(det);
  }
}

/* ══════════════════════════════════════════════════════════════════
   DATABASES
══════════════════════════════════════════════════════════════════ */

let _selectedDb = '';
let _selectedFacts = new Set();

async function refreshDatabaseSelector() {
  try {
    const r = await fetch('/api/databases');
    const dbs = await r.json();
    const sel = $('#db-selector');
    const prev = sel.value;
    sel.innerHTML = '<option value="">— select —</option>';
    dbs.forEach(db => {
      const opt = document.createElement('option');
      opt.value = db.name;
      opt.textContent = `${db.name} (${db.count} facts)`;
      sel.appendChild(opt);
    });
    // Restore selection
    if (prev && dbs.find(d => d.name === prev)) sel.value = prev;
    onDbSelect(sel.value);
  } catch (e) {
    toast('Failed to load databases: ' + e.message, 'error');
  }
}

$('#db-selector').addEventListener('change', e => onDbSelect(e.target.value));

function onDbSelect(name) {
  _selectedDb = name;
  _selectedFacts.clear();
  updateDelBtn();
  const delDbBtn = $('#db-delete-btn');
  if (name) {
    delDbBtn.style.display = '';
    loadFacts(name);
  } else {
    delDbBtn.style.display = 'none';
    $('#db-facts-container').innerHTML = '<div class="empty"><p>Select a database to browse facts.</p></div>';
  }
}

async function loadFacts(dbName, query = '') {
  const container = $('#db-facts-container');
  container.innerHTML = '<div class="empty"><p>Loading…</p></div>';
  try {
    const url = query
      ? `/api/databases/${encodeURIComponent(dbName)}/facts?limit=50&query=${encodeURIComponent(query)}`
      : `/api/databases/${encodeURIComponent(dbName)}/facts?limit=50`;
    const r = await fetch(url);
    const data = await r.json();
    renderFacts(data.facts || [], data.total);
  } catch (e) {
    container.innerHTML = `<div class="empty"><p style="color:var(--error)">${esc(e.message)}</p></div>`;
  }
}

function renderFacts(facts, total) {
  const container = $('#db-facts-container');
  _selectedFacts.clear();
  updateDelBtn();

  if (facts.length === 0) {
    container.innerHTML = '<div class="empty"><p>No facts in this database.</p></div>';
    return;
  }

  const hdr = document.createElement('p');
  hdr.style.cssText = 'font-size:.78rem;color:var(--text-dim);margin-bottom:8px';
  hdr.textContent = `Showing ${facts.length} of ${total}`;

  const table = document.createElement('table');
  table.className = 'facts-table';
  table.innerHTML = `
    <thead>
      <tr>
        <th style="width:28px"><input type="checkbox" id="db-check-all"></th>
        <th>Fact</th>
        <th style="width:110px">ID / Tags</th>
      </tr>
    </thead>
    <tbody id="db-facts-tbody"></tbody>
  `;

  const tbody = table.querySelector('#db-facts-tbody');
  facts.forEach(f => {
    const tr = document.createElement('tr');
    const tags = f.metadata?.tags || '';
    tr.innerHTML = `
      <td><input type="checkbox" class="fact-check" data-id="${esc(f.id)}"></td>
      <td>${esc(f.fact)}</td>
      <td>
        <div class="fact-id" title="${esc(f.id)}">${esc(f.id)}</div>
        ${tags ? `<span class="fact-tag">${esc(tags)}</span>` : ''}
      </td>
    `;
    tbody.appendChild(tr);
  });

  container.innerHTML = '';
  container.appendChild(hdr);
  container.appendChild(table);

  // Check-all toggle
  table.querySelector('#db-check-all').addEventListener('change', e => {
    $$('.fact-check', table).forEach(cb => {
      cb.checked = e.target.checked;
      if (e.target.checked) _selectedFacts.add(cb.dataset.id);
      else _selectedFacts.delete(cb.dataset.id);
    });
    updateDelBtn();
  });

  // Individual checkboxes
  $$('.fact-check', table).forEach(cb => {
    cb.addEventListener('change', () => {
      if (cb.checked) _selectedFacts.add(cb.dataset.id);
      else _selectedFacts.delete(cb.dataset.id);
      updateDelBtn();
    });
  });
}

function updateDelBtn() {
  const btn = $('#db-del-selected-btn');
  if (_selectedFacts.size > 0) {
    btn.style.display = '';
    btn.textContent = `Delete ${_selectedFacts.size} selected`;
  } else {
    btn.style.display = 'none';
  }
}

// Search
$('#db-search-btn').addEventListener('click', () => {
  if (!_selectedDb) return;
  loadFacts(_selectedDb, $('#db-search').value.trim());
});
$('#db-search').addEventListener('keydown', e => {
  if (e.key === 'Enter' && _selectedDb) loadFacts(_selectedDb, e.target.value.trim());
});
$('#db-search-clear-btn').addEventListener('click', () => {
  $('#db-search').value = '';
  if (_selectedDb) loadFacts(_selectedDb);
});

// Refresh button
$('#db-refresh-btn').addEventListener('click', refreshDatabaseSelector);

// Delete selected facts
$('#db-del-selected-btn').addEventListener('click', async () => {
  if (!_selectedDb || _selectedFacts.size === 0) return;
  if (!confirm(`Delete ${_selectedFacts.size} fact(s) from "${_selectedDb}"?`)) return;
  try {
    const r = await fetch(`/api/databases/${encodeURIComponent(_selectedDb)}/facts`, {
      method: 'DELETE',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify([..._selectedFacts]),
    });
    if (!r.ok) throw new Error((await r.json()).error || 'Delete failed');
    toast(`Deleted ${_selectedFacts.size} fact(s).`, 'success');
    loadFacts(_selectedDb);
  } catch (e) {
    toast('Error: ' + e.message, 'error');
  }
});

// Delete entire database
$('#db-delete-btn').addEventListener('click', async () => {
  if (!_selectedDb) return;
  if (!confirm(`Delete database "${_selectedDb}" and all its facts? This cannot be undone.`)) return;
  try {
    const r = await fetch(`/api/databases/${encodeURIComponent(_selectedDb)}`, { method: 'DELETE' });
    if (!r.ok) throw new Error((await r.json()).error || 'Delete failed');
    toast(`Database "${_selectedDb}" deleted.`, 'success');
    refreshDatabaseSelector();
  } catch (e) {
    toast('Error: ' + e.message, 'error');
  }
});

// Add facts
$('#db-add-btn').addEventListener('click', async () => {
  if (!_selectedDb) { toast('Select a database first.', 'error'); return; }

  const factsText = $('#db-facts-text').value;
  const factsFile = $('#db-facts-file').files[0];
  if (!factsText && !factsFile) { toast('Enter facts text or upload a file.', 'error'); return; }

  const btn = $('#db-add-btn');
  btn.disabled = true;
  setStatus('db-add', 'Embedding and adding facts…');

  try {
    const fd = new FormData();
    if (factsFile) fd.append('file', factsFile);
    else fd.append('facts_text', factsText);
    fd.append('tags', $('#db-tags').value);
    fd.append('source', $('#db-source').value || 'user');

    const r = await fetch(`/api/databases/${encodeURIComponent(_selectedDb)}/facts`, {
      method: 'POST', body: fd,
    });
    const j = await r.json();
    if (!r.ok) throw new Error(j.error || 'Submission failed');

    setStatus('db-add', 'Embedding…');
    const result = await pollTask(j.task_id, s => setStatus('db-add', `Status: ${s}…`));
    setStatus('db-add', `Added ${result.added} fact(s).`, 'success');
    toast(`Added ${result.added} fact(s) to "${_selectedDb}".`, 'success');

    // Clear form
    $('#db-facts-text').value = '';
    $('#db-facts-file').value = '';
    $('#db-file-name').textContent = '';
    $('#db-file-name').classList.remove('visible');

    loadFacts(_selectedDb);
    refreshDatabaseSelector();
  } catch (e) {
    setStatus('db-add', e.message, 'error');
  } finally {
    btn.disabled = false;
  }
});

// Create database modal
$('#db-create-btn').addEventListener('click', () => {
  $('#new-db-name').value = '';
  $('#new-db-modal').classList.add('open');
  setTimeout(() => $('#new-db-name').focus(), 50);
});
$('#new-db-cancel').addEventListener('click', () => $('#new-db-modal').classList.remove('open'));
$('#new-db-modal').addEventListener('click', e => {
  if (e.target === $('#new-db-modal')) $('#new-db-modal').classList.remove('open');
});
$('#new-db-confirm').addEventListener('click', async () => {
  const name = $('#new-db-name').value.trim();
  if (!name) { toast('Enter a name.', 'error'); return; }
  try {
    const r = await fetch('/api/databases', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name }),
    });
    const j = await r.json();
    if (!r.ok) throw new Error(j.error || 'Failed');
    toast(`Database "${name}" created.`, 'success');
    $('#new-db-modal').classList.remove('open');
    refreshDatabaseSelector();
    // Auto-select the new database
    setTimeout(() => { $('#db-selector').value = name; onDbSelect(name); }, 300);
  } catch (e) {
    toast('Error: ' + e.message, 'error');
  }
});
$('#new-db-name').addEventListener('keydown', e => {
  if (e.key === 'Enter') $('#new-db-confirm').click();
});

/* ══════════════════════════════════════════════════════════════════
   INLINE PROMPT EDITORS
══════════════════════════════════════════════════════════════════ */

let _promptCache = {};  // name → {template, is_custom, ...}

async function initPromptCache() {
  try {
    const r = await fetch('/api/prompts');
    if (!r.ok) return;
    const list = await r.json();
    list.forEach(p => { _promptCache[p.name] = p; });
  } catch (_) {}
}

/* ── Scene Graph inline prompt ──────────────────────────────────── */

let _sgPrompt = { name: '', serverSys: '', serverUser: '' };

function getSgPromptName() {
  const file = $('#sg-file').files[0];
  const mode = document.querySelector('input[name="sg-mode"]:checked').value;
  if (!file) return `scenegraph_text_${mode}`;
  // Video files use the video-segment quintuple prompt; images use the visual triplet prompt.
  const isVideo = (file.type && file.type.startsWith('video/'))
    || /\.(mp4|mov|m4v|webm|mkv|avi|flv|wmv|mpg|mpeg|3gp)$/i.test(file.name);
  return isVideo ? `scenegraph_video_${mode}` : `scenegraph_visual_${mode}`;
}

function loadSgPromptFromCache() {
  const name = getSgPromptName();
  if (_sgPrompt.name === name) return;
  const p = _promptCache[name];
  if (!p) return;
  _sgPrompt.name = name;
  _sgPrompt.serverSys  = p.template.system;
  _sgPrompt.serverUser = p.template.user;
  $('#sg-prompt-slot').textContent = name;
  $('#sg-prompt-system-ta').value = p.template.system;
  $('#sg-prompt-user-ta').value   = p.template.user;
  $('#sg-prompt-reset').style.display = p.is_custom ? '' : 'none';
  updateSgPromptModifiedBadge();
}

function updateSgPromptModifiedBadge() {
  const modified = $('#sg-prompt-system-ta').value !== _sgPrompt.serverSys
                 || $('#sg-prompt-user-ta').value !== _sgPrompt.serverUser;
  $('#sg-prompt-modified').style.display = modified ? '' : 'none';
}

$$('input[name="sg-mode"]').forEach(r => r.addEventListener('change', () => {
  _sgPrompt.name = '';
  loadSgPromptFromCache();
}));

$('#sg-file').addEventListener('change', () => {
  _sgPrompt.name = '';
  loadSgPromptFromCache();
});

$('#sg-prompt-system-ta').addEventListener('input', updateSgPromptModifiedBadge);
$('#sg-prompt-user-ta').addEventListener('input', updateSgPromptModifiedBadge);

$('#sg-prompt-reset').addEventListener('click', async () => {
  if (!_sgPrompt.name || !confirm(`Reset "${_sgPrompt.name}" to built-in default?`)) return;
  try {
    const r = await fetch(`/api/prompts/${encodeURIComponent(_sgPrompt.name)}`, { method: 'DELETE' });
    if (!r.ok) throw new Error('Reset failed');
    const data = await r.json();
    if (_promptCache[_sgPrompt.name]) {
      _promptCache[_sgPrompt.name].template = data.template;
      _promptCache[_sgPrompt.name].is_custom = false;
    }
    _sgPrompt.serverSys  = data.template.system;
    _sgPrompt.serverUser = data.template.user;
    $('#sg-prompt-system-ta').value = data.template.system;
    $('#sg-prompt-user-ta').value   = data.template.user;
    $('#sg-prompt-reset').style.display = 'none';
    updateSgPromptModifiedBadge();
    toast('Prompt reset to default.', 'success');
  } catch (e) {
    toast('Error: ' + e.message, 'error');
  }
});

/* ── Validate inline prompt ─────────────────────────────────────── */

let _valPrompt = { serverSys: '', serverUser: '' };

function loadValPromptFromCache() {
  const p = _promptCache['validation'];
  if (!p) return;
  _valPrompt.serverSys  = p.template.system;
  _valPrompt.serverUser = p.template.user;
  $('#val-prompt-slot').textContent = 'validation';
  $('#val-prompt-system-ta').value = p.template.system;
  $('#val-prompt-user-ta').value   = p.template.user;
  $('#val-prompt-reset').style.display = p.is_custom ? '' : 'none';
  updateValPromptModifiedBadge();
}

function updateValPromptModifiedBadge() {
  const modified = $('#val-prompt-system-ta').value !== _valPrompt.serverSys
                 || $('#val-prompt-user-ta').value !== _valPrompt.serverUser;
  $('#val-prompt-modified').style.display = modified ? '' : 'none';
}

$('#val-prompt-system-ta').addEventListener('input', updateValPromptModifiedBadge);
$('#val-prompt-user-ta').addEventListener('input', updateValPromptModifiedBadge);

$('#val-prompt-reset').addEventListener('click', async () => {
  if (!confirm('Reset "validation" prompt to built-in default?')) return;
  try {
    const r = await fetch('/api/prompts/validation', { method: 'DELETE' });
    if (!r.ok) throw new Error('Reset failed');
    const data = await r.json();
    if (_promptCache['validation']) {
      _promptCache['validation'].template = data.template;
      _promptCache['validation'].is_custom = false;
    }
    _valPrompt.serverSys  = data.template.system;
    _valPrompt.serverUser = data.template.user;
    $('#val-prompt-system-ta').value = data.template.system;
    $('#val-prompt-user-ta').value   = data.template.user;
    $('#val-prompt-reset').style.display = 'none';
    updateValPromptModifiedBadge();
    toast('Prompt reset to default.', 'success');
  } catch (e) {
    toast('Error: ' + e.message, 'error');
  }
});

/* ── Bootstrap ──────────────────────────────────────────────────── */
initPromptCache().then(() => {
  loadSgPromptFromCache();
  loadValPromptFromCache();
});
