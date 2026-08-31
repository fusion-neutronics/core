// Pure-JS viewer for yamc interactive geometry plots.
//
// Extracted from yamc-plot/src/viewer_html.rs so it can be edited as
// real JavaScript (and reused by the browser-side editor). The Rust
// side of `build_interactive_html` declares these globals *before*
// this file is included:
//   const GEOMETRY_JSON, GEOMETRY_KIND, CUSTOM_COLORS, CELL_NAMES,
//   MATERIAL_NAMES, DISCRETE_COLORS, SOURCE_DIST, BBOX_WIDTHS,
//   PRESAMPLED_B64 (+ PRESAMPLED_PH / PRESAMPLED_PV when non-null),
//   INITIAL, CSG_WORKER_SOURCE,
//   YAMT_WASM_BASE64 + yamtWasmInitSync (mesh path only).
// After this file is included, the Rust side appends `init();`.
//
// Everything that depends on the *specific model* lives in those
// globals; the code here is model-independent.

// Unit scale factors (internal units are cm)
function unitScale(u) { return {"mm":10,"cm":1,"m":0.01,"km":0.00001}[u]||1; }

// Compute pixel dimensions from total pixel count and width aspect ratio
function computePixelDims(total, wH, wV) {
  const aspect = wH / wV;
  const pV = Math.max(1, Math.round(Math.sqrt(total / aspect)));
  const pH = Math.max(1, Math.round(total / pV));
  return [pH, pV];
}

let plotter = null;
let wasmInstance = null;
let rendering = false;
let pendingRender = false;

// State
let state = { ...INITIAL };

// DOM refs
const canvas = document.getElementById('plot-canvas');
const ctx = canvas.getContext('2d');
const container = document.getElementById('canvas-container');
const tooltip = document.getElementById('tooltip');
const status = document.getElementById('status');
const legend = document.getElementById('legend');

// Color mapping
let colorIndex = {};
let idColors = {};

function getColor(id, colorBy) {
  const key = colorBy + ':' + id;
  if (idColors[key]) return idColors[key];
  // Check custom colors
  if (CUSTOM_COLORS[id]) {
    idColors[key] = parseColor(CUSTOM_COLORS[id]);
    return idColors[key];
  }
  // Auto-assign
  if (colorIndex[key] === undefined) {
    colorIndex[key] = Object.keys(colorIndex).length;
  }
  const hex = DISCRETE_COLORS[colorIndex[key] % DISCRETE_COLORS.length];
  idColors[key] = parseColor(hex);
  return idColors[key];
}

function parseColor(c) {
  // Parse hex #rrggbb to [r,g,b]
  if (c.startsWith('#') && c.length === 7) {
    return [parseInt(c.slice(1,3),16), parseInt(c.slice(3,5),16), parseInt(c.slice(5,7),16)];
  }
  // Parse rgb(r,g,b)
  const m = c.match(/rgb\((\d+),\s*(\d+),\s*(\d+)\)/);
  if (m) return [+m[1], +m[2], +m[3]];
  // Fallback: use a canvas to parse named colors
  const tmp = document.createElement('canvas'); tmp.width=1; tmp.height=1;
  const tc = tmp.getContext('2d'); tc.fillStyle = c; tc.fillRect(0,0,1,1);
  const d = tc.getImageData(0,0,1,1).data;
  return [d[0], d[1], d[2]];
}

function renderGrid(data, ph, pv) {
  canvas.width = ph;
  canvas.height = pv;
  const img = ctx.createImageData(ph, pv);
  const d = img.data;
  const colorBy = state.colorBy;
  const outlineMode = state.outline;
  const doOutline = outlineMode !== 'none';

  // Build outline lookup if needed
  let outlineIds = null;
  if (doOutline) {
    outlineIds = new Int32Array(ph * pv);
    for (let i = 0; i < pv; i++) {
      for (let j = 0; j < ph; j++) {
        const idx = (i * ph + j) * 2;
        outlineIds[i * ph + j] = outlineMode === 'cell' ? data[idx] : data[idx + 1];
      }
    }
  }

  const seenIds = new Set();
  for (let i = 0; i < pv; i++) {
    for (let j = 0; j < ph; j++) {
      const idx = (i * ph + j) * 2;
      const cellId = data[idx];
      const matId = data[idx + 1];
      const pixIdx = (i * ph + j) * 4;

      if (cellId === -1) {
        // Outside geometry -- always transparent
        d[pixIdx] = 0; d[pixIdx+1] = 0; d[pixIdx+2] = 0; d[pixIdx+3] = 0;
        continue;
      }

      const displayId = colorBy === 'cell' ? cellId : matId;
      seenIds.add(displayId + ':' + cellId + ':' + matId);

      // Check if this pixel is on an outline boundary
      if (doOutline) {
        const myOutlineId = outlineIds[i * ph + j];
        let isEdge = false;
        const ow = state.outlineWidth;
        for (let d2 = 1; d2 <= ow && !isEdge; d2++) {
          if (j >= d2 && outlineIds[i * ph + j - d2] !== myOutlineId) isEdge = true;
          if (!isEdge && j + d2 < ph && outlineIds[i * ph + j + d2] !== myOutlineId) isEdge = true;
          if (!isEdge && i >= d2 && outlineIds[(i-d2) * ph + j] !== myOutlineId) isEdge = true;
          if (!isEdge && i + d2 < pv && outlineIds[(i+d2) * ph + j] !== myOutlineId) isEdge = true;
        }
        if (isEdge) {
          const oc = state.outlineColor;
          d[pixIdx] = oc[0]; d[pixIdx+1] = oc[1]; d[pixIdx+2] = oc[2]; d[pixIdx+3] = 255;
          continue;
        }
      }

      const rgb = getColor(displayId, colorBy);
      d[pixIdx] = rgb[0]; d[pixIdx+1] = rgb[1]; d[pixIdx+2] = rgb[2]; d[pixIdx+3] = 255;
    }
  }

  ctx.putImageData(img, 0, 0);
  renderSource(ph, pv);
  fitCanvas();
  updateLegend(data, ph, pv);
  updateAxisLabels();
  status.textContent = '';
  document.getElementById('pixel-dims').textContent = 'Grid: ' + ph + ' \u00d7 ' + pv + ' px (PNG size)';
}

function renderSource(ph, pv) {
  if (!SOURCE_DIST || !document.getElementById('source-show').checked) return;
  const tolerance = +document.getElementById('source-tolerance').value || 1.0;
  const maxSamples = +document.getElementById('source-samples').value || 1000;
  const markerSize = +document.getElementById('source-size').value || 3;
  const colorHex = document.getElementById('source-color').value;
  const sc = parseColor(colorHex);
  ensureSourceSamples(maxSamples);
  if (!sourcePoints || sourcePoints.length === 0) return;

  // Slice coordinate (the axis perpendicular to the plot plane)
  const sliceCoord = state.basis === 'xy' ? state.originZ : (state.basis === 'xz' ? state.originY : state.originX);

  ctx.fillStyle = colorHex;
  let count = 0;
  const n = sourcePoints.length;
  for (let k = 0; k < n; k++) {
    const pt = sourcePoints[k];
    const x = pt[0], y = pt[1], z = pt[2];

    // Check distance from slice plane
    let dist;
    if (state.basis === 'xy') dist = Math.abs(z - sliceCoord);
    else if (state.basis === 'xz') dist = Math.abs(y - sliceCoord);
    else dist = Math.abs(x - sliceCoord);
    if (dist > tolerance) continue;

    // Convert to pixel coordinates
    let h, v;
    if (state.basis === 'xy') { h = x; v = y; }
    else if (state.basis === 'xz') { h = x; v = z; }
    else { h = y; v = z; }

    const fracX = (h - state.originH() + state.widthH / 2) / state.widthH;
    const fracY = (v - state.originV() + state.widthV / 2) / state.widthV;
    if (fracX < 0 || fracX > 1 || fracY < 0 || fracY > 1) continue;

    const px = fracX * ph;
    const py = (1 - fracY) * pv;
    ctx.beginPath();
    ctx.arc(px, py, markerSize, 0, 2 * Math.PI);
    ctx.fill();
    count++;
  }
}

function fitCanvas() {
  const cw = container.clientWidth;
  const ch = container.clientHeight;
  if (!canvas.width || !canvas.height) return;
  const aspect = canvas.width / canvas.height;
  // padLeft has to fit (axis label rotated 90° ≈ 18px) + gap + tick text
  // (~4 chars at ~17px = ~50px) + extra. 50 was too tight -- the axis
  // label landed at canvasLeft − 76, i.e. negative coords when canvasLeft
  // ≈ padLeft, and got clipped by the container's `overflow: hidden`.
  const padLeft = 80;
  const padBottom = 50; // space for horizontal ticks + axis label
  let dispW, dispH;
  if (cw / ch > aspect) {
    dispH = ch - padBottom;
    dispW = (ch - padBottom) * aspect;
  } else {
    dispW = cw - padLeft;
    dispH = (cw - padLeft) / aspect;
  }
  if (dispW < 10) dispW = 10;
  if (dispH < 10) dispH = 10;
  canvas.style.width = dispW + 'px';
  canvas.style.height = dispH + 'px';
  // Left-align next to the controls sidebar -- when the window is wide and
  // the canvas is aspect-fit (height-limited for square models), centring
  // used to leave a big gray gap between the controls and the plot.
  canvas.style.left = padLeft + 'px';
  canvas.style.top = ((ch - padBottom - dispH) / 2) + 'px';
  canvas.style.border = '1px solid #999';
  updateAxisLabels();
}

function updateLegend(data, ph, pv) {
  const seen = new Map();
  for (let i = 0; i < pv; i += 3) {
    for (let j = 0; j < ph; j += 3) {
      const idx = (i * ph + j) * 2;
      const cellId = data[idx]; const matId = data[idx + 1];
      if (cellId === -1) continue;
      const displayId = state.colorBy === 'cell' ? cellId : matId;
      if (!seen.has(displayId)) {
        const name = state.colorBy === 'cell'
          ? (CELL_NAMES[cellId] || 'Cell ' + cellId)
          : (matId === -1 ? 'Void' : (MATERIAL_NAMES[matId] || 'Material ' + matId));
        seen.set(displayId, name);
      }
    }
  }
  legend.innerHTML = '';
  const sorted = [...seen.entries()].sort((a,b) => a[0] - b[0]);
  for (const [id, name] of sorted) {
    const rgb = getColor(id, state.colorBy);
    const el = document.createElement('span');
    el.className = 'legend-item';
    const swatch = document.createElement('span');
    swatch.className = 'legend-swatch';
    swatch.style.background = 'rgb('+rgb[0]+','+rgb[1]+','+rgb[2]+')';
    swatch.style.cursor = 'pointer';
    swatch.title = 'Click to change color';
    // Hidden color input
    const picker = document.createElement('input');
    picker.type = 'color';
    picker.value = '#' + rgb.map(c => c.toString(16).padStart(2,'0')).join('');
    picker.style.cssText = 'position:absolute;width:0;height:0;opacity:0;pointer-events:none;';
    swatch.addEventListener('click', () => picker.click());
    picker.addEventListener('input', ((capturedId, capturedSwatch) => (ev) => {
      const hex = ev.target.value;
      const newRgb = parseColor(hex);
      const key = state.colorBy + ':' + capturedId;
      idColors[key] = newRgb;
      capturedSwatch.style.background = hex;
      // Re-render with new color (no re-sampling needed)
      if (window._lastGridData) {
        const [pH, pV] = computePixelDims(state.totalPixels, state.widthH, state.widthV);
        renderGrid(window._lastGridData, pH, pV);
      }
    })(id, swatch));
    el.appendChild(swatch);
    el.appendChild(picker);
    el.appendChild(document.createTextNode(' ' + name));
    legend.appendChild(el);
  }
}

function updateAxisLabels() {
  const u = state.units;
  const us = unitScale(u);
  const basis = state.basis;
  const fontSize = 18;
  const tickFontSize = Math.max(6, fontSize - 1);
  const hLabel = basis === 'yz' ? 'Y' : 'X';
  const vLabel = basis === 'xy' ? 'Y' : 'Z';
  const hLabelEl = document.getElementById('axis-label-h');
  const vLabelEl = document.getElementById('axis-label-v');
  hLabelEl.textContent = hLabel + ' (' + u + ')';
  vLabelEl.textContent = vLabel + ' (' + u + ')';
  hLabelEl.style.fontSize = fontSize + 'px';
  vLabelEl.style.fontSize = fontSize + 'px';

  // Compute canvas display rect
  const rect = canvas.getBoundingClientRect();
  const cRect = container.getBoundingClientRect();
  const canvasLeft = rect.left - cRect.left;
  const canvasTop = rect.top - cRect.top;
  const canvasW = rect.width;
  const canvasH = rect.height;
  if (canvasW < 1 || canvasH < 1) return;

  // Position axis labels centered on the canvas
  hLabelEl.style.left = (canvasLeft + canvasW / 2) + 'px';
  hLabelEl.style.transform = 'translateX(-50%)';
  hLabelEl.style.top = (canvasTop + canvasH + tickFontSize + 8) + 'px';
  vLabelEl.style.left = (canvasLeft - tickFontSize * 4 - 8) + 'px';
  vLabelEl.style.top = (canvasTop + canvasH / 2) + 'px';

  // World extents (in internal cm, then scaled to display units)
  const hMin = (state.originH() - state.widthH / 2) * us;
  const hMax = (state.originH() + state.widthH / 2) * us;
  const vMin = (state.originV() - state.widthV / 2) * us;
  const vMax = (state.originV() + state.widthV / 2) * us;

  // Generate nice ticks
  const hTicks = niceTicks(hMin, hMax, Math.floor(canvasW / 80));
  const vTicks = niceTicks(vMin, vMax, Math.floor(canvasH / 60));

  // Horizontal ticks (along bottom of canvas)
  const hContainer = document.getElementById('tick-container-h');
  hContainer.innerHTML = '';
  for (const val of hTicks) {
    const frac = (val - hMin) / (hMax - hMin);
    if (frac < 0.02 || frac > 0.98) continue;
    // Tick mark line
    const mark = document.createElement('span');
    mark.style.cssText = 'position:absolute;width:1px;height:5px;background:#666;';
    mark.style.left = (canvasLeft + frac * canvasW) + 'px';
    mark.style.top = (canvasTop + canvasH + 1) + 'px';
    hContainer.appendChild(mark);
    // Tick label
    const el = document.createElement('span');
    el.className = 'tick-h';
    el.textContent = formatTick(val);
    el.style.fontSize = tickFontSize + 'px';
    el.style.left = (canvasLeft + frac * canvasW) + 'px';
    el.style.top = (canvasTop + canvasH + 7) + 'px';
    hContainer.appendChild(el);
  }

  // Vertical ticks (along left of canvas)
  const vContainer = document.getElementById('tick-container-v');
  vContainer.innerHTML = '';
  for (const val of vTicks) {
    const frac = (val - vMin) / (vMax - vMin);
    if (frac < 0.02 || frac > 0.98) continue;
    // Tick mark line
    const mark = document.createElement('span');
    mark.style.cssText = 'position:absolute;height:1px;width:5px;background:#666;';
    mark.style.left = (canvasLeft - 6) + 'px';
    mark.style.top = (canvasTop + (1 - frac) * canvasH) + 'px';
    vContainer.appendChild(mark);
    // Tick label
    const el = document.createElement('span');
    el.className = 'tick-v';
    el.textContent = formatTick(val);
    el.style.fontSize = tickFontSize + 'px';
    el.style.left = '0px';
    el.style.width = (canvasLeft - 8) + 'px';
    el.style.top = (canvasTop + (1 - frac) * canvasH) + 'px';
    vContainer.appendChild(el);
  }
}

// Helper: get the horizontal-axis origin component for current basis
state.originH = function() {
  return state.basis === 'yz' ? state.originY : state.originX;
};
state.originV = function() {
  return state.basis === 'xy' ? state.originY : state.originZ;
};

// Compute nice tick values for a range
function niceTicks(lo, hi, maxTicks) {
  if (maxTicks < 2) maxTicks = 2;
  const range = hi - lo;
  if (range <= 0) return [lo];
  const rough = range / maxTicks;
  const pow10 = Math.pow(10, Math.floor(Math.log10(rough)));
  let step;
  const r = rough / pow10;
  if (r < 1.5) step = pow10;
  else if (r < 3.5) step = 2 * pow10;
  else if (r < 7.5) step = 5 * pow10;
  else step = 10 * pow10;
  const start = Math.ceil(lo / step) * step;
  const ticks = [];
  for (let v = start; v <= hi + step * 0.001; v += step) {
    ticks.push(v);
  }
  return ticks;
}

function formatTick(v) {
  if (Math.abs(v) < 1e-10) return '0';
  const av = Math.abs(v);
  if (av >= 1000 || av < 0.01) return v.toExponential(1);
  // Remove trailing zeros
  return parseFloat(v.toPrecision(6)).toString();
}

async function sample() {
  // Mesh-filled CSG cells (issue #291): this sampler works from GEOMETRY_JSON,
  // where a fill carries only an identity fingerprint, so it would draw the bare
  // CSG frame and hide the very body the particles see. The server-rendered
  // raster is the only correct view of such a model, so keep showing it and say
  // why instead of silently drawing something else.
  if (HAS_MESH_FILLS) {
    renderPresampled();
    status.textContent =
      'This model has mesh-filled cells, which the in-browser sampler cannot ' +
      'resolve. Showing the view rendered by yamc; re-run model.plot(...) with ' +
      'the origin / width / basis you want.';
    return;
  }
  if (!plotter) return;
  if (rendering) { pendingRender = true; return; }
  rendering = true;
  status.textContent = 'Sampling...';

  await new Promise(r => setTimeout(r, 0)); // yield to UI

  try {
    const [pH, pV] = computePixelDims(state.totalPixels, state.widthH, state.widthV);
    const sampleT0 = performance.now();
    const data = await plotter.sampleGrid(
      state.basis,
      state.originX, state.originY, state.originZ,
      state.widthH, state.widthV,
      pH, pV
    );
    const sampleMs = performance.now() - sampleT0;
    renderGrid(data, pH, pV);
    status.textContent = `Sampled ${pH}x${pV} (${pH*pV} px) in ${sampleMs.toFixed(1)} ms`;
  } catch (e) {
    status.textContent = 'Error: ' + e;
    console.error(e);
  }

  rendering = false;
  if (pendingRender) {
    pendingRender = false;
    sample();
  }
}

// Read state from controls
function readControls() {
  state.originX = +document.getElementById('origin-x').value;
  state.originY = +document.getElementById('origin-y').value;
  state.originZ = +document.getElementById('origin-z').value;
  state.widthH = +document.getElementById('width-h').value;
  state.widthV = +document.getElementById('width-v').value;
  state.totalPixels = +document.getElementById('total-pixels').value;
  state.colorBy = document.querySelector('input[name="color-by"]:checked').value;
  state.outline = document.querySelector('input[name="outline"]:checked').value;
  state.outlineColor = parseColor(document.getElementById('outline-color').value);
  state.outlineWidth = +document.getElementById('outline-width').value || 1;
  state.units = document.querySelector('input[name="units"]:checked').value;
  const activeBtn = document.querySelector('#basis-btns button.active');
  if (activeBtn) state.basis = activeBtn.dataset.val;
}

function writeControls() {
  document.getElementById('origin-x').value = state.originX;
  document.getElementById('origin-y').value = state.originY;
  document.getElementById('origin-z').value = state.originZ;
  document.getElementById('width-h').value = state.widthH;
  document.getElementById('width-v').value = state.widthV;
  document.getElementById('total-pixels').value = state.totalPixels;
  document.getElementById('outline-color').value = '#' + state.outlineColor.map(c => c.toString(16).padStart(2,'0')).join('');
  document.getElementById('outline-width').value = state.outlineWidth;
}

// --- Apply button: sampling controls mark dirty, Apply triggers re-sample ---
const applyBtn = document.getElementById('apply-btn');
let dirty = false;
// Snapshot of the last-sampled state (the values that require re-sampling)
let sampledSnap = { basis: state.basis, originX: state.originX, originY: state.originY, originZ: state.originZ, widthH: state.widthH, widthV: state.widthV, totalPixels: state.totalPixels };

function checkDirty() {
  const s = state;
  dirty = s.basis !== sampledSnap.basis || s.originX !== sampledSnap.originX ||
    s.originY !== sampledSnap.originY || s.originZ !== sampledSnap.originZ ||
    s.widthH !== sampledSnap.widthH || s.widthV !== sampledSnap.widthV ||
    s.totalPixels !== sampledSnap.totalPixels;
  applyBtn.classList.toggle('dirty', dirty);
}

function applySample() {
  readControls();
  sampledSnap = { basis: state.basis, originX: state.originX, originY: state.originY, originZ: state.originZ, widthH: state.widthH, widthV: state.widthV, totalPixels: state.totalPixels };
  dirty = false;
  applyBtn.classList.remove('dirty');
  sample();
}

// Sync snapshot after interactive actions (pan/zoom) that sample immediately
function syncSnap() {
  sampledSnap = { basis: state.basis, originX: state.originX, originY: state.originY, originZ: state.originZ, widthH: state.widthH, widthV: state.widthV, totalPixels: state.totalPixels };
  dirty = false;
  applyBtn.classList.remove('dirty');
}

applyBtn.addEventListener('click', () => { if (dirty) applySample(); });

// Slice plane buttons -- mark dirty
document.querySelectorAll('#basis-btns button').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('#basis-btns button').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    state.basis = btn.dataset.val;
    checkDirty();
  });
});

// Number inputs (origin, width, pixels) -- mark dirty on change
document.querySelectorAll('#controls input[type=number]').forEach(inp => {
  inp.addEventListener('change', () => { readControls(); checkDirty(); });
});

// Instant controls: color-by, outline, outline-color, outline-width, units -- no re-sampling needed
document.querySelectorAll('input[name="color-by"], input[name="outline"], input[name="units"]').forEach(inp => {
  inp.addEventListener('change', () => {
    readControls();
    if (inp.name === 'units') {
      updateAxisLabels();
      return;
    }
    if (inp.name === 'color-by') {
      // Only reset auto-assignment index; keep idColors so user-picked
      // colours survive a color-by round-trip.
      colorIndex = {};
    }
    if (window._lastGridData) {
      const [pH, pV] = computePixelDims(state.totalPixels, state.widthH, state.widthV);
      renderGrid(window._lastGridData, pH, pV);
    }
  });
});
// Outline color and width -- instant re-render
document.getElementById('outline-color').addEventListener('input', () => {
  readControls();
  if (window._lastGridData) {
    const [pH, pV] = computePixelDims(state.totalPixels, state.widthH, state.widthV);
    renderGrid(window._lastGridData, pH, pV);
  }
});
document.getElementById('outline-width').addEventListener('change', () => {
  readControls();
  if (window._lastGridData) {
    const [pH, pV] = computePixelDims(state.totalPixels, state.widthH, state.widthV);
    renderGrid(window._lastGridData, pH, pV);
  }
});
// Pixel count preview -- update dimensions text as user types
document.getElementById('total-pixels').addEventListener('input', () => {
  const t = +document.getElementById('total-pixels').value || 1;
  const [pH, pV] = computePixelDims(t, state.widthH, state.widthV);
  document.getElementById('pixel-dims').textContent = 'Grid: ' + pH + ' \u00d7 ' + pV + ' px (PNG size)';
});

// Source controls -- instant re-render (no re-sampling of geometry needed)
document.getElementById('source-show').addEventListener('change', () => {
  document.getElementById('source-options').style.display =
    document.getElementById('source-show').checked ? 'block' : 'none';
  if (window._lastGridData) {
    const [pH, pV] = computePixelDims(state.totalPixels, state.widthH, state.widthV);
    renderGrid(window._lastGridData, pH, pV);
  }
});
['source-color', 'source-size', 'source-samples', 'source-tolerance'].forEach(id => {
  const el = document.getElementById(id);
  if (el) el.addEventListener(el.type === 'color' ? 'input' : 'change', () => {
    if (window._lastGridData) {
      const [pH, pV] = computePixelDims(state.totalPixels, state.widthH, state.widthV);
      renderGrid(window._lastGridData, pH, pV);
    }
  });
});

document.getElementById('reset-btn').addEventListener('click', () => {
  Object.assign(state, INITIAL);
  writeControls();
  document.querySelectorAll('#basis-btns button').forEach(b => {
    b.classList.toggle('active', b.dataset.val === state.basis);
  });
  document.querySelector('input[name="color-by"][value="'+state.colorBy+'"]').checked = true;
  document.querySelector('input[name="outline"][value="'+state.outline+'"]').checked = true;
  document.querySelector('input[name="units"][value="'+state.units+'"]').checked = true;
  colorIndex = {};
  idColors = {};
  sampledSnap = { basis: state.basis, originX: state.originX, originY: state.originY, originZ: state.originZ, widthH: state.widthH, widthV: state.widthV, totalPixels: state.totalPixels };
  dirty = false;
  applyBtn.classList.remove('dirty');
  renderPresampled();  // Reset always matches initial state → use presampled
});

// Copy Python code to clipboard
document.getElementById('copy-btn').addEventListener('click', () => {
  const s = state;
  const o = [s.originX, s.originY, s.originZ];
  const oc_hex = '#' + s.outlineColor.map(c => c.toString(16).padStart(2,'0')).join('');

  // Pick the right object name
  const obj = SOURCE_DIST ? 'model' : (GEOMETRY_KIND === 'mesh' ? 'mesh_geometry' : 'geometry');

  const lines = [
    'plot = ' + obj + '.plot(',
    '    origin=(' + o.map(v => v.toFixed(4)).join(', ') + '),',
    '    width=(' + s.widthH.toFixed(4) + ', ' + s.widthV.toFixed(4) + '),',
    '    pixels=' + s.totalPixels + ',',
    '    basis="' + s.basis + '",',
    '    color_by="' + s.colorBy + '",',
    '    outline=' + (s.outline === 'none' ? 'None' : '"' + s.outline + '"') + ',',
    '    axis_units="' + s.units + '",',
  ];

  // Emit colors dict if any custom colours were set via legend picker
  const colorEntries = [];
  for (const [key, rgb] of Object.entries(idColors)) {
    const parts = key.split(':');
    if (parts[0] !== s.colorBy) continue;
    const id = parts[1];
    const hex = '#' + rgb.map(c => c.toString(16).padStart(2,'0')).join('');
    colorEntries.push(id + ': "' + hex + '"');
  }
  if (colorEntries.length > 0) {
    lines.push('    colors={' + colorEntries.join(', ') + '},');
  }

  // contour_kwargs -- only include if outline is not none
  if (s.outline !== 'none') {
    lines.push('    contour_kwargs={"colors": "' + oc_hex + '", "linewidths": ' + s.outlineWidth + '},');
  }

  if (SOURCE_DIST) {
    const srcShow = document.getElementById('source-show').checked;
    const srcSamples = +document.getElementById('source-samples').value || 5000;
    const srcTol = +document.getElementById('source-tolerance').value || 1.0;
    const srcColor = document.getElementById('source-color').value;
    const srcSize = +document.getElementById('source-size').value || 3;
    if (srcShow) {
      lines.push('    n_samples=' + srcSamples + ',');
      lines.push('    plane_tolerance=' + srcTol.toFixed(2) + ',');
      lines.push('    source_kwargs={"color": "' + srcColor + '", "size": ' + srcSize + '},');
    }
  }
  lines.push(')');
  const code = lines.join('\n');
  navigator.clipboard.writeText(code).then(() => {
    const btn = document.getElementById('copy-btn');
    btn.textContent = 'Copied!';
    setTimeout(() => { btn.textContent = 'Copy Python Code'; }, 1500);
  });
});

// Download PNG with axis labels and tick numbers
document.getElementById('download-btn').addEventListener('click', () => {
  const us = unitScale(state.units);
  const hMin = (state.originH() - state.widthH / 2) * us;
  const hMax = (state.originH() + state.widthH / 2) * us;
  const vMin = (state.originV() - state.widthV / 2) * us;
  const vMax = (state.originV() + state.widthV / 2) * us;
  const hLabel = (state.basis === 'yz' ? 'Y' : 'X') + ' (' + state.units + ')';
  const vLabel = (state.basis === 'xy' ? 'Y' : 'Z') + ' (' + state.units + ')';

  const plotW = canvas.width;
  const plotH = canvas.height;
  const padL = 60, padR = 20, padT = 20, padB = 50;
  const totalW = padL + plotW + padR;
  const totalH = padT + plotH + padB;

  const offscreen = document.createElement('canvas');
  offscreen.width = totalW;
  offscreen.height = totalH;
  const oc = offscreen.getContext('2d');

  // Always transparent background
  oc.clearRect(0, 0, totalW, totalH);

  // Draw plot image
  oc.drawImage(canvas, padL, padT);

  // Plot region outline
  oc.strokeStyle = '#999';
  oc.lineWidth = 1;
  oc.strokeRect(padL + 0.5, padT + 0.5, plotW - 1, plotH - 1);

  // Tick styling
  oc.fillStyle = '#222';
  oc.strokeStyle = '#666';
  oc.lineWidth = 1;

  // Compute ticks
  const pngFontSize = 18;
  const pngTickFont = Math.max(6, pngFontSize - 1);
  const hTicks = niceTicks(hMin, hMax, Math.floor(plotW / 80));
  const vTicks = niceTicks(vMin, vMax, Math.floor(plotH / 60));

  // Horizontal ticks
  oc.font = pngTickFont + 'px monospace';
  oc.textAlign = 'center';
  oc.textBaseline = 'top';
  for (const val of hTicks) {
    const frac = (val - hMin) / (hMax - hMin);
    if (frac < 0.02 || frac > 0.98) continue;
    const x = padL + frac * plotW;
    oc.beginPath(); oc.moveTo(x, padT + plotH); oc.lineTo(x, padT + plotH + 4); oc.stroke();
    oc.fillText(formatTick(val), x, padT + plotH + 6);
  }

  // Vertical ticks
  oc.textAlign = 'right';
  oc.textBaseline = 'middle';
  for (const val of vTicks) {
    const frac = (val - vMin) / (vMax - vMin);
    if (frac < 0.02 || frac > 0.98) continue;
    const y = padT + (1 - frac) * plotH;
    oc.beginPath(); oc.moveTo(padL - 4, y); oc.lineTo(padL, y); oc.stroke();
    oc.fillText(formatTick(val), padL - 6, y);
  }

  // Horizontal axis label
  oc.font = (pngFontSize + 2) + 'px sans-serif';
  oc.textAlign = 'center';
  oc.textBaseline = 'top';
  oc.fillText(hLabel, padL + plotW / 2, padT + plotH + 26);

  // Vertical axis label (rotated)
  oc.save();
  oc.translate(14, padT + plotH / 2);
  oc.rotate(-Math.PI / 2);
  oc.textAlign = 'center';
  oc.textBaseline = 'middle';
  oc.fillText(vLabel, 0, 0);
  oc.restore();

  const link = document.createElement('a');
  link.download = 'geometry_plot.png';
  link.href = offscreen.toDataURL('image/png');
  link.click();
});

// Prevent context menu on canvas so right-click drag works for pan
canvas.addEventListener('contextmenu', e => e.preventDefault());

// Interaction modes:
//   Left-click drag  = draw zoom rectangle
//   Right-click drag = pan (drag the view)
//   Scroll wheel     = move slice along perpendicular axis (1 cm per tick)
let panning = false, panStartX, panStartY, panStartState;
let zooming = false, zoomStartX, zoomStartY;

// Zoom rectangle overlay
const zoomRect = document.createElement('div');
zoomRect.style.cssText = 'position:absolute;border:2px dashed #e94560;background:rgba(233,69,96,0.1);display:none;pointer-events:none;z-index:90;';
container.appendChild(zoomRect);

canvas.addEventListener('mousedown', e => {
  if (e.button === 2) {
    // Right-click: pan
    panning = true;
    panStartX = e.clientX;
    panStartY = e.clientY;
    panStartState = { ...state };
    canvas.style.cursor = 'grabbing';
  } else if (e.button === 0) {
    // Left-click: start zoom rectangle
    zooming = true;
    const rect = canvas.getBoundingClientRect();
    zoomStartX = e.clientX;
    zoomStartY = e.clientY;
    zoomRect.style.left = (e.clientX - container.getBoundingClientRect().left) + 'px';
    zoomRect.style.top = (e.clientY - container.getBoundingClientRect().top) + 'px';
    zoomRect.style.width = '0px';
    zoomRect.style.height = '0px';
    zoomRect.style.display = 'block';
  }
});

window.addEventListener('mousemove', e => {
  if (panning) {
    const rect = canvas.getBoundingClientRect();
    const dx = (e.clientX - panStartX) / rect.width * panStartState.widthH;
    const dy = (e.clientY - panStartY) / rect.height * panStartState.widthV;
    // Move origin opposite to drag direction so the view follows the mouse.
    // Horizontal: screen-X and world-H are aligned, so origin -= dx.
    // Vertical: screen-Y and world-V are flipped (top of canvas = highest v),
    // so origin += dy to make "drag down = pan down" (drag-map behavior).
    switch (state.basis) {
      case 'xy': state.originX = panStartState.originX - dx; state.originY = panStartState.originY + dy; break;
      case 'xz': state.originX = panStartState.originX - dx; state.originZ = panStartState.originZ + dy; break;
      case 'yz': state.originY = panStartState.originY - dx; state.originZ = panStartState.originZ + dy; break;
    }
    writeControls();
    syncSnap(); sample();
  }
  if (zooming) {
    const cRect = container.getBoundingClientRect();
    const x0 = Math.min(zoomStartX, e.clientX) - cRect.left;
    const y0 = Math.min(zoomStartY, e.clientY) - cRect.top;
    const w = Math.abs(e.clientX - zoomStartX);
    const h = Math.abs(e.clientY - zoomStartY);
    zoomRect.style.left = x0 + 'px';
    zoomRect.style.top = y0 + 'px';
    zoomRect.style.width = w + 'px';
    zoomRect.style.height = h + 'px';
  }
});

window.addEventListener('mouseup', e => {
  if (panning) {
    panning = false;
    canvas.style.cursor = 'crosshair';
  }
  if (zooming) {
    zooming = false;
    zoomRect.style.display = 'none';
    // Compute the selected region in world coordinates
    const rect = canvas.getBoundingClientRect();
    const x1 = Math.min(zoomStartX, e.clientX), x2 = Math.max(zoomStartX, e.clientX);
    const y1 = Math.min(zoomStartY, e.clientY), y2 = Math.max(zoomStartY, e.clientY);
    const pw = x2 - x1, ph = y2 - y1;
    if (pw < 4 || ph < 4) return; // too small, ignore
    // Fractional positions on the canvas
    const fL = (x1 - rect.left) / rect.width, fR = (x2 - rect.left) / rect.width;
    const fT = (y1 - rect.top) / rect.height, fB = (y2 - rect.top) / rect.height;
    // Convert to world-space offsets from origin
    const hL = (fL - 0.5) * state.widthH, hR = (fR - 0.5) * state.widthH;
    const vT = (0.5 - fT) * state.widthV, vB = (0.5 - fB) * state.widthV;
    const newCenterH = (hL + hR) / 2, newCenterV = (vT + vB) / 2;
    const newW = hR - hL, newH = vT - vB;
    switch (state.basis) {
      case 'xy': state.originX += newCenterH; state.originY += newCenterV; break;
      case 'xz': state.originX += newCenterH; state.originZ += newCenterV; break;
      case 'yz': state.originY += newCenterH; state.originZ += newCenterV; break;
    }
    state.widthH = newW;
    state.widthV = newH;
    writeControls();
    syncSnap(); sample();
  }
});

// Scroll: move slice origin along the perpendicular axis (~2% of bbox per tick)
canvas.addEventListener('wheel', e => {
  e.preventDefault();
  const perpIdx = state.basis === 'xy' ? 2 : (state.basis === 'xz' ? 1 : 0);
  const extent = BBOX_WIDTHS[perpIdx];
  const step = Math.max(0.1, extent * 0.02) * (e.deltaY > 0 ? -1.0 : 1.0);
  switch (state.basis) {
    case 'xy': state.originZ += step; break;
    case 'xz': state.originY += step; break;
    case 'yz': state.originX += step; break;
  }
  writeControls();
  syncSnap(); sample();
}, { passive: false });

// Tooltip: hover
canvas.addEventListener('mousemove', e => {
  if (panning || zooming) { tooltip.style.display = 'none'; return; }
  const rect = canvas.getBoundingClientRect();
  const fracX = (e.clientX - rect.left) / rect.width;
  const fracY = 1.0 - (e.clientY - rect.top) / rect.height;
  const h = (fracX - 0.5) * state.widthH;
  const v = (fracY - 0.5) * state.widthV;
  const us = unitScale(state.units);
  let x, y, z;
  switch (state.basis) {
    case 'xy': x = state.originX + h; y = state.originY + v; z = state.originZ; break;
    case 'xz': x = state.originX + h; y = state.originY; z = state.originZ + v; break;
    case 'yz': x = state.originX; y = state.originY + h; z = state.originZ + v; break;
  }

  // Look up pixel data
  const px = Math.floor(fracX * canvas.width);
  const py = Math.floor((1 - fracY) * canvas.height);
  let cellInfo = '';
  if (px >= 0 && px < canvas.width && py >= 0 && py < canvas.height) {
    const pixData = ctx.getImageData(px, py, 1, 1).data;
    // We need to look up from the grid data -- use a global ref
    if (window._lastGridData) {
      const idx = (py * canvas.width + px) * 2;
      const cid = window._lastGridData[idx];
      const mid = window._lastGridData[idx + 1];
      if (cid !== -1) {
        const cname = CELL_NAMES[cid] || 'Cell ' + cid;
        const mname = mid === -1 ? 'Void' : (MATERIAL_NAMES[mid] || 'Material ' + mid);
        cellInfo = '<br>cell: ' + cname + ' (id=' + cid + ')' + '<br>material: ' + mname + (mid !== -1 ? ' (id=' + mid + ')' : '');
      }
    }
  }

  // Surface hover: tolerance scales with view's world-units-per-pixel so
  // zoom in/out behaves correctly. Half a pixel's world size -- within
  // that, |f| reads as "on the surface".
  let surfaceInfo = '';
  if (SURFACE_TABLE.length > 0) {
    const tol = Math.max(state.widthH / canvas.width, state.widthV / canvas.height) * 0.5;
    const hits = surfacesAtPoint(x, y, z, tol);
    if (hits.length > 0) {
      surfaceInfo = '<br>surface: ' + hits.join(', ');
    }
  }

  const isZoomed = state.originX !== INITIAL.originX || state.originY !== INITIAL.originY ||
    state.originZ !== INITIAL.originZ || state.widthH !== INITIAL.widthH || state.widthV !== INITIAL.widthV;
  const perpAxis = state.basis === 'xy' ? 'Z' : (state.basis === 'xz' ? 'Y' : 'X');
  const hints = [];
  hints.push('Left mouse click and drag to zoom.');
  if (isZoomed) hints.push('Double-click to zoom out.');
  hints.push('Hold right mouse button to pan.');
  hints.push('Mouse wheel to change ' + perpAxis + ' slice origin.');
  const zoomHint = '<br><span style="color:#888;font-size:11px;">' + hints.join('<br>') + '</span>';
  tooltip.innerHTML = 'x: ' + (x*us).toFixed(4) + ' ' + state.units +
    '<br>y: ' + (y*us).toFixed(4) + ' ' + state.units +
    '<br>z: ' + (z*us).toFixed(4) + ' ' + state.units + cellInfo + surfaceInfo + zoomHint;
  tooltip.style.display = 'block';
  // position:fixed -- clientX/Y are viewport coords, so the tooltip
  // floats over the legend / anything else outside the canvas-container.
  tooltip.style.left = (e.clientX + 15) + 'px';
  tooltip.style.top = (e.clientY + 15) + 'px';
});

canvas.addEventListener('mouseleave', () => { tooltip.style.display = 'none'; });

// Resize handler
window.addEventListener('resize', fitCanvas);

// Check if current sampling params match the initial (presampled) state
function matchesPresampled() {
  return presampledData &&
    state.basis === INITIAL.basis &&
    state.originX === INITIAL.originX && state.originY === INITIAL.originY && state.originZ === INITIAL.originZ &&
    state.widthH === INITIAL.widthH && state.widthV === INITIAL.widthV &&
    state.totalPixels === INITIAL.totalPixels;
}

// Double-click on canvas resets origin/width only (preserves pixels, colors, etc.)
canvas.addEventListener('dblclick', () => {
  state.originX = INITIAL.originX;
  state.originY = INITIAL.originY;
  state.originZ = INITIAL.originZ;
  state.widthH = INITIAL.widthH;
  state.widthV = INITIAL.widthV;
  writeControls();
  sampledSnap = { basis: state.basis, originX: state.originX, originY: state.originY, originZ: state.originZ, widthH: state.widthH, widthV: state.widthV, totalPixels: state.totalPixels };
  dirty = false;
  applyBtn.classList.remove('dirty');
  if (matchesPresampled()) { renderPresampled(); } else { sample(); }
});

// Store grid data globally for tooltip lookup
const origRenderGrid = renderGrid;
// Monkey-patch isn't needed -- let's store data in sample() instead

// Decode pre-sampled grid from base64 (little-endian i32 pairs)
let presampledData = null;
let presampledPH = 0, presampledPV = 0;
if (PRESAMPLED_B64) {
  const raw = atob(PRESAMPLED_B64);
  const bytes = new Uint8Array(raw.length);
  for (let i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i);
  presampledData = new Int32Array(bytes.buffer);
  presampledPH = PRESAMPLED_PH;
  presampledPV = PRESAMPLED_PV;
}

// Render the pre-sampled initial view, or fall back to live sampling
function renderPresampled() {
  if (presampledData) {
    window._lastGridData = presampledData;
    renderGrid(presampledData, presampledPH, presampledPV);
  } else {
    sample();
  }
}

// Load WASM (mesh only) and initialize
async function init() {
  // Render pre-sampled data immediately (before WASM loads)
  if (presampledData) {
    renderPresampled();
    // Yield to browser so the canvas renders before we block on WASM
    await new Promise(r => setTimeout(r, 0));
  }

  const geo = GEOMETRY_JSON;

  if (HAS_MESH_FILLS) {
    // Nothing to initialise: `sample()` short-circuits to the server-rendered
    // raster for a filled model (issue #291).
    status.textContent = '';
    return;
  }
  if (GEOMETRY_KIND === 'csg') {
    plotter = new JsCsgPlotter(geo);
  } else {
    // Mesh geometry -- WASM engine (yamt BVH)
    status.textContent = 'Loading WASM...';
    const wasmBytes = Uint8Array.from(atob(YAMT_WASM_BASE64), c => c.charCodeAt(0));
    // Use async compilation to avoid blocking main thread
    const wasmModule = await WebAssembly.compile(wasmBytes);
    yamtWasmInitSync(wasmModule);
    const geoJson = JSON.stringify(geo);
    const wasmPlotter = new YamtWasmMeshPlotter(geoJson);
    plotter = {
      sampleGrid(basis, ox, oy, oz, wh, wv, ph, pv) {
        const data = wasmPlotter.sampleGrid(basis, ox, oy, oz, wh, wv, ph, pv);
        window._lastGridData = data;
        return data;
      },
      boundingBox() { return wasmPlotter.boundingBox(); }
    };
  }
  status.textContent = '';

  // If no pre-sampled data, sample now
  if (!presampledData) {
    sample();
  }
}

// JS CSG Plotter -- pool of Web Workers, one per hardware thread. Falls
// back to main-thread sampling if the browser won't spawn workers (some
// browsers block Worker construction from file:// in certain setups).

class JsCsgPlotter {
  constructor(geo) {
    this.cells = geo.cells;
    this.workers = [];
    this.reqId = 0;
    try {
      const blob = new Blob([CSG_WORKER_SOURCE], {type: 'application/javascript'});
      const url = URL.createObjectURL(blob);
      const n = Math.max(1, Math.min(navigator.hardwareConcurrency || 4, 16));
      for (let i = 0; i < n; i++) {
        const w = new Worker(url);
        w.postMessage({type: 'init', cells: this.cells});
        this.workers.push(w);
      }
      URL.revokeObjectURL(url);
    } catch (e) {
      console.warn('JsCsgPlotter: workers unavailable, running single-threaded', e);
      this.workers = [];
    }
  }

  async sampleGrid(basis, ox, oy, oz, wh, wv, ph, pv) {
    if (this.workers.length === 0) {
      const data = sampleGridCsgSync(this.cells, basis, ox, oy, oz, wh, wv, ph, pv);
      window._lastGridData = data;
      return data;
    }
    const data = new Int32Array(ph * pv * 2);
    const rid = ++this.reqId;
    const n = this.workers.length;
    const rowsPer = Math.ceil(pv / n);
    const promises = [];
    for (let i = 0; i < n; i++) {
      const rowStart = i * rowsPer;
      const rowEnd = Math.min(rowStart + rowsPer, pv);
      if (rowStart >= rowEnd) continue;
      const w = this.workers[i];
      promises.push(new Promise((resolve) => {
        const handler = (ev) => {
          if (ev.data.rid !== rid) return;
          w.removeEventListener('message', handler);
          data.set(ev.data.slab, ev.data.rowStart * ph * 2);
          resolve();
        };
        w.addEventListener('message', handler);
        w.postMessage({type: 'sample', rid, basis, ox, oy, oz, wh, wv, ph, pv, rowStart, rowEnd});
      }));
    }
    await Promise.all(promises);
    window._lastGridData = data;
    return data;
  }

  boundingBox() {
    // Compute from cells -- approximate with surface bounds
    return [0, 0, 0, 10, 10, 10]; // fallback
  }
}

// Single-threaded fallback used when Web Workers can't be spawned.
function sampleGridCsgSync(cells, basis, ox, oy, oz, wh, wv, ph, pv) {
  const data = new Int32Array(ph * pv * 2);
  const halfH = wh / 2, halfV = wv / 2;
  for (let i = 0; i < pv; i++) {
    const v = pv > 1 ? halfV - wv * i / (pv - 1) : 0;
    for (let j = 0; j < ph; j++) {
      const h = ph > 1 ? -halfH + wh * j / (ph - 1) : 0;
      let px, py, pz;
      switch(basis) {
        case 'xy': px = ox+h; py = oy+v; pz = oz; break;
        case 'xz': px = ox+h; py = oy; pz = oz+v; break;
        case 'yz': px = ox; py = oy+h; pz = oz+v; break;
        default: px = ox+h; py = oy+v; pz = oz;
      }
      const idx = (i * ph + j) * 2;
      let found = false;
      for (const cell of cells) {
        if (regionContains(cell.region, px, py, pz)) {
          data[idx] = cell.cell_id;
          data[idx+1] = cell.material_id;
          found = true;
          break;
        }
      }
      if (!found) { data[idx] = -1; data[idx+1] = -1; }
    }
  }
  return data;
}

// Evaluate whether a point is inside a region (CSG boolean tree)
function regionContains(region, x, y, z) {
  return evalExpr(region.expr, x, y, z);
}

function evalExpr(expr, x, y, z) {
  if (expr.Halfspace) {
    const hs = expr.Halfspace;
    if (hs.Above) return surfaceEval(hs.Above, x, y, z) > 0;
    if (hs.Below) return surfaceEval(hs.Below, x, y, z) < 0;
  }
  if (expr.Intersection) {
    return evalExpr(expr.Intersection[0], x, y, z) && evalExpr(expr.Intersection[1], x, y, z);
  }
  if (expr.Union) {
    return evalExpr(expr.Union[0], x, y, z) || evalExpr(expr.Union[1], x, y, z);
  }
  if (expr.Complement) {
    return !evalExpr(expr.Complement, x, y, z);
  }
  return false;
}

// Find every surface within `tol` of the world point (x, y, z) -- used by
// the hover tooltip. O(N) over SURFACE_TABLE; for typical N ≤ 100 this
// is sub-microsecond per call, so we eat the cost on every mousemove
// rather than precomputing a grid (which would be ~100 MB at 4K res).
function surfacesAtPoint(x, y, z, tol) {
  const hits = [];
  for (let i = 0; i < SURFACE_TABLE.length; i++) {
    const entry = SURFACE_TABLE[i];
    if (Math.abs(surfaceEval(entry.surface, x, y, z)) <= tol) {
      hits.push(entry.name);
    }
  }
  return hits;
}

function surfaceEval(surface, x, y, z) {
  const k = surface.kind;
  if (k.Plane) {
    return k.Plane.a * x + k.Plane.b * y + k.Plane.c * z - k.Plane.d;
  }
  if (k.Sphere) {
    const dx = x - k.Sphere.x0, dy = y - k.Sphere.y0, dz = z - k.Sphere.z0;
    return Math.sqrt(dx*dx + dy*dy + dz*dz) - k.Sphere.radius;
  }
  if (k.Cylinder) {
    const ax = k.Cylinder.axis, or = k.Cylinder.origin;
    const vx = x-or[0], vy = y-or[1], vz = z-or[2];
    const dot = vx*ax[0] + vy*ax[1] + vz*ax[2];
    const dx = vx - dot*ax[0], dy = vy - dot*ax[1], dz = vz - dot*ax[2];
    return Math.sqrt(dx*dx + dy*dy + dz*dz) - k.Cylinder.radius;
  }
  if (k.ZTorus) {
    const t = k.ZTorus;
    const dx = x - t.x0, dy = y - t.y0, dz = z - t.z0;
    const rho = Math.sqrt(dx*dx + dy*dy);
    return (rho - t.a) * (rho - t.a) / (t.c * t.c) + dz * dz / (t.b * t.b) - 1.0;
  }
  return 0;
}

