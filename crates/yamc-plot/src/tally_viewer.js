// Pure-JS viewer for yamc mesh-tally plots.
//
// Extracted from yamc-plot/src/tally_html.rs so it can be edited as
// real JavaScript. The Rust side of `build_interactive_tally_html`
// declares these globals *before* including this file:
//
//   const CELL_NAMES, MATERIAL_NAMES, SURFACE_TABLE,
//   MESH_LL, MESH_UR, MESH_DIM, MESH_WIDTH,
//   SLICES, EXTRA_SLICES, SLICE_INDICES, SLICE_DIMS,
//   GEOMETRY_JSON, GEOMETRY_KIND,
//   PRESAMPLED_OUTLINE_B64 (+ _PH/_PV when non-null),
//   OUTLINE_PIXELS, INITIAL_SLICE_BIN, NATIVE_BASIS,
//   INITIAL_SCALING_FACTOR, INITIAL_FONT_SIZE, COLORSCALES,
//   DISPLAY_VALUE, TITLE,
//   let state = { … };
//
// And after this file: any conditional `{wasm_section}` (mesh path),
// then `init();`.

// State-dependent helpers -- must follow the Rust-emitted `let state = {...}`.
function unitScale(u) { return {"mm":10,"cm":1,"m":0.01,"km":0.00001}[u]||1; }

function initView(basis) {
  const axes = basisAxes(basis);
  state.originH = (MESH_LL[axes[0]] + MESH_UR[axes[0]]) / 2;
  state.originV = (MESH_LL[axes[1]] + MESH_UR[axes[1]]) / 2;
  state.widthH = MESH_UR[axes[0]] - MESH_LL[axes[0]];
  state.widthV = MESH_UR[axes[1]] - MESH_LL[axes[1]];
}

// ---- Helpers ----
function basisAxes(basis) {
  // Returns [hAxisIndex, vAxisIndex, fixedAxisIndex]
  if (basis === 'xy') return [0, 1, 2];
  if (basis === 'xz') return [0, 2, 1];
  return [1, 2, 0]; // yz
}

function basisLabels(basis) {
  if (basis === 'xy') return ['X', 'Y', 'Z'];
  if (basis === 'xz') return ['X', 'Z', 'Y'];
  return ['Y', 'Z', 'X'];
}

// ---- DOM refs ----
const canvas = document.getElementById('plot-canvas');
const ctx = canvas.getContext('2d');
const container = document.getElementById('canvas-container');
const tooltip = document.getElementById('tooltip');
const statusEl = document.getElementById('status');
const cbCanvas = document.getElementById('colorbar-canvas');
const cbCtx = cbCanvas.getContext('2d');

// ---- Slice decode ----
function decodeSlice(b64) {
  const raw = atob(b64);
  const bytes = new Uint8Array(raw.length);
  for (let i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i);
  return new Float64Array(bytes.buffer);
}

// Cache decoded slices
const sliceCache = {};
function getSlice(key) {
  if (!sliceCache[key]) {
    if (!SLICES[key]) return null;
    sliceCache[key] = decodeSlice(SLICES[key]);
  }
  return sliceCache[key];
}

// Cache for extra value slices (tooltip only)
const extraSliceCache = {};
function getExtraSlice(valueName, key) {
  const cacheKey = valueName + ':' + key;
  if (!extraSliceCache[cacheKey]) {
    if (!EXTRA_SLICES[valueName] || !EXTRA_SLICES[valueName][key]) return null;
    extraSliceCache[cacheKey] = decodeSlice(EXTRA_SLICES[valueName][key]);
  }
  return extraSliceCache[cacheKey];
}
const EXTRA_VALUE_NAMES = Object.keys(EXTRA_SLICES);

// ---- Cross-basis slice extraction ----
function extractCrossBasisSlice(viewBasis, sliderBinIndex) {
  const viewAxes = basisAxes(viewBasis);
  const viewH = viewAxes[0], viewV = viewAxes[1];
  const nativeAxes = basisAxes(NATIVE_BASIS);
  const nativeH = nativeAxes[0], nativeV = nativeAxes[1], nativeFixed = nativeAxes[2];
  const viewDims = SLICE_DIMS[viewBasis];
  const nH_view = viewDims[0], nV_view = viewDims[1];
  const nativeDims = SLICE_DIMS[NATIVE_BASIS];
  const nH_native = nativeDims[0], nV_native = nativeDims[1];
  const nTotal = nH_view * nV_view;
  const data = new Float64Array(nTotal);
  const mask = new Uint8Array(nTotal);
  const embeddedIndices = SLICE_INDICES[NATIVE_BASIS] || [];
  const embeddedSet = new Set(embeddedIndices);
  for (let row = 0; row < nV_view; row++) {
    for (let col = 0; col < nH_view; col++) {
      const meshIdx = [0, 0, 0];
      meshIdx[viewH] = col;
      meshIdx[viewV] = row;
      meshIdx[viewAxes[2]] = sliderBinIndex;
      const nativeFixedVal = meshIdx[nativeFixed];
      if (!embeddedSet.has(nativeFixedVal)) continue;
      const key = NATIVE_BASIS + ':' + nativeFixedVal;
      const nativeSlice = getSlice(key);
      if (!nativeSlice) continue;
      const nH_idx = meshIdx[nativeH];
      const nV_idx = meshIdx[nativeV];
      if (nH_idx < 0 || nH_idx >= nH_native || nV_idx < 0 || nV_idx >= nV_native) continue;
      const outIdx = row * nH_view + col;
      data[outIdx] = nativeSlice[nV_idx * nH_native + nH_idx];
      mask[outIdx] = 1;
    }
  }
  return { data: data, mask: mask };
}

// ---- Colorscale lookup ----
function sampleColor(csName, t) {
  const stops = COLORSCALES[csName] || COLORSCALES['Viridis'];
  if (t <= stops[0][0]) return stops[0][1];
  if (t >= stops[stops.length-1][0]) return stops[stops.length-1][1];
  for (let i = 1; i < stops.length; i++) {
    if (t <= stops[i][0]) {
      const f = (t - stops[i-1][0]) / (stops[i][0] - stops[i-1][0]);
      const a = stops[i-1][1], b = stops[i][1];
      return [
        Math.round(a[0] + (b[0]-a[0])*f),
        Math.round(a[1] + (b[1]-a[1])*f),
        Math.round(a[2] + (b[2]-a[2])*f)
      ];
    }
  }
  return stops[stops.length-1][1];
}

// Build a 256-entry LUT for fast pixel mapping
function buildLUT(csName, levels) {
  const lut = new Uint8Array(256 * 3);
  for (let i = 0; i < 256; i++) {
    let t = i / 255;
    if (levels > 0) {
      t = Math.floor(t * levels) / (levels - 1 || 1);
      t = Math.min(1, t);
    }
    const rgb = sampleColor(csName, t);
    lut[i*3] = rgb[0]; lut[i*3+1] = rgb[1]; lut[i*3+2] = rgb[2];
  }
  return lut;
}

let currentLUT = buildLUT(state.colorscale, state.levels);

// ---- Heatmap rendering ----
function renderHeatmap(sliceData, nH, nV, mask) {
  // Compute view-to-mesh mapping
  const axes = basisAxes(state.basis);
  const meshMinH = MESH_LL[axes[0]], meshMaxH = MESH_UR[axes[0]];
  const meshMinV = MESH_LL[axes[1]], meshMaxV = MESH_UR[axes[1]];

  const viewMinH = state.originH - state.widthH / 2;
  const viewMaxH = state.originH + state.widthH / 2;
  const viewMinV = state.originV - state.widthV / 2;
  const viewMaxV = state.originV + state.widthV / 2;

  // Canvas pixel dimensions (match the heatmap at 1:1 for crisp rendering,
  // up to a max to avoid huge canvas)
  const aspect = state.widthH / state.widthV;
  const maxDim = 800;
  let cW, cH;
  if (aspect >= 1) {
    cW = Math.min(maxDim, Math.max(nH, 200));
    cH = Math.round(cW / aspect);
  } else {
    cH = Math.min(maxDim, Math.max(nV, 200));
    cW = Math.round(cH * aspect);
  }
  if (cW < 2) cW = 2;
  if (cH < 2) cH = 2;
  canvas.width = cW;
  canvas.height = cH;

  const img = ctx.createImageData(cW, cH);
  const d = img.data;

  // Compute data range (skip zeros/NaN/masked), apply scaling factor
  const sf = state.scalingFactor;
  let dMin = Infinity, dMax = -Infinity;
  for (let k = 0; k < sliceData.length; k++) {
    if (mask && !mask[k]) continue;
    const v = sliceData[k] * sf;
    if (v > 0 && isFinite(v)) {
      if (v < dMin) dMin = v;
      if (v > dMax) dMax = v;
    }
  }
  if (!isFinite(dMin)) { dMin = 0; dMax = 1; }
  if (dMin === dMax) { dMin = dMax * 0.9; if (dMin === 0) dMax = 1; }
  if (state.vmin !== null && isFinite(state.vmin)) dMin = state.vmin;
  if (state.vmax !== null && isFinite(state.vmax)) dMax = state.vmax;
  if (dMin >= dMax) dMax = dMin + 1;
  state.dataMin = dMin;
  state.dataMax = dMax;

  const useLog = state.logScale && dMin > 0;
  const logMin = useLog ? Math.log10(dMin) : dMin;
  const logMax = useLog ? Math.log10(dMax) : dMax;
  const range = logMax - logMin || 1;

  const meshWidthH = MESH_WIDTH[axes[0]];
  const meshWidthV = MESH_WIDTH[axes[1]];

  for (let row = 0; row < cH; row++) {
    // row 0 = top = highest V
    const worldV = viewMaxV - (row + 0.5) / cH * (viewMaxV - viewMinV);
    const meshV = (worldV - meshMinV) / meshWidthV;
    const iv = Math.floor(meshV);

    for (let col = 0; col < cW; col++) {
      const worldH = viewMinH + (col + 0.5) / cW * (viewMaxH - viewMinH);
      const meshH = (worldH - meshMinH) / meshWidthH;
      const ih = Math.floor(meshH);

      const px = (row * cW + col) * 4;

      if (iv < 0 || iv >= nV || ih < 0 || ih >= nH) {
        // Outside mesh -- transparent
        d[px] = 0; d[px+1] = 0; d[px+2] = 0; d[px+3] = 0;
        continue;
      }

      if (mask && !mask[iv * nH + ih]) {
        // No data (cross-basis gap) -- light gray
        d[px] = 208; d[px+1] = 208; d[px+2] = 208; d[px+3] = 255;
        continue;
      }

      const val = sliceData[iv * nH + ih] * sf;
      if (val <= 0 || !isFinite(val)) {
        d[px] = 200; d[px+1] = 200; d[px+2] = 200; d[px+3] = 255;
        continue;
      }

      const norm = useLog
        ? (Math.log10(val) - logMin) / range
        : (val - logMin) / range;
      const ci = Math.max(0, Math.min(255, Math.round(norm * 255)));
      d[px] = currentLUT[ci*3];
      d[px+1] = currentLUT[ci*3+1];
      d[px+2] = currentLUT[ci*3+2];
      d[px+3] = 255;
    }
  }

  ctx.putImageData(img, 0, 0);
}

// ---- Outline rendering ----
// Uses a DOM overlay canvas at full sample resolution (same technique as geometry viewer).
// CSS handles display scaling -- no downsampling, so no gaps at high pixel counts.
let outlinePlotter = null;
const olCanvas = document.getElementById('outline-overlay');
const olCtx = olCanvas.getContext('2d');
// Cached outline sample data for tooltip lookup [cell_id, material_id, ...]
let cachedOutlineData = null;
let cachedOutlinePH = 0, cachedOutlinePV = 0;

function renderOutline() {
  if (!outlinePlotter || state.outline === 'none') {
    olCanvas.width = 0; olCanvas.height = 0;
    return;
  }

  // Compute sample dimensions from outlinePixels, preserving aspect ratio
  const aspect = state.widthH / state.widthV;
  const totalPx = state.outlinePixels;
  const opV = Math.max(1, Math.round(Math.sqrt(totalPx / aspect)));
  const opH = Math.max(1, Math.round(totalPx / opV));

  // Determine the current fixed-axis coordinate from slice
  const axes = basisAxes(state.basis);
  const fixedIdx = axes[2];
  const binIdx = state.sliceIdx;
  const fixedCoord = MESH_LL[fixedIdx] + MESH_WIDTH[fixedIdx] * (binIdx + 0.5);

  let ox, oy, oz;
  if (state.basis === 'xy') { ox = state.originH; oy = state.originV; oz = fixedCoord; }
  else if (state.basis === 'xz') { ox = state.originH; oy = fixedCoord; oz = state.originV; }
  else { ox = fixedCoord; oy = state.originH; oz = state.originV; }

  let data;
  try {
    data = outlinePlotter.sampleGrid(state.basis, ox, oy, oz, state.widthH, state.widthV, opH, opV);
  } catch(e) {
    console.error('Outline sampling error:', e);
    return;
  }

  // Cache for tooltip lookup
  cachedOutlineData = data;
  cachedOutlinePH = opH;
  cachedOutlinePV = opV;

  // Build ID map at sample resolution
  const outlineMode = state.outline;
  const outlineIds = new Int32Array(opH * opV);
  for (let i = 0; i < opV; i++) {
    for (let j = 0; j < opH; j++) {
      const idx = (i * opH + j) * 2;
      outlineIds[i * opH + j] = outlineMode === 'cell' ? data[idx] : data[idx + 1];
    }
  }

  // Render edges at sample resolution (1:1 with canvas pixels, like geometry viewer)
  olCanvas.width = opH;
  olCanvas.height = opV;
  const img = olCtx.createImageData(opH, opV);
  const d = img.data;
  const ow = state.outlineWidth;
  const oc = state.outlineColor;

  for (let i = 0; i < opV; i++) {
    for (let j = 0; j < opH; j++) {
      const myId = outlineIds[i * opH + j];
      if (myId === -1) continue;

      let isEdge = false;
      for (let dd = 1; dd <= ow && !isEdge; dd++) {
        if (j >= dd && outlineIds[i * opH + j - dd] !== myId) isEdge = true;
        if (!isEdge && j + dd < opH && outlineIds[i * opH + j + dd] !== myId) isEdge = true;
        if (!isEdge && i >= dd && outlineIds[(i - dd) * opH + j] !== myId) isEdge = true;
        if (!isEdge && i + dd < opV && outlineIds[(i + dd) * opH + j] !== myId) isEdge = true;
      }

      if (isEdge) {
        const px = (i * opH + j) * 4;
        d[px] = oc[0]; d[px+1] = oc[1]; d[px+2] = oc[2]; d[px+3] = 255;
      }
    }
  }

  olCtx.putImageData(img, 0, 0);
  // CSS positions and scales the overlay to match the heatmap (set in fitCanvas)
}

// ---- Colorbar ----
function renderColorbar() {
  const cbContainer = document.getElementById('colorbar-container');
  cbContainer.style.display = state.showColorbar ? '' : 'none';
  if (!state.showColorbar) return;
  const h = 256;
  cbCanvas.height = h;
  const img = cbCtx.createImageData(16, h);
  const d = img.data;
  for (let i = 0; i < h; i++) {
    const t = 1 - i / (h - 1); // top = max
    const ci = Math.round(t * 255);
    for (let j = 0; j < 16; j++) {
      const px = (i * 16 + j) * 4;
      d[px] = currentLUT[ci*3]; d[px+1] = currentLUT[ci*3+1]; d[px+2] = currentLUT[ci*3+2]; d[px+3] = 255;
    }
  }
  cbCtx.putImageData(img, 0, 0);

  // Position colorbar to match the plot area height
  const canvasRect = canvas.getBoundingClientRect();
  const containerRect = document.getElementById('canvas-area').getBoundingClientRect();
  const canvasTop = canvasRect.top - containerRect.top;
  const canvasH = canvasRect.height;
  cbCanvas.style.top = canvasTop + 'px';
  const cbH = canvasH;
  cbCanvas.style.height = cbH + 'px';

  const tickFs = Math.max(7, state.fontSize - 3);

  // Ticks
  const tickContainer = document.getElementById('colorbar-ticks');
  tickContainer.innerHTML = '';
  const useLog = state.logScale && state.dataMin > 0;
  const nTicks = 5;
  for (let i = 0; i <= nTicks; i++) {
    const frac = i / nTicks;
    let val;
    if (useLog) {
      val = Math.pow(10, Math.log10(state.dataMin) + frac * (Math.log10(state.dataMax) - Math.log10(state.dataMin)));
    } else {
      val = state.dataMin + frac * (state.dataMax - state.dataMin);
    }
    const y = canvasTop + (1 - frac) * cbH;
    const el = document.createElement('span');
    el.className = 'cb-tick';
    el.style.fontSize = tickFs + 'px';
    el.textContent = val.toExponential(1);
    el.style.top = y + 'px';
    tickContainer.appendChild(el);
  }

  // Label
  const label = document.getElementById('colorbar-label');
  label.textContent = state.colorbarLabel;
  label.style.fontSize = state.fontSize + 'px';
  label.style.right = '2px';
  label.style.top = (canvasTop + cbH / 2) + 'px';
}

// ---- Composite render ----
function renderFrame() {
  const dims = SLICE_DIMS[state.basis];
  const nH = dims[0], nV = dims[1];
  const binIdx = state.sliceIdx;
  let sliceData, mask;

  if (state.basis === NATIVE_BASIS) {
    const key = NATIVE_BASIS + ':' + binIdx;
    sliceData = getSlice(key);
    if (sliceData) {
      mask = null;
    } else {
      sliceData = new Float64Array(nH * nV);
      mask = new Uint8Array(nH * nV);
    }
  } else {
    const result = extractCrossBasisSlice(state.basis, binIdx);
    sliceData = result.data;
    mask = result.mask;
  }

  state.currentSliceData = sliceData;
  state.currentSliceMask = mask;

  renderHeatmap(sliceData, nH, nV, mask);
  renderOutline();
  fitCanvas();
  renderColorbar();
  updateAxisLabels();
  statusEl.textContent = '';
}

// ---- Canvas fitting (CSS scaling) ----
function fitCanvas() {
  const cw = container.clientWidth;
  const ch = container.clientHeight;
  if (!canvas.width || !canvas.height) return;
  const aspect = canvas.width / canvas.height;
  const padLeft = 75, padBottom = 50;
  let dispW, dispH;
  if ((cw - padLeft) / (ch - padBottom) > aspect) {
    dispH = ch - padBottom;
    dispW = dispH * aspect;
  } else {
    dispW = cw - padLeft;
    dispH = dispW / aspect;
  }
  if (dispW < 10) dispW = 10;
  if (dispH < 10) dispH = 10;
  canvas.style.width = dispW + 'px';
  canvas.style.height = dispH + 'px';
  // Left-align next to the controls sidebar. Used to push the canvas
  // flush against the right-side colorbar -- looked tidy on narrow
  // windows but on a wide screen left a big gray strip between
  // controls and plot. Empty space now sits between canvas and
  // colorbar instead.
  const canvasLeft = padLeft + 'px';
  const canvasTop = ((ch - padBottom - dispH) / 2) + 'px';
  canvas.style.left = canvasLeft;
  canvas.style.top = canvasTop;
  canvas.style.border = '1px solid #999';
  // Position outline overlay to match
  const olCanvas = document.getElementById('outline-overlay');
  olCanvas.style.width = dispW + 'px';
  olCanvas.style.height = dispH + 'px';
  olCanvas.style.left = canvasLeft;
  olCanvas.style.top = canvasTop;
}

// ---- Axis labels and ticks ----
function updateAxisLabels() {
  const u = state.units;
  const us = unitScale(u);
  const fs = state.fontSize;
  const tickFs = Math.max(8, fs - 2);
  const labels = basisLabels(state.basis);
  const hLabel = labels[0], vLabel = labels[1];
  const hLabelEl = document.getElementById('axis-label-h');
  const vLabelEl = document.getElementById('axis-label-v');
  hLabelEl.textContent = hLabel + ' (' + u + ')';
  vLabelEl.textContent = vLabel + ' (' + u + ')';
  hLabelEl.style.fontSize = fs + 'px';
  vLabelEl.style.fontSize = fs + 'px';

  const rect = canvas.getBoundingClientRect();
  const cRect = container.getBoundingClientRect();
  const canvasLeft = rect.left - cRect.left;
  const canvasTop = rect.top - cRect.top;
  const canvasW = rect.width;
  const canvasH = rect.height;
  if (canvasW < 1 || canvasH < 1) return;

  hLabelEl.style.left = (canvasLeft + canvasW / 2) + 'px';
  hLabelEl.style.transform = 'translateX(-50%)';
  hLabelEl.style.top = (canvasTop + canvasH + tickFs + 12) + 'px';
  vLabelEl.style.left = (canvasLeft - 68) + 'px';
  vLabelEl.style.top = (canvasTop + canvasH / 2) + 'px';

  const hMin = (state.originH - state.widthH / 2) * us;
  const hMax = (state.originH + state.widthH / 2) * us;
  const vMin = (state.originV - state.widthV / 2) * us;
  const vMax = (state.originV + state.widthV / 2) * us;

  const hTicks = niceTicks(hMin, hMax, Math.floor(canvasW / 80));
  const vTicks = niceTicks(vMin, vMax, Math.floor(canvasH / 60));

  const hContainer = document.getElementById('tick-container-h');
  hContainer.innerHTML = '';
  for (const val of hTicks) {
    const frac = (val - hMin) / (hMax - hMin);
    if (frac < 0.02 || frac > 0.98) continue;
    const mark = document.createElement('span');
    mark.style.cssText = 'position:absolute;width:1px;height:5px;background:#666;';
    mark.style.left = (canvasLeft + frac * canvasW) + 'px';
    mark.style.top = (canvasTop + canvasH + 1) + 'px';
    hContainer.appendChild(mark);
    const el = document.createElement('span');
    el.className = 'tick-h';
    el.style.fontSize = tickFs + 'px';
    el.textContent = formatTick(val);
    el.style.left = (canvasLeft + frac * canvasW) + 'px';
    el.style.top = (canvasTop + canvasH + 7) + 'px';
    hContainer.appendChild(el);
  }

  const vContainer = document.getElementById('tick-container-v');
  vContainer.innerHTML = '';
  for (const val of vTicks) {
    const frac = (val - vMin) / (vMax - vMin);
    if (frac < 0.02 || frac > 0.98) continue;
    const mark = document.createElement('span');
    mark.style.cssText = 'position:absolute;height:1px;width:5px;background:#666;';
    mark.style.left = (canvasLeft - 6) + 'px';
    mark.style.top = (canvasTop + (1 - frac) * canvasH) + 'px';
    vContainer.appendChild(mark);
    const el = document.createElement('span');
    el.className = 'tick-v';
    el.style.fontSize = tickFs + 'px';
    el.textContent = formatTick(val);
    el.style.left = '0px';
    el.style.width = (canvasLeft - 8) + 'px';
    el.style.top = (canvasTop + (1 - frac) * canvasH) + 'px';
    vContainer.appendChild(el);
  }
}

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
  for (let v = start; v <= hi + step * 0.001; v += step) ticks.push(v);
  return ticks;
}

function formatTick(v) {
  if (Math.abs(v) < 1e-10) return '0';
  const av = Math.abs(v);
  if (av >= 1000 || av < 0.01) return v.toExponential(1);
  return parseFloat(v.toPrecision(6)).toString();
}

// ---- Slice navigation ----
function updateSlider() {
  const slider = document.getElementById('slice-slider');
  const axes = basisAxes(state.basis);
  const fixedDim = MESH_DIM[axes[2]];
  slider.max = Math.max(0, fixedDim - 1);
  slider.value = state.sliceIdx;
  updateSliceInfo();
}

function updateSliceInfo() {
  const info = document.getElementById('slice-info');
  const axes = basisAxes(state.basis);
  const fixedIdx = axes[2];
  const labels = basisLabels(state.basis);
  const us = unitScale(state.units);
  const binIdx = state.sliceIdx;
  const coord = MESH_LL[fixedIdx] + MESH_WIDTH[fixedIdx] * (binIdx + 0.5);
  let text = 'Bin ' + binIdx + ' (' + labels[2] + ' = ' + (coord * us).toFixed(4) + ' ' + state.units + ')';

  if (state.basis === NATIVE_BASIS) {
    const embeddedSet = new Set(SLICE_INDICES[NATIVE_BASIS] || []);
    if (!embeddedSet.has(binIdx)) {
      text += ' \u2014 No embedded data';
    }
  } else {
    const embeddedIndices = SLICE_INDICES[NATIVE_BASIS] || [];
    const nativeAxes = basisAxes(NATIVE_BASIS);
    const total = MESH_DIM[nativeAxes[2]];
    text += ' \u2014 Partial: ' + embeddedIndices.length + ' of ' + total + ' from ' + NATIVE_BASIS.toUpperCase() + ' slices';
  }
  info.textContent = text;
}

// ---- Zoom/Pan ----
canvas.addEventListener('contextmenu', e => e.preventDefault());

let panning = false, panStartX, panStartY, panStartOriginH, panStartOriginV;
let zooming = false, zoomStartX, zoomStartY;

const zoomRect = document.createElement('div');
zoomRect.style.cssText = 'position:absolute;border:2px dashed #e94560;background:rgba(233,69,96,0.1);display:none;pointer-events:none;z-index:90;';
container.appendChild(zoomRect);

canvas.addEventListener('mousedown', e => {
  if (e.button === 2) {
    panning = true;
    panStartX = e.clientX; panStartY = e.clientY;
    panStartOriginH = state.originH; panStartOriginV = state.originV;
    canvas.style.cursor = 'grabbing';
  } else if (e.button === 0) {
    zooming = true;
    zoomStartX = e.clientX; zoomStartY = e.clientY;
    zoomRect.style.left = (e.clientX - container.getBoundingClientRect().left) + 'px';
    zoomRect.style.top = (e.clientY - container.getBoundingClientRect().top) + 'px';
    zoomRect.style.width = '0px'; zoomRect.style.height = '0px';
    zoomRect.style.display = 'block';
  }
});

window.addEventListener('mousemove', e => {
  if (panning) {
    const rect = canvas.getBoundingClientRect();
    const dx = (e.clientX - panStartX) / rect.width * state.widthH;
    const dy = (e.clientY - panStartY) / rect.height * state.widthV;
    state.originH = panStartOriginH - dx;
    state.originV = panStartOriginV + dy; // canvas Y is inverted
    renderFrame();
  }
  if (zooming) {
    const cRect = container.getBoundingClientRect();
    const x0 = Math.min(zoomStartX, e.clientX) - cRect.left;
    const y0 = Math.min(zoomStartY, e.clientY) - cRect.top;
    const w = Math.abs(e.clientX - zoomStartX);
    const h = Math.abs(e.clientY - zoomStartY);
    zoomRect.style.left = x0 + 'px'; zoomRect.style.top = y0 + 'px';
    zoomRect.style.width = w + 'px'; zoomRect.style.height = h + 'px';
  }
});

window.addEventListener('mouseup', e => {
  if (panning) { panning = false; canvas.style.cursor = 'crosshair'; }
  if (zooming) {
    zooming = false; zoomRect.style.display = 'none';
    const rect = canvas.getBoundingClientRect();
    const x1 = Math.min(zoomStartX, e.clientX), x2 = Math.max(zoomStartX, e.clientX);
    const y1 = Math.min(zoomStartY, e.clientY), y2 = Math.max(zoomStartY, e.clientY);
    if (x2 - x1 < 4 || y2 - y1 < 4) return;
    const fL = (x1 - rect.left) / rect.width, fR = (x2 - rect.left) / rect.width;
    const fT = (y1 - rect.top) / rect.height, fB = (y2 - rect.top) / rect.height;
    const hL = (fL - 0.5) * state.widthH, hR = (fR - 0.5) * state.widthH;
    const vT = (0.5 - fT) * state.widthV, vB = (0.5 - fB) * state.widthV;
    state.originH += (hL + hR) / 2;
    state.originV += (vT + vB) / 2;
    state.widthH = hR - hL;
    state.widthV = vT - vB;
    renderFrame();
  }
});

// Scroll wheel: zoom in/out centered on cursor
canvas.addEventListener('wheel', e => {
  e.preventDefault();
  const factor = e.deltaY > 0 ? 1.15 : 1 / 1.15;
  const rect = canvas.getBoundingClientRect();
  const fracX = (e.clientX - rect.left) / rect.width;
  const fracY = 1 - (e.clientY - rect.top) / rect.height;
  const worldH = state.originH + (fracX - 0.5) * state.widthH;
  const worldV = state.originV + (fracY - 0.5) * state.widthV;
  state.widthH *= factor;
  state.widthV *= factor;
  state.originH = worldH - (fracX - 0.5) * state.widthH;
  state.originV = worldV - (fracY - 0.5) * state.widthV;
  renderFrame();
}, { passive: false });

// Double-click: reset view
canvas.addEventListener('dblclick', () => {
  initView(state.basis);
  renderFrame();
});

// ---- Tooltip ----
canvas.addEventListener('mousemove', e => {
  if (panning || zooming) { tooltip.style.display = 'none'; return; }
  const rect = canvas.getBoundingClientRect();
  const fracX = (e.clientX - rect.left) / rect.width;
  const fracY = 1.0 - (e.clientY - rect.top) / rect.height;
  const worldH = state.originH + (fracX - 0.5) * state.widthH;
  const worldV = state.originV + (fracY - 0.5) * state.widthV;
  const us = unitScale(state.units);
  const labels = basisLabels(state.basis);

  // Look up mesh bin value
  const axes = basisAxes(state.basis);
  const meshH = (worldH - MESH_LL[axes[0]]) / MESH_WIDTH[axes[0]];
  const meshV = (worldV - MESH_LL[axes[1]]) / MESH_WIDTH[axes[1]];
  const ih = Math.floor(meshH), iv = Math.floor(meshV);
  const dims = SLICE_DIMS[state.basis];
  let valStr = '';
  if (ih >= 0 && ih < dims[0] && iv >= 0 && iv < dims[1] && state.currentSliceData) {
    const idx = iv * dims[0] + ih;
    if (state.currentSliceMask && !state.currentSliceMask[idx]) {
      valStr = '<br>value: no data';
    } else {
      const val = state.currentSliceData[idx] * state.scalingFactor;
      valStr = '<br>' + DISPLAY_VALUE + ': ' + val.toExponential(4);
      // Show extra values (standard_deviation, relative_error, etc.)
      const sliceKey = state.basis === NATIVE_BASIS
        ? NATIVE_BASIS + ':' + state.sliceIdx
        : null;
      for (const evName of EXTRA_VALUE_NAMES) {
        let extraData = null;
        if (state.basis === NATIVE_BASIS && sliceKey) {
          extraData = getExtraSlice(evName, sliceKey);
        }
        if (extraData) {
          const ev = extraData[idx] * (evName === 'relative_error' ? 1 : state.scalingFactor);
          valStr += '<br>' + evName + ': ' + ev.toExponential(4);
        }
      }
    }
  }

  // Look up cell/material from cached outline data
  let geoStr = '';
  if (cachedOutlineData && cachedOutlinePH > 0) {
    const gFracX = (worldH - (state.originH - state.widthH / 2)) / state.widthH;
    const gFracY = (worldV - (state.originV - state.widthV / 2)) / state.widthV;
    const gj = Math.floor(gFracX * cachedOutlinePH);
    const gi = Math.floor((1 - gFracY) * cachedOutlinePV);
    if (gi >= 0 && gi < cachedOutlinePV && gj >= 0 && gj < cachedOutlinePH) {
      const gIdx = (gi * cachedOutlinePH + gj) * 2;
      const cellId = cachedOutlineData[gIdx];
      const matId = cachedOutlineData[gIdx + 1];
      if (cellId >= 0) {
        const cellName = CELL_NAMES[cellId];
        geoStr += '<br>Cell: ' + cellId + (cellName ? ' (' + cellName + ')' : '');
      }
      if (matId >= 0) {
        const matName = MATERIAL_NAMES[matId];
        geoStr += '<br>Material: ' + matId + (matName ? ' (' + matName + ')' : '');
      } else if (cellId >= 0) {
        geoStr += '<br>Material: void';
      }
    }
  }

  // Surface-hover: reconstruct the full 3D world point (worldH/worldV
  // are slice-plane coords; third coord is the slice's mesh-bin center
  // on the perpendicular axis). Tolerance scales with the world-per-
  // pixel ratio so the criterion behaves the same at any zoom.
  let surfaceStr = '';
  if (SURFACE_TABLE.length > 0) {
    const fixedIdx = axes[2];
    const sliceCoord = MESH_LL[fixedIdx] + MESH_WIDTH[fixedIdx] * (state.sliceIdx + 0.5);
    const pt = [0, 0, 0];
    pt[axes[0]] = worldH;
    pt[axes[1]] = worldV;
    pt[fixedIdx] = sliceCoord;
    const tol = Math.max(state.widthH / canvas.width, state.widthV / canvas.height) * 0.5;
    const hits = surfacesAtPoint(pt[0], pt[1], pt[2], tol);
    if (hits.length > 0) {
      surfaceStr = '<br>surface: ' + hits.join(', ');
    }
  }

  const hints = ['Left drag: zoom', 'Right drag: pan', 'Scroll: zoom', 'Double-click: reset'];
  const hintStr = '<br><span style="color:#888;font-size:11px;">' + hints.join('<br>') + '</span>';
  tooltip.innerHTML = labels[0] + ': ' + (worldH * us).toFixed(4) + ' ' + state.units +
    '<br>' + labels[1] + ': ' + (worldV * us).toFixed(4) + ' ' + state.units + valStr + geoStr + surfaceStr + hintStr;
  tooltip.style.display = 'block';
  // position:fixed -- clientX/Y are viewport coords, tooltip floats over
  // the colorbar / legend instead of being clipped by overflow:hidden.
  tooltip.style.left = (e.clientX + 15) + 'px';
  tooltip.style.top = (e.clientY + 15) + 'px';
});
canvas.addEventListener('mouseleave', () => { tooltip.style.display = 'none'; });

// ---- Controls ----
// Basis buttons
document.querySelectorAll('#basis-btns button').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('#basis-btns button').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    state.basis = btn.dataset.val;
    state.sliceIdx = 0;
    initView(state.basis);
    updateSlider();
    renderFrame();
  });
});

// Slice slider
document.getElementById('slice-slider').addEventListener('input', e => {
  state.sliceIdx = +e.target.value;
  updateSliceInfo();
  renderFrame();
});

// Scale
document.querySelectorAll('input[name="scale"]').forEach(inp => {
  inp.addEventListener('change', () => {
    state.logScale = inp.value === 'log';
    renderFrame();
  });
});

// Value range
document.getElementById('vmin-input').addEventListener('change', () => {
  const v = document.getElementById('vmin-input').value;
  state.vmin = (v === '' || v === null) ? null : +v;
  renderFrame();
});
document.getElementById('vmax-input').addEventListener('change', () => {
  const v = document.getElementById('vmax-input').value;
  state.vmax = (v === '' || v === null) ? null : +v;
  renderFrame();
});

// Scaling factor
document.getElementById('scaling-factor').addEventListener('change', () => {
  const v = +document.getElementById('scaling-factor').value;
  state.scalingFactor = (v && isFinite(v)) ? v : 1;
  renderFrame();
});

// Colorscale
document.getElementById('colorscale-select').addEventListener('change', e => {
  state.colorscale = e.target.value;
  currentLUT = buildLUT(state.colorscale, state.levels);
  renderFrame();
});

// Levels
document.getElementById('levels-select').addEventListener('change', e => {
  state.levels = +e.target.value;
  currentLUT = buildLUT(state.colorscale, state.levels);
  renderFrame();
});

// Outline
document.querySelectorAll('input[name="outline"]').forEach(inp => {
  inp.addEventListener('change', () => {
    state.outline = inp.value;
    renderFrame();
  });
});
document.getElementById('outline-color').addEventListener('input', () => {
  const hex = document.getElementById('outline-color').value;
  state.outlineColor = [parseInt(hex.slice(1,3),16), parseInt(hex.slice(3,5),16), parseInt(hex.slice(5,7),16)];
  renderFrame();
});
document.getElementById('outline-width').addEventListener('change', () => {
  state.outlineWidth = +document.getElementById('outline-width').value || 1;
  renderFrame();
});
document.getElementById('outline-pixels').addEventListener('change', () => {
  state.outlinePixels = Math.max(1000, +document.getElementById('outline-pixels').value || OUTLINE_PIXELS);
  renderFrame();
});

// Units
document.querySelectorAll('input[name="units"]').forEach(inp => {
  inp.addEventListener('change', () => {
    state.units = inp.value;
    updateSliceInfo();
    updateAxisLabels();
  });
});

// Font size
document.getElementById('font-size').addEventListener('change', () => {
  state.fontSize = Math.max(6, +document.getElementById('font-size').value || INITIAL_FONT_SIZE);
  renderFrame();
});

// Colorbar label
document.getElementById('colorbar-label-input').addEventListener('change', () => {
  state.colorbarLabel = document.getElementById('colorbar-label-input').value;
  renderColorbar();
});

document.getElementById('show-colorbar').addEventListener('change', () => {
  state.showColorbar = document.getElementById('show-colorbar').checked;
  renderColorbar();
});

// Reset view
document.getElementById('reset-btn').addEventListener('click', () => {
  state.basis = NATIVE_BASIS;
  state.sliceIdx = INITIAL_SLICE_BIN;
  initView(state.basis);
  updateSlider();
  document.querySelectorAll('#basis-btns button').forEach(b => {
    b.classList.toggle('active', b.dataset.val === state.basis);
  });
  renderFrame();
  if (presampledOutlineData) {
    renderPresampledGrid();
    fitCanvas();
  }
});

// Copy Python code
document.getElementById('copy-btn').addEventListener('click', () => {
  const s = state;
  const nativeIndices = SLICE_INDICES[NATIVE_BASIS] || [];
  const sliceList = nativeIndices.join(', ');
  const lines = [
    'plot = tally.interactive_plot(',
    '    basis="' + NATIVE_BASIS + '",',
    '    slices=[' + sliceList + '],',
    '    log_scale=' + (s.logScale ? 'True' : 'False') + ',',
    '    colorscale="' + s.colorscale + '",',
    '    outline=' + (s.outline === 'none' ? 'None' : '"' + s.outline + '"') + ',',
    '    axis_units="' + s.units + '",',
    '    show_colorbar=' + (s.showColorbar ? 'True' : 'False') + ',',
    ')',
  ];
  navigator.clipboard.writeText(lines.join('\n')).then(() => {
    const btn = document.getElementById('copy-btn');
    btn.textContent = 'Copied!';
    setTimeout(() => { btn.textContent = 'Copy Python Code'; }, 1500);
  });
});

// Download PNG
document.getElementById('download-btn').addEventListener('click', () => {
  const us = unitScale(state.units);
  const hMin = (state.originH - state.widthH / 2) * us;
  const hMax = (state.originH + state.widthH / 2) * us;
  const vMin = (state.originV - state.widthV / 2) * us;
  const vMax = (state.originV + state.widthV / 2) * us;
  const labels = basisLabels(state.basis);
  const hLabel = labels[0] + ' (' + state.units + ')';
  const vLabel = labels[1] + ' (' + state.units + ')';

  // Use CSS display size (what the user sees) instead of internal canvas size
  const dispRect = canvas.getBoundingClientRect();
  const plotW = Math.round(dispRect.width * (window.devicePixelRatio || 1));
  const plotH = Math.round(dispRect.height * (window.devicePixelRatio || 1));
  const padL = 80, padR = 120, padT = 40, padB = 60;
  const totalW = padL + plotW + padR;
  const totalH = padT + plotH + padB;

  // Re-render heatmap at full resolution into a temp canvas
  const plotCanvas = document.createElement('canvas');
  plotCanvas.width = plotW; plotCanvas.height = plotH;
  const pc = plotCanvas.getContext('2d');
  const axes = basisAxes(state.basis);
  const meshMinH = MESH_LL[axes[0]], meshMaxH = MESH_UR[axes[0]];
  const meshMinV = MESH_LL[axes[1]], meshMaxV = MESH_UR[axes[1]];
  const viewMinH2 = state.originH - state.widthH / 2;
  const viewMaxH2 = state.originH + state.widthH / 2;
  const viewMinV2 = state.originV - state.widthV / 2;
  const viewMaxV2 = state.originV + state.widthV / 2;
  const dims = SLICE_DIMS[state.basis];
  const nH = dims[0], nV = dims[1];
  const sf = state.scalingFactor;
  const pxUseLog = state.logScale && state.dataMin > 0;
  const pxLogMin = pxUseLog ? Math.log10(state.dataMin) : state.dataMin;
  const pxLogMax = pxUseLog ? Math.log10(state.dataMax) : state.dataMax;
  const pxRange = pxLogMax - pxLogMin || 1;
  const pxMeshWidthH = MESH_WIDTH[axes[0]];
  const pxMeshWidthV = MESH_WIDTH[axes[1]];
  const pxImg = pc.createImageData(plotW, plotH);
  const pxD = pxImg.data;
  const sliceData = state.currentSliceData;
  const sliceMask = state.currentSliceMask;
  for (let row = 0; row < plotH; row++) {
    const worldV = viewMaxV2 - (row + 0.5) / plotH * (viewMaxV2 - viewMinV2);
    const iv = Math.floor((worldV - meshMinV) / pxMeshWidthV);
    for (let col = 0; col < plotW; col++) {
      const worldH = viewMinH2 + (col + 0.5) / plotW * (viewMaxH2 - viewMinH2);
      const ih = Math.floor((worldH - meshMinH) / pxMeshWidthH);
      const px = (row * plotW + col) * 4;
      if (iv < 0 || iv >= nV || ih < 0 || ih >= nH) {
        pxD[px] = 0; pxD[px+1] = 0; pxD[px+2] = 0; pxD[px+3] = 0; continue;
      }
      if (sliceMask && !sliceMask[iv * nH + ih]) {
        pxD[px] = 208; pxD[px+1] = 208; pxD[px+2] = 208; pxD[px+3] = 255; continue;
      }
      const val = sliceData[iv * nH + ih] * sf;
      if (val <= 0 || !isFinite(val)) {
        pxD[px] = 200; pxD[px+1] = 200; pxD[px+2] = 200; pxD[px+3] = 255; continue;
      }
      const norm = pxUseLog ? (Math.log10(val) - pxLogMin) / pxRange : (val - pxLogMin) / pxRange;
      const ci = Math.max(0, Math.min(255, Math.round(norm * 255)));
      pxD[px] = currentLUT[ci*3]; pxD[px+1] = currentLUT[ci*3+1]; pxD[px+2] = currentLUT[ci*3+2]; pxD[px+3] = 255;
    }
  }
  pc.putImageData(pxImg, 0, 0);

  // Composite geometry outline at high resolution
  if (outlinePlotter && state.outline !== 'none') {
    const olAspect = plotW / plotH;
    const olTotalPx = state.outlinePixels;
    const olOpV = Math.max(1, Math.round(Math.sqrt(olTotalPx / olAspect)));
    const olOpH = Math.max(1, Math.round(olTotalPx / olOpV));
    const olAxes = basisAxes(state.basis);
    const olFixedIdx = olAxes[2];
    const olBinIdx = state.sliceIdx;
    const olFixedCoord = MESH_LL[olFixedIdx] + MESH_WIDTH[olFixedIdx] * (olBinIdx + 0.5);
    let olOx, olOy, olOz;
    if (state.basis === 'xy') { olOx = state.originH; olOy = state.originV; olOz = olFixedCoord; }
    else if (state.basis === 'xz') { olOx = state.originH; olOy = olFixedCoord; olOz = state.originV; }
    else { olOx = olFixedCoord; olOy = state.originH; olOz = state.originV; }
    try {
      const olData = outlinePlotter.sampleGrid(state.basis, olOx, olOy, olOz, state.widthH, state.widthV, olOpH, olOpV);
      const olMode = state.outline;
      const olIds = new Int32Array(olOpH * olOpV);
      for (let i = 0; i < olOpV; i++) {
        for (let j = 0; j < olOpH; j++) {
          const idx2 = (i * olOpH + j) * 2;
          olIds[i * olOpH + j] = olMode === 'cell' ? olData[idx2] : olData[idx2 + 1];
        }
      }
      // Render edges at sample resolution (1:1), then scale to plot canvas
      const olW = state.outlineWidth;
      const olC = state.outlineColor;
      const olImg = new ImageData(olOpH, olOpV);
      const olD = olImg.data;
      for (let i = 0; i < olOpV; i++) {
        for (let j = 0; j < olOpH; j++) {
          const myId = olIds[i * olOpH + j];
          if (myId === -1) continue;
          let isEdge = false;
          for (let dd = 1; dd <= olW && !isEdge; dd++) {
            if (j >= dd && olIds[i * olOpH + j - dd] !== myId) isEdge = true;
            if (!isEdge && j + dd < olOpH && olIds[i * olOpH + j + dd] !== myId) isEdge = true;
            if (!isEdge && i >= dd && olIds[(i - dd) * olOpH + j] !== myId) isEdge = true;
            if (!isEdge && i + dd < olOpV && olIds[(i + dd) * olOpH + j] !== myId) isEdge = true;
          }
          if (isEdge) {
            const olPx = (i * olOpH + j) * 4;
            olD[olPx] = olC[0]; olD[olPx+1] = olC[1]; olD[olPx+2] = olC[2]; olD[olPx+3] = 255;
          }
        }
      }
      const olTempCanvas = document.createElement('canvas');
      olTempCanvas.width = olOpH; olTempCanvas.height = olOpV;
      const olTempCtx = olTempCanvas.getContext('2d');
      olTempCtx.putImageData(olImg, 0, 0);
      pc.drawImage(olTempCanvas, 0, 0, plotW, plotH);
    } catch(e) { console.error('PNG outline error:', e); }
  }

  const offscreen = document.createElement('canvas');
  offscreen.width = totalW; offscreen.height = totalH;
  const oc = offscreen.getContext('2d');
  oc.fillStyle = '#ffffff';
  oc.fillRect(0, 0, totalW, totalH);
  oc.drawImage(plotCanvas, padL, padT);
  oc.strokeStyle = '#999'; oc.lineWidth = 1;
  oc.strokeRect(padL + 0.5, padT + 0.5, plotW - 1, plotH - 1);

  const fs = state.fontSize;
  const tickFs = Math.max(8, fs - 2);

  // Title
  oc.fillStyle = '#222'; oc.font = (fs + 2) + 'px sans-serif'; oc.textAlign = 'center';
  oc.fillText(TITLE, padL + plotW/2, 18);

  // Ticks
  oc.fillStyle = '#222'; oc.strokeStyle = '#666'; oc.lineWidth = 1;
  const hTicks = niceTicks(hMin, hMax, Math.floor(plotW / 80));
  const vTicks = niceTicks(vMin, vMax, Math.floor(plotH / 60));
  oc.font = tickFs + 'px monospace'; oc.textAlign = 'center'; oc.textBaseline = 'top';
  for (const val of hTicks) {
    const frac = (val - hMin) / (hMax - hMin);
    if (frac < 0.02 || frac > 0.98) continue;
    const x = padL + frac * plotW;
    oc.beginPath(); oc.moveTo(x, padT + plotH); oc.lineTo(x, padT + plotH + 4); oc.stroke();
    oc.fillText(formatTick(val), x, padT + plotH + 6);
  }
  oc.textAlign = 'right'; oc.textBaseline = 'middle';
  for (const val of vTicks) {
    const frac = (val - vMin) / (vMax - vMin);
    if (frac < 0.02 || frac > 0.98) continue;
    const y = padT + (1 - frac) * plotH;
    oc.beginPath(); oc.moveTo(padL - 4, y); oc.lineTo(padL, y); oc.stroke();
    oc.fillText(formatTick(val), padL - 6, y);
  }
  oc.font = (fs + 1) + 'px sans-serif'; oc.textAlign = 'center'; oc.textBaseline = 'top';
  oc.fillText(hLabel, padL + plotW / 2, padT + plotH + tickFs + 10);
  oc.save();
  oc.translate(14, padT + plotH / 2);
  oc.rotate(-Math.PI / 2);
  oc.fillText(vLabel, 0, 0);
  oc.restore();

  // Colorbar in PNG
  const cbW = 14, cbH = plotH;
  const cbX = padL + plotW + 10;
  for (let i = 0; i < cbH; i++) {
    const t = 1 - i / (cbH - 1);
    const ci = Math.round(t * 255);
    oc.fillStyle = 'rgb(' + currentLUT[ci*3] + ',' + currentLUT[ci*3+1] + ',' + currentLUT[ci*3+2] + ')';
    oc.fillRect(cbX, padT + i, cbW, 1);
  }
  oc.strokeRect(cbX, padT, cbW, cbH);
  oc.font = Math.max(7, fs - 3) + 'px monospace'; oc.textAlign = 'left'; oc.textBaseline = 'middle';
  const useLog = state.logScale && state.dataMin > 0;
  for (let i = 0; i <= 5; i++) {
    const frac = i / 5;
    let val;
    if (useLog) val = Math.pow(10, Math.log10(state.dataMin) + frac * (Math.log10(state.dataMax) - Math.log10(state.dataMin)));
    else val = state.dataMin + frac * (state.dataMax - state.dataMin);
    oc.fillText(val.toExponential(1), cbX + cbW + 3, padT + (1-frac) * cbH);
  }
  // Colorbar label in PNG
  oc.font = fs + 'px sans-serif'; oc.textAlign = 'center';
  oc.save();
  oc.translate(cbX + cbW + 70, padT + cbH / 2);
  oc.rotate(-Math.PI / 2);
  oc.fillText(state.colorbarLabel, 0, 0);
  oc.restore();

  const link = document.createElement('a');
  link.download = 'tally_plot.png';
  link.href = offscreen.toDataURL('image/png');
  link.click();
});

// ---- Resize ----
window.addEventListener('resize', () => { fitCanvas(); updateAxisLabels(); renderColorbar(); });

// ---- CSG plotter (same as geometry viewer) ----
class JsCsgPlotter {
  constructor(geo) { this.cells = geo.cells; }
  sampleGrid(basis, ox, oy, oz, wh, wv, ph, pv) {
    const data = new Int32Array(ph * pv * 2);
    const halfH = wh / 2, halfV = wv / 2;
    for (let i = 0; i < pv; i++) {
      const v = pv > 1 ? halfV - wv * i / (pv - 1) : 0;
      for (let j = 0; j < ph; j++) {
        const h = ph > 1 ? -halfH + wh * j / (ph - 1) : 0;
        let px, py, pz;
        switch(basis) {
          case 'xy': px=ox+h; py=oy+v; pz=oz; break;
          case 'xz': px=ox+h; py=oy; pz=oz+v; break;
          case 'yz': px=ox; py=oy+h; pz=oz+v; break;
          default: px=ox+h; py=oy+v; pz=oz;
        }
        const idx = (i * ph + j) * 2;
        let found = false;
        for (const cell of this.cells) {
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
}

function regionContains(region, x, y, z) { return evalExpr(region.expr, x, y, z); }

function evalExpr(expr, x, y, z) {
  if (expr.Halfspace) {
    const hs = expr.Halfspace;
    if (hs.Above) return surfaceEval(hs.Above, x, y, z) > 0;
    if (hs.Below) return surfaceEval(hs.Below, x, y, z) < 0;
  }
  if (expr.Intersection) return evalExpr(expr.Intersection[0], x, y, z) && evalExpr(expr.Intersection[1], x, y, z);
  if (expr.Union) return evalExpr(expr.Union[0], x, y, z) || evalExpr(expr.Union[1], x, y, z);
  if (expr.Complement) return !evalExpr(expr.Complement, x, y, z);
  return false;
}

function surfacesAtPoint(x, y, z, tol) {
  // Same JS-side surface lookup the geometry viewer's tooltip uses.
  // Iterates SURFACE_TABLE; for typical N ≤ 100 surfaces this is
  // sub-microsecond per mousemove.
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
  if (k.Plane) return k.Plane.a*x + k.Plane.b*y + k.Plane.c*z - k.Plane.d;
  if (k.Sphere) { const dx=x-k.Sphere.x0,dy=y-k.Sphere.y0,dz=z-k.Sphere.z0; return Math.sqrt(dx*dx+dy*dy+dz*dz)-k.Sphere.radius; }
  if (k.Cylinder) { const ax=k.Cylinder.axis,or=k.Cylinder.origin; const vx=x-or[0],vy=y-or[1],vz=z-or[2]; const dot=vx*ax[0]+vy*ax[1]+vz*ax[2]; const dx=vx-dot*ax[0],dy=vy-dot*ax[1],dz=vz-dot*ax[2]; return Math.sqrt(dx*dx+dy*dy+dz*dz)-k.Cylinder.radius; }
  if (k.ZTorus) { const t=k.ZTorus; const dx=x-t.x0,dy=y-t.y0,dz=z-t.z0; const rho=Math.sqrt(dx*dx+dy*dy); return (rho-t.a)*(rho-t.a)/(t.c*t.c)+dz*dz/(t.b*t.b)-1.0; }
  return 0;
}


// ---- Init ----
// ---- Init ----
// Decode presampled outline for instant first render
let presampledOutlineData = null;
if (PRESAMPLED_OUTLINE_B64) {
  const raw = atob(PRESAMPLED_OUTLINE_B64);
  const bytes = new Uint8Array(raw.length);
  for (let i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i);
  presampledOutlineData = new Int32Array(bytes.buffer);
}

function renderPresampledGrid() {
  if (!presampledOutlineData || state.outline === 'none') return;
  const opH = PRESAMPLED_OUTLINE_PH;
  const opV = PRESAMPLED_OUTLINE_PV;
  // Cache for tooltip lookup
  cachedOutlineData = presampledOutlineData;
  cachedOutlinePH = opH;
  cachedOutlinePV = opV;
  const outlineMode = state.outline;
  const outlineIds = new Int32Array(opH * opV);
  for (let i = 0; i < opV; i++) {
    for (let j = 0; j < opH; j++) {
      const idx = (i * opH + j) * 2;
      outlineIds[i * opH + j] = outlineMode === 'cell' ? presampledOutlineData[idx] : presampledOutlineData[idx + 1];
    }
  }
  olCanvas.width = opH;
  olCanvas.height = opV;
  const img = olCtx.createImageData(opH, opV);
  const d = img.data;
  const ow = state.outlineWidth;
  const oc = state.outlineColor;
  for (let i = 0; i < opV; i++) {
    for (let j = 0; j < opH; j++) {
      const myId = outlineIds[i * opH + j];
      if (myId === -1) continue;
      let isEdge = false;
      for (let dd = 1; dd <= ow && !isEdge; dd++) {
        if (j >= dd && outlineIds[i * opH + j - dd] !== myId) isEdge = true;
        if (!isEdge && j + dd < opH && outlineIds[i * opH + j + dd] !== myId) isEdge = true;
        if (!isEdge && i >= dd && outlineIds[(i - dd) * opH + j] !== myId) isEdge = true;
        if (!isEdge && i + dd < opV && outlineIds[(i + dd) * opH + j] !== myId) isEdge = true;
      }
      if (isEdge) {
        const px = (i * opH + j) * 4;
        d[px] = oc[0]; d[px+1] = oc[1]; d[px+2] = oc[2]; d[px+3] = 255;
      }
    }
  }
  olCtx.putImageData(img, 0, 0);
}

async function init() {
  initView(state.basis);
  // Set initial slice to the requested bin index (direct bin index)
  state.sliceIdx = INITIAL_SLICE_BIN;
  updateSlider();

  // Render initial frame with presampled outline (instant)
  renderFrame();
  if (presampledOutlineData) {
    renderPresampledGrid();
    fitCanvas();
  }

  // Load outline plotter if geometry provided (for subsequent slices)
  if (GEOMETRY_JSON) {
    if (GEOMETRY_KIND === 'csg') {
      outlinePlotter = new JsCsgPlotter(GEOMETRY_JSON);
      if (!presampledOutlineData) renderFrame(); // only re-render if no presampled
    } else if (GEOMETRY_KIND === 'mesh' && typeof YAMT_WASM_BASE64 !== 'undefined') {
      statusEl.textContent = 'Loading WASM...';
      try {
        const wasmBytes = Uint8Array.from(atob(YAMT_WASM_BASE64), c => c.charCodeAt(0));
        const wasmModule = await WebAssembly.compile(wasmBytes);
        yamtWasmInitSync(wasmModule);
        const geoJson = JSON.stringify(GEOMETRY_JSON);
        const wasmPlotter = new YamtWasmMeshPlotter(geoJson);
        outlinePlotter = {
          sampleGrid(basis, ox, oy, oz, wh, wv, ph, pv) {
            return wasmPlotter.sampleGrid(basis, ox, oy, oz, wh, wv, ph, pv);
          }
        };
        if (!presampledOutlineData) renderFrame();
      } catch(e) {
        console.error('WASM load error:', e);
        statusEl.textContent = 'WASM failed: ' + e;
      }
    }
  }
}

