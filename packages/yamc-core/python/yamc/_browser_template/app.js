const WASM_B64 = "__WASM_B64__";
const MODEL_JSON = __MODEL_JSON__;
// EMBEDDED_XS = { nuclide_name: { filename: base64_bytes, ... }, ... }
// May be `{}` if the engineer chose `embed_cross_sections=False`. When
// non-empty, those nuclides are installed into the in-memory storage at
// init so they don't need to be fetched from the network.
const EMBEDDED_XS = __EMBEDDED_XS__;
// EMBEDDED_PHOTON_XS = { element_symbol: { filename: base64_bytes, ... } }
// Per-element photon data (e.g. "Fe", "Be"). Same idea as EMBEDDED_XS but
// keyed by element; `{}` unless the engineer embedded photon data for a
// transport_secondary_photons model. Installed under /<El>.arrow/ at init.
const EMBEDDED_PHOTON_XS = __EMBEDDED_PHOTON_XS__;
// SECTION_FLAGS = {geometry, surfaces, materials, tallies} -- each bool.
// When false, the matching `<fieldset data-section="...">` is removed
// from the DOM (and its editor code is skipped). Set by the engineer
// via Model.to_html(show_geometry=, show_surfaces=, ...).
const SECTION_FLAGS = __SECTION_FLAGS__;

// Drop hidden sections before anything else binds to their DOM nodes.
// The editors below early-return when their root element is missing.
for (const [section, visible] of Object.entries(SECTION_FLAGS)) {
  if (visible) continue;
  for (const el of document.querySelectorAll(`[data-section="${section}"]`)) {
    el.remove();
  }
}

// Delegated fullscreen-button handler. Buttons carry
// `data-target="<iframe id>"`; click → request fullscreen on the
// matching iframe. The iframes themselves carry the fullscreen
// permission attribute so requestFullscreen() is granted. Browsers
// handle Esc to exit.
document.addEventListener("click", (e) => {
  const btn = e.target.closest(".fullscreen-btn");
  if (!btn) return;
  const target = document.getElementById(btn.dataset.target);
  if (!target || !target.requestFullscreen) return;
  target.requestFullscreen().catch((err) => {
    console.warn("fullscreen request failed:", err);
  });
});

// --- wasm-bindgen JS glue (inlined as a module via blob URL) ---
const JS_SRC = __JS_SRC_JSON__;
const jsBlob = new Blob([JS_SRC], { type: "application/javascript" });
const mod = await import(URL.createObjectURL(jsBlob));
const init = mod.default;
const { WasmSimulation } = mod;

function b64ToBytes(b64) {
  const bin = atob(b64);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}
const wasmBytes = b64ToBytes(WASM_B64);
const wasmResp = new Response(wasmBytes, { headers: { "Content-Type": "application/wasm" } });
await init(wasmResp);

const sim = new WasmSimulation();
sim.load_model_json(MODEL_JSON);

// Show the engineer's pre-run mesh-tally snapshot on load whenever
// it's present -- paired with the seeded result-panel text from the
// same pre-run, both views read consistently. The Simulate handler
// also unhides it (in case some path landed without seeded results).
{
  const f = document.getElementById('tally-plot-fieldset');
  const iframe = document.getElementById('tally-plot-iframe');
  if (f && iframe && iframe.getAttribute('srcdoc')) {
    f.hidden = false;
  }
}

// Install any embedded cross-section data into the in-memory storage.
// Each entry: EMBEDDED_XS[nuclide][filename] is base64 bytes. Path
// convention `/<nuclide>.arrow/<filename>` matches what the Rust side
// expects when read_nuclear_data runs.
for (const [nuc, files] of Object.entries(EMBEDDED_XS)) {
  for (const [filename, b64] of Object.entries(files)) {
    sim.add_file(`/${nuc}.arrow/${filename}`, b64ToBytes(b64));
  }
}
// Embedded photon data is per-element; register under /<El>.arrow/.
for (const [el, files] of Object.entries(EMBEDDED_PHOTON_XS)) {
  for (const [filename, b64] of Object.entries(files)) {
    sim.add_file(`/${el}.arrow/${filename}`, b64ToBytes(b64));
  }
}

// Working copy of the model -- the materials editor mutates this and
// pushes it back through `sim.load_model_json` on Apply.
const model = JSON.parse(MODEL_JSON);
const csgGeom = (model.geometry && model.geometry.Csg) || null;

// Track which nuclides we have data files for in the in-memory storage.
// Pre-populated with any embedded XS nuclides, then grown by Fetch.
const loadedNuclides = new Set(Object.keys(EMBEDDED_XS));
// Photon data we hold, keyed by element symbol (parallel to loadedNuclides).
const loadedElements = new Set(Object.keys(EMBEDDED_PHOTON_XS));
// Whether the recipient is *allowed* to fetch -- distinct from
// "nothing left to fetch". An engineer can ship a model that already has
// all its data embedded; the fetch button then shows "All cross sections
// embedded" and stays disabled even if the recipient adds a nuclide that
// happens to be in EMBEDDED_XS (unlikely path; UI is intentionally simple).
const ALL_EMBEDDED_AT_BUILD =
  (Object.keys(EMBEDDED_XS).length > 0 || Object.keys(EMBEDDED_PHOTON_XS).length > 0)
  && sim.model_required_nuclides().split(",").filter(Boolean)
       .every((n) => EMBEDDED_XS[n] !== undefined)
  && sim.model_required_elements().split(",").filter(Boolean)
       .every((e) => EMBEDDED_PHOTON_XS[e] !== undefined);

const summary = document.getElementById("model-summary");
function renderSummary() {
  const required = sim.model_required_nuclides().split(",").filter(Boolean);
  const cellCount = csgGeom ? csgGeom.cells.length : 0;
  const matCount = csgGeom ? csgGeom.materials.length : 0;
  summary.textContent =
`Cells: ${cellCount}  Materials: ${matCount}  Sources: ${model.sources.length}  Tallies: ${model.tallies.length}
Required nuclides: ${required.join(", ") || "(none)"}
Default total particles: ${model.total_particles ?? 1000}  (seed ${model.seed ?? 1})`;
  return required;
}

function refreshSimGate() {
  const required = sim.model_required_nuclides().split(",").filter(Boolean);
  const reqElements = sim.model_required_elements().split(",").filter(Boolean);
  const missing = required.filter((n) => !loadedNuclides.has(n))
    .concat(reqElements.filter((e) => !loadedElements.has(e)));
  document.getElementById("sim").disabled = missing.length > 0;
  const dlBtn = document.getElementById("dl");
  const statusEl = document.getElementById("status");
  if (missing.length === 0) {
    // Everything required is already in memory. Grey out fetch.
    dlBtn.disabled = true;
    if (ALL_EMBEDDED_AT_BUILD) {
      dlBtn.textContent = "All cross sections embedded";
      if (statusEl) statusEl.textContent = "Cross sections were embedded, html is offline-ready.";
    } else {
      dlBtn.textContent = "All cross sections loaded";
    }
  } else {
    dlBtn.disabled = false;
    dlBtn.textContent =
      loadedNuclides.size === 0 ? "Fetch cross sections" : `Fetch missing (${missing.join(", ")})`;
  }
  return missing;
}

renderSummary();

// Pre-fill the run-parameter input with the model's default.
document.getElementById("total-particles").value = model.total_particles ?? 1000;

// --- Live geometry & source plot (full viewer via wasm plotHtml) ---
//
// On init, ask wasm for the full interactive-viewer HTML for the
// currently loaded model and inject it into the iframe. On every
// Apply (materials / surfaces / tallies), re-call plotHtml so the
// viewer reflects the post-edit geometry. The same viewer code that
// model.plot()._repr_html_() ships in Python -- same look, same
// controls, same features (pan/zoom/legend/colour-by-cell/...).
const PLOT = (() => {
  const iframe = document.getElementById('plot-iframe');
  if (!iframe) return { redraw: () => {} }; // geometry section hidden
  // Show a placeholder while we wait for the deferred build below;
  // the iframe is a ~80 KB HTML doc that takes the browser tens of ms
  // to parse + the viewer JS hundreds more to spin up.
  iframe.srcdoc = '<div style="color:#888;font:13px system-ui;padding:1em;">Building plot…</div>';
  let pending = false;
  function redraw() {
    // Defer to the next frame so the editor UI paints first, and so
    // a burst of Apply-driven redraws (mat → surf → tally) coalesces
    // into a single iframe rebuild.
    if (pending) return;
    pending = true;
    requestAnimationFrame(() => {
      pending = false;
      try {
        iframe.srcdoc = sim.plotHtml('{}');
      } catch (e) {
        iframe.srcdoc = '<pre style="color:#c00;padding:1em;font:13px monospace;">'
          + 'plotHtml failed: ' + String(e).replace(/</g, '&lt;') + '</pre>';
      }
    });
  }
  redraw();
  return { redraw };
})();

// --- Material editor ---
const matsHost = document.getElementById("materials-edit");
function renderMaterialsEditor() {
  if (!matsHost) return; // section hidden by SECTION_FLAGS
  matsHost.innerHTML = "";
  if (!csgGeom || csgGeom.materials.length === 0) {
    matsHost.textContent = "This model has no editable materials.";
    document.getElementById("apply-mats").disabled = true;
    return;
  }
  csgGeom.materials.forEach((mat, mIdx) => {
    const card = document.createElement("div");
    card.className = "mat-card";

    const head = document.createElement("div");
    head.className = "mat-name";
    head.textContent = mat.name || `material ${mIdx + 1}`;
    card.appendChild(head);

    // Density
    const densRow = document.createElement("div");
    densRow.className = "mat-row";
    const densLbl = document.createElement("span");
    densLbl.textContent = `Density (${mat.density_units || "g/cm3"}):`;
    const densInput = document.createElement("input");
    densInput.type = "number"; densInput.step = "any"; densInput.value = mat.density;
    densInput.dataset.matIdx = mIdx; densInput.dataset.role = "density";
    densRow.appendChild(densLbl); densRow.appendChild(densInput);
    card.appendChild(densRow);

    // Composition rows
    const compLbl = document.createElement("div");
    compLbl.style.marginTop = "0.4em";
    compLbl.textContent = `Composition (${mat.fraction_type} fraction):`;
    card.appendChild(compLbl);

    const order = (mat.nuclide_input_order && mat.nuclide_input_order.length)
      ? mat.nuclide_input_order.slice()
      : Object.keys(mat.nuclides);
    const compBox = document.createElement("div");
    compBox.dataset.matIdx = mIdx;
    function addRow(name, frac) {
      const row = document.createElement("div");
      row.className = "mat-row";
      const nameInput = document.createElement("input");
      nameInput.type = "text"; nameInput.value = name; nameInput.dataset.role = "nuc-name";
      nameInput.placeholder = "Li6";
      const fracInput = document.createElement("input");
      fracInput.type = "number"; fracInput.step = "any"; fracInput.value = frac;
      fracInput.dataset.role = "nuc-frac"; fracInput.placeholder = "0.5";
      const rm = document.createElement("button");
      rm.className = "rm"; rm.textContent = "✕"; rm.title = "remove";
      rm.onclick = () => row.remove();
      row.appendChild(nameInput); row.appendChild(fracInput); row.appendChild(rm);
      compBox.appendChild(row);
    }
    order.forEach((n) => addRow(n, mat.nuclides[n]));
    card.appendChild(compBox);

    const addBtn = document.createElement("button");
    addBtn.textContent = "+ add nuclide";
    addBtn.style.marginTop = "0.3em";
    addBtn.onclick = () => addRow("", "");
    card.appendChild(addBtn);

    const meta = document.createElement("div");
    meta.className = "mat-meta";
    meta.textContent = `fraction_type=${mat.fraction_type}  temperature=${mat.temperature}  (read-only)`;
    card.appendChild(meta);

    matsHost.appendChild(card);
  });
}
renderMaterialsEditor();

// --- Surface editor (CSG only) ---
// Walks every cell.region.expr, collects each `Halfspace` leaf, then
// deduplicates by JSON of `{kind, boundary}` so a surface shared
// between cells (e.g. the inner sphere of a shell) shows up as ONE
// card and edits propagate to every occurrence.
function walkRegionHalfspaces(node, visit) {
  if (!node || typeof node !== "object") return;
  if (node.Halfspace !== undefined) {
    const side = node.Halfspace.Below !== undefined ? "Below" : "Above";
    visit(node.Halfspace[side]);
    return;
  }
  if (node.Union !== undefined) { walkRegionHalfspaces(node.Union[0], visit); walkRegionHalfspaces(node.Union[1], visit); return; }
  if (node.Intersection !== undefined) { walkRegionHalfspaces(node.Intersection[0], visit); walkRegionHalfspaces(node.Intersection[1], visit); return; }
  if (node.Complement !== undefined) { walkRegionHalfspaces(node.Complement, visit); return; }
}

function collectSurfaces() {
  // Map from JSON-key -> { kindName, params, boundary, name, occurrences: [surfaceRef] }.
  // Key includes `name` and `surface_id` so surfaces the engineer labelled
  // differently stay distinct even if their parameters coincide.
  const out = new Map();
  if (!csgGeom) return out;
  for (const cell of csgGeom.cells) {
    walkRegionHalfspaces(cell.region.expr, (surf) => {
      const key = JSON.stringify({
        kind: surf.kind,
        boundary: surf.boundary,
        name: surf.name ?? null,
        surface_id: surf.surface_id ?? null,
      });
      let group = out.get(key);
      if (!group) {
        const kindName = Object.keys(surf.kind)[0];
        const params = surf.kind[kindName];
        group = {
          kindName,
          spec: surfaceSpec(kindName, params),
          boundary: surf.boundary,
          name: surf.name ?? null,
          surfaceId: surf.surface_id ?? null,
          occurrences: [],
        };
        out.set(key, group);
      }
      group.occurrences.push(surf);
    });
  }
  return out;
}

function surfaceSpec(kindName, p) {
  switch (kindName) {
    case "Sphere":
      return [
        { name: "x0", type: "scalar", value: p.x0 },
        { name: "y0", type: "scalar", value: p.y0 },
        { name: "z0", type: "scalar", value: p.z0 },
        { name: "radius", type: "scalar", value: p.radius },
      ];
    case "Plane":
      return [
        { name: "a", type: "scalar", value: p.a },
        { name: "b", type: "scalar", value: p.b },
        { name: "c", type: "scalar", value: p.c },
        { name: "d", type: "scalar", value: p.d },
      ];
    case "Cylinder":
      return [
        { name: "axis", type: "vec3", value: p.axis.slice() },
        { name: "origin", type: "vec3", value: p.origin.slice() },
        { name: "radius", type: "scalar", value: p.radius },
      ];
    case "ZTorus":
      return [
        { name: "x0", type: "scalar", value: p.x0 },
        { name: "y0", type: "scalar", value: p.y0 },
        { name: "z0", type: "scalar", value: p.z0 },
        { name: "a", type: "scalar", value: p.a },
        { name: "b", type: "scalar", value: p.b },
        { name: "c", type: "scalar", value: p.c },
      ];
    default:
      return [];
  }
}

let surfaceGroups = collectSurfaces();
const surfsHost = document.getElementById("surfaces-edit");
const surfsFs = document.getElementById("surfaces-fieldset");

function renderSurfacesEditor() {
  if (!surfsHost) return; // section hidden by SECTION_FLAGS
  surfsHost.innerHTML = "";
  if (!csgGeom || surfaceGroups.size === 0) {
    surfsFs.hidden = true;
    return;
  }
  surfsFs.hidden = false;
  let idx = 0;
  for (const [key, group] of surfaceGroups) {
    idx += 1;
    const card = document.createElement("div");
    card.className = "surf-card";
    const head = document.createElement("div");
    head.className = "surf-head";
    // Prefer the engineer-provided name; fall back to surface_id, then to "#idx".
    const label = group.name
      ? `${group.name}  (${group.kindName})`
      : group.surfaceId != null
        ? `surface ${group.surfaceId}  (${group.kindName})`
        : `${group.kindName} #${idx}`;
    head.textContent = `${label}  -- boundary: ${group.boundary}`;
    card.appendChild(head);

    for (const field of group.spec) {
      const row = document.createElement("div");
      row.className = "surf-row";
      const lbl = document.createElement("label");
      lbl.textContent = field.name;
      row.appendChild(lbl);
      if (field.type === "scalar") {
        const inp = document.createElement("input");
        inp.type = "number"; inp.step = "any"; inp.value = field.value;
        inp.dataset.field = field.name;
        row.appendChild(inp);
      } else {
        // vec3 -- three inputs
        for (let i = 0; i < 3; i++) {
          const inp = document.createElement("input");
          inp.type = "number"; inp.step = "any"; inp.value = field.value[i];
          inp.dataset.field = field.name; inp.dataset.idx = String(i);
          row.appendChild(inp);
        }
      }
      card.appendChild(row);
    }

    const occ = document.createElement("div");
    occ.className = "surf-occ";
    occ.textContent = `Used by ${group.occurrences.length} halfspace${group.occurrences.length === 1 ? "" : "s"}.`;
    card.appendChild(occ);

    card.dataset.surfKey = key;
    surfsHost.appendChild(card);
  }
}
renderSurfacesEditor();

const applySurfsBtn = document.getElementById("apply-surfs");
if (applySurfsBtn) applySurfsBtn.onclick = () => {
  const surfStatus = document.getElementById("surf-status");
  surfStatus.classList.remove("err");
  // Build new params per card, then write to every occurrence in the group.
  try {
    surfsHost.querySelectorAll(".surf-card").forEach((card) => {
      const key = card.dataset.surfKey;
      const group = surfaceGroups.get(key);
      if (!group) return;
      // Read the form back into a params object matching the kind's schema.
      const newParams = {};
      for (const field of group.spec) {
        if (field.type === "scalar") {
          const inp = card.querySelector(`input[data-field="${field.name}"]`);
          const v = parseFloat(inp.value);
          if (!Number.isFinite(v)) throw new Error(`${group.kindName}: ${field.name} not a number`);
          newParams[field.name] = v;
        } else {
          const arr = [0, 1, 2].map((i) => {
            const inp = card.querySelector(`input[data-field="${field.name}"][data-idx="${i}"]`);
            const v = parseFloat(inp.value);
            if (!Number.isFinite(v)) throw new Error(`${group.kindName}: ${field.name}[${i}] not a number`);
            return v;
          });
          newParams[field.name] = arr;
        }
      }
      // Sanity-checks per kind.
      if (group.kindName === "Sphere" && !(newParams.radius > 0)) throw new Error("Sphere radius must be > 0");
      if (group.kindName === "Cylinder" && !(newParams.radius > 0)) throw new Error("Cylinder radius must be > 0");
      if (group.kindName === "Cylinder") {
        const ax = newParams.axis;
        if (ax[0] * ax[0] + ax[1] * ax[1] + ax[2] * ax[2] < 1e-12) throw new Error("Cylinder axis cannot be zero vector");
      }
      // Push to every occurrence -- same identity reference across cells.
      for (const surf of group.occurrences) {
        surf.kind[group.kindName] = newParams;
      }
    });
  } catch (e) {
    surfStatus.textContent = e.message;
    surfStatus.classList.add("err");
    return;
  }
  // Reload the updated model.
  try {
    sim.load_model_json(JSON.stringify(model));
  } catch (e) {
    surfStatus.textContent = `load_model_json failed: ${e} -- model may be in an inconsistent state, refresh the page.`;
    surfStatus.classList.add("err");
    return;
  }
  // Re-collect surface groups because the JSON keys (which include params) just changed.
  surfaceGroups = collectSurfaces();
  renderSurfacesEditor();
  renderSummary();
  PLOT.redraw();
  surfStatus.textContent = "Applied. If the simulation crashes or gives nonsense, your geometry change probably broke a cell.";
};

// --- Tally / score editor ---
// Curated list of presets. Each entry has a human label + the exact
// Score JSON that yamc round-trips. Add another to the bottom and it
// shows up in the dropdown -- the recipient doesn't need a yamc install.
const SCORE_PRESETS = [
  { label: "flux",                              json: { Flux: null } },
  { label: "heating",                           json: { Heating: null } },
  { label: "heating-local",                     json: { HeatingLocal: null } },
  { label: "damage-energy",                     json: { DamageEnergy: null } },
  { label: "H1-production (MT=203)",            json: { Production: { mt: 203 } } },
  { label: "H2-production (MT=204)",            json: { Production: { mt: 204 } } },
  { label: "H3-production (tritium, MT=205)",   json: { Production: { mt: 205 } } },
  { label: "He3-production (MT=206)",           json: { Production: { mt: 206 } } },
  { label: "He4-production (MT=207)",           json: { Production: { mt: 207 } } },
  { label: "total (MT=1)",                      json: { ReactionRate: { mt: 1,   display_name: "total" } } },
  { label: "elastic (MT=2)",                    json: { ReactionRate: { mt: 2,   display_name: "elastic" } } },
  { label: "inelastic (MT=4)",                  json: { ReactionRate: { mt: 4,   display_name: "inelastic" } } },
  { label: "(n,2n) (MT=16)",                    json: { ReactionRate: { mt: 16,  display_name: null } } },
  { label: "(n,3n) (MT=17)",                    json: { ReactionRate: { mt: 17,  display_name: null } } },
  { label: "fission (MT=18)",                   json: { ReactionRate: { mt: 18,  display_name: "fission" } } },
  { label: "absorption (MT=27)",                json: { ReactionRate: { mt: 27,  display_name: "absorption" } } },
  { label: "(n,γ) (MT=102)",                    json: { ReactionRate: { mt: 102, display_name: null } } },
  { label: "(n,p) (MT=103)",                    json: { ReactionRate: { mt: 103, display_name: null } } },
  { label: "(n,d) (MT=104)",                    json: { ReactionRate: { mt: 104, display_name: null } } },
  { label: "(n,t) (MT=105)",                    json: { ReactionRate: { mt: 105, display_name: null } } },
  { label: "(n,α) (MT=107)",                    json: { ReactionRate: { mt: 107, display_name: null } } },
  { label: "coherent-scatter (photon)",         json: { PhotonXS: { component: "Coherent" } } },
  { label: "incoherent-scatter (photon)",       json: { PhotonXS: { component: "Incoherent" } } },
  { label: "pair-production (photon)",          json: { PhotonXS: { component: "PairProduction" } } },
  { label: "photoelectric (photon)",            json: { PhotonXS: { component: "Photoelectric" } } },
];

// Populate the <datalist> so the input gets browser-native autocomplete.
{
  const dl = document.getElementById("score-presets");
  for (const p of SCORE_PRESETS) {
    const opt = document.createElement("option");
    opt.value = p.label;
    dl.appendChild(opt);
  }
}

// Find a human label for an existing score's JSON (by deep equality of
// the score shape). Falls back to a JSON-ified blob.
function scoreLabel(scoreJson) {
  const key = JSON.stringify(scoreJson);
  for (const p of SCORE_PRESETS) {
    if (JSON.stringify(p.json) === key) return p.label;
  }
  // ReactionRate with an arbitrary MT we don't have a label for.
  if (scoreJson.ReactionRate && typeof scoreJson.ReactionRate.mt === "number") {
    return `MT=${scoreJson.ReactionRate.mt} (ReactionRate)`;
  }
  return JSON.stringify(scoreJson);
}

const talliesHost = document.getElementById("tallies-edit");
const talliesFs = document.getElementById("tallies-fieldset");

function renderTalliesEditor() {
  if (!talliesHost) return; // section hidden by SECTION_FLAGS
  talliesHost.innerHTML = "";
  if (!model.tallies || model.tallies.length === 0) {
    talliesFs.hidden = true;
    return;
  }
  talliesFs.hidden = false;
  model.tallies.forEach((tally, tIdx) => {
    const card = document.createElement("div");
    card.className = "tally-card";
    card.dataset.tallyIdx = tIdx;

    const head = document.createElement("div");
    head.className = "tally-head";
    head.textContent = tally.name || `tally ${tIdx + 1}`;
    card.appendChild(head);

    // Current scores as chips
    const chipsBox = document.createElement("div");
    function renderChips() {
      chipsBox.innerHTML = "";
      tally.scores.forEach((s, sIdx) => {
        const chip = document.createElement("span");
        chip.className = "score-chip";
        chip.appendChild(document.createTextNode(scoreLabel(s)));
        const rm = document.createElement("button");
        rm.className = "rm"; rm.textContent = "✕"; rm.title = "remove";
        rm.onclick = () => {
          tally.scores.splice(sIdx, 1);
          renderChips();
        };
        chip.appendChild(rm);
        chipsBox.appendChild(chip);
      });
      if (tally.scores.length === 0) {
        const blank = document.createElement("span");
        blank.style.color = "#c00";
        blank.textContent = "(no scores -- add at least one before Apply)";
        chipsBox.appendChild(blank);
      }
    }
    renderChips();
    card.appendChild(chipsBox);

    // Add-score row
    const addRow = document.createElement("div");
    addRow.className = "tally-add";
    const input = document.createElement("input");
    input.type = "text"; input.setAttribute("list", "score-presets");
    input.placeholder = "Type to search (e.g. H3-prod, (n,gamma), 102, …)";
    input.size = 40;
    const addBtn = document.createElement("button");
    addBtn.textContent = "+ add score";
    addBtn.onclick = () => {
      const v = input.value.trim();
      if (!v) return;
      // Look up by label first; then accept a raw MT integer as a fallback.
      const preset = SCORE_PRESETS.find((p) => p.label === v);
      let newScore;
      if (preset) {
        newScore = JSON.parse(JSON.stringify(preset.json));
      } else if (/^\d+$/.test(v)) {
        newScore = { ReactionRate: { mt: parseInt(v, 10), display_name: null } };
      } else {
        // Treat as a substring match against the preset list.
        const partial = SCORE_PRESETS.find((p) => p.label.toLowerCase().includes(v.toLowerCase()));
        if (!partial) {
          tallyStatus.textContent = `Unknown score: ${v}`;
          tallyStatus.classList.add("err");
          return;
        }
        newScore = JSON.parse(JSON.stringify(partial.json));
      }
      // Reject duplicates within the same tally.
      const key = JSON.stringify(newScore);
      if (tally.scores.some((s) => JSON.stringify(s) === key)) {
        tallyStatus.textContent = "That score is already in this tally.";
        tallyStatus.classList.add("err");
        return;
      }
      tallyStatus.classList.remove("err");
      tallyStatus.textContent = "";
      tally.scores.push(newScore);
      renderChips();
      input.value = "";
    };
    addRow.appendChild(input); addRow.appendChild(addBtn);
    card.appendChild(addRow);

    const meta = document.createElement("div");
    meta.className = "tally-meta";
    const filt = (tally.filters || []).map((f) => Object.keys(f)[0]).join(", ") || "(none)";
    meta.textContent = `Filters: ${filt}  (read-only)`;
    card.appendChild(meta);

    talliesHost.appendChild(card);
  });
}
const tallyStatus = document.getElementById("tally-status");
renderTalliesEditor();

const applyTalliesBtn = document.getElementById("apply-tallies");
if (applyTalliesBtn) applyTalliesBtn.onclick = () => {
  tallyStatus.classList.remove("err");
  // Validate: every tally needs at least one score.
  for (let i = 0; i < model.tallies.length; i++) {
    if (model.tallies[i].scores.length === 0) {
      tallyStatus.textContent = `Tally "${model.tallies[i].name || i + 1}" has no scores.`;
      tallyStatus.classList.add("err");
      return;
    }
  }
  try {
    sim.load_model_json(JSON.stringify(model));
  } catch (e) {
    tallyStatus.textContent = `load_model_json failed: ${e}`;
    tallyStatus.classList.add("err");
    return;
  }
  tallyStatus.textContent = "Applied. Click Simulate to see the new score breakdown.";
};

const applyMatsBtn = document.getElementById("apply-mats");
if (applyMatsBtn) applyMatsBtn.onclick = () => {
  const matStatus = document.getElementById("mat-status");
  matStatus.classList.remove("err");
  // Read each material card back into the working model.
  try {
    matsHost.querySelectorAll(".mat-card").forEach((card, mIdx) => {
      const mat = csgGeom.materials[mIdx];
      const density = parseFloat(card.querySelector('input[data-role="density"]').value);
      if (!(density > 0)) throw new Error(`${mat.name || "material " + (mIdx + 1)}: density must be > 0`);
      const rows = card.querySelectorAll('.mat-row input[data-role="nuc-name"]');
      const nuclides = {};
      const order = [];
      rows.forEach((nameInput) => {
        const name = nameInput.value.trim();
        if (!name) return; // skip blank rows
        const fracInput = nameInput.parentElement.querySelector('input[data-role="nuc-frac"]');
        const frac = parseFloat(fracInput.value);
        if (!(frac > 0)) throw new Error(`${mat.name || "material " + (mIdx + 1)}: ${name} fraction must be > 0`);
        if (name in nuclides) throw new Error(`${mat.name || "material " + (mIdx + 1)}: ${name} listed twice`);
        nuclides[name] = frac;
        order.push(name);
      });
      if (order.length === 0) throw new Error(`${mat.name || "material " + (mIdx + 1)}: at least one nuclide required`);
      mat.density = density;
      mat.nuclides = nuclides;
      mat.nuclide_input_order = order;
    });
  } catch (e) {
    matStatus.textContent = e.message;
    matStatus.classList.add("err");
    return;
  }
  // Reload into wasm sim.
  try {
    sim.load_model_json(JSON.stringify(model));
  } catch (e) {
    matStatus.textContent = `load_model_json failed: ${e}`;
    matStatus.classList.add("err");
    return;
  }
  renderSummary();
  PLOT.redraw();
  const missing = refreshSimGate();
  if (missing.length === 0) {
    matStatus.textContent = "Applied. Materials updated.";
  } else {
    matStatus.textContent = `Applied. New nuclide(s) needed: ${missing.join(", ")} -- fetch before simulating.`;
  }
};

// --- Minimal tar reader (POSIX ustar). ---
function parseTar(bytes) {
  const dec = new TextDecoder("ascii");
  const files = [];
  let off = 0;
  while (off + 512 <= bytes.length) {
    if (bytes[off] === 0) break;
    const nameRaw = bytes.subarray(off, off + 100);
    const nul = nameRaw.indexOf(0);
    const name = dec.decode(nameRaw.subarray(0, nul === -1 ? 100 : nul));
    const sizeStr = dec.decode(bytes.subarray(off + 124, off + 124 + 11)).replace(/[^\d]/g, "");
    const size = parseInt(sizeStr, 8) || 0;
    const typeflag = String.fromCharCode(bytes[off + 156] || 0);
    if ((typeflag === "0" || typeflag === "\0") && size > 0) {
      files.push({ name, data: bytes.subarray(off + 512, off + 512 + size) });
    }
    off += 512 + Math.ceil(size / 512) * 512;
  }
  return files;
}

// --- "Fetch cross sections" -- fetches whichever required nuclides
// aren't already in the in-memory storage. Re-runs cleanly after the
// material editor adds new nuclides.
const dlBtn = document.getElementById("dl");
const status = document.getElementById("status");
dlBtn.onclick = async () => {
  status.classList.remove("err");
  // Build a combined work list: neutron data is per-nuclide, photon data
  // per-element. Each fetches /<kind>/<name>.arrow.tar and registers files
  // under /<name>.arrow/ (the tar's internal prefix is stripped either way).
  const reqNuc = sim.model_required_nuclides().split(",").filter(Boolean);
  const reqEl = sim.model_required_elements().split(",").filter(Boolean);
  const jobs = [
    ...reqNuc.filter((n) => !loadedNuclides.has(n)).map((name) => ({ name, kind: "neutron" })),
    ...reqEl.filter((e) => !loadedElements.has(e)).map((name) => ({ name, kind: "photon" })),
  ];
  if (jobs.length === 0) {
    status.textContent = "All required cross sections already loaded.";
    refreshSimGate();
    return;
  }
  dlBtn.disabled = true;
  for (let i = 0; i < jobs.length; i++) {
    const { name, kind } = jobs[i];
    status.textContent = `Fetching ${name} (${kind}) (${i + 1}/${jobs.length})…`;
    try {
      const url = `https://yamc-data.xsplot.com/endf-b8.1/${kind}/${encodeURIComponent(name)}.arrow.tar`;
      const resp = await fetch(url);
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const buf = new Uint8Array(await resp.arrayBuffer());
      const prefixRe = new RegExp(`^${name.replace(/[.*+?^${}()|[\\]\\\\]/g, "\\$&")}\\.arrow/?`);
      for (const { name: tarName, data } of parseTar(buf)) {
        const rel = tarName.replace(prefixRe, "");
        sim.add_file(`/${name}.arrow/${rel}`, data);
      }
      if (kind === "photon") loadedElements.add(name);
      else loadedNuclides.add(name);
    } catch (e) {
      status.textContent = `Failed on ${name}: ${e.message}`;
      status.classList.add("err");
      refreshSimGate();
      return;
    }
  }
  status.textContent =
    `Loaded ${sim.file_count()} files (${loadedNuclides.size} nuclide(s), ${loadedElements.size} element(s)).`;
  dlBtn.textContent = "Fetch cross sections";
  refreshSimGate();
};

// Initial gate evaluation -- Simulate stays disabled until fetch runs.
refreshSimGate();

// --- "Simulate" button ---
const simBtn = document.getElementById("sim");
const simStatus = document.getElementById("sim-status");
const resultEl = document.getElementById("result");
const SEED = 42n;
simBtn.onclick = () => {
  const totalParticles = parseInt(document.getElementById("total-particles").value, 10);
  simBtn.disabled = true;
  simStatus.textContent = "Running…";
  setTimeout(() => {
    try {
      const t0 = performance.now();
      // Per-history Welford variance: a single batch over all histories
      // gives the correct std, so we pass batches=1 and the full count.
      const json = sim.simulate_transport(totalParticles, 1, SEED);
      const dt = performance.now() - t0;
      const r = JSON.parse(json);
      simStatus.textContent = `Done in ${dt.toFixed(0)} ms.`;
      if (r.status === "ok") {
        const lines = [];
        for (const t of r.tallies) {
          lines.push(`Tally: ${t.name || "(unnamed)"}`);
          for (const s of t.scores) {
            const meanStr = Number.isFinite(s.mean) ? s.mean.toExponential(4) : String(s.mean);
            const stdStr = Number.isFinite(s.std) ? s.std.toExponential(4) : String(s.std);
            lines.push(`  ${s.name.padEnd(20)}  mean ${meanStr}   std ${stdStr}`);
          }
        }
        lines.push("");
        lines.push(`total particles: ${r.particles * r.batches}  (seed ${r.seed})`);
        resultEl.textContent = lines.join("\n");
        resultEl.classList.remove("err");
        // Refresh the mesh-tally plot from THIS run. Find the first
        // tally that has a mesh filter (tally with a "mesh" field in
        // the model JSON) and ask wasm for its current plot HTML.
        // Silently no-ops when the model has no mesh tally -- the
        // recipient just gets the scalar result text above.
        const tplotFs = document.getElementById('tally-plot-fieldset');
        const tplotIfr = document.getElementById('tally-plot-iframe');
        if (tplotFs && tplotIfr) {
          // Tally's mesh lives inside a Filter::Mesh node, not on the
          // tally itself -- `t.filters[i].Mesh.mesh` after JSON round-trip.
          const meshTallyIdx = model.tallies.findIndex(t =>
            Array.isArray(t.filters) && t.filters.some(f => f && f.Mesh)
          );
          if (meshTallyIdx >= 0) {
            try {
              const tparams = JSON.stringify({ tally_index: meshTallyIdx });
              tplotIfr.srcdoc = sim.tallyPlotHtml(tparams);
              tplotFs.hidden = false;
              // Replace the now-stale "engineer's pre-run" warning
              // with a fresh-results banner once it's been refreshed.
              const fsLegend = tplotFs.querySelector('legend');
              if (fsLegend) fsLegend.textContent = 'Mesh tally -- your latest run';
              const oldWarn = tplotFs.querySelector('div[style*="fff8e1"]');
              if (oldWarn) oldWarn.remove();
            } catch (e) {
              // Fall back to the engineer's pre-run snapshot if the
              // wasm call fails (e.g., mesh-geometry overlay).
              if (tplotIfr.getAttribute('srcdoc')) tplotFs.hidden = false;
            }
          } else if (tplotIfr.getAttribute('srcdoc')) {
            tplotFs.hidden = false;
          }
        }
      } else {
        resultEl.textContent = `Error: ${r.message || JSON.stringify(r)}`;
        resultEl.classList.add("err");
      }
    } catch (e) {
      resultEl.textContent = `Exception: ${e.message}`;
      resultEl.classList.add("err");
      simStatus.textContent = "";
    } finally {
      simBtn.disabled = false;
    }
  }, 0);
};
