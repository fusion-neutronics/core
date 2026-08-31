/* @ts-self-types="./yamc.d.ts" */

export class WasmConfig {
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        WasmConfigFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_wasmconfig_free(ptr, 0);
    }
    /**
     * @returns {any}
     */
    static get_cross_sections() {
        const ret = wasm.wasmconfig_get_cross_sections();
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    /**
     * @param {any} js_map
     */
    static set_cross_sections(js_map) {
        const ret = wasm.wasmconfig_set_cross_sections(js_map);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
}
if (Symbol.dispose) WasmConfig.prototype[Symbol.dispose] = WasmConfig.prototype.free;

export class WasmElement {
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        WasmElementFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_wasmelement_free(ptr, 0);
    }
    /**
     * @returns {Array<any>}
     */
    getNuclides() {
        const ret = wasm.wasmelement_getNuclides(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {string}
     */
    get name() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.wasmelement_name(this.__wbg_ptr);
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * @param {string} name
     */
    constructor(name) {
        const ptr0 = passStringToWasm0(name, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.wasmelement_new(ptr0, len0);
        this.__wbg_ptr = ret;
        WasmElementFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
}
if (Symbol.dispose) WasmElement.prototype[Symbol.dispose] = WasmElement.prototype.free;

export class WasmMaterial {
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        WasmMaterialFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_wasmmaterial_free(ptr, 0);
    }
    /**
     * @param {string} element
     * @param {number} fraction
     * @param {string | null} [fraction_type]
     * @param {number | null} [enrichment]
     * @param {string | null} [enrichment_target]
     * @param {string | null} [enrichment_type]
     */
    add_element(element, fraction, fraction_type, enrichment, enrichment_target, enrichment_type) {
        const ptr0 = passStringToWasm0(element, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        var ptr1 = isLikeNone(fraction_type) ? 0 : passStringToWasm0(fraction_type, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        var len1 = WASM_VECTOR_LEN;
        var ptr2 = isLikeNone(enrichment_target) ? 0 : passStringToWasm0(enrichment_target, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        var len2 = WASM_VECTOR_LEN;
        var ptr3 = isLikeNone(enrichment_type) ? 0 : passStringToWasm0(enrichment_type, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        var len3 = WASM_VECTOR_LEN;
        const ret = wasm.wasmmaterial_add_element(this.__wbg_ptr, ptr0, len0, fraction, ptr1, len1, !isLikeNone(enrichment), isLikeNone(enrichment) ? 0 : enrichment, ptr2, len2, ptr3, len3);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * @param {string} nuclide
     * @param {number} fraction
     * @param {string | null} [fraction_type]
     */
    add_nuclide(nuclide, fraction, fraction_type) {
        const ptr0 = passStringToWasm0(nuclide, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        var ptr1 = isLikeNone(fraction_type) ? 0 : passStringToWasm0(fraction_type, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        var len1 = WASM_VECTOR_LEN;
        const ret = wasm.wasmmaterial_add_nuclide(this.__wbg_ptr, ptr0, len0, fraction, ptr1, len1);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * @returns {any}
     */
    get_atoms_per_barn_cm() {
        const ret = wasm.wasmmaterial_get_atoms_per_barn_cm(this.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    /**
     * @returns {Array<any>}
     */
    get_nuclides() {
        const ret = wasm.wasmmaterial_get_nuclides(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {number} reaction
     * @param {string | null} [temperature]
     * @returns {Array<any>}
     */
    macroscopicCrossSection(reaction, temperature) {
        var ptr0 = isLikeNone(temperature) ? 0 : passStringToWasm0(temperature, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        var len0 = WASM_VECTOR_LEN;
        const ret = wasm.wasmmaterial_macroscopicCrossSection(this.__wbg_ptr, reaction, ptr0, len0);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    /**
     * @param {number} energy
     * @returns {number | undefined}
     */
    mean_free_path_neutron(energy) {
        const ret = wasm.wasmmaterial_mean_free_path_neutron(this.__wbg_ptr, energy);
        return ret[0] === 0 ? undefined : ret[1];
    }
    constructor() {
        const ret = wasm.wasmmaterial_new();
        this.__wbg_ptr = ret;
        WasmMaterialFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @returns {Array<any>}
     */
    reaction_mts() {
        const ret = wasm.wasmmaterial_reaction_mts(this.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    /**
     * @param {number} energy
     * @param {bigint | null} [seed]
     * @returns {string}
     */
    sampleInteractingNuclide(energy, seed) {
        let deferred2_0;
        let deferred2_1;
        try {
            const ret = wasm.wasmmaterial_sampleInteractingNuclide(this.__wbg_ptr, energy, !isLikeNone(seed), isLikeNone(seed) ? BigInt(0) : seed);
            var ptr1 = ret[0];
            var len1 = ret[1];
            if (ret[3]) {
                ptr1 = 0; len1 = 0;
                throw takeFromExternrefTable0(ret[2]);
            }
            deferred2_0 = ptr1;
            deferred2_1 = len1;
            return getStringFromWasm0(ptr1, len1);
        } finally {
            wasm.__wbindgen_free(deferred2_0, deferred2_1, 1);
        }
    }
    /**
     * @param {number} energy
     * @returns {number | undefined}
     */
    sample_distance_to_collision(energy) {
        const ret = wasm.wasmmaterial_sample_distance_to_collision(this.__wbg_ptr, energy);
        return ret[0] === 0 ? undefined : ret[1];
    }
    /**
     * @param {string} unit
     * @param {number | null} [value]
     */
    set_density(unit, value) {
        const ptr0 = passStringToWasm0(unit, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.wasmmaterial_set_density(this.__wbg_ptr, ptr0, len0, !isLikeNone(value), isLikeNone(value) ? 0 : value);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * @param {string} temperature
     */
    set_temperature(temperature) {
        const ptr0 = passStringToWasm0(temperature, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        wasm.wasmmaterial_set_temperature(this.__wbg_ptr, ptr0, len0);
    }
    /**
     * @param {number} value
     */
    set_volume(value) {
        const ret = wasm.wasmmaterial_set_volume(this.__wbg_ptr, value);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * @returns {string}
     */
    to_string() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.wasmmaterial_to_string(this.__wbg_ptr);
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
}
if (Symbol.dispose) WasmMaterial.prototype[Symbol.dispose] = WasmMaterial.prototype.free;

export class WasmNuclide {
    static __wrap(ptr) {
        const obj = Object.create(WasmNuclide.prototype);
        obj.__wbg_ptr = ptr;
        WasmNuclideFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        WasmNuclideFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_wasmnuclide_free(ptr, 0);
    }
    /**
     * @param {string} temperature
     * @returns {Array<any>}
     */
    get_available_reactions(temperature) {
        const ptr0 = passStringToWasm0(temperature, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.wasmnuclide_get_available_reactions(this.__wbg_ptr, ptr0, len0);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    /**
     * @returns {Array<any>}
     */
    get_available_temperatures() {
        const ret = wasm.wasmnuclide_get_available_temperatures(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {string}
     */
    get_name() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.wasmnuclide_get_name(this.__wbg_ptr);
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * @param {string} _name
     * @param {string} _path
     * @returns {WasmNuclide}
     */
    static load(_name, _path) {
        const ptr0 = passStringToWasm0(_name, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passStringToWasm0(_path, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len1 = WASM_VECTOR_LEN;
        const ret = wasm.wasmnuclide_load(ptr0, len0, ptr1, len1);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return WasmNuclide.__wrap(ret[0]);
    }
}
if (Symbol.dispose) WasmNuclide.prototype[Symbol.dispose] = WasmNuclide.prototype.free;

export class WasmReaction {
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        WasmReactionFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_wasmreaction_free(ptr, 0);
    }
    /**
     * @returns {Array<any>}
     */
    get_cross_section() {
        const ret = wasm.wasmreaction_get_cross_section(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {Array<any>}
     */
    get_energy() {
        const ret = wasm.wasmreaction_get_energy(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {Array<any>}
     */
    get_interpolation() {
        const ret = wasm.wasmreaction_get_interpolation(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {number}
     */
    get_threshold_idx() {
        const ret = wasm.wasmreaction_get_threshold_idx(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @param {number} threshold_idx
     */
    constructor(threshold_idx) {
        const ret = wasm.wasmreaction_new(threshold_idx);
        this.__wbg_ptr = ret;
        WasmReactionFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @param {Float64Array} cross_section
     */
    set_cross_section(cross_section) {
        const ptr0 = passArrayF64ToWasm0(cross_section, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        wasm.wasmreaction_set_cross_section(this.__wbg_ptr, ptr0, len0);
    }
    /**
     * @param {Float64Array} energy
     */
    set_energy(energy) {
        const ptr0 = passArrayF64ToWasm0(energy, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        wasm.wasmreaction_set_energy(this.__wbg_ptr, ptr0, len0);
    }
    /**
     * @param {Int32Array} interpolation
     */
    set_interpolation(interpolation) {
        const ptr0 = passArray32ToWasm0(interpolation, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        wasm.wasmreaction_set_interpolation(this.__wbg_ptr, ptr0, len0);
    }
    /**
     * @returns {any}
     */
    to_json() {
        const ret = wasm.wasmreaction_to_json(this.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
}
if (Symbol.dispose) WasmReaction.prototype[Symbol.dispose] = WasmReaction.prototype.free;

/**
 * Browser-facing handle. Owns the in-memory `Storage` backend plus the
 * currently-loaded model (if any).
 */
export class WasmSimulation {
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        WasmSimulationFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_wasmsimulation_free(ptr, 0);
    }
    /**
     * Add a single virtual file at `path`. Called once per file in each
     * extracted nuclide `.arrow.tar` before `simulate_transport`.
     * @param {string} path
     * @param {Uint8Array} bytes
     */
    add_file(path, bytes) {
        const ptr0 = passStringToWasm0(path, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passArray8ToWasm0(bytes, wasm.__wbindgen_malloc);
        const len1 = WASM_VECTOR_LEN;
        wasm.wasmsimulation_add_file(this.__wbg_ptr, ptr0, len0, ptr1, len1);
    }
    /**
     * Bounding box of the loaded geometry as `[cx, cy, cz, wx, wy, wz]`
     * (center + widths in cm). Used by the JS viewer to seed initial view
     * origin/width and to scale wheel-zoom step size.
     * @returns {Float64Array}
     */
    boundingBox() {
        const ret = wasm.wasmsimulation_boundingBox(this.__wbg_ptr);
        if (ret[3]) {
            throw takeFromExternrefTable0(ret[2]);
        }
        var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v1;
    }
    /**
     * Number of files currently held in the in-memory backend.
     * @returns {number}
     */
    file_count() {
        const ret = wasm.wasmsimulation_file_count(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * CsgGeometry JSON for the *visualization* -- same shape Python's
     * `model.plot()` embeds as `GEOMETRY_JSON`. Lets the JS viewer build
     * its cell/material name maps + legend from the currently loaded
     * model. Errors if no model is loaded or the model is mesh-backed.
     * @returns {string}
     */
    geometryJson() {
        let deferred2_0;
        let deferred2_1;
        try {
            const ret = wasm.wasmsimulation_geometryJson(this.__wbg_ptr);
            var ptr1 = ret[0];
            var len1 = ret[1];
            if (ret[3]) {
                ptr1 = 0; len1 = 0;
                throw takeFromExternrefTable0(ret[2]);
            }
            deferred2_0 = ptr1;
            deferred2_1 = len1;
            return getStringFromWasm0(ptr1, len1);
        } finally {
            wasm.__wbindgen_free(deferred2_0, deferred2_1, 1);
        }
    }
    /**
     * Load a model from its JSON representation (as produced by Python's
     * `model.save(path)` / `model.export(html)`). Replaces any previously
     * loaded model. Returns an error string on parse failure.
     * @param {string} json
     */
    load_model_json(json) {
        const ptr0 = passStringToWasm0(json, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.wasmsimulation_load_model_json(this.__wbg_ptr, ptr0, len0);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * Element symbols the loaded model needs *photon* data for, comma-
     * joined. Empty when the model has no photons in flight (no photon
     * source and no secondary-photon production), or no model loaded.
     * Photon data is per-element (`Fe`, not `Fe56`); the JS host fetches
     * `endf-b8.1/photon/<El>.arrow.tar` for each and registers the files
     * under `/<El>.arrow/`, mirroring the neutron convention.
     * @returns {string}
     */
    model_required_elements() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.wasmsimulation_model_required_elements(this.__wbg_ptr);
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * Diagnostic: list of nuclide names referenced by the loaded model's
     * materials. JS uses this to know which `.arrow.tar` files to fetch.
     * Returns a comma-joined string; empty if no model is loaded.
     * @returns {string}
     */
    model_required_nuclides() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.wasmsimulation_model_required_nuclides(this.__wbg_ptr);
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * Construct a new simulation. Installs a fresh in-memory `Storage`
     * backend as the process-global `yamc_nuclide` storage.
     */
    constructor() {
        const ret = wasm.wasmsimulation_new();
        this.__wbg_ptr = ret;
        WasmSimulationFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * Build the full interactive-viewer HTML for the currently loaded
     * model -- same shape Python's `model.plot()._repr_html_()` produces.
     * The editor mounts this in an iframe; on every Apply it re-calls
     * `plotHtml` and rewrites `iframe.srcdoc` so the viewer reflects the
     * post-edit geometry/source.
     *
     * `params_json` accepts an optional initial view spec -- same keys as
     * `sampleSlice` (`origin`, `width`, `pixels`, `basis`). Empty / "{}"
     * auto-fits to the model's bounding box.
     * @param {string} params_json
     * @returns {string}
     */
    plotHtml(params_json) {
        let deferred3_0;
        let deferred3_1;
        try {
            const ptr0 = passStringToWasm0(params_json, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
            const len0 = WASM_VECTOR_LEN;
            const ret = wasm.wasmsimulation_plotHtml(this.__wbg_ptr, ptr0, len0);
            var ptr2 = ret[0];
            var len2 = ret[1];
            if (ret[3]) {
                ptr2 = 0; len2 = 0;
                throw takeFromExternrefTable0(ret[2]);
            }
            deferred3_0 = ptr2;
            deferred3_1 = len2;
            return getStringFromWasm0(ptr2, len2);
        } finally {
            wasm.__wbindgen_free(deferred3_0, deferred3_1, 1);
        }
    }
    /**
     * Sample a 2D slice through the loaded model's geometry.
     *
     * `params_json` accepts:
     * ```json
     * { "origin":  [ox, oy, oz],
     *   "width":   [wh, wv],
     *   "pixels":  [ph, pv],
     *   "basis":   "xy" | "xz" | "yz" }
     * ```
     *
     * Returns an interleaved `[cell_id, mat_id, cell_id, mat_id, ...]`
     * flat `Vec<i32>` of length `ph * pv * 2`, rows top-to-bottom. `-1`
     * for void cells / material-less cells. Matches the wire format the
     * inline `JsCsgPlotter.sampleGrid(...)` produces, so the same
     * viewer JS consumes both.
     * @param {string} params_json
     * @returns {Int32Array}
     */
    sampleSlice(params_json) {
        const ptr0 = passStringToWasm0(params_json, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.wasmsimulation_sampleSlice(this.__wbg_ptr, ptr0, len0);
        if (ret[3]) {
            throw takeFromExternrefTable0(ret[2]);
        }
        var v2 = getArrayI32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v2;
    }
    /**
     * Sample `n` source-particle positions using the loaded model's source
     * distribution. Returns a flat `[x0,y0,z0, x1,y1,z1, ...]` of length
     * `n * 3`. Drives the JS viewer's source-overlay dot cloud -- using the
     * real Rust sampler avoids the drift the inline JS sampler warns about.
     * @param {number} n
     * @param {bigint} seed
     * @returns {Float64Array}
     */
    sampleSourcePoints(n, seed) {
        const ret = wasm.wasmsimulation_sampleSourcePoints(this.__wbg_ptr, n, seed);
        if (ret[3]) {
            throw takeFromExternrefTable0(ret[2]);
        }
        var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v1;
    }
    /**
     * Run the loaded model. `particles`, `batches`, `seed` override the
     * model's defaults. Returns a JSON result string with per-tally,
     * per-score breakdown:
     *
     * ```json
     * {
     *   "status": "ok",
     *   "particles": ..., "batches": ..., "seed": ...,
     *   "tallies": [
     *     {
     *       "name": "tritium",
     *       "scores": [{ "name": "105", "mean": ..., "std": ... }, ...]
     *     },
     *     ...
     *   ]
     * }
     * ```
     *
     * On failure: `{"status":"error","message":...}`.
     * @param {number} particles
     * @param {number} batches
     * @param {bigint} seed
     * @returns {string}
     */
    simulate_transport(particles, batches, seed) {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.wasmsimulation_simulate_transport(this.__wbg_ptr, particles, batches, seed);
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * Build an interactive mesh-tally plot HTML from the currently
     * loaded model's *current* tally state. Used by the editor's
     * Simulate handler to refresh the tally-plot iframe after each
     * in-browser run.
     *
     * Args (all in `params_json`):
     * - `tally_index` (usize): which tally to plot (default 0)
     * - `basis` ("xy" | "xz" | "yz", default "xz")
     * - `score_index` (usize, default 0)
     * - `outline` ("cell" | "material" | "none", default "cell")
     *
     * Returns the full self-contained interactive tally viewer HTML --
     * same shape Python's `tally.plot()._repr_html_()` produces.
     * Errors if the tally has no MeshFilter or no model is loaded.
     * @param {string} params_json
     * @returns {string}
     */
    tallyPlotHtml(params_json) {
        let deferred3_0;
        let deferred3_1;
        try {
            const ptr0 = passStringToWasm0(params_json, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
            const len0 = WASM_VECTOR_LEN;
            const ret = wasm.wasmsimulation_tallyPlotHtml(this.__wbg_ptr, ptr0, len0);
            var ptr2 = ret[0];
            var len2 = ret[1];
            if (ret[3]) {
                ptr2 = 0; len2 = 0;
                throw takeFromExternrefTable0(ret[2]);
            }
            deferred3_0 = ptr2;
            deferred3_1 = len2;
            return getStringFromWasm0(ptr2, len2);
        } finally {
            wasm.__wbindgen_free(deferred3_0, deferred3_1, 1);
        }
    }
}
if (Symbol.dispose) WasmSimulation.prototype[Symbol.dispose] = WasmSimulation.prototype.free;

/**
 * @returns {any}
 */
export function element_names() {
    const ret = wasm.element_names();
    return ret;
}

/**
 * @returns {any}
 */
export function element_nuclides() {
    const ret = wasm.element_nuclides();
    return ret;
}

/**
 * @returns {any}
 */
export function natural_abundance() {
    const ret = wasm.natural_abundance();
    return ret;
}

/**
 * @param {string} name
 * @param {string} path
 * @returns {WasmNuclide}
 */
export function wasm_load_nuclide(name, path) {
    const ptr0 = passStringToWasm0(name, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passStringToWasm0(path, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.wasm_load_nuclide(ptr0, len0, ptr1, len1);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return WasmNuclide.__wrap(ret[0]);
}

export function wasm_start() {
    wasm.wasm_start();
}
function __wbg_get_imports() {
    const import0 = {
        __proto__: null,
        __wbg_Error_bce6d499ff0a4aff: function(arg0, arg1) {
            const ret = Error(getStringFromWasm0(arg0, arg1));
            return ret;
        },
        __wbg___wbindgen_debug_string_edece8177ad01481: function(arg0, arg1) {
            const ret = debugString(arg1);
            const ptr1 = passStringToWasm0(ret, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
            const len1 = WASM_VECTOR_LEN;
            getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
            getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
        },
        __wbg___wbindgen_is_string_dde0fd9020db4434: function(arg0) {
            const ret = typeof(arg0) === 'string';
            return ret;
        },
        __wbg___wbindgen_string_get_d109740c0d18f4d7: function(arg0, arg1) {
            const obj = arg1;
            const ret = typeof(obj) === 'string' ? obj : undefined;
            var ptr1 = isLikeNone(ret) ? 0 : passStringToWasm0(ret, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
            var len1 = WASM_VECTOR_LEN;
            getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
            getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
        },
        __wbg___wbindgen_throw_9c31b086c2b26051: function(arg0, arg1) {
            throw new Error(getStringFromWasm0(arg0, arg1));
        },
        __wbg_error_a6fa202b58aa1cd3: function(arg0, arg1) {
            let deferred0_0;
            let deferred0_1;
            try {
                deferred0_0 = arg0;
                deferred0_1 = arg1;
                console.error(getStringFromWasm0(arg0, arg1));
            } finally {
                wasm.__wbindgen_free(deferred0_0, deferred0_1, 1);
            }
        },
        __wbg_getRandomValues_76dfc69825c9c552: function() { return handleError(function (arg0, arg1) {
            globalThis.crypto.getRandomValues(getArrayU8FromWasm0(arg0, arg1));
        }, arguments); },
        __wbg_get_98fdf51d029a75eb: function(arg0, arg1) {
            const ret = arg0[arg1 >>> 0];
            return ret;
        },
        __wbg_get_dcf82ab8aad1a593: function() { return handleError(function (arg0, arg1) {
            const ret = Reflect.get(arg0, arg1);
            return ret;
        }, arguments); },
        __wbg_instanceof_Object_03924e0dbda74bd8: function(arg0) {
            let result;
            try {
                result = arg0 instanceof Object;
            } catch (_) {
                result = false;
            }
            const ret = result;
            return ret;
        },
        __wbg_keys_682010b680c9b1f8: function(arg0) {
            const ret = Object.keys(arg0);
            return ret;
        },
        __wbg_length_2591a0f4f659a55c: function(arg0) {
            const ret = arg0.length;
            return ret;
        },
        __wbg_new_02d162bc6cf02f60: function() {
            const ret = new Object();
            return ret;
        },
        __wbg_new_070df68d66325372: function() {
            const ret = new Map();
            return ret;
        },
        __wbg_new_227d7c05414eb861: function() {
            const ret = new Error();
            return ret;
        },
        __wbg_new_310879b66b6e95e1: function() {
            const ret = new Array();
            return ret;
        },
        __wbg_new_from_slice_02962bf7778cf945: function(arg0, arg1) {
            const ret = new Float64Array(getArrayF64FromWasm0(arg0, arg1));
            return ret;
        },
        __wbg_parse_2c1cad6215e84999: function() { return handleError(function (arg0, arg1) {
            const ret = JSON.parse(getStringFromWasm0(arg0, arg1));
            return ret;
        }, arguments); },
        __wbg_push_b77c476b01548d0a: function(arg0, arg1) {
            const ret = arg0.push(arg1);
            return ret;
        },
        __wbg_set_6be42768c690e380: function(arg0, arg1, arg2) {
            arg0[arg1] = arg2;
        },
        __wbg_set_78ea6a19f4818587: function(arg0, arg1, arg2) {
            arg0[arg1 >>> 0] = arg2;
        },
        __wbg_set_facb7a5914e0fa39: function(arg0, arg1, arg2) {
            const ret = arg0.set(arg1, arg2);
            return ret;
        },
        __wbg_stack_3b0d974bbf31e44f: function(arg0, arg1) {
            const ret = arg1.stack;
            const ptr1 = passStringToWasm0(ret, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
            const len1 = WASM_VECTOR_LEN;
            getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
            getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
        },
        __wbindgen_cast_0000000000000001: function(arg0) {
            // Cast intrinsic for `F64 -> Externref`.
            const ret = arg0;
            return ret;
        },
        __wbindgen_cast_0000000000000002: function(arg0, arg1) {
            // Cast intrinsic for `Ref(String) -> Externref`.
            const ret = getStringFromWasm0(arg0, arg1);
            return ret;
        },
        __wbindgen_init_externref_table: function() {
            const table = wasm.__wbindgen_externrefs;
            const offset = table.grow(4);
            table.set(0, undefined);
            table.set(offset + 0, undefined);
            table.set(offset + 1, null);
            table.set(offset + 2, true);
            table.set(offset + 3, false);
        },
    };
    return {
        __proto__: null,
        "./yamc_bg.js": import0,
    };
}

const WasmConfigFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wasmconfig_free(ptr, 1));
const WasmElementFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wasmelement_free(ptr, 1));
const WasmMaterialFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wasmmaterial_free(ptr, 1));
const WasmNuclideFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wasmnuclide_free(ptr, 1));
const WasmReactionFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wasmreaction_free(ptr, 1));
const WasmSimulationFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wasmsimulation_free(ptr, 1));

function addToExternrefTable0(obj) {
    const idx = wasm.__externref_table_alloc();
    wasm.__wbindgen_externrefs.set(idx, obj);
    return idx;
}

function debugString(val) {
    // primitive types
    const type = typeof val;
    if (type == 'number' || type == 'boolean' || val == null) {
        return  `${val}`;
    }
    if (type == 'string') {
        return `"${val}"`;
    }
    if (type == 'symbol') {
        const description = val.description;
        if (description == null) {
            return 'Symbol';
        } else {
            return `Symbol(${description})`;
        }
    }
    if (type == 'function') {
        const name = val.name;
        if (typeof name == 'string' && name.length > 0) {
            return `Function(${name})`;
        } else {
            return 'Function';
        }
    }
    // objects
    if (Array.isArray(val)) {
        const length = val.length;
        let debug = '[';
        if (length > 0) {
            debug += debugString(val[0]);
        }
        for(let i = 1; i < length; i++) {
            debug += ', ' + debugString(val[i]);
        }
        debug += ']';
        return debug;
    }
    // Test for built-in
    const builtInMatches = /\[object ([^\]]+)\]/.exec(toString.call(val));
    let className;
    if (builtInMatches && builtInMatches.length > 1) {
        className = builtInMatches[1];
    } else {
        // Failed to match the standard '[object ClassName]'
        return toString.call(val);
    }
    if (className == 'Object') {
        // we're a user defined class or Object
        // JSON.stringify avoids problems with cycles, and is generally much
        // easier than looping through ownProperties of `val`.
        try {
            return 'Object(' + JSON.stringify(val) + ')';
        } catch (_) {
            return 'Object';
        }
    }
    // errors
    if (val instanceof Error) {
        return `${val.name}: ${val.message}\n${val.stack}`;
    }
    // TODO we could test for more things here, like `Set`s and `Map`s.
    return className;
}

function getArrayF64FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getFloat64ArrayMemory0().subarray(ptr / 8, ptr / 8 + len);
}

function getArrayI32FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getInt32ArrayMemory0().subarray(ptr / 4, ptr / 4 + len);
}

function getArrayU8FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getUint8ArrayMemory0().subarray(ptr / 1, ptr / 1 + len);
}

let cachedDataViewMemory0 = null;
function getDataViewMemory0() {
    if (cachedDataViewMemory0 === null || cachedDataViewMemory0.buffer.detached === true || (cachedDataViewMemory0.buffer.detached === undefined && cachedDataViewMemory0.buffer !== wasm.memory.buffer)) {
        cachedDataViewMemory0 = new DataView(wasm.memory.buffer);
    }
    return cachedDataViewMemory0;
}

let cachedFloat64ArrayMemory0 = null;
function getFloat64ArrayMemory0() {
    if (cachedFloat64ArrayMemory0 === null || cachedFloat64ArrayMemory0.byteLength === 0) {
        cachedFloat64ArrayMemory0 = new Float64Array(wasm.memory.buffer);
    }
    return cachedFloat64ArrayMemory0;
}

let cachedInt32ArrayMemory0 = null;
function getInt32ArrayMemory0() {
    if (cachedInt32ArrayMemory0 === null || cachedInt32ArrayMemory0.byteLength === 0) {
        cachedInt32ArrayMemory0 = new Int32Array(wasm.memory.buffer);
    }
    return cachedInt32ArrayMemory0;
}

function getStringFromWasm0(ptr, len) {
    return decodeText(ptr >>> 0, len);
}

let cachedUint32ArrayMemory0 = null;
function getUint32ArrayMemory0() {
    if (cachedUint32ArrayMemory0 === null || cachedUint32ArrayMemory0.byteLength === 0) {
        cachedUint32ArrayMemory0 = new Uint32Array(wasm.memory.buffer);
    }
    return cachedUint32ArrayMemory0;
}

let cachedUint8ArrayMemory0 = null;
function getUint8ArrayMemory0() {
    if (cachedUint8ArrayMemory0 === null || cachedUint8ArrayMemory0.byteLength === 0) {
        cachedUint8ArrayMemory0 = new Uint8Array(wasm.memory.buffer);
    }
    return cachedUint8ArrayMemory0;
}

function handleError(f, args) {
    try {
        return f.apply(this, args);
    } catch (e) {
        const idx = addToExternrefTable0(e);
        wasm.__wbindgen_exn_store(idx);
    }
}

function isLikeNone(x) {
    return x === undefined || x === null;
}

function passArray32ToWasm0(arg, malloc) {
    const ptr = malloc(arg.length * 4, 4) >>> 0;
    getUint32ArrayMemory0().set(arg, ptr / 4);
    WASM_VECTOR_LEN = arg.length;
    return ptr;
}

function passArray8ToWasm0(arg, malloc) {
    const ptr = malloc(arg.length * 1, 1) >>> 0;
    getUint8ArrayMemory0().set(arg, ptr / 1);
    WASM_VECTOR_LEN = arg.length;
    return ptr;
}

function passArrayF64ToWasm0(arg, malloc) {
    const ptr = malloc(arg.length * 8, 8) >>> 0;
    getFloat64ArrayMemory0().set(arg, ptr / 8);
    WASM_VECTOR_LEN = arg.length;
    return ptr;
}

function passStringToWasm0(arg, malloc, realloc) {
    if (realloc === undefined) {
        const buf = cachedTextEncoder.encode(arg);
        const ptr = malloc(buf.length, 1) >>> 0;
        getUint8ArrayMemory0().subarray(ptr, ptr + buf.length).set(buf);
        WASM_VECTOR_LEN = buf.length;
        return ptr;
    }

    let len = arg.length;
    let ptr = malloc(len, 1) >>> 0;

    const mem = getUint8ArrayMemory0();

    let offset = 0;

    for (; offset < len; offset++) {
        const code = arg.charCodeAt(offset);
        if (code > 0x7F) break;
        mem[ptr + offset] = code;
    }
    if (offset !== len) {
        if (offset !== 0) {
            arg = arg.slice(offset);
        }
        ptr = realloc(ptr, len, len = offset + arg.length * 3, 1) >>> 0;
        const view = getUint8ArrayMemory0().subarray(ptr + offset, ptr + len);
        const ret = cachedTextEncoder.encodeInto(arg, view);

        offset += ret.written;
        ptr = realloc(ptr, len, offset, 1) >>> 0;
    }

    WASM_VECTOR_LEN = offset;
    return ptr;
}

function takeFromExternrefTable0(idx) {
    const value = wasm.__wbindgen_externrefs.get(idx);
    wasm.__externref_table_dealloc(idx);
    return value;
}

let cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true, fatal: true });
cachedTextDecoder.decode();
const MAX_SAFARI_DECODE_BYTES = 2146435072;
let numBytesDecoded = 0;
function decodeText(ptr, len) {
    numBytesDecoded += len;
    if (numBytesDecoded >= MAX_SAFARI_DECODE_BYTES) {
        cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true, fatal: true });
        cachedTextDecoder.decode();
        numBytesDecoded = len;
    }
    return cachedTextDecoder.decode(getUint8ArrayMemory0().subarray(ptr, ptr + len));
}

const cachedTextEncoder = new TextEncoder();

if (!('encodeInto' in cachedTextEncoder)) {
    cachedTextEncoder.encodeInto = function (arg, view) {
        const buf = cachedTextEncoder.encode(arg);
        view.set(buf);
        return {
            read: arg.length,
            written: buf.length
        };
    };
}

let WASM_VECTOR_LEN = 0;

let wasmModule, wasmInstance, wasm;
function __wbg_finalize_init(instance, module) {
    wasmInstance = instance;
    wasm = instance.exports;
    wasmModule = module;
    cachedDataViewMemory0 = null;
    cachedFloat64ArrayMemory0 = null;
    cachedInt32ArrayMemory0 = null;
    cachedUint32ArrayMemory0 = null;
    cachedUint8ArrayMemory0 = null;
    wasm.__wbindgen_start();
    return wasm;
}

async function __wbg_load(module, imports) {
    if (typeof Response === 'function' && module instanceof Response) {
        if (typeof WebAssembly.instantiateStreaming === 'function') {
            try {
                return await WebAssembly.instantiateStreaming(module, imports);
            } catch (e) {
                const validResponse = module.ok && expectedResponseType(module.type);

                if (validResponse && module.headers.get('Content-Type') !== 'application/wasm') {
                    console.warn("`WebAssembly.instantiateStreaming` failed because your server does not serve Wasm with `application/wasm` MIME type. Falling back to `WebAssembly.instantiate` which is slower. Original error:\n", e);

                } else { throw e; }
            }
        }

        const bytes = await module.arrayBuffer();
        return await WebAssembly.instantiate(bytes, imports);
    } else {
        const instance = await WebAssembly.instantiate(module, imports);

        if (instance instanceof WebAssembly.Instance) {
            return { instance, module };
        } else {
            return instance;
        }
    }

    function expectedResponseType(type) {
        switch (type) {
            case 'basic': case 'cors': case 'default': return true;
        }
        return false;
    }
}

function initSync(module) {
    if (wasm !== undefined) return wasm;


    if (module !== undefined) {
        if (Object.getPrototypeOf(module) === Object.prototype) {
            ({module} = module)
        } else {
            console.warn('using deprecated parameters for `initSync()`; pass a single object instead')
        }
    }

    const imports = __wbg_get_imports();
    if (!(module instanceof WebAssembly.Module)) {
        module = new WebAssembly.Module(module);
    }
    const instance = new WebAssembly.Instance(module, imports);
    return __wbg_finalize_init(instance, module);
}

async function __wbg_init(module_or_path) {
    if (wasm !== undefined) return wasm;


    if (module_or_path !== undefined) {
        if (Object.getPrototypeOf(module_or_path) === Object.prototype) {
            ({module_or_path} = module_or_path)
        } else {
            console.warn('using deprecated parameters for the initialization function; pass a single object instead')
        }
    }

    if (module_or_path === undefined) {
        module_or_path = new URL('yamc_bg.wasm', import.meta.url);
    }
    const imports = __wbg_get_imports();

    if (typeof module_or_path === 'string' || (typeof Request === 'function' && module_or_path instanceof Request) || (typeof URL === 'function' && module_or_path instanceof URL)) {
        module_or_path = fetch(module_or_path);
    }

    const { instance, module } = await __wbg_load(await module_or_path, imports);

    return __wbg_finalize_init(instance, module);
}

export { initSync, __wbg_init as default };
