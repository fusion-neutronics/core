//! Host-side packing for coupled neutron->photon production (slice S4b).
//!
//! [`CoupledPhotonInputs`] concatenates each material's per-material
//! [`GpuPhotonProductionXs`] (built by
//! [`crate::neutron::xs::photon_production::extract_photon_production_xs`]) into
//! the flat, material-major buffers the neutron transport kernel reads at the
//! photon-emission site, plus the per-material COUNT + base-OFFSET tables that
//! key one material's sub-table from the next.
//!
//! # Why per-material variable-size packing (not pad-to-global-max)
//!
//! `n_product` / `n_photon_rxn` / `n_continuous` vary hugely per material (~532
//! products for an Fe56 material, 1 for a non-emitter). Padding every material
//! to the global maximum would waste most of the buffer. Instead each
//! material's sub-table is appended end-to-end and addressed by a per-material
//! base offset. To keep the shared `#[cube]` samplers
//! ([`sample_photon_product`](crate::photon::production_select::sample_photon_product),
//! [`sample_photon_kinematics`](crate::photon::production_kinematics::sample_photon_kinematics))
//! material-local with only a single base parameter, the GLOBAL fields are
//! rewritten on the host as the sub-tables are concatenated:
//!
//! - `prod_rxn_idx[i]` gets the material's rxn-row base added, so it indexes
//!   the concatenated `rxn_xs` directly,
//! - `prod_dist_slot[i]` gets the material's continuous-slot base added, so
//!   the kinematics sampler reads the right `ct_*` slot,
//!
//! and `sample_photon_product` is given `prod_base` so its walk runs over
//! `[prod_base, prod_base + n_product)`. Every other per-product index
//! (`prod_eout_kind`, the `pa_*` rows at `global_p * AE`) is naturally global
//! once the product index itself is global.
//!
//! # Coupled-off default
//!
//! [`CoupledPhotonInputs::coupled_off`] builds a minimal size-1 dummy table
//! with the gate flag `0`. The kernel then draws ZERO photon RNG and writes
//! ZERO bank records, so a coupled-off launch is byte-identical to a
//! neutron-only run. Existing callers pass this.

use crate::neutron::xs::photon_production::GpuPhotonProductionXs;

/// Flat, material-major coupled neutron->photon production buffers + the
/// per-material count / base-offset tables, ready to upload to the transport
/// kernel. See the module docs for the concatenation + global-rewrite scheme.
#[derive(Debug, Clone)]
pub struct CoupledPhotonInputs {
    /// Runtime gate (1 element). `1` enables photon emission, `0` disables it
    /// (the kernel then does no photon RNG and no bank writes).
    pub coupled_enabled: Vec<u32>,

    /// Aggregate photon-production macro xs, concatenated material-major;
    /// material `m`'s `n_grid` row starts at `pp_base_per_material[m]`.
    pub photon_prod: Vec<f64>,
    /// Per-(material, photon reaction) macro production xs, concatenated
    /// material-major (row-major within a material). Material `m`'s first row
    /// is at `rxn_base_per_material[m]` (element units).
    pub rxn_xs: Vec<f64>,
    /// GLOBAL reaction-row index per product (the material's rxn base already
    /// folded in), concatenated material-major.
    pub prod_rxn_idx: Vec<u32>,
    /// Per-product yield curve on the grid (product-major), concatenated.
    pub prod_yield_grid: Vec<f64>,
    /// Per-product outgoing-energy kind / line params, concatenated.
    pub prod_eout_kind: Vec<u32>,
    pub prod_line_energy: Vec<f64>,
    pub prod_primary_flag: Vec<i32>,
    pub prod_awr: Vec<f64>,
    /// GLOBAL continuous-tabular slot per product (material ct base folded in).
    pub prod_dist_slot: Vec<u32>,

    /// Per-product angular table (tight CSR, issue #104), concatenated across
    /// materials. `pa_ae_offset` (one per global product) is the GLOBAL ae-row
    /// base into the tight per-row arrays; `pa_mu_offset` (one per global
    /// ae-row) is the GLOBAL mu-point base into the tight per-point arrays.
    /// Both are rewritten global on concatenation.
    pub pa_n_energies: Vec<u32>,
    pub pa_ae_offset: Vec<u32>,
    pub pa_mu_offset: Vec<u32>,
    pub pa_energy_grid: Vec<f64>,
    pub pa_n_mu: Vec<u32>,
    pub pa_mu: Vec<f64>,
    pub pa_cdf: Vec<f64>,
    pub pa_pdf: Vec<f64>,
    pub pa_interp: Vec<u32>,

    /// Per-continuous-slot outgoing-energy table (tight CSR, issue #104),
    /// concatenated across materials. `ct_ae_offset` (one per global continuous
    /// slot) is the GLOBAL ae-row base into the tight per-row arrays;
    /// `ct_x_offset` (one per global ae-row) is the GLOBAL x-point base into the
    /// tight per-point arrays. Both are rewritten global on concatenation.
    pub ct_ae_offset: Vec<u32>,
    pub ct_x_offset: Vec<u32>,
    pub ct_energy_grid: Vec<f64>,
    pub ct_n_x: Vec<u32>,
    pub ct_x: Vec<f64>,
    pub ct_cdf: Vec<f64>,
    pub ct_p: Vec<f64>,
    pub ct_interp: Vec<u32>,
    pub ct_n_discrete: Vec<u32>,
    pub ct_n_eout: Vec<u32>,
    pub ct_hist: Vec<u32>,

    /// Per-material product count and base offsets into the concatenated
    /// buffers above. `n_product_per_material[m]` products start at
    /// `prod_base_per_material[m]` (per-product arrays); the aggregate
    /// `photon_prod` row starts at `pp_base_per_material[m]` (element units).
    pub n_product_per_material: Vec<u32>,
    pub prod_base_per_material: Vec<u32>,
    pub pp_base_per_material: Vec<u32>,
}

impl CoupledPhotonInputs {
    /// Coupled-OFF default: a single size-1 dummy table with the gate flag
    /// `0`. The kernel reads none of it (the emission block is gated on
    /// `coupled_enabled[0] == 1`); it exists only so the buffers are non-empty
    /// (cubecl rejects zero-length bindings). `n_materials` materials each map
    /// to the same dummy sub-table at base 0.
    pub fn coupled_off(n_materials: usize, n_grid: usize) -> Self {
        // A single dummy continuous slot with one zero ae-row / one zero
        // x-point. The product's `ct_n_eout` is 0, so the kernel reads no
        // rows (and emission is gated off anyway); the one-row / one-point
        // sizing exists only to keep the `ct_*` buffers non-empty (wgpu
        // rejects zero-length storage bindings and the host does not pad a
        // sentinel for these). No fixed per-axis cap dependency.
        let ct_ae = 1usize;
        let ct_x = 1usize;
        // A single padded product / rxn row / continuous slot, all zero.
        CoupledPhotonInputs {
            coupled_enabled: vec![0u32],
            photon_prod: vec![0.0; n_grid.max(1)],
            rxn_xs: vec![0.0; n_grid.max(1)],
            prod_rxn_idx: vec![0u32],
            prod_yield_grid: vec![0.0; n_grid.max(1)],
            prod_eout_kind: vec![0u32],
            prod_line_energy: vec![0.0],
            prod_primary_flag: vec![0i32],
            prod_awr: vec![0.0],
            prod_dist_slot: vec![0u32],
            // Tight CSR (issue #104): one isotropic-empty product (n_energies
            // 0) at ae-row base 0; no rows, so `pa_mu_offset` and the per-row /
            // per-point arrays are empty (host pads a sentinel). Kernel reads
            // none of it (emission is gated on `coupled_enabled[0] == 1`).
            pa_n_energies: vec![0u32; 1],
            pa_ae_offset: vec![0u32],
            pa_mu_offset: Vec::new(),
            pa_energy_grid: Vec::new(),
            pa_n_mu: Vec::new(),
            pa_mu: Vec::new(),
            pa_cdf: Vec::new(),
            pa_pdf: Vec::new(),
            pa_interp: Vec::new(),
            // Tight CSR offsets describing this dummy slot: the single
            // continuous slot starts at ae-row 0, and its one row is one
            // point wide. Kernel reads none of it.
            ct_ae_offset: vec![0u32],
            ct_x_offset: (0..ct_ae).map(|r| (r * ct_x) as u32).collect(),
            ct_energy_grid: vec![0.0; ct_ae],
            ct_n_x: vec![0u32; ct_ae],
            ct_x: vec![0.0; ct_ae * ct_x],
            ct_cdf: vec![0.0; ct_ae * ct_x],
            ct_p: vec![0.0; ct_ae * ct_x],
            ct_interp: vec![0u32; ct_ae],
            ct_n_discrete: vec![0u32; ct_ae],
            ct_n_eout: vec![0u32; 1],
            ct_hist: vec![0u32; 1],
            // Every material points at the single dummy sub-table at base 0.
            n_product_per_material: vec![1u32; n_materials.max(1)],
            prod_base_per_material: vec![0u32; n_materials.max(1)],
            pp_base_per_material: vec![0u32; n_materials.max(1)],
        }
    }

    /// Concatenate per-material [`GpuPhotonProductionXs`] tables into the flat
    /// kernel buffers and set the gate flag to `1` (coupled ON). Every table
    /// must share the same `n_grid` (the shared log-energy grid the kernel
    /// uses). The per-product GLOBAL fields (`prod_rxn_idx`, `prod_dist_slot`)
    /// are rewritten with each material's rxn-row / continuous-slot base so the
    /// shared samplers stay material-local with a single `prod_base`.
    pub fn from_materials(tables: &[GpuPhotonProductionXs]) -> Self {
        assert!(!tables.is_empty(), "need at least one material table");
        let n_grid = tables[0].n_grid;
        for t in tables {
            assert_eq!(
                t.n_grid, n_grid,
                "all material photon tables must share the shared n_grid"
            );
        }

        let mut out = CoupledPhotonInputs {
            coupled_enabled: vec![1u32],
            photon_prod: Vec::new(),
            rxn_xs: Vec::new(),
            prod_rxn_idx: Vec::new(),
            prod_yield_grid: Vec::new(),
            prod_eout_kind: Vec::new(),
            prod_line_energy: Vec::new(),
            prod_primary_flag: Vec::new(),
            prod_awr: Vec::new(),
            prod_dist_slot: Vec::new(),
            pa_n_energies: Vec::new(),
            pa_ae_offset: Vec::new(),
            pa_mu_offset: Vec::new(),
            pa_energy_grid: Vec::new(),
            pa_n_mu: Vec::new(),
            pa_mu: Vec::new(),
            pa_cdf: Vec::new(),
            pa_pdf: Vec::new(),
            pa_interp: Vec::new(),
            ct_ae_offset: Vec::new(),
            ct_x_offset: Vec::new(),
            ct_energy_grid: Vec::new(),
            ct_n_x: Vec::new(),
            ct_x: Vec::new(),
            ct_cdf: Vec::new(),
            ct_p: Vec::new(),
            ct_interp: Vec::new(),
            ct_n_discrete: Vec::new(),
            ct_n_eout: Vec::new(),
            ct_hist: Vec::new(),
            n_product_per_material: Vec::with_capacity(tables.len()),
            prod_base_per_material: Vec::with_capacity(tables.len()),
            pp_base_per_material: Vec::with_capacity(tables.len()),
        };

        for t in tables {
            // Bases for THIS material's sub-tables (element units, computed
            // BEFORE appending this material's data).
            let prod_base = out.prod_eout_kind.len() as u32; // per-product index
            let rxn_row_base = (out.rxn_xs.len() / n_grid) as u32; // rxn-row index
            let pp_base = out.photon_prod.len() as u32; // aggregate row offset
            let ct_slot_base = out.ct_n_eout.len() as u32; // continuous-slot index

            out.n_product_per_material.push(t.n_product as u32);
            out.prod_base_per_material.push(prod_base);
            out.pp_base_per_material.push(pp_base);

            // Aggregate + reaction rows.
            out.photon_prod.extend_from_slice(&t.photon_prod);
            out.rxn_xs.extend_from_slice(&t.rxn_xs);

            // Per-product fields. `prod_rxn_idx` is rewritten GLOBAL by adding
            // the material's rxn-row base; `prod_dist_slot` is rewritten GLOBAL
            // by adding the continuous-slot base (sentinel left untouched).
            for &r in &t.prod_rxn_idx {
                out.prod_rxn_idx.push(r + rxn_row_base);
            }
            out.prod_yield_grid.extend_from_slice(&t.prod_yield_grid);
            out.prod_eout_kind.extend_from_slice(&t.prod_eout_kind);
            out.prod_line_energy.extend_from_slice(&t.prod_line_energy);
            out.prod_primary_flag
                .extend_from_slice(&t.prod_primary_flag);
            out.prod_awr.extend_from_slice(&t.prod_awr);
            for &s in &t.prod_dist_slot {
                if s == crate::neutron::xs::photon_production::PHOTON_DIST_SLOT_NONE {
                    out.prod_dist_slot.push(s);
                } else {
                    out.prod_dist_slot.push(s + ct_slot_base);
                }
            }

            // Per-product angular table (tight CSR, issue #104). The CSR bases
            // are rewritten GLOBAL: each material's ae-row offsets shift by the
            // running ae-row count, and its mu-point offsets by the running
            // mu-point count (both captured BEFORE extending), mirroring the
            // `ct_*` family above.
            let pa_ae_row_base = out.pa_n_mu.len() as u32;
            let pa_mu_point_base = out.pa_mu.len() as u32;
            for &a in &t.pa_ae_offset {
                out.pa_ae_offset.push(a + pa_ae_row_base);
            }
            for &m in &t.pa_mu_offset {
                out.pa_mu_offset.push(m + pa_mu_point_base);
            }
            out.pa_n_energies.extend_from_slice(&t.pa_n_energies);
            out.pa_energy_grid.extend_from_slice(&t.pa_energy_grid);
            out.pa_n_mu.extend_from_slice(&t.pa_n_mu);
            out.pa_mu.extend_from_slice(&t.pa_mu);
            out.pa_cdf.extend_from_slice(&t.pa_cdf);
            out.pa_pdf.extend_from_slice(&t.pa_pdf);
            out.pa_interp.extend_from_slice(&t.pa_interp);

            // Per-continuous-slot eout table (tight CSR, issue #104). The CSR
            // bases are rewritten GLOBAL: each material's ae-row offsets shift
            // by the running ae-row count, and its x-point offsets by the
            // running x-point count (both captured BEFORE extending).
            let ae_row_base = out.ct_n_x.len() as u32;
            let x_point_base = out.ct_x.len() as u32;
            for &a in &t.ct_ae_offset {
                out.ct_ae_offset.push(a + ae_row_base);
            }
            for &x in &t.ct_x_offset {
                out.ct_x_offset.push(x + x_point_base);
            }
            out.ct_energy_grid.extend_from_slice(&t.ct_energy_grid);
            out.ct_n_x.extend_from_slice(&t.ct_n_x);
            out.ct_x.extend_from_slice(&t.ct_x);
            out.ct_cdf.extend_from_slice(&t.ct_cdf);
            out.ct_p.extend_from_slice(&t.ct_p);
            out.ct_interp.extend_from_slice(&t.ct_interp);
            out.ct_n_discrete.extend_from_slice(&t.ct_n_discrete);
            out.ct_n_eout.extend_from_slice(&t.ct_n_eout);
            out.ct_hist.extend_from_slice(&t.ct_hist);
        }

        out
    }
}

/// Post-launch contents of the device particle bank (S4b produces it; S5
/// drains it). `count` is the total slot reservations (may exceed `capacity`);
/// `overflow > 0` is a hard error per the bank contract. The f64 / u32 records
/// are stride [`crate::common::particle_bank::BANK_F64_STRIDE`] /
/// [`crate::common::particle_bank::BANK_U32_STRIDE`]; slots
/// `0..min(count, capacity)` are written, the rest zero.
#[derive(Debug, Clone, Default)]
pub struct PhotonBankResult {
    pub count: u64,
    pub overflow: u64,
    pub bank_f64: Vec<f64>,
    pub bank_u32: Vec<u32>,
}
