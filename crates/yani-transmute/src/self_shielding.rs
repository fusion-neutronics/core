//! Resonance self-shielding of the collapse.
//!
//! [`compute_multigroup_reaction_rates`](crate::compute_multigroup_reaction_rates)
//! folds the caller's spectrum against the pointwise cross section as if that
//! spectrum were the flux *inside* the material. What a caller usually has is
//! the flux in the surrounding field. Inside a lump the two differ wherever the
//! total cross section is large: the flux is depressed across a resonance, so
//! the true reaction rate is below what the unshielded fold reports.
//!
//! What is corrected here is the shape of the flux **inside** each group. The
//! group totals the caller supplied are preserved exactly, because they are the
//! measurement or the transport result and this has no business rescaling them.
//! So the correction is a per-group, per-reaction weighting, and a nuclide with
//! no resonances in a group comes back with the number it had before.
//!
//! # The escape term has to be given, not guessed
//!
//! Both methods need to know how easily a neutron leaves the lump, which is
//! geometry, and a [`Material`] has none. So [`Shielding`] takes a mean chord
//! length `4V/S` -- twice the thickness for a thin slab, `4R/3` for a sphere --
//! and no shape is ever inferred. Absent means no shielding at all and a result
//! bit-identical to a run without this module, which is the same "absent is not
//! zero" rule the uncertainty work uses: a run that did not shield must say so
//! rather than quietly look like one that did.
//!
//! # One method, and the one that was measured and rejected
//!
//! The flux inside the lump solves
//!
//! ```text
//! (Sigma_t(u) + Sigma_e) phi(u) = int_{u-du}^{u} Sigma_s(u') phi(u')
//!                                 e^{-(u-u')} / (1-alpha) du'  +  Sigma_e
//! ```
//!
//! marched down in lethargy, so the in-scattering source is whatever the flux
//! above it actually was. It makes no assumption about resonances being narrow.
//!
//! The cheaper narrow-resonance approximation, which takes that source at its
//! asymptotic `1/E` form, was implemented, measured against this solve, and
//! deliberately not shipped. It over-shields anything carrying strong *elastic*
//! resonances, whose real in-scattering fills the capture dips an asymptotic
//! source leaves empty: on a 1/E field at 50 um it is 41% low on Co59, 63% low
//! on W186 and 16% low on Mn55, while agreeing inside 1% on Rh103, Fe56, Eu151
//! and Nb93. A measurement settles it rather than the comparison alone --
//! W186(n,gamma) in FNG-tung is 1.29 b, where this solve gives C/E 0.80 and
//! narrow resonance gives 0.46, twice as far from the measurement as applying
//! no correction at all.
//!
//! Since whether a run is in the safe case depends on its field and composition
//! rather than on anything the caller can state up front, offering the
//! approximation would mostly offer a way to be wrong by a factor of two. It is
//! not here.
//!
//! It costs nothing to ship: `reactions.arrow` already carries MT=1 and MT=2,
//! so this adds no bytes to what is downloaded.

use std::collections::HashMap;

use yamc_materials::material::Material;
use yamc_nuclide::reaction::Reaction;

/// Total cross section, the mixture denominator of the flux depression.
const MT_TOTAL: i32 = 1;
/// Elastic scattering, the in-scattering source of the slowing-down solve.
///
/// Named here rather than at the call site so the two MTs this module needs are
/// stated together; [`crate::multigroup`] looks the reaction up with it.
pub const MT_ELASTIC: i32 = 2;

/// Lethargy steps per collision window. 24 converges the march to 0.3%.
const STEPS_PER_WINDOW: usize = 24;
/// Windows of asymptotic flux before the march is trusted.
const BURN_IN_WINDOWS: usize = 3;
/// Below this the resolved resonances are gone and shielding with them.
const TOP_ENERGY_EV: f64 = 2.0e7;
const BOTTOM_ENERGY_EV: f64 = 1.0e-5;

/// The form of the lump, for turning a volume into a mean chord.
///
/// The chord is `4V/S`, so the shape decides how much surface a given volume
/// hides behind. A sphere is the extreme: it has the least surface for its
/// volume, so it gives the longest chord and the most shielding of any shape.
/// Assuming one is therefore not a neutral default but an upper bound, which is
/// why the shape is asked for rather than picked.
///
/// Each variant carries exactly the dimension a volume cannot supply, and no
/// more, so nothing is ever specified twice and there is no disagreement to
/// police. A sphere and a cube are fixed by their volume alone; a foil needs its
/// thickness, and a cylinder or a wire its radius.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Shape {
    /// Fixed by volume: `r = (3V/4pi)^(1/3)`, chord `4r/3`.
    Sphere,
    /// Fixed by volume: `a = V^(1/3)`, chord `2a/3`.
    Cube,
    /// A slab of this thickness in cm. Chord `2t`, whatever its area, which is
    /// why a thin foil's shielding does not depend on the volume at all.
    Foil { thickness_cm: f64 },
    /// A cylinder of this radius in cm; the height follows from the volume.
    Cylinder { radius_cm: f64 },
    /// A cylinder long enough for its ends not to matter. Chord `2r`.
    Wire { radius_cm: f64 },
}

impl Shape {
    /// The mean chord in cm, or why the volume was needed and missing.
    ///
    /// `Foil` and `Wire` never consult the volume, so they work on a material
    /// that has none; the others say which quantity they wanted rather than
    /// failing generically.
    pub fn chord_cm(&self, volume_cm3: Option<f64>) -> Result<f64, String> {
        let volume = || -> Result<f64, String> {
            match volume_cm3 {
                Some(v) if v > 0.0 && v.is_finite() => Ok(v),
                Some(v) => Err(format!(
                    "self-shielding with this shape needs a positive volume, got {v} cm^3"
                )),
                None => Err(concat!(
                    "self-shielding with this shape needs the material's volume in cm^3, ",
                    "which sets its size. Give Material(..., volume=...), or use a shape ",
                    "that carries its own dimension (a foil's thickness or a wire's radius).",
                )
                .to_string()),
            }
        };
        match *self {
            Shape::Sphere => {
                let radius = (3.0 * volume()? / (4.0 * std::f64::consts::PI)).cbrt();
                Ok(4.0 * radius / 3.0)
            }
            Shape::Cube => Ok(2.0 * volume()?.cbrt() / 3.0),
            Shape::Foil { thickness_cm } => Ok(2.0 * thickness_cm),
            Shape::Cylinder { radius_cm } => {
                let height = volume()? / (std::f64::consts::PI * radius_cm * radius_cm);
                // 4V/S for a closed cylinder, which tends to 2r as it lengthens.
                Ok(2.0 * radius_cm * height / (radius_cm + height))
            }
            Shape::Wire { radius_cm } => Ok(2.0 * radius_cm),
        }
    }
}

/// A request to self-shield the collapse.
#[derive(Debug, Clone, Copy)]
pub struct Shielding {
    /// Mean chord length `4V/S`, in cm. Twice the thickness for a thin slab.
    pub chord_cm: f64,
}

impl Shielding {
    pub fn new(chord_cm: f64) -> Result<Self, String> {
        if !chord_cm.is_finite() || chord_cm <= 0.0 {
            return Err(format!(
                "chord must be a positive length in cm, got {chord_cm}. \
                 It is the mean chord 4V/S: twice the thickness for a thin slab, \
                 4R/3 for a sphere. Omit it entirely to run unshielded."
            ));
        }
        Ok(Self { chord_cm })
    }

    /// Macroscopic escape cross section `1/chord`, in 1/cm.
    fn escape(&self) -> f64 {
        1.0 / self.chord_cm
    }
}

/// What the shielding did, for the run to report rather than imply.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ShieldingInfo {
    pub method: Option<String>,
    pub chord_cm: Option<f64>,
    /// Nuclides whose rates were weighted by a shielded flux.
    pub shielded: Vec<String>,
    /// Nuclides skipped because the data needed was not there, with the reason.
    pub not_shielded: HashMap<String, String>,
    /// Smallest per-reaction factor applied, so a run states its own strength.
    pub strongest_factor: Option<f64>,
    /// Nuclides a dilute run did not shield that look like they needed it,
    /// with the strongest suppression each could have seen.
    ///
    /// Populated only when no chord was given, which is the case #564 objects
    /// to: today such a run over-predicts silently. This turns the silence into
    /// a statement without inventing the geometry that would be needed to act
    /// on it.
    pub would_shield: HashMap<String, f64>,
}

impl ShieldingInfo {
    /// Fold another spectrum's report into this one.
    ///
    /// A schedule may name more than one spectrum, and each is collapsed
    /// separately. The driver used to assign the report of each over the last,
    /// so with two spectra only the second one's nuclides were ever named --
    /// which is the whole point of #564's report, silently halved. Issue #576.
    ///
    /// Every field folds the way it is built: the two maps are keyed by nuclide
    /// and take the strongest claim, `shielded` is a set of names in first-seen
    /// order, and `strongest_factor` is a minimum. `method` and `chord_cm` are
    /// one value for the whole run, so the first non-empty one wins.
    pub fn merge(&mut self, other: ShieldingInfo) {
        self.method = self.method.take().or(other.method);
        self.chord_cm = self.chord_cm.or(other.chord_cm);
        for name in other.shielded {
            if !self.shielded.contains(&name) {
                self.shielded.push(name);
            }
        }
        for (name, reason) in other.not_shielded {
            self.not_shielded.entry(name).or_insert(reason);
        }
        for (name, bound) in other.would_shield {
            let entry = self.would_shield.entry(name).or_insert(bound);
            if bound < *entry {
                *entry = bound;
            }
        }
        if let Some(factor) = other.strongest_factor {
            self.strongest_factor =
                Some(self.strongest_factor.map_or(factor, |f: f64| f.min(factor)));
        }
    }
}

/// An indicator of how much a nuclide's own resonances could suppress a reaction.
///
/// Cheap enough to run on every dilute transmutation, because it reads only the
/// reaction already loaded: this cross section is treated as its own absorber,
/// and no total, no elastic and no geometry are touched. The weight is
/// `1 / (1 + N * sigma_x(E))`, so a cross section flat across a group returns 1
/// to rounding and only real structure moves it.
///
/// **This is an indicator, not the correction and not a strict bound.** The unit
/// background in the denominator stands in for a geometry the dilute run does
/// not have, so the number scales with how strongly a nuclide's own resonances
/// could bite without predicting what a given lump would actually see. It reads
/// low for a strong resonance absorber and 1 for a smooth one, which is what a
/// warning needs; it is not a factor to correct anything by.
///
/// On the FNS position-3 spectrum it orders the foils the way the physics does
/// -- Al silent, Fe56 0.96, Rh103 0.91, Au197 0.82, Ta181 0.58 -- and lands at
/// 0.54 for Re187, whose measured shielded factor at a 0.3 cm chord is 0.546.
/// That agreement is a coincidence of this spectrum, not a promise.
pub fn strongest_possible_suppression(
    reaction: &Reaction,
    number_density: f64,
    boundaries: &[f64],
    flux: &[f64],
) -> f64 {
    if number_density <= 0.0 {
        return 1.0;
    }
    let mut shielded = 0.0;
    let mut dilute = 0.0;
    for (g, &phi) in flux.iter().enumerate() {
        if phi <= 0.0 || g + 1 >= boundaries.len() {
            continue;
        }
        let (lo, hi) = (boundaries[g], boundaries[g + 1]);
        let terms = crate::multigroup::walk_group(reaction, lo, hi, None, Some(number_density));
        if let Some((s, d)) = terms.bound_contribution(phi) {
            shielded += s;
            dilute += d;
        }
    }
    if dilute > 0.0 {
        (shielded / dilute).min(1.0)
    } else {
        1.0
    }
}

/// The flux shape inside the lump, on its own lethargy grid.
///
/// Held as `(energy descending, phi per unit lethargy)`. Group averaging reads
/// it by interpolation, so the march grid and the data grid stay independent.
pub struct FluxShape {
    energy: Vec<f64>,
    phi: Vec<f64>,
}

impl FluxShape {
    /// Flux per unit lethargy at `e`, by linear interpolation in log energy.
    pub(crate) fn at(&self, e: f64) -> f64 {
        if self.energy.is_empty() {
            return 1.0;
        }
        // `energy` descends, so the first index at or below `e` brackets it.
        let n = self.energy.len();
        if e >= self.energy[0] {
            return self.phi[0];
        }
        if e <= self.energy[n - 1] {
            return self.phi[n - 1];
        }
        let mut lo = 0usize;
        let mut hi = n - 1;
        while hi - lo > 1 {
            let mid = (lo + hi) / 2;
            if self.energy[mid] >= e {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let (e0, e1) = (self.energy[lo], self.energy[hi]);
        let (p0, p1) = (self.phi[lo], self.phi[hi]);
        if e0 <= e1 {
            return p0;
        }
        let t = (e0.ln() - e.ln()) / (e0.ln() - e1.ln());
        p0 + t * (p1 - p0)
    }
}

/// Mixture total cross section, sampled on demand.
///
/// The flux depression is driven by everything in the material, not by the
/// nuclide whose reaction is being collapsed, so a foil of one element and the
/// same element diluted in a matrix shield differently. `sigma_0 = 0` for a
/// mono-isotopic pure element is the reason the composition term cannot ship
/// without the escape term: on its own it is the infinite medium.
pub struct MixtureTotal<'a> {
    entries: Vec<(f64, &'a Reaction)>,
}

impl<'a> MixtureTotal<'a> {
    /// Macroscopic total at `e`, in 1/cm, with densities in atoms/barn-cm.
    fn at(&self, e: f64) -> f64 {
        self.entries
            .iter()
            .map(|(density, reaction)| density * reaction.cross_section_at(e).unwrap_or(0.0))
            .sum()
    }
}

/// Gather the material's total cross sections at the temperature in use.
pub fn mixture_total<'a>(
    material: &'a Material,
    densities: &HashMap<String, f64>,
    temperature_of: &dyn Fn(&str) -> Option<String>,
) -> MixtureTotal<'a> {
    let mut entries = Vec::new();
    for (name, &density) in densities.iter() {
        if density <= 0.0 {
            continue;
        }
        let Some(nuclide_data) = material.nuclide_data.get(name) else {
            continue;
        };
        let Some(temperature) = temperature_of(name) else {
            continue;
        };
        let Some(reactions) = nuclide_data.reactions_for_temp(&temperature) else {
            continue;
        };
        if let Some(total) = reactions.get(&MT_TOTAL) {
            entries.push((density, total.as_ref()));
        }
    }
    MixtureTotal { entries }
}

/// `alpha = ((A-1)/(A+1))^2`, the fraction of energy a neutron keeps at most.
fn alpha_of(mass_number: f64) -> f64 {
    if mass_number <= 1.0 {
        return 0.0;
    }
    let r = (mass_number - 1.0) / (mass_number + 1.0);
    r * r
}

/// Solve for the flux shape inside the lump.
///
/// `elastic` is the shielded nuclide's own MT=2, which is the in-scattering
/// source that fills its resonance dips. A mixture's other scatterers also
/// contribute, and leaving them out under-fills the dips, so this is
/// conservative in the direction of over-shielding rather than under.
pub fn flux_shape(
    mixture: &MixtureTotal<'_>,
    elastic: Option<&Reaction>,
    elastic_density: f64,
    mass_number: f64,
    shielding: &Shielding,
) -> FluxShape {
    let escape = shielding.escape();
    {
        {
            let alpha = alpha_of(mass_number);
            let window = if alpha > 0.0 { (1.0 / alpha).ln() } else { 5.0 };
            let step = window / STEPS_PER_WINDOW as f64;
            let span = (TOP_ENERGY_EV / BOTTOM_ENERGY_EV).ln();
            let n = (span / step).ceil() as usize + 1;

            let mut energy = Vec::with_capacity(n);
            let mut total = Vec::with_capacity(n);
            let mut scatter = Vec::with_capacity(n);
            for i in 0..n {
                let e = TOP_ENERGY_EV * (-(i as f64) * step).exp();
                energy.push(e);
                total.push(mixture.at(e) + escape);
                scatter.push(
                    elastic_density * elastic.and_then(|r| r.cross_section_at(e)).unwrap_or(0.0),
                );
            }

            // Trapezoid weights for the in-scattering integral, and the
            // self-scatter term held back so it can be solved implicitly.
            let width = STEPS_PER_WINDOW;
            let kernel: Vec<f64> = (1..=width)
                .map(|k| (-(k as f64) * step).exp() / (1.0 - alpha) * step)
                .collect();
            let self_weight = 0.5 * step / (1.0 - alpha);

            let mut phi = vec![1.0; n];
            let burn = (width * BURN_IN_WINDOWS).min(n);
            for i in burn..n {
                let lo = i.saturating_sub(width);
                let mut source = 0.0;
                for (k, j) in (lo..i).rev().enumerate() {
                    source += scatter[j] * kernel[k];
                }
                let denominator = total[i] - scatter[i] * self_weight;
                phi[i] = if denominator > 0.0 {
                    (source + escape) / denominator
                } else {
                    phi[i - 1]
                };
            }
            FluxShape { energy, phi }
        }
    }
}

/// Group-averaged cross section under a shielded flux shape.
///
/// The dilute form is the same integral with `phi = 1`, so the ratio of the two
/// is the shielding factor, and a group with no structure returns its
/// unshielded value to rounding.
pub fn shielded_group_averaged_xs(
    reaction: &Reaction,
    shape: &FluxShape,
    e_lo: f64,
    e_hi: f64,
) -> f64 {
    if e_lo >= e_hi {
        return 0.0;
    }
    crate::multigroup::walk_group(reaction, e_lo, e_hi, Some(shape), None).shielded()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_thin_lump_is_not_shielded() {
        // Escape swamps any cross section, so the flux shape goes flat and the
        // group average returns to its dilute value.
        let shielding = Shielding::new(1.0e-8).unwrap();
        assert!(shielding.escape() > 1.0e7);
    }

    #[test]
    fn a_chord_must_be_a_positive_length() {
        assert!(Shielding::new(0.0).is_err());
        assert!(Shielding::new(-1.0).is_err());
        assert!(Shielding::new(f64::NAN).is_err());
        assert!(Shielding::new(0.1).is_ok());
    }

    #[test]
    fn a_sphere_and_a_cube_are_fixed_by_their_volume() {
        // 4/3 pi r^3 = 1 cm^3 -> r = 0.6204, chord 4r/3.
        let chord = Shape::Sphere.chord_cm(Some(1.0)).unwrap();
        assert!((chord - 4.0 * 0.62035 / 3.0).abs() < 1e-4, "{chord}");
        // a = 1 cm, chord 2a/3.
        assert!((Shape::Cube.chord_cm(Some(1.0)).unwrap() - 2.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn a_sphere_hides_the_most_material_of_any_shape() {
        // Least surface for a given volume, so the longest chord. A cube of the
        // same volume must come out shorter.
        let sphere = Shape::Sphere.chord_cm(Some(8.0)).unwrap();
        let cube = Shape::Cube.chord_cm(Some(8.0)).unwrap();
        assert!(sphere > cube, "sphere {sphere} should exceed cube {cube}");
    }

    #[test]
    fn a_foil_and_a_wire_need_no_volume() {
        assert_eq!(Shape::Foil { thickness_cm: 0.01 }.chord_cm(None), Ok(0.02));
        assert_eq!(Shape::Wire { radius_cm: 0.05 }.chord_cm(None), Ok(0.1));
    }

    #[test]
    fn the_shapes_that_need_a_volume_say_so() {
        for shape in [
            Shape::Sphere,
            Shape::Cube,
            Shape::Cylinder { radius_cm: 1.0 },
        ] {
            let err = shape.chord_cm(None).unwrap_err();
            assert!(err.contains("volume"), "{err}");
        }
        assert!(Shape::Sphere.chord_cm(Some(0.0)).is_err());
    }

    #[test]
    fn a_long_cylinder_tends_to_a_wire() {
        // Same radius, volume large enough that the ends stop mattering.
        let radius = 0.05;
        let long = Shape::Cylinder { radius_cm: radius }
            .chord_cm(Some(100.0))
            .unwrap();
        let wire = Shape::Wire { radius_cm: radius }.chord_cm(None).unwrap();
        assert!((long - wire).abs() < 0.01 * wire, "{long} vs {wire}");
    }

    #[test]
    fn alpha_is_the_heavy_nucleus_limit() {
        assert!(alpha_of(197.0) > 0.97);
        assert!(alpha_of(1.0) == 0.0);
        assert!(alpha_of(12.0) < alpha_of(56.0));
    }
}
