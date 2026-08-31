//! Parametric tokamak plasma source.
//!
//! Turns a handful of plasma parameters (major/minor radius, elongation,
//! triangularity, confinement mode, and the ion density / temperature
//! profiles) into a set of ring sources that reproduce the neutron emission
//! of a tokamak plasma.
//!
//! The profile models follow Fausser et al., "Tokamak D-T neutron source
//! models for different plasma physics confinement modes", Fus. Eng. Des. 87
//! (2012) 787, doi:10.1016/j.fusengdes.2012.02.025 -- the same models the
//! `openmc-plasma-source` package implements, so the two can be compared
//! parameter for parameter.
//!
//! The plasma is discretised on a cylindrical (R, phi, Z) mesh. A dense grid
//! in plasma `(a, alpha)` coordinates is forward-mapped to `(R, Z)`, weighted
//! by the local volume element and neutron source density, and binned into
//! the mesh; every non-empty voxel becomes one [`Source`] whose spatial
//! distribution is that voxel and whose energy is the Ballabio spectrum at
//! the voxel's emission-weighted ion temperature. One source is emitted per
//! (voxel, reaction), so the D-D and D-T components keep their own spectra
//! and strengths.
//!
//! Within a voxel the position is sampled uniformly in R rather than
//! uniformly in volume, so emission is flat across the voxel instead of
//! rising with R. The bias is of order `dR / 2R` -- a few parts in a thousand
//! at the default resolution, and smaller on a finer mesh.

use crate::distribution::angular::AngularDistribution;
use crate::distribution::energy::{fusion_neutron_spectrum, FusionReactants, Uniform};
use crate::distribution::spatial::{CylindricalRing, Univariate};
use crate::source::{Source, SourceEnergyDistribution, SourceSpatialDistribution};

/// Plasma confinement mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfinementMode {
    /// Low confinement: a single parabolic profile with no pedestal.
    L,
    /// High confinement: a pedestal profile.
    H,
    /// Advanced confinement: the same profile shape as [`ConfinementMode::H`].
    A,
}

impl ConfinementMode {
    /// Parse the one-letter mode name used by the Python API.
    pub fn parse(name: &str) -> Result<Self, String> {
        match name {
            "L" => Ok(Self::L),
            "H" => Ok(Self::H),
            "A" => Ok(Self::A),
            other => Err(format!(
                "mode must be one of \"L\", \"H\" or \"A\" (got {other:?})"
            )),
        }
    }
}

/// A fuel ion species.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum FuelIon {
    Deuterium,
    Tritium,
}

impl FuelIon {
    /// Parse the one-letter species name used by the Python API.
    pub fn parse(name: &str) -> Result<Self, String> {
        match name {
            "D" => Ok(Self::Deuterium),
            "T" => Ok(Self::Tritium),
            other => Err(format!(
                "fuel species must be \"D\" or \"T\" (got {other:?})"
            )),
        }
    }
}

/// Maxwellian-averaged fusion reactivity `<sigma v>` in m^3/s.
///
/// Bosch & Hale, "Improved formulas for fusion cross-sections and thermal
/// reactivities", Nucl. Fusion 32 (1992) 611, Table VII. The fit is quoted
/// for 0.2 keV <= T_i <= 100 keV; below that the exponential drives the
/// result to zero, which is the physically right limit even where the fit is
/// no longer accurate.
///
/// `FusionReactants::DD` is the neutron-producing D(d,n)3He branch only --
/// the D(d,p)T branch makes no neutrons and is not counted.
pub fn reactivity(ion_temperature: f64, reactants: FusionReactants) -> f64 {
    if ion_temperature <= 0.0 {
        return 0.0;
    }
    // (B_G in keV^1/2, m_r c^2 in keV, C1..C7)
    let (bg, mrc2, c) = match reactants {
        FusionReactants::DT => (
            34.3827_f64,
            1_124_656.0_f64,
            [
                1.173_02e-9,
                1.513_61e-2,
                7.518_86e-2,
                4.606_43e-3,
                1.35e-2,
                -1.067_5e-4,
                1.366e-5,
            ],
        ),
        FusionReactants::DD => (
            31.397_0_f64,
            937_814.0_f64,
            [
                5.433_6e-12,
                5.857_78e-3,
                7.682_22e-3,
                0.0,
                -2.964e-6,
                0.0,
                0.0,
            ],
        ),
    };

    let t = ion_temperature * 1.0e-3; // the fit is in keV
    let theta = t
        / (1.0 - (t * (c[1] + t * (c[3] + t * c[5]))) / (1.0 + t * (c[2] + t * (c[4] + t * c[6]))));
    let xi = (bg * bg / (4.0 * theta)).cbrt();
    // The fit gives cm^3/s; the profiles are in m^-3, so return m^3/s.
    c[0] * theta * (xi / (mrc2 * t * t * t)).sqrt() * (-3.0 * xi).exp() * 1.0e-6
}

/// Neutron source density in neutrons/s/m^3.
///
/// `ion_density` is the *reacting pair* density in m^-6: `n_D * n_T` for D-T,
/// `0.5 * n_D^2` for D-D.
pub fn neutron_source_density(
    ion_density: f64,
    ion_temperature: f64,
    reactants: FusionReactants,
) -> f64 {
    ion_density * reactivity(ion_temperature, reactants)
}

/// Map plasma `(a, alpha)` coordinates onto `(R, Z)` in cm.
///
/// `a` is the minor radius of the flux surface and `alpha` the poloidal
/// angle. The mapping applies triangularity, elongation and the Shafranov
/// shift, which falls off quadratically from the magnetic axis.
pub fn convert_a_alpha_to_r_z(
    a: f64,
    alpha: f64,
    shafranov_factor: f64,
    minor_radius: f64,
    major_radius: f64,
    triangularity: f64,
    elongation: f64,
) -> (f64, f64) {
    let shafranov_shift = shafranov_factor * (1.0 - (a / minor_radius).powi(2));
    let r = major_radius + a * (alpha + triangularity * alpha.sin()).cos() + shafranov_shift;
    let z = elongation * a * alpha.sin();
    (r, z)
}

/// A parametric tokamak plasma.
///
/// Lengths are in cm, ion densities in m^-3, ion temperatures in eV and
/// angles in radians. Build one and call [`TokamakPlasma::sources`] for the
/// ring sources, or the profile methods for the density / temperature curves
/// the source is built from.
#[derive(Debug, Clone, PartialEq)]
pub struct TokamakPlasma {
    /// Plasma major radius (cm).
    pub major_radius: f64,
    /// Plasma minor radius (cm).
    pub minor_radius: f64,
    /// Plasma elongation (dimensionless).
    pub elongation: f64,
    /// Plasma triangularity (dimensionless).
    pub triangularity: f64,
    /// Confinement mode.
    pub mode: ConfinementMode,
    /// Ion density at the plasma centre (m^-3).
    pub ion_density_centre: f64,
    /// Ion density peaking factor (dimensionless).
    pub ion_density_peaking_factor: f64,
    /// Ion density at the pedestal (m^-3).
    pub ion_density_pedestal: f64,
    /// Ion density at the separatrix (m^-3).
    pub ion_density_separatrix: f64,
    /// Ion temperature at the plasma centre (eV).
    pub ion_temperature_centre: f64,
    /// Ion temperature peaking factor (dimensionless, `alpha_T` in Fausser).
    pub ion_temperature_peaking_factor: f64,
    /// Ion temperature beta exponent (dimensionless, `beta_T` in Fausser).
    pub ion_temperature_beta: f64,
    /// Ion temperature at the pedestal (eV).
    pub ion_temperature_pedestal: f64,
    /// Ion temperature at the separatrix (eV).
    pub ion_temperature_separatrix: f64,
    /// Minor radius at the pedestal (cm).
    pub pedestal_radius: f64,
    /// Shafranov factor (cm): the outward radial shift of the magnetic axis.
    pub shafranov_factor: f64,
    /// Toroidal angle at which the plasma sector starts (radians).
    pub start_angle: f64,
    /// Toroidal extent of the plasma sector (radians). Negative extends the
    /// sector the other way from `start_angle`.
    pub rotation_angle: f64,
    /// Number of `(R, Z)` mesh bins the plasma is discretised into. The
    /// toroidal direction carries no structure (the plasma is axisymmetric),
    /// so it is a single bin covering the sector.
    pub mesh_resolution: (usize, usize),
    /// Points per dimension of the internal `(a, alpha)` grid that is
    /// forward-mapped onto the mesh.
    pub grid_density: usize,
    /// Fuel species and their atom fractions, which must sum to 1.
    pub fuel: Vec<(FuelIon, f64)>,
}

impl Default for TokamakPlasma {
    /// The optional parameters only: a full torus, a 100 x 100 mesh, a 500 x
    /// 500 mapping grid and 50:50 D-T fuel. Every plasma parameter is left at
    /// zero and must be set, since there is no meaningful default tokamak.
    fn default() -> Self {
        Self {
            major_radius: 0.0,
            minor_radius: 0.0,
            elongation: 0.0,
            triangularity: 0.0,
            mode: ConfinementMode::H,
            ion_density_centre: 0.0,
            ion_density_peaking_factor: 0.0,
            ion_density_pedestal: 0.0,
            ion_density_separatrix: 0.0,
            ion_temperature_centre: 0.0,
            ion_temperature_peaking_factor: 0.0,
            ion_temperature_beta: 0.0,
            ion_temperature_pedestal: 0.0,
            ion_temperature_separatrix: 0.0,
            pedestal_radius: 0.0,
            shafranov_factor: 0.0,
            start_angle: 0.0,
            rotation_angle: std::f64::consts::TAU,
            mesh_resolution: (100, 100),
            grid_density: 500,
            fuel: vec![(FuelIon::Deuterium, 0.5), (FuelIon::Tritium, 0.5)],
        }
    }
}

impl TokamakPlasma {
    /// Ion density (m^-3) on the flux surface of minor radius `a` (cm).
    pub fn ion_density(&self, a: f64) -> Result<f64, String> {
        self.check_minor_radius(a)?;
        Ok(match self.mode {
            ConfinementMode::L => {
                self.ion_density_centre
                    * (1.0 - (a / self.minor_radius).powi(2)).powf(self.ion_density_peaking_factor)
            }
            ConfinementMode::H | ConfinementMode::A => {
                if a < self.pedestal_radius {
                    (self.ion_density_centre - self.ion_density_pedestal)
                        * (1.0 - (a / self.pedestal_radius).powi(2))
                            .powf(self.ion_density_peaking_factor)
                        + self.ion_density_pedestal
                } else {
                    (self.ion_density_pedestal - self.ion_density_separatrix)
                        * (self.minor_radius - a)
                        / (self.minor_radius - self.pedestal_radius)
                        + self.ion_density_separatrix
                }
            }
        })
    }

    /// Ion temperature (eV) on the flux surface of minor radius `a` (cm).
    pub fn ion_temperature(&self, a: f64) -> Result<f64, String> {
        self.check_minor_radius(a)?;
        Ok(match self.mode {
            ConfinementMode::L => {
                self.ion_temperature_centre
                    * (1.0 - (a / self.minor_radius).powi(2))
                        .powf(self.ion_temperature_peaking_factor)
            }
            ConfinementMode::H | ConfinementMode::A => {
                if a < self.pedestal_radius {
                    self.ion_temperature_pedestal
                        + (self.ion_temperature_centre - self.ion_temperature_pedestal)
                            * (1.0 - (a / self.pedestal_radius).powf(self.ion_temperature_beta))
                                .powf(self.ion_temperature_peaking_factor)
                } else {
                    self.ion_temperature_separatrix
                        + (self.ion_temperature_pedestal - self.ion_temperature_separatrix)
                            * (self.minor_radius - a)
                            / (self.minor_radius - self.pedestal_radius)
                }
            }
        })
    }

    /// Map `(a, alpha)` onto `(R, Z)` in cm using this plasma's shape.
    pub fn convert_a_alpha_to_r_z(&self, a: f64, alpha: f64) -> (f64, f64) {
        convert_a_alpha_to_r_z(
            a,
            alpha,
            self.shafranov_factor,
            self.minor_radius,
            self.major_radius,
            self.triangularity,
            self.elongation,
        )
    }

    /// The neutron-producing reactions this fuel supports, each paired with
    /// the factor that turns an ion density `n_i` into the reacting pair
    /// density: `factor * n_i^2`.
    ///
    /// T(t,2n)4He is not included: yamc has no T-T neutron spectrum model,
    /// where D(d,n)3He and T(d,n)4He both have Ballabio spectra. For D-T fuel
    /// the T-T yield is a small fraction of the D-T yield, but the omission is
    /// a real difference from `openmc-plasma-source`, which carries a T-T
    /// component from NeSST. The other difference to expect when comparing
    /// the two is a birth peak a few tens of keV apart:
    /// [`fusion_neutron_spectrum`] takes the zero-temperature neutron energy
    /// from the reaction Q value, where `openmc-plasma-source` uses
    /// Ballabio's tabulated 14.021 / 2.4495 MeV.
    pub fn reactions(&self) -> Result<Vec<(FusionReactants, f64)>, String> {
        let deuterium = self.fuel_fraction(FuelIon::Deuterium);
        let tritium = self.fuel_fraction(FuelIon::Tritium);
        if deuterium == 0.0 {
            return Err(
                "fuel must contain deuterium: T-T fusion is not supported (yamc has no \
                 T(t,2n) neutron spectrum model)"
                    .to_string(),
            );
        }
        let mut reactions = Vec::new();
        if tritium > 0.0 {
            reactions.push((FusionReactants::DT, deuterium * tritium));
        }
        reactions.push((FusionReactants::DD, 0.5 * deuterium * deuterium));
        Ok(reactions)
    }

    /// Build the ring sources describing this plasma's neutron emission.
    ///
    /// Every non-empty `(R, Z)` mesh voxel becomes one source per reaction,
    /// sampling uniformly across the voxel in R and Z and across the toroidal
    /// sector in phi. Strengths are normalised to sum to 1, so the absolute
    /// neutron rate is set by the simulation's source rate, not here.
    pub fn sources(&self) -> Result<Vec<Source>, String> {
        self.validate()?;
        let reactions = self.reactions()?;

        let (n_r, n_z) = self.mesh_resolution;
        let shift = self.shafranov_factor.abs();
        let r_min = self.major_radius - self.minor_radius - shift;
        let r_max = self.major_radius + self.minor_radius + shift;
        let z_max = self.elongation * self.minor_radius;
        let r_edges = linspace(r_min, r_max, n_r + 1);
        let z_edges = linspace(-z_max, z_max, n_z + 1);

        let n_cells = n_r * n_z;
        let mut strengths = vec![vec![0.0_f64; n_cells]; reactions.len()];
        let mut temperature_weights = vec![vec![0.0_f64; n_cells]; reactions.len()];

        let grid_density = self.grid_density as f64;
        let da = self.minor_radius / grid_density;
        let dalpha = std::f64::consts::TAU / grid_density;
        let cell_area = da * dalpha;

        // Density, temperature and hence source density depend only on the
        // minor radius, so they are evaluated once per flux surface and reused
        // for every poloidal angle on it.
        let mut source_densities = vec![0.0_f64; reactions.len()];
        for i in 0..self.grid_density {
            let a = (i as f64 + 0.5) * da;
            let density = self.ion_density(a)?;
            let temperature = self.ion_temperature(a)?;
            for (k, (reactants, pair_factor)) in reactions.iter().enumerate() {
                source_densities[k] = neutron_source_density(
                    pair_factor * density * density,
                    temperature,
                    *reactants,
                );
            }
            if source_densities.iter().all(|&value| value <= 0.0) {
                continue;
            }

            for j in 0..self.grid_density {
                let alpha = (j as f64 + 0.5) * dalpha;
                let (r, z) = self.convert_a_alpha_to_r_z(a, alpha);
                let (Some(bin_r), Some(bin_z)) = (
                    bin_index(r, r_min, r_max, n_r),
                    bin_index(z, -z_max, z_max, n_z),
                ) else {
                    continue;
                };
                let cell = bin_r * n_z + bin_z;
                // The (a, alpha) grid is uniform, so each point stands for the
                // plasma volume it represents: the toroidal factor R times the
                // poloidal Jacobian |d(R, Z)/d(a, alpha)|. Without the R factor
                // the inboard side of the plasma is over-weighted.
                let volume = r * self.poloidal_jacobian(a, alpha) * cell_area;
                for (k, source_density) in source_densities.iter().enumerate() {
                    let weight = source_density * volume;
                    strengths[k][cell] += weight;
                    temperature_weights[k][cell] += weight * temperature;
                }
            }
        }

        let total: f64 = strengths.iter().flat_map(|row| row.iter()).sum();
        if total <= 0.0 {
            return Err(
                "total neutron source density is zero: the ion temperatures or densities are \
                 too low to produce fusion reactions"
                    .to_string(),
            );
        }

        let (phi_low, phi_high) = self.toroidal_bounds();
        let phi = Univariate::Uniform(Uniform::new(phi_low, phi_high)?);

        let mut sources = Vec::new();
        for (k, (reactants, _)) in reactions.iter().enumerate() {
            for cell in 0..n_cells {
                let strength = strengths[k][cell];
                if strength <= 0.0 {
                    continue;
                }
                // Emission-weighted mean ion temperature of the voxel, which
                // sets the Ballabio spectrum sampled from it.
                let temperature = temperature_weights[k][cell] / strength;
                let energy = fusion_neutron_spectrum(temperature, *reactants)?;
                let (bin_r, bin_z) = (cell / n_z, cell % n_z);
                let space = CylindricalRing::new(
                    Univariate::Uniform(Uniform::new(r_edges[bin_r], r_edges[bin_r + 1])?),
                    phi.clone(),
                    Univariate::Uniform(Uniform::new(z_edges[bin_z], z_edges[bin_z + 1])?),
                    [0.0, 0.0, 0.0],
                );
                sources.push(Source {
                    space: SourceSpatialDistribution::CylindricalRing(Box::new(space)),
                    angle: AngularDistribution::Isotropic,
                    energy: SourceEnergyDistribution::Normal(energy),
                    strength: strength / total,
                });
            }
        }
        Ok(sources)
    }

    /// `|d(R, Z)/d(a, alpha)|`, the poloidal area element of the mapping.
    fn poloidal_jacobian(&self, a: f64, alpha: f64) -> f64 {
        let theta = alpha + self.triangularity * alpha.sin();
        let dtheta_dalpha = 1.0 + self.triangularity * alpha.cos();
        let dr_da =
            theta.cos() - 2.0 * self.shafranov_factor * a / (self.minor_radius * self.minor_radius);
        let dr_dalpha = -a * theta.sin() * dtheta_dalpha;
        let dz_da = self.elongation * alpha.sin();
        let dz_dalpha = self.elongation * a * alpha.cos();
        (dr_da * dz_dalpha - dr_dalpha * dz_da).abs()
    }

    /// The toroidal sector as an ascending `(low, high)` pair of angles. The
    /// angles are not wrapped into `[0, 2*pi)`: a ring source takes the sine
    /// and cosine of whatever it samples, so a sector crossing the seam needs
    /// no special case.
    fn toroidal_bounds(&self) -> (f64, f64) {
        let end = self.start_angle + self.rotation_angle;
        if self.rotation_angle < 0.0 {
            (end, self.start_angle)
        } else {
            (self.start_angle, end)
        }
    }

    fn fuel_fraction(&self, ion: FuelIon) -> f64 {
        self.fuel
            .iter()
            .filter(|(species, _)| *species == ion)
            .map(|(_, fraction)| fraction)
            .sum()
    }

    fn check_minor_radius(&self, a: f64) -> Result<(), String> {
        if !(0.0..=self.minor_radius).contains(&a) {
            return Err(format!(
                "minor radius position must be between 0 and the plasma minor radius {} cm \
                 (got {a} cm)",
                self.minor_radius
            ));
        }
        Ok(())
    }

    /// Check the parameters that [`TokamakPlasma::sources`] relies on, so a
    /// bad plasma fails before any sampling rather than producing NaNs.
    pub fn validate(&self) -> Result<(), String> {
        if self.major_radius <= 0.0 {
            return Err(format!(
                "major_radius must be positive (got {})",
                self.major_radius
            ));
        }
        if self.minor_radius <= 0.0 {
            return Err(format!(
                "minor_radius must be positive (got {})",
                self.minor_radius
            ));
        }
        if self.minor_radius >= self.major_radius {
            return Err(format!(
                "minor_radius must be less than major_radius (got {} >= {})",
                self.minor_radius, self.major_radius
            ));
        }
        if self.elongation <= 0.0 {
            return Err(format!(
                "elongation must be positive (got {})",
                self.elongation
            ));
        }
        if !(-1.0..=1.0).contains(&self.triangularity) {
            return Err(format!(
                "triangularity must be between -1 and 1 (got {})",
                self.triangularity
            ));
        }
        if self.pedestal_radius >= self.minor_radius {
            return Err(format!(
                "pedestal_radius must be less than minor_radius (got {} >= {})",
                self.pedestal_radius, self.minor_radius
            ));
        }
        if self.shafranov_factor.abs() >= 0.5 * self.minor_radius {
            return Err(format!(
                "abs(shafranov_factor) must be less than half the minor radius (got {} >= {})",
                self.shafranov_factor.abs(),
                0.5 * self.minor_radius
            ));
        }
        for (name, value) in [
            ("ion_density_centre", self.ion_density_centre),
            ("ion_density_pedestal", self.ion_density_pedestal),
            ("ion_density_separatrix", self.ion_density_separatrix),
        ] {
            if value <= 0.0 {
                return Err(format!("{name} must be positive (got {value})"));
            }
        }
        for (name, value) in [
            ("ion_temperature_centre", self.ion_temperature_centre),
            ("ion_temperature_pedestal", self.ion_temperature_pedestal),
            (
                "ion_temperature_separatrix",
                self.ion_temperature_separatrix,
            ),
        ] {
            if value < 0.0 {
                return Err(format!("{name} must not be negative (got {value})"));
            }
        }
        let two_pi = std::f64::consts::TAU;
        if !(-two_pi..=two_pi).contains(&self.start_angle) {
            return Err(format!(
                "start_angle must be between -2*pi and 2*pi (got {})",
                self.start_angle
            ));
        }
        if !(-two_pi..=two_pi).contains(&self.rotation_angle) || self.rotation_angle == 0.0 {
            return Err(format!(
                "rotation_angle must be a non-zero value between -2*pi and 2*pi (got {})",
                self.rotation_angle
            ));
        }
        if self.mesh_resolution.0 == 0 || self.mesh_resolution.1 == 0 {
            return Err(format!(
                "mesh_resolution must be positive in both directions (got {:?})",
                self.mesh_resolution
            ));
        }
        if self.grid_density == 0 {
            return Err("grid_density must be positive (got 0)".to_string());
        }
        if self.fuel.is_empty() {
            return Err("fuel must contain at least one species".to_string());
        }
        for (species, fraction) in &self.fuel {
            if *fraction <= 0.0 || *fraction > 1.0 {
                return Err(format!(
                    "fuel fractions must be greater than 0 and at most 1 (got {fraction} for \
                     {species:?})"
                ));
            }
        }
        let sum: f64 = self.fuel.iter().map(|(_, fraction)| fraction).sum();
        if (sum - 1.0).abs() > 1.0e-9 {
            return Err(format!("fuel fractions must sum to 1 (got {sum})"));
        }
        Ok(())
    }
}

/// `count` evenly spaced values from `start` to `stop` inclusive.
fn linspace(start: f64, stop: f64, count: usize) -> Vec<f64> {
    let last = (count - 1) as f64;
    (0..count)
        .map(|i| start + (stop - start) * (i as f64) / last)
        .collect()
}

/// Index of the uniform bin holding `value`, or `None` when it falls outside
/// `[low, high]`. The top edge belongs to the last bin, matching the
/// half-open-except-at-the-end convention of a histogram.
fn bin_index(value: f64, low: f64, high: f64, bins: usize) -> Option<usize> {
    if value < low || value > high {
        return None;
    }
    let position = (value - low) / (high - low) * bins as f64;
    Some((position as usize).min(bins - 1))
}
