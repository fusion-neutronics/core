//! Flattening a [`Univariate`] into the `(x, p, c)` triple the format stores.
//!
//! Every ragged distribution in `distributions.arrow` is written the same way:
//! one flat `f64` array holding each column end to end, plus an offset array
//! saying where each sub-table starts. This turns one distribution into the
//! row of that layout.
//!
//! # Discrete lines come first
//!
//! An outgoing energy distribution may carry discrete photon lines alongside a
//! continuous spectrum, which the parser represents as a
//! [`Univariate::Mixture`] of a [`Discrete`] and a [`Tabular`]. The format has
//! no mixture: it stores one table and an `n_discrete` count saying how many
//! of the leading points are lines. So the discrete part must be written
//! first, and `n_discrete` must equal its length.
//!
//! Getting that order wrong does not fail to load. It produces a distribution
//! where `n_discrete` points of a continuous spectrum are sampled as discrete
//! lines and the real lines are smeared, which is a plausible-looking spectrum
//! with the wrong shape.

use endf::univariate::{Interpolation, Univariate};

/// One sub-table in the flat layout.
pub struct Flat {
    pub x: Vec<f64>,
    pub p: Vec<f64>,
    pub c: Vec<f64>,
    /// 1 for histogram, 2 for linear-linear, which is the only distinction the
    /// format draws.
    pub interp: i32,
    pub n_discrete: usize,
}

impl Flat {
    /// The number of points, which is what the offsets step by.
    pub fn len(&self) -> usize {
        self.x.len()
    }

    pub fn is_empty(&self) -> bool {
        self.x.is_empty()
    }
}

/// The interpolation code the format uses.
fn code(interp: Interpolation) -> i32 {
    match interp {
        Interpolation::Histogram => 1,
        _ => 2,
    }
}

/// The cumulative distribution, taken from the file where it gave one.
///
/// Recomputing it instead would be a silent change: an ACE table's CDF is what
/// the sampler was built against, and it need not agree bit for bit with the
/// integral of the density beside it.
fn cdf(stored: &Option<Vec<f64>>, x: &[f64], p: &[f64], interp: Interpolation) -> Vec<f64> {
    if let Some(c) = stored {
        if c.len() == x.len() {
            return c.clone();
        }
    }
    let mut c = Vec::with_capacity(x.len());
    let mut acc = 0.0;
    c.push(0.0);
    for i in 1..x.len() {
        let dx = x[i] - x[i - 1];
        acc += match interp {
            Interpolation::Histogram => p[i - 1] * dx,
            _ => 0.5 * (p[i - 1] + p[i]) * dx,
        };
        c.push(acc);
    }
    c
}

/// Flatten a distribution into one sub-table.
pub fn flatten(dist: &Univariate) -> Flat {
    match dist {
        Univariate::Tabular(t) => Flat {
            c: cdf(&t.c, &t.x, &t.p, t.interpolation),
            x: t.x.clone(),
            p: t.p.clone(),
            interp: code(t.interpolation),
            n_discrete: 0,
        },
        Univariate::Discrete(d) => {
            let n = d.x.len();
            Flat {
                c: cdf(&d.c, &d.x, &d.p, Interpolation::Histogram),
                x: d.x.clone(),
                p: d.p.clone(),
                // A discrete line is sampled by its own rule, so the
                // interpolation code is not consulted; histogram is what the
                // ACE reader assigns.
                interp: 1,
                n_discrete: n,
            }
        }
        // Isotropic. Written with the histogram code, which is what the
        // published files hold: a constant density samples the same either
        // way, so this is a convention rather than a physical claim.
        Univariate::Uniform(u) => {
            let density = if u.b > u.a { 1.0 / (u.b - u.a) } else { 0.0 };
            Flat {
                x: vec![u.a, u.b],
                p: vec![density, density],
                c: vec![0.0, 1.0],
                interp: 1,
                n_discrete: 0,
            }
        }
        // Discrete lines and a continuous spectrum in one distribution. The
        // components are concatenated in the order the parser built them,
        // which is discrete first, because `n_discrete` counts from the front.
        Univariate::Mixture(m) => {
            let mut out = Flat {
                x: Vec::new(),
                p: Vec::new(),
                c: Vec::new(),
                interp: 2,
                n_discrete: 0,
            };
            let mut discrete_seen = 0;
            let mut continuous_interp = None;
            for part in &m.distribution {
                let flat = flatten(part);
                if flat.n_discrete > 0 {
                    discrete_seen += flat.n_discrete;
                } else if continuous_interp.is_none() {
                    continuous_interp = Some(flat.interp);
                }
                out.x.extend_from_slice(&flat.x);
                out.p.extend_from_slice(&flat.p);
                out.c.extend_from_slice(&flat.c);
            }
            out.n_discrete = discrete_seen;
            // The continuous component's scheme, since that is the part the
            // code actually interpolates.
            out.interp = continuous_interp.unwrap_or(2);
            out
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use endf::univariate::{Discrete, Mixture, Tabular};

    #[test]
    fn a_tabular_keeps_the_cdf_the_file_gave() {
        // Deliberately inconsistent with the density: the point is that it is
        // taken verbatim rather than recomputed, so an ACE table round-trips.
        let t = Tabular::with_cdf(
            vec![0.0, 1.0, 2.0],
            vec![0.5, 0.5, 0.0],
            Interpolation::LinearLinear,
            vec![0.0, 0.4, 0.9],
        );
        let flat = flatten(&Univariate::Tabular(t));
        assert_eq!(flat.c, vec![0.0, 0.4, 0.9]);
        assert_eq!(flat.interp, 2);
        assert_eq!(flat.n_discrete, 0);
    }

    #[test]
    fn a_mixture_puts_its_discrete_lines_first() {
        // The ordering that matters. `n_discrete` counts from the front, so
        // writing the continuous part first would make the reader treat two
        // spectrum points as photon lines and the lines as spectrum, which
        // loads cleanly and samples wrongly.
        let mut discrete = Discrete::new(vec![1.0e6, 2.0e6], vec![0.3, 0.2]);
        discrete.c = Some(vec![0.3, 0.5]);
        let continuous = Tabular::with_cdf(
            vec![0.0, 5.0e6],
            vec![1.0e-7, 1.0e-7],
            Interpolation::Histogram,
            vec![0.5, 1.0],
        );
        let mixture = Univariate::Mixture(Mixture::new(
            vec![0.5, 0.5],
            vec![
                Univariate::Discrete(discrete),
                Univariate::Tabular(continuous),
            ],
        ));

        let flat = flatten(&mixture);
        assert_eq!(flat.n_discrete, 2);
        assert_eq!(flat.x, vec![1.0e6, 2.0e6, 0.0, 5.0e6]);
        assert_eq!(flat.p, vec![0.3, 0.2, 1.0e-7, 1.0e-7]);
        assert_eq!(flat.c, vec![0.3, 0.5, 0.5, 1.0]);
        // The continuous component's scheme, not the discrete part's.
        assert_eq!(flat.interp, 1);
    }

    #[test]
    fn a_cdf_is_built_when_the_file_gave_none() {
        let t = Tabular::new(
            vec![0.0, 1.0, 2.0],
            vec![0.0, 1.0, 0.0],
            Interpolation::LinearLinear,
        );
        let flat = flatten(&Univariate::Tabular(t));
        assert_eq!(flat.c, vec![0.0, 0.5, 1.0]);
    }
}
