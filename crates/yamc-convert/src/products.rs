//! `products.arrow`: the secondary particles each reaction emits.
//!
//! One row per (reaction, product). The distributions themselves are in
//! `distributions.arrow`, joined on `(reaction_mt, product_idx, dist_idx)`, so
//! the two files must be written from the same traversal in the same order.
//! [`crate::distributions::write_distributions`] walks the reactions exactly
//! as this does, and `n_distribution` here is what tells the reader how many
//! `dist_idx` values to look for.

use std::error::Error;
use std::path::Path;

use endf::product::Yield;
use endf::IncidentNeutron;

use crate::sections::*;

/// The reactions to write products for, in MT order.
///
/// Shared with the distribution writer so the two files cannot disagree about
/// which product is index 3 of MT 16.
pub fn product_rows(data: &IncidentNeutron) -> Vec<(i32, usize, &endf::product::Product)> {
    // Plain MT order, and deliberately NOT the published files' order.
    //
    // Those are written from a Python dict, which preserves insertion order:
    // the ACE table's MTR list first, then the reactions the parser adds
    // afterwards (the photon-production sums, then `SUMMED_IF_ABSENT`, then
    // the rest). B10 ends up as `2, 51..85, 102, 103, 801, 4` and U240 puts
    // MT 3 and MT 18 last. A `BTreeMap` does not carry that history, so the
    // order is not reproducible here without changing the parser's public
    // types.
    //
    // It also does not mean anything: the reader joins products and
    // distributions on `(reaction_mt, product_idx)` and never looks at row
    // position. Sorting by MT is deterministic and easy to reason about, which
    // is worth more than matching a byte layout that carries no information.
    // A comparison against published data should key on the join columns
    // rather than the row index.
    let mut out = Vec::new();
    for (&mt, rx) in &data.reactions {
        for (idx, product) in rx.products.iter().enumerate() {
            out.push((mt, idx, product));
        }
    }
    out
}

/// A yield as `(type, data, shape, breakpoints, interpolation)`.
///
/// The two forms share one `data` column: a polynomial writes its coefficients
/// and shape `[n]`, a tabulated yield writes x then y end to end with shape
/// `[2, n]`. The reader splits on `shape[1]`, so a wrong shape silently halves
/// the multiplicity rather than failing.
fn yield_columns(y: &Yield) -> (&'static str, Vec<f64>, Vec<i32>, Vec<i32>, Vec<i32>) {
    match y {
        Yield::Polynomial(p) => (
            "Polynomial",
            p.coefficients.clone(),
            vec![p.coefficients.len() as i32],
            Vec::new(),
            Vec::new(),
        ),
        Yield::Tabulated(t) => {
            let mut data = t.x.clone();
            data.extend_from_slice(&t.y);
            (
                "Tabulated1D",
                data,
                vec![2, t.x.len() as i32],
                t.breakpoints.clone(),
                t.interpolation.clone(),
            )
        }
    }
}

/// Write `products.arrow`.
pub fn write_products(data: &IncidentNeutron, dir: &Path) -> Result<(), Box<dyn Error>> {
    let rows = product_rows(data);

    let mut reaction_mt = Vec::with_capacity(rows.len());
    let mut product_idx = Vec::with_capacity(rows.len());
    let mut particle = Vec::with_capacity(rows.len());
    let mut mode = Vec::with_capacity(rows.len());
    let mut decay_rate = Vec::with_capacity(rows.len());
    let mut n_distribution = Vec::with_capacity(rows.len());
    let mut yield_type = Vec::with_capacity(rows.len());
    let mut yield_data = Vec::with_capacity(rows.len());
    let mut yield_shape = Vec::with_capacity(rows.len());
    let mut yield_breakpoints = Vec::with_capacity(rows.len());
    let mut yield_interpolation = Vec::with_capacity(rows.len());

    for (mt, idx, product) in &rows {
        // The reader knows two particle types and refuses a third rather than
        // dropping the product, so anything else has to stop here with the
        // name in the message.
        if product.name != "neutron" && product.name != "photon" {
            return Err(format!(
                "MT {mt} product {idx} is a {:?}, which the neutron format has \
                 no column for; it holds neutrons and photons only",
                product.name
            )
            .into());
        }

        let (ty, ydata, yshape, ybreak, yinterp) = yield_columns(&product.yield_);

        reaction_mt.push(*mt);
        product_idx.push(*idx as i32);
        particle.push(product.name.clone());
        mode.push(product.emission_mode.name().to_string());
        decay_rate.push(product.decay_rate);
        n_distribution.push(product.distribution.len() as i32);
        yield_type.push(ty.to_string());
        yield_data.push(ydata);
        yield_shape.push(yshape);
        yield_breakpoints.push(ybreak);
        yield_interpolation.push(yinterp);
    }

    write_section(
        &dir.join("products.arrow"),
        "products.arrow",
        vec![
            ints(&reaction_mt),
            ints(&product_idx),
            strings(&particle),
            strings(&mode),
            floats(&decay_rate),
            ints(&n_distribution),
            strings(&yield_type),
            float_lists(&yield_data),
            int_lists(&yield_shape),
            int_lists(&yield_breakpoints),
            int_lists(&yield_interpolation),
        ],
    )
}
