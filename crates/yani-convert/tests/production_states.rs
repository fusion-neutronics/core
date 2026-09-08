//! Which final states an evaluation lists, read without any decay data.
//!
//! The fixture is TENDL-2017's Ir191, trimmed. It is here because its (n,2n)
//! is one of the two channels whose short state list breaks the FNS
//! benchmark: it lists Ir190's ground state and the 26 keV isomer and not the
//! 377 keV one, which is the state that carries the measured heat. So this
//! test documents a real library gap as well as exercising the read.

use endf::Material;

macro_rules! fixture {
    ($name:literal) => {
        include_bytes!(concat!("../../endf/fixtures/", $name))
    };
}

fn material(compressed: &[u8]) -> Material {
    let mut out = Vec::new();
    lzma_rs::xz_decompress(&mut &compressed[..], &mut out).expect("fixture decompresses");
    Material::from_str(&String::from_utf8(out).expect("fixture is UTF-8")).expect("fixture parses")
}

#[test]
fn the_states_a_reaction_lists_are_reported_in_level_order() {
    let ir191 = material(fixture!("n-077_Ir_191_trimmed.endf.xz"));
    let channels = yani_convert::production::extract_production(&ir191);
    assert!(!channels.is_empty());
    for channel in &channels {
        assert_eq!(channel.parent, "Ir191");
        assert!(
            channel
                .states
                .windows(2)
                .all(|w| w[0].level_index <= w[1].level_index),
            "MT{} states are out of level order",
            channel.mt
        );
    }

    let n2n = channels
        .iter()
        .find(|c| c.mt == 16)
        .expect("the fixture keeps MF=8/10 for MT=16");
    assert_eq!(n2n.reaction.as_deref(), Some("(n,2n)"));
    // Ground state and the 26 keV isomer. Ir190 also has a 377 keV isomer,
    // which JEFF-4.0, ENDF/B-VIII.1, JENDL-5.0 and TENDL-2025 all list and
    // this evaluation does not. That absence is the finding: nothing else
    // about the channel looks wrong.
    let energies: Vec<f64> = n2n.states.iter().map(|s| s.excitation_energy).collect();
    assert_eq!(energies.len(), 2, "{energies:?}");
    assert_eq!(energies[0], 0.0);
    assert!((energies[1] - 26_100.0).abs() < 100.0, "{energies:?}");
    assert!(n2n.excited().count() == 1);

    // The product is named at its ground state even for the excited entry,
    // since naming the isomer would need decay data this read never touches.
    assert!(n2n.states.iter().all(|s| s.product == "Ir190"));
    assert!(n2n.states.iter().all(|s| s.source == "cross_section"));
}

#[test]
fn a_yield_channel_is_labelled_as_one() {
    let ir191 = material(fixture!("n-077_Ir_191_trimmed.endf.xz"));
    let channels = yani_convert::production::extract_production(&ir191);
    let capture = channels
        .iter()
        .find(|c| c.mt == 102)
        .expect("the fixture keeps MF=9 for MT=102");
    assert_eq!(capture.reaction.as_deref(), Some("(n,gamma)"));
    assert!(
        capture.states.iter().all(|s| s.source == "yield"),
        "capture is given as MF=9 yields here"
    );
    assert!(capture.excited().count() >= 1);
}

/// An evaluation with no MF=1 MT=451 cannot name its parent, so it reports
/// nothing rather than guessing a name from the filename.
#[test]
fn a_decay_evaluation_has_no_production_to_report() {
    let decay = material(fixture!("dec-049_In_116m1.endf.xz"));
    assert!(yani_convert::production::extract_production(&decay).is_empty());
}
