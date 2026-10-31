//! Element names, physical constants and the reaction sum rules.
//!
//! The tables here are generated from the Python package's `data.py` rather
//! than retyped, so the two cannot drift apart through a transcription slip.

use crate::error::{Error, Result};

/// Boltzmann constant, eV per kelvin.
pub const K_BOLTZMANN: f64 = 8.617333262e-5;

/// eV per MeV.
pub const EV_PER_MEV: f64 = 1.0e6;

/// Chemical symbol by atomic number. Index 0 is the neutron, `n`.
pub const ATOMIC_SYMBOL: [&str; 119] = [
    "n", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S",
    "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge",
    "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "In", "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd",
    "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm",
    "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn",
    "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
];

/// Element name to chemical symbol, as the Python package spells them.
pub const ELEMENT_SYMBOL: [(&str, &str); 123] = [
    ("neutron", "n"),
    ("hydrogen", "H"),
    ("helium", "He"),
    ("lithium", "Li"),
    ("beryllium", "Be"),
    ("boron", "B"),
    ("carbon", "C"),
    ("nitrogen", "N"),
    ("oxygen", "O"),
    ("fluorine", "F"),
    ("neon", "Ne"),
    ("sodium", "Na"),
    ("magnesium", "Mg"),
    ("aluminium", "Al"),
    ("aluminum", "Al"),
    ("silicon", "Si"),
    ("phosphorus", "P"),
    ("sulfur", "S"),
    ("sulphur", "S"),
    ("chlorine", "Cl"),
    ("argon", "Ar"),
    ("potassium", "K"),
    ("calcium", "Ca"),
    ("scandium", "Sc"),
    ("titanium", "Ti"),
    ("vanadium", "V"),
    ("chromium", "Cr"),
    ("manganese", "Mn"),
    ("iron", "Fe"),
    ("cobalt", "Co"),
    ("nickel", "Ni"),
    ("copper", "Cu"),
    ("zinc", "Zn"),
    ("gallium", "Ga"),
    ("germanium", "Ge"),
    ("arsenic", "As"),
    ("selenium", "Se"),
    ("bromine", "Br"),
    ("krypton", "Kr"),
    ("rubidium", "Rb"),
    ("strontium", "Sr"),
    ("yttrium", "Y"),
    ("zirconium", "Zr"),
    ("niobium", "Nb"),
    ("molybdenum", "Mo"),
    ("technetium", "Tc"),
    ("ruthenium", "Ru"),
    ("rhodium", "Rh"),
    ("palladium", "Pd"),
    ("silver", "Ag"),
    ("cadmium", "Cd"),
    ("indium", "In"),
    ("tin", "Sn"),
    ("antimony", "Sb"),
    ("tellurium", "Te"),
    ("iodine", "I"),
    ("xenon", "Xe"),
    ("caesium", "Cs"),
    ("cesium", "Cs"),
    ("barium", "Ba"),
    ("lanthanum", "La"),
    ("cerium", "Ce"),
    ("praseodymium", "Pr"),
    ("neodymium", "Nd"),
    ("promethium", "Pm"),
    ("samarium", "Sm"),
    ("europium", "Eu"),
    ("gadolinium", "Gd"),
    ("terbium", "Tb"),
    ("dysprosium", "Dy"),
    ("holmium", "Ho"),
    ("erbium", "Er"),
    ("thulium", "Tm"),
    ("ytterbium", "Yb"),
    ("lutetium", "Lu"),
    ("hafnium", "Hf"),
    ("tantalum", "Ta"),
    ("tungsten", "W"),
    ("wolfram", "W"),
    ("rhenium", "Re"),
    ("osmium", "Os"),
    ("iridium", "Ir"),
    ("platinum", "Pt"),
    ("gold", "Au"),
    ("mercury", "Hg"),
    ("thallium", "Tl"),
    ("lead", "Pb"),
    ("bismuth", "Bi"),
    ("polonium", "Po"),
    ("astatine", "At"),
    ("radon", "Rn"),
    ("francium", "Fr"),
    ("radium", "Ra"),
    ("actinium", "Ac"),
    ("thorium", "Th"),
    ("protactinium", "Pa"),
    ("uranium", "U"),
    ("neptunium", "Np"),
    ("plutonium", "Pu"),
    ("americium", "Am"),
    ("curium", "Cm"),
    ("berkelium", "Bk"),
    ("californium", "Cf"),
    ("einsteinium", "Es"),
    ("fermium", "Fm"),
    ("mendelevium", "Md"),
    ("nobelium", "No"),
    ("lawrencium", "Lr"),
    ("rutherfordium", "Rf"),
    ("dubnium", "Db"),
    ("seaborgium", "Sg"),
    ("bohrium", "Bh"),
    ("hassium", "Hs"),
    ("meitnerium", "Mt"),
    ("darmstadtium", "Ds"),
    ("roentgenium", "Rg"),
    ("copernicium", "Cn"),
    ("nihonium", "Nh"),
    ("flerovium", "Fl"),
    ("moscovium", "Mc"),
    ("livermorium", "Lv"),
    ("tennessine", "Ts"),
    ("oganesson", "Og"),
];

/// Reactions whose cross section is the sum of others, from ENDF-102.
pub const SUM_RULES: [(i32, &[i32]); 15] = [
    (1, &[2, 3]),
    (
        3,
        &[
            4, 5, 11, 16, 17, 22, 23, 24, 25, 27, 28, 29, 30, 32, 33, 34, 35, 36, 37, 41, 42, 44,
            45, 152, 153, 154, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168,
            169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 183, 184, 185, 186,
            187, 188, 189, 190, 194, 195, 196, 198, 199, 200,
        ],
    ),
    (
        4,
        &[
            50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71,
            72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91,
        ],
    ),
    (
        16,
        &[
            875, 876, 877, 878, 879, 880, 881, 882, 883, 884, 885, 886, 887, 888, 889, 890, 891,
        ],
    ),
    (18, &[19, 20, 21, 38]),
    (27, &[18, 101]),
    (
        101,
        &[
            102, 103, 104, 105, 106, 107, 108, 109, 111, 112, 113, 114, 115, 116, 117, 155, 182,
            191, 192, 193, 197,
        ],
    ),
    (
        103,
        &[
            600, 601, 602, 603, 604, 605, 606, 607, 608, 609, 610, 611, 612, 613, 614, 615, 616,
            617, 618, 619, 620, 621, 622, 623, 624, 625, 626, 627, 628, 629, 630, 631, 632, 633,
            634, 635, 636, 637, 638, 639, 640, 641, 642, 643, 644, 645, 646, 647, 648, 649,
        ],
    ),
    (
        104,
        &[
            650, 651, 652, 653, 654, 655, 656, 657, 658, 659, 660, 661, 662, 663, 664, 665, 666,
            667, 668, 669, 670, 671, 672, 673, 674, 675, 676, 677, 678, 679, 680, 681, 682, 683,
            684, 685, 686, 687, 688, 689, 690, 691, 692, 693, 694, 695, 696, 697, 698, 699,
        ],
    ),
    (
        105,
        &[
            700, 701, 702, 703, 704, 705, 706, 707, 708, 709, 710, 711, 712, 713, 714, 715, 716,
            717, 718, 719, 720, 721, 722, 723, 724, 725, 726, 727, 728, 729, 730, 731, 732, 733,
            734, 735, 736, 737, 738, 739, 740, 741, 742, 743, 744, 745, 746, 747, 748, 749,
        ],
    ),
    (
        106,
        &[
            750, 751, 752, 753, 754, 755, 756, 757, 758, 759, 760, 761, 762, 763, 764, 765, 766,
            767, 768, 769, 770, 771, 772, 773, 774, 775, 776, 777, 778, 779, 780, 781, 782, 783,
            784, 785, 786, 787, 788, 789, 790, 791, 792, 793, 794, 795, 796, 797, 798, 799,
        ],
    ),
    (
        107,
        &[
            800, 801, 802, 803, 804, 805, 806, 807, 808, 809, 810, 811, 812, 813, 814, 815, 816,
            817, 818, 819, 820, 821, 822, 823, 824, 825, 826, 827, 828, 829, 830, 831, 832, 833,
            834, 835, 836, 837, 838, 839, 840, 841, 842, 843, 844, 845, 846, 847, 848, 849,
        ],
    ),
    (501, &[502, 504, 516, 522]),
    (516, &[515, 517]),
    (
        522,
        &[
            534, 535, 536, 537, 538, 539, 540, 541, 542, 543, 544, 545, 546, 547, 548, 549, 550,
            551, 552, 553, 554, 555, 556, 557, 558, 559, 560, 561, 562, 563, 564, 565, 566, 567,
            568, 569, 570, 571, 572,
        ],
    ),
];

/// The atomic number of a chemical symbol, e.g. `"Am"` gives 95.
pub fn atomic_number(symbol: &str) -> Option<u32> {
    ATOMIC_SYMBOL
        .iter()
        .position(|&s| s == symbol)
        .map(|z| z as u32)
}

/// The reactions MT is the sum of, if it is a summed reaction.
pub fn sum_rule(mt: i32) -> Option<&'static [i32]> {
    SUM_RULES.iter().find(|&&(k, _)| k == mt).map(|&(_, v)| v)
}

/// A nuclide's name in GNDS convention, e.g. `gnds_name(95, 242, 1)` gives
/// `"Am242_m1"`.
pub fn gnds_name(z: u32, a: u32, m: u32) -> String {
    let symbol = ATOMIC_SYMBOL.get(z as usize).copied().unwrap_or("?");
    if m > 0 {
        format!("{symbol}{a}_m{m}")
    } else {
        format!("{symbol}{a}")
    }
}

/// The atomic number, mass number and metastable state of a GNDS name.
///
/// The inverse of [`gnds_name`]; `"Am242_m1"` gives `(95, 242, 1)`.
pub fn zam(name: &str) -> Result<(u32, u32, u32)> {
    // Equivalent to the Python reader's `([A-Zn][a-z]*)(\d+)((?:_[em]\d+)?)`,
    // matched by hand so the crate stays dependency-free.
    let bytes = name.as_bytes();
    let bad = || Error::BadNuclideName {
        name: name.to_string(),
    };

    // The symbol: an upper-case letter (or a lone `n`) then lower-case ones.
    let mut i = match bytes.first() {
        Some(c) if c.is_ascii_uppercase() || *c == b'n' => 1,
        _ => return Err(bad()),
    };
    while i < bytes.len() && bytes[i].is_ascii_lowercase() {
        i += 1;
    }
    let symbol = &name[..i];

    // The mass number.
    let start = i;
    while i < bytes.len() && bytes[i].is_ascii_digit() {
        i += 1;
    }
    if i == start {
        return Err(bad());
    }
    let a: u32 = name[start..i].parse().map_err(|_| bad())?;

    // An optional `_m<n>` or `_e<n>` state.
    let metastable = if i == bytes.len() {
        0
    } else {
        let rest = &name[i..];
        let digits = rest
            .strip_prefix("_m")
            .or_else(|| rest.strip_prefix("_e"))
            .ok_or_else(bad)?;
        if digits.is_empty() || !digits.bytes().all(|c| c.is_ascii_digit()) {
            return Err(bad());
        }
        digits.parse().map_err(|_| bad())?
    };

    let z = atomic_number(symbol).ok_or_else(|| Error::UnknownElement {
        symbol: symbol.to_string(),
    })?;
    Ok((z, a, metastable))
}

/// A temperature rendered the way the data files name it, e.g. `"294K"`.
///
/// Ties round to even, because Python's `round` does and this string is used
/// as a dictionary key: `1200.5` has to give `"1200K"` on both sides or the
/// two readers disagree about which temperature a table belongs to.
/// `f64::round` rounds half away from zero and would give `"1201K"`.
pub fn temperature_str(t: f64) -> String {
    let floor = t.floor();
    let fraction = t - floor;
    // Round up when past the halfway point, and on an exact tie only when
    // doing so lands on an even number.
    let up = fraction > 0.5 || (fraction == 0.5 && (floor as i64) % 2 != 0);
    format!("{}K", floor as i64 + i64::from(up))
}

/// Python's `str()` of a float.
///
/// Two places need it, and both would be wrong without it. The decay mode
/// encoding packs a chain of modes as the digits of a decimal and decodes it
/// by stripping the zeros and the point, which only works because `str()`
/// always writes a fractional part, `10.0` keeps its trailing zero where a
/// bare shortest-round-trip format gives `10` and loses it. An NJOY input deck
/// interpolates temperatures the same way, so `900.0` has to stay `900.0`.
pub fn python_float_str(value: f64) -> String {
    let s = format!("{value}");
    if s.contains('.')
        || s.contains('e')
        || s.contains('E')
        || s.contains("inf")
        || s.contains("NaN")
    {
        s
    } else {
        format!("{s}.0")
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn formats_floats_the_way_python_str_does() {
        // Python always writes a fractional part; Rust's shortest form does
        // not, and both the decay mode encoding and the NJOY deck depend on
        // it being there.
        assert_eq!(python_float_str(1.0), "1.0");
        assert_eq!(python_float_str(10.0), "10.0");
        assert_eq!(python_float_str(0.0), "0.0");
        assert_eq!(python_float_str(900.0), "900.0");
        assert_eq!(python_float_str(-0.0), "-0.0");
        assert_eq!(python_float_str(1234567.0), "1234567.0");
        // A value that already has one is left alone.
        assert_eq!(python_float_str(293.6), "293.6");
        assert_eq!(python_float_str(1.5), "1.5");
    }

    use super::*;

    #[test]
    fn symbols_and_numbers_round_trip() {
        assert_eq!(ATOMIC_SYMBOL[0], "n");
        assert_eq!(ATOMIC_SYMBOL[95], "Am");
        assert_eq!(ATOMIC_SYMBOL[118], "Og");
        for (z, &symbol) in ATOMIC_SYMBOL.iter().enumerate() {
            assert_eq!(atomic_number(symbol), Some(z as u32), "{symbol}");
        }
        assert_eq!(atomic_number("Xx"), None);
    }

    #[test]
    fn gnds_names_round_trip() {
        for (z, a, m, name) in [
            (95, 242, 1, "Am242_m1"),
            (95, 244, 0, "Am244"),
            (1, 1, 0, "H1"),
            (0, 1, 0, "n1"),
        ] {
            assert_eq!(gnds_name(z, a, m), name);
            assert_eq!(zam(name).unwrap(), (z, a, m));
        }
    }

    #[test]
    fn zam_rejects_what_is_not_a_nuclide() {
        for bad in ["", "Am", "242", "_m1", "Am242_x1", "Am242_m", "Xx242"] {
            assert!(zam(bad).is_err(), "{bad:?} should not parse");
        }
        // An excited state uses _e rather than _m and is read the same way.
        assert_eq!(zam("Am242_e2").unwrap(), (95, 242, 2));
    }

    #[test]
    fn sum_rules_are_looked_up_by_mt() {
        assert_eq!(sum_rule(1), Some(&[2, 3][..]));
        assert_eq!(sum_rule(18).unwrap(), &[19, 20, 21, 38]);
        // MT=4 is the sum of the discrete inelastic levels.
        assert_eq!(sum_rule(4).unwrap().len(), 42);
        assert_eq!(sum_rule(2), None);
    }

    #[test]
    fn temperatures_are_named_as_the_data_files_name_them() {
        assert_eq!(temperature_str(293.6), "294K");
        assert_eq!(temperature_str(900.0), "900K");
        assert_eq!(temperature_str(294.4), "294K");
        // Ties go to even, as Python's round does. f64::round would give
        // 1201K here and 1202K below, and the string is used as a key.
        assert_eq!(temperature_str(1200.5), "1200K");
        assert_eq!(temperature_str(1201.5), "1202K");
        assert_eq!(temperature_str(-0.5), "0K");
    }
}
