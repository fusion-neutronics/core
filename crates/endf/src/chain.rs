//! Depletion chains: which nuclides turn into which, and how fast.
//!
//! Built from three sub-libraries at once (decay, fission product yields and
//! neutron reactions) because a chain is the join of all three: what a
//! nuclide decays into, what a neutron turns it into, and what its fission
//! leaves behind.
//!
//! The XML serialisation and the burnup matrix of the Python package are not
//! here. Neither reads a nuclear data format: one needs an XML writer and the
//! other sparse linear algebra, and both belong in a consumer.

use std::collections::BTreeMap;

use crate::data::{gnds_name, zam, ATOMIC_SYMBOL};
use crate::decay::{Decay, FissionProductYields};
use crate::error::{Error, Result};
use crate::material::Material;
use crate::reaction::FISSION_MTS;

/// One transmutation reaction: which MTs mean it, what it does to the mass and
/// atomic numbers, and what else comes out.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReactionInfo {
    /// The reaction's name, e.g. `"(n,2n)"`.
    pub name: &'static str,
    /// Every MT that means this reaction. More than one where the format
    /// numbers the levels separately.
    pub mts: &'static [i32],
    /// Change in the mass number.
    pub delta_a: i64,
    /// Change in the atomic number.
    pub delta_z: i64,
    /// The light nuclides emitted alongside.
    pub secondaries: &'static [&'static str],
}

/// Every transmutation reaction a chain can follow.
pub const REACTIONS: [ReactionInfo; 84] = [
    ReactionInfo {
        name: "(n,2nd)",
        mts: &[11],
        delta_a: -3,
        delta_z: -1,
        secondaries: &["H2"],
    },
    ReactionInfo {
        name: "(n,2n)",
        mts: &[
            16, 875, 876, 877, 878, 879, 880, 881, 882, 883, 884, 885, 886, 887, 888, 889, 890, 891,
        ],
        delta_a: -1,
        delta_z: 0,
        secondaries: &[],
    },
    ReactionInfo {
        name: "(n,3n)",
        mts: &[17],
        delta_a: -2,
        delta_z: 0,
        secondaries: &[],
    },
    ReactionInfo {
        name: "(n,na)",
        mts: &[22],
        delta_a: -4,
        delta_z: -2,
        secondaries: &["He4"],
    },
    ReactionInfo {
        name: "(n,n3a)",
        mts: &[23],
        delta_a: -12,
        delta_z: -6,
        secondaries: &["He4", "He4", "He4"],
    },
    ReactionInfo {
        name: "(n,2na)",
        mts: &[24],
        delta_a: -5,
        delta_z: -2,
        secondaries: &["He4"],
    },
    ReactionInfo {
        name: "(n,3na)",
        mts: &[25],
        delta_a: -6,
        delta_z: -2,
        secondaries: &["He4"],
    },
    ReactionInfo {
        name: "(n,np)",
        mts: &[28],
        delta_a: -1,
        delta_z: -1,
        secondaries: &["H1"],
    },
    ReactionInfo {
        name: "(n,n2a)",
        mts: &[29],
        delta_a: -8,
        delta_z: -4,
        secondaries: &["He4", "He4"],
    },
    ReactionInfo {
        name: "(n,2n2a)",
        mts: &[30],
        delta_a: -9,
        delta_z: -4,
        secondaries: &["He4", "He4"],
    },
    ReactionInfo {
        name: "(n,nd)",
        mts: &[32],
        delta_a: -2,
        delta_z: -1,
        secondaries: &["H2"],
    },
    ReactionInfo {
        name: "(n,nt)",
        mts: &[33],
        delta_a: -3,
        delta_z: -1,
        secondaries: &["H3"],
    },
    ReactionInfo {
        name: "(n,n3He)",
        mts: &[34],
        delta_a: -3,
        delta_z: -2,
        secondaries: &["He3"],
    },
    ReactionInfo {
        name: "(n,nd2a)",
        mts: &[35],
        delta_a: -10,
        delta_z: -5,
        secondaries: &["H2", "He4", "He4"],
    },
    ReactionInfo {
        name: "(n,nt2a)",
        mts: &[36],
        delta_a: -11,
        delta_z: -5,
        secondaries: &["H3", "He4", "He4"],
    },
    ReactionInfo {
        name: "(n,4n)",
        mts: &[37],
        delta_a: -3,
        delta_z: 0,
        secondaries: &[],
    },
    ReactionInfo {
        name: "(n,2np)",
        mts: &[41],
        delta_a: -2,
        delta_z: -1,
        secondaries: &["H1"],
    },
    ReactionInfo {
        name: "(n,3np)",
        mts: &[42],
        delta_a: -3,
        delta_z: -1,
        secondaries: &["H1"],
    },
    ReactionInfo {
        name: "(n,n2p)",
        mts: &[44],
        delta_a: -2,
        delta_z: -2,
        secondaries: &["H1", "H1"],
    },
    ReactionInfo {
        name: "(n,npa)",
        mts: &[45],
        delta_a: -5,
        delta_z: -3,
        secondaries: &["H1", "He4"],
    },
    ReactionInfo {
        name: "(n,gamma)",
        mts: &[102],
        delta_a: 1,
        delta_z: 0,
        secondaries: &[],
    },
    ReactionInfo {
        name: "(n,p)",
        mts: &[
            103, 600, 601, 602, 603, 604, 605, 606, 607, 608, 609, 610, 611, 612, 613, 614, 615,
            616, 617, 618, 619, 620, 621, 622, 623, 624, 625, 626, 627, 628, 629, 630, 631, 632,
            633, 634, 635, 636, 637, 638, 639, 640, 641, 642, 643, 644, 645, 646, 647, 648, 649,
        ],
        delta_a: 0,
        delta_z: -1,
        secondaries: &["H1"],
    },
    ReactionInfo {
        name: "(n,d)",
        mts: &[
            104, 650, 651, 652, 653, 654, 655, 656, 657, 658, 659, 660, 661, 662, 663, 664, 665,
            666, 667, 668, 669, 670, 671, 672, 673, 674, 675, 676, 677, 678, 679, 680, 681, 682,
            683, 684, 685, 686, 687, 688, 689, 690, 691, 692, 693, 694, 695, 696, 697, 698, 699,
        ],
        delta_a: -1,
        delta_z: -1,
        secondaries: &["H2"],
    },
    ReactionInfo {
        name: "(n,t)",
        mts: &[
            105, 700, 701, 702, 703, 704, 705, 706, 707, 708, 709, 710, 711, 712, 713, 714, 715,
            716, 717, 718, 719, 720, 721, 722, 723, 724, 725, 726, 727, 728, 729, 730, 731, 732,
            733, 734, 735, 736, 737, 738, 739, 740, 741, 742, 743, 744, 745, 746, 747, 748, 749,
        ],
        delta_a: -2,
        delta_z: -1,
        secondaries: &["H3"],
    },
    ReactionInfo {
        name: "(n,3He)",
        mts: &[
            106, 750, 751, 752, 753, 754, 755, 756, 757, 758, 759, 760, 761, 762, 763, 764, 765,
            766, 767, 768, 769, 770, 771, 772, 773, 774, 775, 776, 777, 778, 779, 780, 781, 782,
            783, 784, 785, 786, 787, 788, 789, 790, 791, 792, 793, 794, 795, 796, 797, 798, 799,
        ],
        delta_a: -2,
        delta_z: -2,
        secondaries: &["He3"],
    },
    ReactionInfo {
        name: "(n,a)",
        mts: &[
            107, 800, 801, 802, 803, 804, 805, 806, 807, 808, 809, 810, 811, 812, 813, 814, 815,
            816, 817, 818, 819, 820, 821, 822, 823, 824, 825, 826, 827, 828, 829, 830, 831, 832,
            833, 834, 835, 836, 837, 838, 839, 840, 841, 842, 843, 844, 845, 846, 847, 848, 849,
        ],
        delta_a: -3,
        delta_z: -2,
        secondaries: &["He4"],
    },
    ReactionInfo {
        name: "(n,2a)",
        mts: &[108],
        delta_a: -7,
        delta_z: -4,
        secondaries: &["He4", "He4"],
    },
    ReactionInfo {
        name: "(n,3a)",
        mts: &[109],
        delta_a: -11,
        delta_z: -6,
        secondaries: &["He4", "He4", "He4"],
    },
    ReactionInfo {
        name: "(n,2p)",
        mts: &[111],
        delta_a: -1,
        delta_z: -2,
        secondaries: &["H1", "H1"],
    },
    ReactionInfo {
        name: "(n,pa)",
        mts: &[112],
        delta_a: -4,
        delta_z: -3,
        secondaries: &["H1", "He4"],
    },
    ReactionInfo {
        name: "(n,t2a)",
        mts: &[113],
        delta_a: -10,
        delta_z: -5,
        secondaries: &["H3", "He4", "He4"],
    },
    ReactionInfo {
        name: "(n,d2a)",
        mts: &[114],
        delta_a: -9,
        delta_z: -5,
        secondaries: &["H2", "He4", "He4"],
    },
    ReactionInfo {
        name: "(n,pd)",
        mts: &[115],
        delta_a: -2,
        delta_z: -2,
        secondaries: &["H1", "H2"],
    },
    ReactionInfo {
        name: "(n,pt)",
        mts: &[116],
        delta_a: -3,
        delta_z: -2,
        secondaries: &["H1", "H3"],
    },
    ReactionInfo {
        name: "(n,da)",
        mts: &[117],
        delta_a: -5,
        delta_z: -3,
        secondaries: &["H2", "He4"],
    },
    ReactionInfo {
        name: "(n,5n)",
        mts: &[152],
        delta_a: -4,
        delta_z: 0,
        secondaries: &[],
    },
    ReactionInfo {
        name: "(n,6n)",
        mts: &[153],
        delta_a: -5,
        delta_z: 0,
        secondaries: &[],
    },
    ReactionInfo {
        name: "(n,2nt)",
        mts: &[154],
        delta_a: -4,
        delta_z: -1,
        secondaries: &["H3"],
    },
    ReactionInfo {
        name: "(n,ta)",
        mts: &[155],
        delta_a: -6,
        delta_z: -3,
        secondaries: &["H3", "He4"],
    },
    ReactionInfo {
        name: "(n,4np)",
        mts: &[156],
        delta_a: -4,
        delta_z: -1,
        secondaries: &["H1"],
    },
    ReactionInfo {
        name: "(n,3nd)",
        mts: &[157],
        delta_a: -4,
        delta_z: -1,
        secondaries: &["H2"],
    },
    ReactionInfo {
        name: "(n,nda)",
        mts: &[158],
        delta_a: -6,
        delta_z: -3,
        secondaries: &["H2", "He4"],
    },
    ReactionInfo {
        name: "(n,2npa)",
        mts: &[159],
        delta_a: -6,
        delta_z: -3,
        secondaries: &["H1", "He4"],
    },
    ReactionInfo {
        name: "(n,7n)",
        mts: &[160],
        delta_a: -6,
        delta_z: 0,
        secondaries: &[],
    },
    ReactionInfo {
        name: "(n,8n)",
        mts: &[161],
        delta_a: -7,
        delta_z: 0,
        secondaries: &[],
    },
    ReactionInfo {
        name: "(n,5np)",
        mts: &[162],
        delta_a: -5,
        delta_z: -1,
        secondaries: &["H1"],
    },
    ReactionInfo {
        name: "(n,6np)",
        mts: &[163],
        delta_a: -6,
        delta_z: -1,
        secondaries: &["H1"],
    },
    ReactionInfo {
        name: "(n,7np)",
        mts: &[164],
        delta_a: -7,
        delta_z: -1,
        secondaries: &["H1"],
    },
    ReactionInfo {
        name: "(n,4na)",
        mts: &[165],
        delta_a: -7,
        delta_z: -2,
        secondaries: &["He4"],
    },
    ReactionInfo {
        name: "(n,5na)",
        mts: &[166],
        delta_a: -8,
        delta_z: -2,
        secondaries: &["He4"],
    },
    ReactionInfo {
        name: "(n,6na)",
        mts: &[167],
        delta_a: -9,
        delta_z: -2,
        secondaries: &["He4"],
    },
    ReactionInfo {
        name: "(n,7na)",
        mts: &[168],
        delta_a: -10,
        delta_z: -2,
        secondaries: &["He4"],
    },
    ReactionInfo {
        name: "(n,4nd)",
        mts: &[169],
        delta_a: -5,
        delta_z: -1,
        secondaries: &["H2"],
    },
    ReactionInfo {
        name: "(n,5nd)",
        mts: &[170],
        delta_a: -6,
        delta_z: -1,
        secondaries: &["H2"],
    },
    ReactionInfo {
        name: "(n,6nd)",
        mts: &[171],
        delta_a: -7,
        delta_z: -1,
        secondaries: &["H2"],
    },
    ReactionInfo {
        name: "(n,3nt)",
        mts: &[172],
        delta_a: -5,
        delta_z: -1,
        secondaries: &["H3"],
    },
    ReactionInfo {
        name: "(n,4nt)",
        mts: &[173],
        delta_a: -6,
        delta_z: -1,
        secondaries: &["H3"],
    },
    ReactionInfo {
        name: "(n,5nt)",
        mts: &[174],
        delta_a: -7,
        delta_z: -1,
        secondaries: &["H3"],
    },
    ReactionInfo {
        name: "(n,6nt)",
        mts: &[175],
        delta_a: -8,
        delta_z: -1,
        secondaries: &["H3"],
    },
    ReactionInfo {
        name: "(n,2n3He)",
        mts: &[176],
        delta_a: -4,
        delta_z: -2,
        secondaries: &["He3"],
    },
    ReactionInfo {
        name: "(n,3n3He)",
        mts: &[177],
        delta_a: -5,
        delta_z: -2,
        secondaries: &["He3"],
    },
    ReactionInfo {
        name: "(n,4n3He)",
        mts: &[178],
        delta_a: -6,
        delta_z: -2,
        secondaries: &["He3"],
    },
    ReactionInfo {
        name: "(n,3n2p)",
        mts: &[179],
        delta_a: -4,
        delta_z: -2,
        secondaries: &["H1", "H1"],
    },
    ReactionInfo {
        name: "(n,3n2a)",
        mts: &[180],
        delta_a: -10,
        delta_z: -4,
        secondaries: &["He4", "He4"],
    },
    ReactionInfo {
        name: "(n,3npa)",
        mts: &[181],
        delta_a: -7,
        delta_z: -3,
        secondaries: &["H1", "He4"],
    },
    ReactionInfo {
        name: "(n,dt)",
        mts: &[182],
        delta_a: -4,
        delta_z: -2,
        secondaries: &["H2", "H3"],
    },
    ReactionInfo {
        name: "(n,npd)",
        mts: &[183],
        delta_a: -3,
        delta_z: -2,
        secondaries: &["H1", "H2"],
    },
    ReactionInfo {
        name: "(n,npt)",
        mts: &[184],
        delta_a: -4,
        delta_z: -2,
        secondaries: &["H1", "H3"],
    },
    ReactionInfo {
        name: "(n,ndt)",
        mts: &[185],
        delta_a: -5,
        delta_z: -2,
        secondaries: &["H2", "H3"],
    },
    ReactionInfo {
        name: "(n,np3He)",
        mts: &[186],
        delta_a: -4,
        delta_z: -3,
        secondaries: &["H1", "He3"],
    },
    ReactionInfo {
        name: "(n,nd3He)",
        mts: &[187],
        delta_a: -5,
        delta_z: -3,
        secondaries: &["H2", "He3"],
    },
    ReactionInfo {
        name: "(n,nt3He)",
        mts: &[188],
        delta_a: -6,
        delta_z: -3,
        secondaries: &["H3", "He3"],
    },
    ReactionInfo {
        name: "(n,nta)",
        mts: &[189],
        delta_a: -7,
        delta_z: -3,
        secondaries: &["H3", "He4"],
    },
    ReactionInfo {
        name: "(n,2n2p)",
        mts: &[190],
        delta_a: -3,
        delta_z: -2,
        secondaries: &["H1", "H1"],
    },
    ReactionInfo {
        name: "(n,p3He)",
        mts: &[191],
        delta_a: -4,
        delta_z: -3,
        secondaries: &["H1", "He3"],
    },
    ReactionInfo {
        name: "(n,d3He)",
        mts: &[192],
        delta_a: -5,
        delta_z: -3,
        secondaries: &["H2", "He3"],
    },
    ReactionInfo {
        name: "(n,3Hea)",
        mts: &[193],
        delta_a: -6,
        delta_z: -4,
        secondaries: &["He3", "He4"],
    },
    ReactionInfo {
        name: "(n,4n2p)",
        mts: &[194],
        delta_a: -5,
        delta_z: -2,
        secondaries: &["H1", "H1"],
    },
    ReactionInfo {
        name: "(n,4n2a)",
        mts: &[195],
        delta_a: -11,
        delta_z: -4,
        secondaries: &["He4", "He4"],
    },
    ReactionInfo {
        name: "(n,4npa)",
        mts: &[196],
        delta_a: -8,
        delta_z: -3,
        secondaries: &["H1", "He4"],
    },
    ReactionInfo {
        name: "(n,3p)",
        mts: &[197],
        delta_a: -2,
        delta_z: -3,
        secondaries: &["H1", "H1", "H1"],
    },
    ReactionInfo {
        name: "(n,n3p)",
        mts: &[198],
        delta_a: -3,
        delta_z: -3,
        secondaries: &["H1", "H1", "H1"],
    },
    ReactionInfo {
        name: "(n,3n2pa)",
        mts: &[199],
        delta_a: -8,
        delta_z: -4,
        secondaries: &["H1", "H1", "He4"],
    },
    ReactionInfo {
        name: "(n,5n2p)",
        mts: &[200],
        delta_a: -6,
        delta_z: -2,
        secondaries: &["H1", "H1"],
    },
];

/// The reactions a chain includes unless told otherwise. Fission is always
/// included where the evaluation has it.
pub const DEFAULT_REACTIONS: [&str; 6] =
    ["(n,2n)", "(n,3n)", "(n,4n)", "(n,gamma)", "(n,p)", "(n,a)"];

/// Look a reaction up by name.
pub fn reaction_info(name: &str) -> Option<&'static ReactionInfo> {
    REACTIONS.iter().find(|r| r.name == name)
}

/// Scale evaluated decay branching ratios so they sum to one.
///
/// Evaluated ratios often miss unity by a little. The residual goes into the
/// *largest* branch, which it perturbs fractionally least; putting it in an
/// arbitrary branch instead can move a 1e-9 branch by orders of magnitude.
/// Ratios that already sum to one are left exactly as evaluated.
pub fn normalise_branch_ratios(branch_ratios: &mut [f64]) {
    if branch_ratios.is_empty() {
        return;
    }
    let total: f64 = branch_ratios.iter().sum();
    // The same closeness test Python's `math.isclose` makes by default.
    let close = (total - 1.0).abs() <= 1e-9 * total.abs().max(1.0);
    if close {
        return;
    }
    let (i, &max) =
        branch_ratios
            .iter()
            .enumerate()
            .fold((0, &branch_ratios[0]), |(bi, bv), (i, v)| {
                if v > bv {
                    (i, v)
                } else {
                    (bi, bv)
                }
            });
    branch_ratios[i] = max - total + 1.0;
}

/// One decay path out of a nuclide.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct DecayPath {
    /// The chain of modes, joined with commas as the chain format writes it.
    pub kind: String,
    /// The nuclide left behind. `None` where the product is a bare neutron.
    pub target: Option<String>,
    pub branching_ratio: f64,
}

/// One neutron-induced path out of a nuclide.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ReactionPath {
    /// The reaction's name, or `"fission"`.
    pub kind: String,
    /// The nuclide left behind. `None` for fission, which has no single one.
    pub target: Option<String>,
    /// Q value in eV.
    pub q_value: f64,
    pub branching_ratio: f64,
}

/// Fission yields at one incident energy, by product name.
pub type FissionYields = BTreeMap<String, f64>;

/// One nuclide's place in a chain.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Nuclide {
    /// GNDS name, e.g. `"Am242_m1"`.
    pub name: String,
    /// Half-life in seconds. `None` for a stable nuclide.
    pub half_life: Option<f64>,
    /// The evaluation's stated standard deviation on the half-life, in seconds.
    ///
    /// `None` where the evaluation published none, which is NOT zero: an
    /// unstated uncertainty and one measured to be negligible are different
    /// claims, and a consumer that cannot tell them apart reports the first
    /// as confidence (issue #515).
    pub half_life_uncertainty: Option<f64>,
    /// Average energy per decay in eV.
    pub decay_energy: f64,
    /// The evaluation's standard deviation on `decay_energy`, in eV.
    ///
    /// Quadrature over the three recoverable-heat components, which is what
    /// MT=457 supports: it publishes a sigma per component and no covariance
    /// between them. `None` where the evaluation stated none, which is not
    /// zero.
    pub decay_energy_uncertainty: Option<f64>,
    /// Where `decay_energy` came from: [`DECAY_ENERGY_EVALUATED`] for an
    /// evaluated decay scheme, [`DECAY_ENERGY_PLACEHOLDER`] for the Q/3
    /// stand-in some libraries write for nuclides nobody has evaluated (see
    /// [`crate::decay::placeholder_mean_energies`]), or `filled:<library>`
    /// after [`Chain::fill_placeholder_decay_energies`] replaced a placeholder
    /// from another library. `None` for a stable nuclide, which has no decay
    /// energy to source.
    pub decay_energy_source: Option<String>,
    pub decay_modes: Vec<DecayPath>,
    pub reactions: Vec<ReactionPath>,
    /// Fission yields by incident energy in eV. Empty when the nuclide does
    /// not fission, or when its yields are borrowed from another nuclide.
    pub yield_data: BTreeMap<String, FissionYields>,
    /// The nuclide whose yields stand in for this one's, where the library has
    /// none of its own.
    pub borrowed_yields_from: Option<String>,
}

impl Nuclide {
    pub fn new(name: &str) -> Nuclide {
        Nuclide {
            name: name.to_string(),
            ..Default::default()
        }
    }

    /// Everything that does not add up, at the given tolerance.
    ///
    /// Decay branching ratios sum to one, the branches of each reaction sum to
    /// one, and fission yields sum to two (two fragments per fission). An
    /// empty result means the nuclide is consistent.
    pub fn validate(&self, tolerance: f64) -> Vec<String> {
        let mut problems = Vec::new();
        let mut check = |property: &str, actual: f64, expected: f64| {
            if !(expected - tolerance..=expected + tolerance).contains(&actual) {
                problems.push(format!(
                    "Nuclide {} has {property} that sum to {actual} instead of \
                     {expected} +/- {tolerance:7.4e}",
                    self.name
                ));
            }
        };

        if !self.decay_modes.is_empty() {
            let total: f64 = self.decay_modes.iter().map(|m| m.branching_ratio).sum();
            check("decay mode branch ratios", total, 1.0);
        }

        // Each reaction's branches are their own sum, so a nuclide with two
        // reactions is not expected to sum to two.
        let mut kinds: Vec<&str> = self.reactions.iter().map(|r| r.kind.as_str()).collect();
        kinds.sort_unstable();
        kinds.dedup();
        for kind in kinds {
            let total: f64 = self
                .reactions
                .iter()
                .filter(|r| r.kind == kind)
                .map(|r| r.branching_ratio)
                .sum();
            check(&format!("{kind} reaction branch ratios"), total, 1.0);
        }

        for (energy, yields) in &self.yield_data {
            let total: f64 = yields.values().sum();
            check(&format!("fission yields (E = {energy} eV)"), total, 2.0);
        }

        problems
    }

    /// The incident energies the fission yields are given at, in eV.
    ///
    /// The keys are the energies formatted as the Python package writes them,
    /// so they order as text; this gives them back as numbers.
    pub fn yield_energies(&self) -> Vec<f64> {
        let mut out: Vec<f64> = self
            .yield_data
            .keys()
            .filter_map(|k| k.parse().ok())
            .collect();
        out.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        out
    }
}

/// A depletion chain: every nuclide, and the paths between them.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Chain {
    pub nuclides: Vec<Nuclide>,
}

/// `decay_energy_source` of a nuclide whose average energies were evaluated.
pub const DECAY_ENERGY_EVALUATED: &str = "evaluation";
/// `decay_energy_source` of a nuclide carrying the Q/3 placeholder.
pub const DECAY_ENERGY_PLACEHOLDER: &str = "placeholder";

/// What [`Chain::fill_placeholder_decay_energies`] did, nuclide by nuclide.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct DecayEnergyFill {
    /// The library the replacements came from.
    pub library: String,
    /// Placeholders replaced: `(nuclide, before, after)` in eV.
    pub replaced: Vec<(String, f64, f64)>,
    /// Placeholders left alone because the other library's record for the
    /// same name has a half-life more than a quarter away from this one's, so
    /// it may describe a different isomer: libraries do not agree on which of
    /// two close-lying isomers is `m1`.
    pub half_life_mismatch: Vec<String>,
    /// Placeholders the other library could not improve on: absent there, or
    /// a placeholder there too.
    pub unfilled: Vec<String>,
}

impl Chain {
    pub fn new() -> Chain {
        Chain::default()
    }

    /// Replace placeholder average decay energies with another library's
    /// evaluated ones.
    ///
    /// Only the mean decay energy and its uncertainty move; the half-life and
    /// the decay modes stay as this chain's own library gave them, since the
    /// placeholder records carry evaluated half-lives and the point is to
    /// change as little as possible. A replacement is taken only where the
    /// other library has a real decay scheme for the same nuclide name and a
    /// half-life within 25% of this one's, and each nuclide's
    /// `decay_energy_source` records what happened. `library` names the fill
    /// in that record and in the returned report.
    pub fn fill_placeholder_decay_energies(
        &mut self,
        fill: &[Material],
        library: &str,
    ) -> Result<DecayEnergyFill> {
        let mut other: BTreeMap<String, Decay> = BTreeMap::new();
        for material in fill {
            let data = Decay::from_material(material)?;
            other.insert(data.nuclide.name.clone(), data);
        }

        let mut report = DecayEnergyFill {
            library: library.to_string(),
            ..Default::default()
        };
        for nuclide in &mut self.nuclides {
            if nuclide.decay_energy_source.as_deref() != Some(DECAY_ENERGY_PLACEHOLDER) {
                continue;
            }
            let candidate = other.get(&nuclide.name).filter(|d| {
                !d.nuclide.stable && !d.mean_energy_placeholder && d.decay_energy().0 > 0.0
            });
            let Some(candidate) = candidate else {
                report.unfilled.push(nuclide.name.clone());
                continue;
            };
            let same_state = match (nuclide.half_life, candidate.half_life) {
                (Some(ours), Some((theirs, _))) if ours > 0.0 && theirs > 0.0 => {
                    (theirs / ours - 1.0).abs() <= 0.25
                }
                _ => false,
            };
            if !same_state {
                report.half_life_mismatch.push(nuclide.name.clone());
                continue;
            }
            let (energy, sigma) = candidate.decay_energy();
            report
                .replaced
                .push((nuclide.name.clone(), nuclide.decay_energy, energy));
            nuclide.decay_energy = energy;
            nuclide.decay_energy_uncertainty = (sigma > 0.0).then_some(sigma);
            nuclide.decay_energy_source = Some(format!("filled:{library}"));
        }
        Ok(report)
    }

    pub fn contains(&self, name: &str) -> bool {
        self.nuclides.iter().any(|n| n.name == name)
    }

    pub fn get(&self, name: &str) -> Option<&Nuclide> {
        self.nuclides.iter().find(|n| n.name == name)
    }

    pub fn len(&self) -> usize {
        self.nuclides.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nuclides.is_empty()
    }

    /// Build a chain from decay, fission product yield and neutron
    /// evaluations.
    ///
    /// `reactions` names the transmutation reactions to follow; pass
    /// [`DEFAULT_REACTIONS`] for the usual set. Fission is always followed
    /// where an evaluation has it.
    pub fn from_endf(
        decay: &[Material],
        fpy: &[Material],
        neutron: &[Material],
        reactions: &[&str],
    ) -> Result<Chain> {
        // What each target's neutron evaluation says each channel's Q value
        // is. QI is the Q of the channel actually populated; QM, the
        // mass-difference Q, is not the same thing and a few evaluations give
        // it with the opposite sign.
        let mut q_values: BTreeMap<String, BTreeMap<i32, f64>> = BTreeMap::new();
        for material in neutron {
            let Some(meta) = material.mf1_mt451() else {
                continue;
            };
            let (z, a) = (meta.za / 1000, meta.za % 1000);
            let name = gnds_name(z as u32, a as u32, meta.liso as u32);
            let entry = q_values.entry(name).or_default();
            for &(mf, mt) in material.section_data.keys() {
                if mf == 3 {
                    if let Some(section) = material.mf3(mt) {
                        entry.insert(mt, section.qi);
                    }
                }
            }
        }

        let mut decay_data: BTreeMap<String, Decay> = BTreeMap::new();
        for material in decay {
            let data = Decay::from_material(material)?;
            // The neutron's own decay data is not a chain nuclide.
            if data.nuclide.atomic_number == 0 {
                continue;
            }
            decay_data.insert(data.nuclide.name.clone(), data);
        }

        let mut fpy_data: BTreeMap<String, FissionProductYields> = BTreeMap::new();
        for material in fpy {
            let Some(meta) = material.mf1_mt451() else {
                continue;
            };
            let (z, a) = (meta.za / 1000, meta.za % 1000);
            let name = gnds_name(z as u32, a as u32, meta.liso as u32);
            fpy_data.insert(name, FissionProductYields::from_material(material)?);
        }

        // Nuclides come out ordered by Z, then A, then metastable state, which
        // is what `zam` gives.
        let mut parents: Vec<&String> = decay_data.keys().collect();
        parents.sort_by_key(|name| zam(name).unwrap_or((0, 0, 0)));

        let mut chain = Chain::new();
        for parent in parents {
            let data = &decay_data[parent];
            let mut nuclide = Nuclide::new(parent);

            let half_life = data.half_life.map(|(t, _)| t).unwrap_or(0.0);
            if !data.nuclide.stable && half_life != 0.0 {
                nuclide.half_life = Some(half_life);
                // The sigma sits beside the half-life in MF=8 MT=457. A zero
                // there is how the format says "not stated": an evaluation
                // that had measured the uncertainty to be zero would be
                // claiming an exact half-life, which nothing does. Mapping it
                // straight through would make `Some(0.0)` -- known to be
                // exact -- out of 292 of the 3562 unstable nuclides in
                // ENDF/B-8.1, which is the confidence this column exists to
                // avoid manufacturing.
                nuclide.half_life_uncertainty = data.half_life.map(|(_, s)| s).filter(|s| *s > 0.0);
                // One call, so the value and its sigma cannot come from two
                // different evaluations of the same quadrature.
                let (energy, energy_sigma) = data.decay_energy();
                nuclide.decay_energy = energy;
                nuclide.decay_energy_uncertainty = (energy_sigma > 0.0).then_some(energy_sigma);
                nuclide.decay_energy_source = Some(
                    if data.mean_energy_placeholder {
                        DECAY_ENERGY_PLACEHOLDER
                    } else {
                        DECAY_ENERGY_EVALUATED
                    }
                    .to_string(),
                );

                let mut ratios: Vec<f64> = Vec::new();
                let mut ids: Vec<(String, Option<String>)> = Vec::new();
                for mode in &data.modes {
                    let daughter = mode.daughter();
                    let target = match &daughter {
                        Some(d) if decay_data.contains_key(d) => Some(d.clone()),
                        Some(d) => replace_missing(d, &decay_data),
                        None => None,
                    };
                    ratios.push(mode.branching_ratio.0);
                    ids.push((mode.modes.join(","), target));
                }
                normalise_branch_ratios(&mut ratios);
                for (ratio, (kind, target)) in ratios.into_iter().zip(ids) {
                    nuclide.decay_modes.push(DecayPath {
                        kind,
                        target,
                        branching_ratio: ratio,
                    });
                }
            }

            let mut fissionable = false;
            if let Some(available) = q_values.get(parent) {
                for name in reactions {
                    let Some(info) = reaction_info(name) else {
                        continue;
                    };
                    if !info.mts.iter().any(|mt| available.contains_key(mt)) {
                        continue;
                    }
                    let a = data.nuclide.mass_number + info.delta_a;
                    let z = data.nuclide.atomic_number + info.delta_z;
                    let symbol = ATOMIC_SYMBOL.get(z as usize).copied().unwrap_or("?");
                    let mut daughter = Some(format!("{symbol}{a}"));
                    if let Some(d) = &daughter {
                        if !decay_data.contains_key(d) {
                            daughter = replace_missing(d, &decay_data);
                        }
                    }

                    // The first MT of the reaction that the evaluation has.
                    let q_value = info
                        .mts
                        .iter()
                        .find_map(|mt| available.get(mt).copied())
                        .unwrap_or(0.0);

                    nuclide.reactions.push(ReactionPath {
                        kind: name.to_string(),
                        target: daughter,
                        q_value,
                        branching_ratio: 1.0,
                    });
                }

                if FISSION_MTS.iter().any(|mt| available.contains_key(mt)) {
                    nuclide.reactions.push(ReactionPath {
                        kind: "fission".to_string(),
                        target: None,
                        q_value: available.get(&18).copied().unwrap_or(0.0),
                        branching_ratio: 1.0,
                    });
                    fissionable = true;
                }
            }

            if fissionable {
                match fpy_data.get(parent) {
                    Some(fpy) => {
                        let energies: Vec<f64> = if fpy.energies.is_empty() {
                            vec![0.0]
                        } else {
                            fpy.energies.clone()
                        };
                        for (energy, table) in energies.iter().zip(&fpy.independent) {
                            let mut yields: FissionYields = BTreeMap::new();
                            for product in table {
                                let name = if decay_data.contains_key(&product.name) {
                                    Some(product.name.clone())
                                } else {
                                    replace_missing(&product.name, &decay_data)
                                };
                                // A product with no stand-in (a bare
                                // neutron) is dropped rather than named.
                                if let Some(name) = name {
                                    *yields.entry(name).or_insert(0.0) += product.yield_.0;
                                }
                            }
                            nuclide.yield_data.insert(energy_key(*energy), yields);
                        }
                    }
                    None => {
                        nuclide.borrowed_yields_from =
                            Some(replace_missing_fpy(parent, &fpy_data, &decay_data));
                    }
                }
            }

            chain.nuclides.push(nuclide);
        }

        // Fill in the borrowed yields, now that every nuclide exists.
        let borrowed: Vec<(usize, String)> = chain
            .nuclides
            .iter()
            .enumerate()
            .filter_map(|(i, n)| n.borrowed_yields_from.clone().map(|from| (i, from)))
            .collect();
        for (i, from) in borrowed {
            if let Some(source) = chain.get(&from) {
                let yields = source.yield_data.clone();
                chain.nuclides[i].yield_data = yields;
            }
        }

        Ok(chain)
    }

    /// The branching ratios of one reaction, by parent and target.
    pub fn branch_ratios(&self, reaction: &str) -> BTreeMap<String, BTreeMap<String, f64>> {
        let mut out = BTreeMap::new();
        for nuclide in &self.nuclides {
            let branches: BTreeMap<String, f64> = nuclide
                .reactions
                .iter()
                .filter(|r| r.kind == reaction)
                .filter_map(|r| r.target.clone().map(|t| (t, r.branching_ratio)))
                .collect();
            if !branches.is_empty() {
                out.insert(nuclide.name.clone(), branches);
            }
        }
        out
    }

    /// Set the branching ratios of one reaction.
    ///
    /// Every branch of that reaction on the named parents is replaced. A
    /// parent not in the chain is an error rather than silently ignored: the
    /// caller has the wrong chain or the wrong name.
    pub fn set_branch_ratios(
        &mut self,
        branch_ratios: &BTreeMap<String, BTreeMap<String, f64>>,
        reaction: &str,
    ) -> Result<()> {
        for (parent, branches) in branch_ratios {
            let Some(nuclide) = self.nuclides.iter_mut().find(|n| &n.name == parent) else {
                return Err(Error::BadNuclideName {
                    name: parent.clone(),
                });
            };
            let q_value = nuclide
                .reactions
                .iter()
                .find(|r| r.kind == reaction)
                .map(|r| r.q_value)
                .unwrap_or(0.0);
            nuclide.reactions.retain(|r| r.kind != reaction);
            for (target, &ratio) in branches {
                nuclide.reactions.push(ReactionPath {
                    kind: reaction.to_string(),
                    target: Some(target.clone()),
                    q_value,
                    branching_ratio: ratio,
                });
            }
        }
        Ok(())
    }

    /// Everything in the chain that does not add up, nuclide by nuclide.
    ///
    /// An empty result means every branching ratio and yield is consistent.
    pub fn validate(&self, tolerance: f64) -> Vec<String> {
        self.nuclides
            .iter()
            .flat_map(|n| n.validate(tolerance))
            .collect()
    }

    /// The chain reachable from a set of starting nuclides.
    ///
    /// `level` bounds how many steps to follow; `None` follows to the end.
    /// Nuclides outside the reduced set keep their paths only where the target
    /// is also inside it, so the result is closed.
    pub fn reduce(&self, initial: &[&str], level: Option<usize>) -> Chain {
        let mut reachable: Vec<String> = Vec::new();
        let mut frontier: Vec<String> = initial.iter().map(|s| s.to_string()).collect();
        let mut depth = 0;
        while !frontier.is_empty() && level.map_or(true, |l| depth <= l) {
            let mut next = Vec::new();
            for name in frontier {
                if reachable.contains(&name) || !self.contains(&name) {
                    continue;
                }
                reachable.push(name.clone());
                let nuclide = self.get(&name).expect("just checked");
                for target in nuclide
                    .decay_modes
                    .iter()
                    .filter_map(|m| m.target.clone())
                    .chain(nuclide.reactions.iter().filter_map(|r| r.target.clone()))
                {
                    next.push(target);
                }
            }
            frontier = next;
            depth += 1;
        }

        let mut out = Chain::new();
        for name in &reachable {
            let mut nuclide = self.get(name).expect("reachable").clone();
            nuclide.decay_modes.retain(|m| match &m.target {
                Some(t) => reachable.contains(t),
                None => true,
            });
            nuclide.reactions.retain(|r| match &r.target {
                Some(t) => reachable.contains(t),
                None => true,
            });
            out.nuclides.push(nuclide);
        }
        out.nuclides
            .sort_by_key(|n| zam(&n.name).unwrap_or((0, 0, 0)));
        out
    }
}

/// The key a fission yield energy is stored under.
///
/// The Python package keys the yields by the float itself; this keys them by
/// its shortest round-tripping decimal, so the map has a total order and the
/// value survives the trip.
fn energy_key(energy: f64) -> String {
    format!("{energy}")
}

/// A stand-in for a product with no decay data.
///
/// Walks towards stability until it reaches a nuclide the library knows: down
/// by alpha decay above Z=98, and by beta otherwise, in whichever direction
/// the element's longest-lived isotope lies.
///
/// `None` when there is no stand-in: a bare neutron, which is simply dropped,
/// or a walk that leaves the table of elements without finding one. The
/// second happens when the decay library is small enough that the direction
/// cannot be judged, the Python reader indexes past the end and raises
/// `KeyError: -1` there; see issue #22.
pub fn replace_missing(product: &str, decay_data: &BTreeMap<String, Decay>) -> Option<String> {
    let (z, a, state) = zam(product).ok()?;
    let mut a = a as i64;
    let mut z = z as i64;
    let symbol = ATOMIC_SYMBOL.get(z as usize).copied()?;

    // A neutron is not replaced by anything.
    if z == 0 {
        return None;
    }

    // The ground state, where the product was metastable.
    let mut product = if state > 0 {
        format!("{symbol}{a}")
    } else {
        product.to_string()
    };

    // The longest-lived isotope of this element says which way stability lies.
    let mut half_life = 0.0;
    let mut mass_longest_lived = a;
    for (nuclide, data) in decay_data {
        let Some((mass, _)) = same_element(nuclide, symbol) else {
            continue;
        };
        if data.nuclide.stable {
            mass_longest_lived = mass;
            break;
        }
        let t = data.half_life.map_or(0.0, |(t, _)| t);
        if t > half_life {
            mass_longest_lived = mass;
            half_life = t;
        }
    }
    let beta_minus = mass_longest_lived < a;

    while !decay_data.contains_key(&product) {
        if z > 98 {
            z -= 2;
            a -= 4;
        } else if beta_minus {
            z += 1;
        } else {
            z -= 1;
        }
        if z < 1 || a < 1 {
            return None;
        }
        product = format!("{}{a}", ATOMIC_SYMBOL.get(z as usize).copied()?);
    }
    Some(product)
}

/// Whether a GNDS name is an isotope of the given element, and its mass.
fn same_element(name: &str, symbol: &str) -> Option<(i64, bool)> {
    let rest = name.strip_prefix(symbol)?;
    let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
    if digits.is_empty() {
        return None;
    }
    let tail = &rest[digits.len()..];
    // Only a metastable suffix may follow, or nothing.
    if !tail.is_empty() && !tail.starts_with("_m") {
        return None;
    }
    Some((digits.parse().ok()?, !tail.is_empty()))
}

/// A stand-in set of fission yields for an actinide the library has none for.
///
/// Tries the metastable state, then isotones in either direction, and falls
/// back to U235, whose yields every library has.
pub fn replace_missing_fpy(
    actinide: &str,
    fpy_data: &BTreeMap<String, FissionProductYields>,
    decay_data: &BTreeMap<String, Decay>,
) -> String {
    let Ok((z, a, m)) = zam(actinide) else {
        return "U235".to_string();
    };
    let (z, a) = (z as i64, a as i64);

    if m == 0 {
        let metastable = gnds_name(z as u32, a as u32, 1);
        if fpy_data.contains_key(&metastable) {
            return metastable;
        }
    }

    // Isotones: the neutron number held fixed while Z moves.
    for step in [1i64, -1] {
        let (mut z, mut a) = (z, a);
        let mut isotone = actinide.to_string();
        while decay_data.contains_key(&isotone) {
            z += step;
            a += step;
            if z < 0 {
                break;
            }
            isotone = gnds_name(z as u32, a as u32, 0);
            if fpy_data.contains_key(&isotone) {
                return isotone;
            }
        }
    }

    "U235".to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_reaction_table_matches_the_format() {
        // A few entries, spot-checked against what the reaction does.
        let two_n = reaction_info("(n,2n)").unwrap();
        assert_eq!((two_n.delta_a, two_n.delta_z), (-1, 0));
        // The level reactions all mean the same transmutation.
        assert!(two_n.mts.contains(&16));
        assert!(two_n.mts.contains(&891));

        let capture = reaction_info("(n,gamma)").unwrap();
        assert_eq!((capture.delta_a, capture.delta_z), (1, 0));
        assert!(capture.secondaries.is_empty());

        let alpha = reaction_info("(n,a)").unwrap();
        assert_eq!((alpha.delta_a, alpha.delta_z), (-3, -2));
        assert_eq!(alpha.secondaries, ["He4"]);

        assert!(reaction_info("(n,nonsense)").is_none());

        // Every reaction conserves nucleons: the change in A plus what the
        // secondaries carry away accounts for the incident neutron.
        for rx in &REACTIONS {
            let carried: i64 = rx
                .secondaries
                .iter()
                .map(|s| match *s {
                    "H1" => 1,
                    "H2" => 2,
                    "H3" => 3,
                    "He3" => 3,
                    "He4" => 4,
                    other => panic!("unexpected secondary {other}"),
                })
                .sum();
            assert!(
                rx.delta_a + carried <= 1,
                "{} gains nucleons from nowhere",
                rx.name
            );
        }
    }

    #[test]
    fn branch_ratios_are_normalised_into_the_largest_branch() {
        // The residual goes into the largest branch, which it moves least.
        let mut br = vec![0.7, 0.2, 0.05];
        normalise_branch_ratios(&mut br);
        assert_eq!(br.iter().sum::<f64>(), 1.0);
        assert_eq!(br[1..], [0.2, 0.05], "only the largest branch moves");

        // A tiny branch beside a large one keeps its order of magnitude.
        let mut br = vec![0.99, 1.0e-9];
        normalise_branch_ratios(&mut br);
        assert_eq!(br[1], 1.0e-9);
        assert_eq!(br.iter().sum::<f64>(), 1.0);

        // Ratios that already sum to one are left exactly as evaluated.
        let mut br = vec![0.25, 0.75];
        normalise_branch_ratios(&mut br);
        assert_eq!(br, [0.25, 0.75]);

        let mut br: Vec<f64> = Vec::new();
        normalise_branch_ratios(&mut br);
        assert!(br.is_empty());
    }

    #[test]
    fn a_missing_product_walks_to_one_the_library_has() {
        // An empty library has nothing to walk to, and the walk stops rather
        // than running off the table; see issue #22.
        let empty = BTreeMap::new();
        assert_eq!(replace_missing("Cd116", &empty), None);
        // A neutron has no stand-in at all.
        assert_eq!(replace_missing("n1", &empty), None);
    }

    #[test]
    fn a_reduced_chain_is_closed() {
        let mut chain = Chain::new();
        for (name, target) in [("A", Some("B")), ("B", Some("C")), ("C", None)] {
            let mut n = Nuclide::new(name);
            if let Some(target) = target {
                n.decay_modes.push(DecayPath {
                    kind: "beta-".to_string(),
                    target: Some(target.to_string()),
                    branching_ratio: 1.0,
                });
            }
            chain.nuclides.push(n);
        }
        // Unbounded: everything reachable comes along.
        let all = chain.reduce(&["A"], None);
        assert_eq!(
            all.nuclides
                .iter()
                .map(|n| n.name.as_str())
                .collect::<Vec<_>>(),
            ["A", "B", "C"]
        );

        // One step: A and B, and A's path to B survives because B is in.
        let one = chain.reduce(&["A"], Some(1));
        assert_eq!(
            one.nuclides
                .iter()
                .map(|n| n.name.as_str())
                .collect::<Vec<_>>(),
            ["A", "B"]
        );
        assert_eq!(one.get("A").unwrap().decay_modes.len(), 1);
        // B's path to C does not: C is outside, so the chain stays closed.
        assert!(one.get("B").unwrap().decay_modes.is_empty());

        // A starting nuclide the chain does not have contributes nothing.
        assert!(chain.reduce(&["Z"], None).is_empty());
    }
}
