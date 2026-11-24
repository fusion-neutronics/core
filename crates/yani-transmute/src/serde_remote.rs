//! Serde forms of types from crates that do not depend on serde, for the
//! transmutation results file ([`crate::results_arrow`]). Each mirrors its
//! type with serde's `remote` derive, which fails to compile if the type
//! gains a variant or field, so the mirrors cannot drift.

use std::collections::BTreeMap;

use endf::resonance_covariance::{Location, Quantity};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use yamc_element::displacement::{DisplacementEnergy, DisplacementEnergySource};

#[derive(Serialize, Deserialize)]
#[serde(remote = "Location")]
pub(crate) enum LocationDef {
    Orbital { section: usize, index: usize },
    SpinGroup { group: usize, index: usize },
    Range,
    Channel { group: usize, channel: usize },
    Unresolved { orbital: usize, spin: usize },
}

#[derive(Serialize, Deserialize)]
#[serde(remote = "Quantity")]
pub(crate) enum QuantityDef {
    Energy,
    NeutronWidth,
    CaptureWidth,
    FissionWidth,
    SecondFissionWidth,
    CompetitiveWidth,
    LevelSpacing,
    ReducedNeutronWidth,
    ChannelWidth(usize),
    ScatteringRadius,
}

#[derive(Serialize, Deserialize)]
#[serde(remote = "DisplacementEnergySource")]
enum DisplacementEnergySourceDef {
    AstmE521,
    OecdNea2015,
    User,
}

#[derive(Serialize, Deserialize)]
#[serde(remote = "DisplacementEnergy")]
struct DisplacementEnergyDef {
    energy_ev: f64,
    #[serde(with = "DisplacementEnergySourceDef")]
    source: DisplacementEnergySource,
}

#[derive(Serialize, Deserialize)]
struct Energy(#[serde(with = "DisplacementEnergyDef")] DisplacementEnergy);

/// `#[serde(with)]` for a map of displacement energies.
pub(crate) mod displacement_energies {
    use super::*;

    pub(crate) fn serialize<S: Serializer>(
        map: &BTreeMap<String, DisplacementEnergy>,
        s: S,
    ) -> Result<S::Ok, S::Error> {
        s.collect_map(map.iter().map(|(k, v)| (k, Energy(*v))))
    }

    pub(crate) fn deserialize<'de, D: Deserializer<'de>>(
        d: D,
    ) -> Result<BTreeMap<String, DisplacementEnergy>, D::Error> {
        let map = BTreeMap::<String, Energy>::deserialize(d)?;
        Ok(map.into_iter().map(|(k, Energy(v))| (k, v)).collect())
    }
}

/// `#[serde(with)]` for a map whose keys are not strings (a tuple, say),
/// which JSON cannot key an object by: written as a list of `[key, value]`.
pub(crate) mod pairs {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub(crate) fn serialize<M, K, V, S>(map: &M, s: S) -> Result<S::Ok, S::Error>
    where
        for<'a> &'a M: IntoIterator<Item = (&'a K, &'a V)>,
        K: Serialize,
        V: Serialize,
        S: Serializer,
    {
        s.collect_seq(map)
    }

    pub(crate) fn deserialize<'de, M, K, V, D>(d: D) -> Result<M, D::Error>
    where
        M: FromIterator<(K, V)>,
        K: Deserialize<'de>,
        V: Deserialize<'de>,
        D: Deserializer<'de>,
    {
        Ok(Vec::<(K, V)>::deserialize(d)?.into_iter().collect())
    }
}

/// The paths of the floats in `value` that JSON cannot hold: NaN and the
/// infinities, which it writes as `null` and so reads back as something else.
pub(crate) fn non_finite<T: Serialize + ?Sized>(value: &T) -> Vec<String> {
    let mut probe = probe::Probe::default();
    // The probe writes nothing and its only error is serde's own.
    let _ = value.serialize(&mut probe);
    probe.found
}

mod probe {
    use serde::ser::{self, Serialize};

    /// A serializer that writes nothing and records the path to every
    /// non-finite float.
    #[derive(Default)]
    pub(super) struct Probe {
        path: Vec<String>,
        pub(super) found: Vec<String>,
    }

    #[derive(Debug)]
    pub(super) struct Found;

    impl std::fmt::Display for Found {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("non-finite float")
        }
    }
    impl std::error::Error for Found {}
    impl ser::Error for Found {
        fn custom<T: std::fmt::Display>(_: T) -> Self {
            Found
        }
    }

    impl Probe {
        fn at<T: Serialize + ?Sized>(&mut self, key: String, value: &T) -> Result<(), Found> {
            self.path.push(key);
            value.serialize(&mut *self)?;
            self.path.pop();
            Ok(())
        }
    }

    /// Indexes sequence elements, so the path says which.
    pub(super) struct Seq<'a> {
        probe: &'a mut Probe,
        index: usize,
    }

    impl Seq<'_> {
        fn next<T: Serialize + ?Sized>(&mut self, value: &T) -> Result<(), Found> {
            let key = self.index.to_string();
            self.index += 1;
            self.probe.at(key, value)
        }
    }

    macro_rules! ok {
        ($($name:ident($($t:ty),*)),* $(,)?) => {
            $(fn $name(self, $(_: $t),*) -> Result<(), Found> { Ok(()) })*
        };
    }

    impl<'a> ser::Serializer for &'a mut Probe {
        type Ok = ();
        type Error = Found;
        type SerializeSeq = Seq<'a>;
        type SerializeTuple = Seq<'a>;
        type SerializeTupleStruct = Seq<'a>;
        type SerializeTupleVariant = Seq<'a>;
        type SerializeMap = Seq<'a>;
        type SerializeStruct = Seq<'a>;
        type SerializeStructVariant = Seq<'a>;

        ok!(
            serialize_bool(bool),
            serialize_i8(i8),
            serialize_i16(i16),
            serialize_i32(i32),
            serialize_i64(i64),
            serialize_u8(u8),
            serialize_u16(u16),
            serialize_u32(u32),
            serialize_u64(u64),
            serialize_char(char),
            serialize_str(&str),
            serialize_bytes(&[u8]),
            serialize_none(),
            serialize_unit(),
            serialize_unit_struct(&'static str),
            serialize_unit_variant(&'static str, u32, &'static str),
        );

        fn serialize_f32(self, v: f32) -> Result<(), Found> {
            self.serialize_f64(f64::from(v))
        }
        fn serialize_f64(self, v: f64) -> Result<(), Found> {
            if !v.is_finite() {
                self.found.push(self.path.join("."));
            }
            Ok(())
        }
        fn serialize_some<T: Serialize + ?Sized>(self, v: &T) -> Result<(), Found> {
            v.serialize(self)
        }
        fn serialize_newtype_struct<T: Serialize + ?Sized>(
            self,
            _: &'static str,
            v: &T,
        ) -> Result<(), Found> {
            v.serialize(self)
        }
        fn serialize_newtype_variant<T: Serialize + ?Sized>(
            self,
            _: &'static str,
            _: u32,
            variant: &'static str,
            v: &T,
        ) -> Result<(), Found> {
            self.at(variant.to_string(), v)
        }
        fn serialize_seq(self, _: Option<usize>) -> Result<Seq<'a>, Found> {
            Ok(Seq {
                probe: self,
                index: 0,
            })
        }
        fn serialize_tuple(self, _: usize) -> Result<Seq<'a>, Found> {
            Ok(Seq {
                probe: self,
                index: 0,
            })
        }
        fn serialize_tuple_struct(self, _: &'static str, _: usize) -> Result<Seq<'a>, Found> {
            Ok(Seq {
                probe: self,
                index: 0,
            })
        }
        fn serialize_tuple_variant(
            self,
            _: &'static str,
            _: u32,
            _: &'static str,
            _: usize,
        ) -> Result<Seq<'a>, Found> {
            Ok(Seq {
                probe: self,
                index: 0,
            })
        }
        fn serialize_map(self, _: Option<usize>) -> Result<Seq<'a>, Found> {
            Ok(Seq {
                probe: self,
                index: 0,
            })
        }
        fn serialize_struct(self, _: &'static str, _: usize) -> Result<Seq<'a>, Found> {
            Ok(Seq {
                probe: self,
                index: 0,
            })
        }
        fn serialize_struct_variant(
            self,
            _: &'static str,
            _: u32,
            _: &'static str,
            _: usize,
        ) -> Result<Seq<'a>, Found> {
            Ok(Seq {
                probe: self,
                index: 0,
            })
        }
    }

    macro_rules! seq {
        ($($tr:ident :: $m:ident),*) => {$(
            impl ser::$tr for Seq<'_> {
                type Ok = ();
                type Error = Found;
                fn $m<T: Serialize + ?Sized>(&mut self, v: &T) -> Result<(), Found> {
                    self.next(v)
                }
                fn end(self) -> Result<(), Found> {
                    Ok(())
                }
            }
        )*};
    }
    seq!(
        SerializeSeq::serialize_element,
        SerializeTuple::serialize_element,
        SerializeTupleStruct::serialize_field,
        SerializeTupleVariant::serialize_field
    );

    impl ser::SerializeMap for Seq<'_> {
        type Ok = ();
        type Error = Found;
        fn serialize_key<T: Serialize + ?Sized>(&mut self, k: &T) -> Result<(), Found> {
            // Keys are named in the path by their position; a float key is
            // checked like any other float.
            k.serialize(&mut *self.probe)
        }
        fn serialize_value<T: Serialize + ?Sized>(&mut self, v: &T) -> Result<(), Found> {
            self.next(v)
        }
        fn end(self) -> Result<(), Found> {
            Ok(())
        }
    }

    macro_rules! fields {
        ($($tr:ident),*) => {$(
            impl ser::$tr for Seq<'_> {
                type Ok = ();
                type Error = Found;
                fn serialize_field<T: Serialize + ?Sized>(&mut self, key: &'static str, v: &T) -> Result<(), Found> {
                    self.probe.at(key.to_string(), v)
                }
                fn end(self) -> Result<(), Found> {
                    Ok(())
                }
            }
        )*};
    }
    fields!(SerializeStruct, SerializeStructVariant);
}

/// An `f64` JSON can hold whatever its value: a number when finite, else
/// `"inf"`, `"-inf"` or `"NaN"`.
#[derive(Clone, Copy)]
struct Float(f64);

impl Serialize for Float {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match self.0 {
            v if v.is_finite() => s.serialize_f64(v),
            v if v.is_nan() => s.serialize_str("NaN"),
            v if v > 0.0 => s.serialize_str("inf"),
            _ => s.serialize_str("-inf"),
        }
    }
}

impl<'de> Deserialize<'de> for Float {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Repr {
            Number(f64),
            Text(String),
        }
        match Repr::deserialize(d)? {
            Repr::Number(v) => Ok(Float(v)),
            Repr::Text(t) => match t.as_str() {
                "NaN" => Ok(Float(f64::NAN)),
                "inf" => Ok(Float(f64::INFINITY)),
                "-inf" => Ok(Float(f64::NEG_INFINITY)),
                _ => Err(serde::de::Error::custom(format!("{t:?} is not a number"))),
            },
        }
    }
}

/// `#[serde(with)]` for an `Option<(f64, f64)>` that may hold an infinity.
pub(crate) mod non_finite_pair {
    use super::*;

    pub(crate) fn serialize<S: Serializer>(
        v: &Option<(f64, f64)>,
        s: S,
    ) -> Result<S::Ok, S::Error> {
        v.map(|(a, b)| (Float(a), Float(b))).serialize(s)
    }

    pub(crate) fn deserialize<'de, D: Deserializer<'de>>(
        d: D,
    ) -> Result<Option<(f64, f64)>, D::Error> {
        Ok(Option::<(Float, Float)>::deserialize(d)?.map(|(a, b)| (a.0, b.0)))
    }
}
