//! A contiguous `f64` array that can share its allocation with whoever
//! produced the bytes (issue #476, task 1).
//!
//! Every large numeric array on [`Nuclide`](crate::nuclide::Nuclide),
//! [`Reaction`](crate::reaction::Reaction) and
//! [`FastXSGrid`](crate::nuclide::FastXSGrid) used to be a `Vec<f64>`, which
//! meant the Arrow reader had exactly one way to hand data over: decode the
//! column, then `.iter().copied().collect()` it into a fresh allocation. Every
//! consumer paid for that copy, native and browser alike, and the copy was
//! unavoidable because the field type demanded ownership of a `Vec`.
//!
//! [`F64Buffer`] wraps [`arrow_buffer::ScalarBuffer<f64>`], which is
//! "`Arc<Vec<f64>>` with O(1) slicing", so the same field can now hold either:
//!
//! * a view onto an Arrow column's values buffer ([`F64Buffer::share`]), or
//! * a view onto another `F64Buffer` ([`F64Buffer::slice`] /
//!   [`F64Buffer::tail`]), or
//! * a private allocation ([`F64Buffer::from_slice`], `From<Vec<f64>>`),
//!
//! and consumers cannot tell the difference: the type derefs to `&[f64]`.
//!
//! ## Why a newtype rather than `ScalarBuffer<f64>` directly
//!
//! `ScalarBuffer` implements neither `Default` nor serde's traits, and both are
//! load-bearing here: `FastXSGrid` derives `Default` across two dozen fields,
//! and `Nuclide`/`Reaction` derive `Serialize`/`Deserialize` (the wasm bindings
//! hand reaction data to JS as JSON). Wrapping once puts those impls in
//! one place instead of a `#[serde(with = ...)]` module per field shape.
//!
//! ## Sharing keeps the *whole* parent alive
//!
//! A `ScalarBuffer` view holds an `Arc` on the entire backing allocation, not
//! just the bytes it spans. Sharing a narrow view of a wide Arrow column
//! therefore pins bytes the loader had decided to drop, so it is only a win
//! when the load retains essentially all of the column. That decision belongs
//! to the caller, which is why this type offers both [`share`](Self::share) and
//! [`from_slice`](Self::from_slice) rather than picking for you; see
//! `LoadScope::is_unfiltered` and its use in `nuclide_arrow`.

use std::ops::Deref;

use arrow_buffer::ScalarBuffer;
use serde::de::{SeqAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// A contiguous, immutable `f64` array with O(1) clone and slice.
///
/// Derefs to `&[f64]`, so indexing, iteration, `len()`, `windows()` and
/// subslicing all work as they did when these fields were `Vec<f64>`.
#[derive(Clone)]
pub struct F64Buffer(ScalarBuffer<f64>);

impl F64Buffer {
    /// Share an Arrow column's values buffer: no copy, no allocation.
    ///
    /// The returned buffer keeps the whole parent allocation alive. Prefer
    /// [`from_slice`](Self::from_slice) when the caller is about to drop a much
    /// larger batch and keep only this slice of it.
    pub fn share(values: ScalarBuffer<f64>) -> Self {
        Self(values)
    }

    /// Copy `values` into a private allocation, releasing whatever it came from.
    pub fn from_slice(values: &[f64]) -> Self {
        Self(values.to_vec().into())
    }

    /// An empty buffer. Same as [`Default`], available in const-ish contexts
    /// where naming the type reads better.
    pub fn empty() -> Self {
        Self(Vec::new().into())
    }

    /// `len` elements starting at `offset`, sharing this buffer's allocation.
    ///
    /// # Panics
    /// If `offset + len` exceeds the buffer length.
    pub fn slice(&self, offset: usize, len: usize) -> Self {
        Self(self.0.slice(offset, len))
    }

    /// Everything from `offset` to the end, sharing this buffer's allocation.
    /// Returns an empty buffer when `offset` is past the end, which is what the
    /// threshold-index truncation wants.
    pub fn tail(&self, offset: usize) -> Self {
        if offset >= self.0.len() {
            return Self::empty();
        }
        self.slice(offset, self.0.len() - offset)
    }

    /// The elements as a slice. Equivalent to derefing; spelled out for call
    /// sites where inference needs the help.
    pub fn as_slice(&self) -> &[f64] {
        &self.0
    }

    /// The underlying Arrow buffer, for handing to Arrow writers or to code
    /// that wants to share it onwards.
    pub fn inner(&self) -> &ScalarBuffer<f64> {
        &self.0
    }

    /// Whether `self` and `other` are views onto the same allocation, at any
    /// offset. Compares the allocation base rather than the view pointer, so a
    /// [`tail`](Self::tail) of a buffer still reports as sharing it.
    ///
    /// The only way to observe that sharing happened rather than copying, so
    /// the loader's tests assert on it.
    pub fn shares_with(&self, other: &Self) -> bool {
        self.0.inner().data_ptr() == other.0.inner().data_ptr()
    }

    /// Bytes of backing allocation this view keeps alive, which is the whole
    /// parent and not just the span. Diagnostics only.
    pub fn allocation_bytes(&self) -> usize {
        self.0.inner().capacity()
    }
}

impl Default for F64Buffer {
    fn default() -> Self {
        Self::empty()
    }
}

impl Deref for F64Buffer {
    type Target = [f64];
    fn deref(&self) -> &[f64] {
        &self.0
    }
}

impl AsRef<[f64]> for F64Buffer {
    fn as_ref(&self) -> &[f64] {
        &self.0
    }
}

impl std::fmt::Debug for F64Buffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Print as a slice: `Vec<f64>`'s old output, so debug dumps and
        // insta-style snapshots read the same as before.
        self.0.as_ref().fmt(f)
    }
}

impl From<Vec<f64>> for F64Buffer {
    /// Zero-copy: the `Vec`'s allocation becomes the buffer's.
    fn from(values: Vec<f64>) -> Self {
        Self(values.into())
    }
}

impl From<ScalarBuffer<f64>> for F64Buffer {
    fn from(values: ScalarBuffer<f64>) -> Self {
        Self::share(values)
    }
}

impl From<&[f64]> for F64Buffer {
    fn from(values: &[f64]) -> Self {
        Self::from_slice(values)
    }
}

impl<const N: usize> From<[f64; N]> for F64Buffer {
    fn from(values: [f64; N]) -> Self {
        Self::from_slice(&values)
    }
}

impl From<F64Buffer> for Vec<f64> {
    fn from(buffer: F64Buffer) -> Self {
        buffer.0.into()
    }
}

impl FromIterator<f64> for F64Buffer {
    fn from_iter<I: IntoIterator<Item = f64>>(iter: I) -> Self {
        Self(iter.into_iter().collect())
    }
}

impl<'a> IntoIterator for &'a F64Buffer {
    type Item = &'a f64;
    type IntoIter = std::slice::Iter<'a, f64>;
    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

// Comparison against anything slice-shaped, in both directions, so the
// `assert_eq!(nuclide.energy_grid(..), vec![..])` style in the test suite keeps
// compiling.
impl<S: AsRef<[f64]> + ?Sized> PartialEq<S> for F64Buffer {
    fn eq(&self, other: &S) -> bool {
        self.as_slice() == other.as_ref()
    }
}

impl PartialEq<F64Buffer> for Vec<f64> {
    fn eq(&self, other: &F64Buffer) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl PartialEq<F64Buffer> for [f64] {
    fn eq(&self, other: &F64Buffer) -> bool {
        self == other.as_slice()
    }
}

impl<const N: usize> PartialEq<F64Buffer> for [f64; N] {
    fn eq(&self, other: &F64Buffer) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl Serialize for F64Buffer {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        // Same wire shape a `Vec<f64>` produced, so JSON consumers (the wasm
        // reaction bindings) see no change.
        serializer.collect_seq(self.as_slice())
    }
}

impl<'de> Deserialize<'de> for F64Buffer {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct SeqVisitor;

        impl<'de> Visitor<'de> for SeqVisitor {
            type Value = F64Buffer;

            fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str("a sequence of f64")
            }

            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<F64Buffer, A::Error> {
                let mut values = Vec::with_capacity(seq.size_hint().unwrap_or(0));
                while let Some(value) = seq.next_element()? {
                    values.push(value);
                }
                Ok(F64Buffer::from(values))
            }
        }

        deserializer.deserialize_seq(SeqVisitor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_and_empty_are_empty() {
        assert!(F64Buffer::default().is_empty());
        assert_eq!(F64Buffer::empty().len(), 0);
    }

    #[test]
    fn derefs_to_a_slice() {
        let buffer = F64Buffer::from(vec![1.0, 2.0, 3.0]);
        assert_eq!(buffer.len(), 3);
        assert_eq!(buffer[1], 2.0);
        assert_eq!(buffer.iter().sum::<f64>(), 6.0);
        assert_eq!(&buffer[1..], &[2.0, 3.0]);
    }

    #[test]
    fn slicing_shares_the_allocation() {
        let buffer = F64Buffer::from(vec![1.0, 2.0, 3.0, 4.0]);
        let tail = buffer.tail(2);
        assert_eq!(tail, [3.0, 4.0]);
        assert!(tail.shares_with(&buffer));
        assert!(buffer.clone().shares_with(&buffer));
    }

    #[test]
    fn copying_does_not_share() {
        let buffer = F64Buffer::from(vec![1.0, 2.0, 3.0]);
        let copy = F64Buffer::from_slice(&buffer);
        assert_eq!(copy, buffer);
        assert!(!copy.shares_with(&buffer));
    }

    #[test]
    fn tail_past_the_end_is_empty() {
        let buffer = F64Buffer::from(vec![1.0, 2.0]);
        assert!(buffer.tail(2).is_empty());
        assert!(buffer.tail(9).is_empty());
        assert!(F64Buffer::empty().tail(0).is_empty());
    }

    #[test]
    fn compares_against_slices_both_ways() {
        let buffer = F64Buffer::from(vec![1.0, 2.0]);
        let owned = vec![1.0, 2.0];
        assert_eq!(buffer, owned);
        assert_eq!(owned, buffer);
        assert_eq!(buffer, [1.0, 2.0]);
        assert_eq!([1.0, 2.0], buffer);
        assert_ne!(buffer, vec![1.0]);
    }

    #[test]
    fn collects_from_an_iterator() {
        let buffer: F64Buffer = (0..4).map(|i| i as f64).collect();
        assert_eq!(buffer, [0.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn serialises_as_a_bare_sequence() {
        let buffer = F64Buffer::from(vec![1.5, 2.5]);
        let json = serde_json::to_string(&buffer).unwrap();
        assert_eq!(json, "[1.5,2.5]");
        let round_tripped: F64Buffer = serde_json::from_str(&json).unwrap();
        assert_eq!(round_tripped, buffer);
    }

    #[test]
    fn a_view_reports_the_whole_parent_allocation() {
        // The property that makes `share` a caller's decision rather than the
        // buffer's: a two-element view of a 1024-element column keeps all 1024
        // alive.
        let parent = F64Buffer::from(vec![0.0; 1024]);
        let view = parent.slice(0, 2);
        assert_eq!(view.len(), 2);
        assert_eq!(view.allocation_bytes(), 1024 * std::mem::size_of::<f64>());
    }
}
