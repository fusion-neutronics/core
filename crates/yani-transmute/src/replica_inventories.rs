//! Every replica's per-step inventories, held compactly.
//!
//! An ensemble keeps each replica's density of every nuclide at every step, so
//! that a quantity summed over nuclides (activity, decay heat, a photon
//! spectrum) can be evaluated per replica and keep the correlations between
//! nuclides. Held as one `HashMap<String, f64>` per replica and step, each
//! entry cost a name allocated afresh and a hash slot, about 85 bytes, for an
//! 8-byte density: roughly 0.5 GB per material at default convergence.
//!
//! Here each name is held once per store, and a row (one replica at one step)
//! is its densities in a flat array beside the indices of the nuclides they
//! belong to. Those indices are the same from one replica to the next at a
//! step, nearly always, and then the row shares the previous replica's rather
//! than holding its own, so a row costs its densities.
//!
//! A nuclide absent from a row reads as zero, as it did from a map. The
//! serialized form is the map layout it replaces, so a result written before
//! reads unchanged.

use std::collections::HashMap;
use std::sync::Arc;

/// The layout a store is serialized as: `[replica][step][nuclide]`.
type Maps = Vec<Vec<HashMap<String, f64>>>;

/// `[replica][step][nuclide]` densities.
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
#[serde(from = "Maps", into = "Maps")]
pub(crate) struct Inventories {
    /// Each nuclide name once, by index.
    names: Vec<Arc<str>>,
    /// Name to index.
    index: HashMap<Arc<str>, u32>,
    /// `[replica][step]`.
    rows: Vec<Vec<Row>>,
}

/// One replica's densities at one step.
#[derive(Clone, Debug)]
struct Row {
    /// The nuclides held, as name indices in ascending order; shared with
    /// every other row that holds the same ones.
    keys: Arc<[u32]>,
    /// Their densities, beside `keys`.
    values: Box<[f64]>,
}

impl Row {
    fn get(&self, key: u32) -> Option<f64> {
        self.keys
            .binary_search(&key)
            .ok()
            .map(|position| self.values[position])
    }
}

impl Inventories {
    /// How many replicas are held.
    pub(crate) fn len(&self) -> usize {
        self.rows.len()
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    fn intern(&mut self, name: &str) -> u32 {
        if let Some(&key) = self.index.get(name) {
            return key;
        }
        let key = u32::try_from(self.names.len()).expect("fewer than 2^32 nuclide names");
        let name: Arc<str> = name.into();
        self.names.push(Arc::clone(&name));
        self.index.insert(name, key);
        key
    }

    /// Add one replica's per-step inventories.
    pub(crate) fn push(&mut self, per_step: &[HashMap<String, f64>]) {
        let sorted: Vec<Vec<(u32, f64)>> = per_step
            .iter()
            .map(|densities| {
                let mut pairs: Vec<(u32, f64)> = densities
                    .iter()
                    .map(|(name, &value)| (self.intern(name), value))
                    .collect();
                pairs.sort_unstable_by_key(|&(key, _)| key);
                pairs
            })
            .collect();
        let previous = self.rows.last();
        let mut rows: Vec<Row> = Vec::with_capacity(sorted.len());
        for (step, pairs) in sorted.into_iter().enumerate() {
            let same = |row: &Row| {
                row.keys.len() == pairs.len()
                    && row.keys.iter().zip(&pairs).all(|(k, (key, _))| k == key)
            };
            // The previous replica at this step, else this replica's previous
            // step: where a row's nuclides nearly always repeat.
            let keys = previous
                .and_then(|p| p.get(step))
                .filter(|row| same(row))
                .or_else(|| rows.last().filter(|row| same(row)))
                .map(|row| Arc::clone(&row.keys))
                .unwrap_or_else(|| pairs.iter().map(|&(key, _)| key).collect());
            let values = pairs.iter().map(|&(_, value)| value).collect();
            rows.push(Row { keys, values });
        }
        self.rows.push(rows);
    }

    /// `nuclide`'s density at `step` in every replica, in replica order, onto
    /// the end of `out`: zero where a replica does not hold it or has no such
    /// step.
    pub(crate) fn densities_into(&self, step: usize, nuclide: &str, out: &mut Vec<f64>) {
        let Some(&key) = self.index.get(nuclide) else {
            out.extend(std::iter::repeat_n(0.0, self.rows.len()));
            return;
        };
        out.extend(self.rows.iter().map(|replica| {
            replica
                .get(step)
                .and_then(|row| row.get(key))
                .unwrap_or(0.0)
        }));
    }

    /// `nuclide`'s density at `step` in every replica, in replica order.
    pub(crate) fn densities(&self, step: usize, nuclide: &str) -> Vec<f64> {
        let mut out = Vec::with_capacity(self.rows.len());
        self.densities_into(step, nuclide, &mut out);
        out
    }

    fn map(&self, row: &Row) -> HashMap<String, f64> {
        row.keys
            .iter()
            .zip(row.values.iter())
            .map(|(&key, &value)| (self.names[key as usize].to_string(), value))
            .collect()
    }

    /// Replica `replica`'s inventory at every step it has.
    pub(crate) fn replica(&self, replica: usize) -> Vec<HashMap<String, f64>> {
        self.rows
            .get(replica)
            .into_iter()
            .flatten()
            .map(|row| self.map(row))
            .collect()
    }

    /// Every replica's inventory at `step`, in replica order, leaving out a
    /// replica with no such step.
    pub(crate) fn at_step(&self, step: usize) -> Vec<HashMap<String, f64>> {
        self.rows
            .iter()
            .filter_map(|replica| replica.get(step))
            .map(|row| self.map(row))
            .collect()
    }
}

impl From<Maps> for Inventories {
    fn from(maps: Maps) -> Self {
        let mut out = Inventories::default();
        for replica in &maps {
            out.push(replica);
        }
        out
    }
}

impl From<Inventories> for Maps {
    fn from(store: Inventories) -> Self {
        (0..store.len()).map(|r| store.replica(r)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn maps() -> Maps {
        let row = |pairs: &[(&str, f64)]| -> HashMap<String, f64> {
            pairs.iter().map(|(n, v)| (n.to_string(), *v)).collect()
        };
        vec![
            vec![row(&[("Fe56", 1.0), ("Mn56", 2.0)]), row(&[("Fe56", 0.9)])],
            vec![
                row(&[("Fe56", 1.1), ("Mn56", 2.1)]),
                row(&[("Fe56", 0.8), ("Co60", 0.0)]),
            ],
            // A replica that stopped a step early.
            vec![row(&[("Mn56", 2.2), ("Fe56", 1.2)])],
        ]
    }

    #[test]
    fn every_density_reads_back_and_an_absence_reads_as_zero() {
        let maps = maps();
        let store = Inventories::from(maps.clone());
        assert_eq!(store.len(), 3);
        assert_eq!(store.densities(0, "Mn56"), vec![2.0, 2.1, 2.2]);
        assert_eq!(store.densities(1, "Fe56"), vec![0.9, 0.8, 0.0]);
        assert_eq!(store.densities(1, "Co60"), vec![0.0, 0.0, 0.0]);
        assert_eq!(store.densities(0, "Xe135"), vec![0.0, 0.0, 0.0]);
        for (r, replica) in maps.iter().enumerate() {
            assert_eq!(&store.replica(r), replica);
        }
        assert_eq!(
            store.at_step(1),
            vec![maps[0][1].clone(), maps[1][1].clone()]
        );
        // A density held as zero stays held, as it was in a map.
        assert_eq!(store.at_step(1)[1].get("Co60"), Some(&0.0));
    }

    #[test]
    fn rows_holding_the_same_nuclides_share_their_keys() {
        let store = Inventories::from(maps());
        assert!(Arc::ptr_eq(&store.rows[0][0].keys, &store.rows[1][0].keys));
        assert!(Arc::ptr_eq(&store.rows[1][0].keys, &store.rows[2][0].keys));
        assert!(!Arc::ptr_eq(&store.rows[0][1].keys, &store.rows[1][1].keys));
    }

    #[test]
    fn the_serialized_form_is_the_map_layout() {
        let maps = maps();
        let store = Inventories::from(maps.clone());
        let json = serde_json::to_value(&store).unwrap();
        assert_eq!(json, serde_json::to_value(&maps).unwrap());
        let back: Inventories = serde_json::from_value(json).unwrap();
        for (r, replica) in maps.iter().enumerate() {
            assert_eq!(&back.replica(r), replica);
        }
    }
}
