use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use yamc::track::{ParticleTrack, TrackEvent, TrackStorage};

/// A single step in a particle's recorded track -- e.g. its birth, a
/// collision, or a surface crossing. Exposes the particle and parent IDs,
/// generation, batch and history indices, the event type, position
/// `(x, y, z)` and direction `(u, v, w)`, incoming and outgoing energy,
/// statistical weight, the cell ID, and (for collisions) the reaction MT
/// and nuclide.
#[gen_stub_pyclass]
#[pyclass(module = "yamc._core", name = "TrackEvent", from_py_object)]
#[derive(Clone)]
pub struct PyTrackEvent {
    inner: TrackEvent,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyTrackEvent {
    #[getter]
    fn particle_id(&self) -> u64 {
        self.inner.particle_id
    }

    #[getter]
    fn parent_id(&self) -> Option<u64> {
        self.inner.parent_id
    }

    #[getter]
    fn generation(&self) -> u32 {
        self.inner.generation
    }

    #[getter]
    fn batch(&self) -> usize {
        self.inner.batch
    }

    #[getter]
    fn history(&self) -> usize {
        self.inner.history
    }

    #[getter]
    fn event_type(&self) -> String {
        self.inner.event_type.to_string()
    }

    #[getter]
    fn position(&self) -> [f64; 3] {
        self.inner.position
    }

    #[getter]
    fn x(&self) -> f64 {
        self.inner.position[0]
    }

    #[getter]
    fn y(&self) -> f64 {
        self.inner.position[1]
    }

    #[getter]
    fn z(&self) -> f64 {
        self.inner.position[2]
    }

    #[getter]
    fn direction(&self) -> [f64; 3] {
        self.inner.direction
    }

    #[getter]
    fn energy_in(&self) -> f64 {
        self.inner.energy_in
    }

    #[getter]
    fn energy_out(&self) -> f64 {
        self.inner.energy_out
    }

    #[getter]
    fn weight(&self) -> f64 {
        self.inner.weight
    }

    #[getter]
    fn cell_id(&self) -> Option<u32> {
        self.inner.cell_id
    }

    #[getter]
    fn reaction_mt(&self) -> Option<i32> {
        self.inner.reaction_mt
    }

    #[getter]
    fn nuclide(&self) -> Option<String> {
        self.inner.nuclide.clone()
    }

    #[getter]
    fn birth_reaction(&self) -> Option<String> {
        self.inner.birth_reaction.clone()
    }

    #[getter]
    fn distribution(&self) -> Option<String> {
        self.inner.distribution.clone()
    }

    #[getter]
    fn energy_dist(&self) -> Option<String> {
        self.inner.energy_dist.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "TrackEvent(type={}, particle_id={}, E_in={:.4e}, E_out={:.4e})",
            self.inner.event_type,
            self.inner.particle_id,
            self.inner.energy_in,
            self.inner.energy_out
        )
    }

    /// Convert to dict for DataFrame creation
    fn to_dict(&self, py: Python<'_>) -> PyResult<pyo3::Py<pyo3::types::PyAny>> {
        let dict = PyDict::new(py);
        dict.set_item("particle_id", self.inner.particle_id)?;
        dict.set_item("parent_id", self.inner.parent_id)?;
        dict.set_item("generation", self.inner.generation)?;
        dict.set_item("batch", self.inner.batch)?;
        dict.set_item("history", self.inner.history)?;
        dict.set_item("event_type", self.inner.event_type.to_string())?;
        dict.set_item("x", self.inner.position[0])?;
        dict.set_item("y", self.inner.position[1])?;
        dict.set_item("z", self.inner.position[2])?;
        dict.set_item("u", self.inner.direction[0])?;
        dict.set_item("v", self.inner.direction[1])?;
        dict.set_item("w", self.inner.direction[2])?;
        dict.set_item("energy_in", self.inner.energy_in)?;
        dict.set_item("energy_out", self.inner.energy_out)?;
        dict.set_item("weight", self.inner.weight)?;
        dict.set_item("cell_id", self.inner.cell_id)?;
        dict.set_item("reaction_mt", self.inner.reaction_mt)?;
        dict.set_item("nuclide", self.inner.nuclide.clone())?;
        dict.set_item("birth_reaction", self.inner.birth_reaction.clone())?;
        dict.set_item("distribution", self.inner.distribution.clone())?;
        dict.set_item("energy_dist", self.inner.energy_dist.clone())?;
        Ok(dict.into())
    }
}

impl From<TrackEvent> for PyTrackEvent {
    fn from(event: TrackEvent) -> Self {
        PyTrackEvent { inner: event }
    }
}

/// One particle's full history: the ordered list of `TrackEvent` steps it
/// went through, with its particle ID, parent ID, and generation. Access
/// the steps via `track.events`; `len(track)` is the number of events.
#[gen_stub_pyclass]
#[pyclass(module = "yamc._core", name = "ParticleTrack", from_py_object)]
#[derive(Clone)]
pub struct PyParticleTrack {
    inner: ParticleTrack,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyParticleTrack {
    #[getter]
    fn particle_id(&self) -> u64 {
        self.inner.particle_id
    }

    #[getter]
    fn parent_id(&self) -> Option<u64> {
        self.inner.parent_id
    }

    #[getter]
    fn generation(&self) -> u32 {
        self.inner.generation
    }

    #[getter]
    fn events(&self) -> Vec<PyTrackEvent> {
        self.inner
            .events
            .iter()
            .cloned()
            .map(PyTrackEvent::from)
            .collect()
    }

    fn __len__(&self) -> usize {
        self.inner.events.len()
    }

    fn __repr__(&self) -> String {
        format!(
            "ParticleTrack(particle_id={}, generation={}, events={})",
            self.inner.particle_id,
            self.inner.generation,
            self.inner.events.len()
        )
    }
}

impl From<ParticleTrack> for PyParticleTrack {
    fn from(track: ParticleTrack) -> Self {
        PyParticleTrack { inner: track }
    }
}

/// All particle tracks recorded by a simulation. Access the per-particle
/// tracks via `.tracks`, the total step count via `total_events()`, and
/// `to_dataframe_records()` to flatten every event into dict rows for a
/// pandas DataFrame. `len(tracks)` is the number of histories.
#[gen_stub_pyclass]
#[pyclass(module = "yamc._core", name = "Tracks", from_py_object)]
#[derive(Clone)]
pub struct PyTracks {
    pub inner: TrackStorage,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyTracks {
    /// Get all particle tracks
    #[getter]
    fn tracks(&self) -> Vec<PyParticleTrack> {
        self.inner
            .tracks
            .iter()
            .cloned()
            .map(PyParticleTrack::from)
            .collect()
    }

    /// Total number of events across all tracks
    fn total_events(&self) -> usize {
        self.inner.total_events()
    }

    fn __len__(&self) -> usize {
        self.inner.tracks.len()
    }

    fn __repr__(&self) -> String {
        format!(
            "Tracks(histories={}, total_events={})",
            self.inner.tracks.len(),
            self.inner.total_events()
        )
    }

    /// Convert all events to a flat list of dicts for DataFrame creation
    ///
    /// Examples:
    ///     df = pd.DataFrame(tracks.to_dataframe_records())
    fn to_dataframe_records(&self, py: Python<'_>) -> PyResult<pyo3::Py<pyo3::types::PyAny>> {
        let list = PyList::empty(py);
        for track in &self.inner.tracks {
            for event in &track.events {
                let dict = PyDict::new(py);
                dict.set_item("particle_id", event.particle_id)?;
                dict.set_item("parent_id", event.parent_id)?;
                dict.set_item("generation", event.generation)?;
                dict.set_item("batch", event.batch)?;
                dict.set_item("history", event.history)?;
                dict.set_item("event_type", event.event_type.to_string())?;
                dict.set_item("x", event.position[0])?;
                dict.set_item("y", event.position[1])?;
                dict.set_item("z", event.position[2])?;
                dict.set_item("u", event.direction[0])?;
                dict.set_item("v", event.direction[1])?;
                dict.set_item("w", event.direction[2])?;
                dict.set_item("energy_in", event.energy_in)?;
                dict.set_item("energy_out", event.energy_out)?;
                dict.set_item("weight", event.weight)?;
                dict.set_item("cell_id", event.cell_id)?;
                dict.set_item("reaction_mt", event.reaction_mt)?;
                dict.set_item("nuclide", event.nuclide.clone())?;
                dict.set_item("birth_reaction", event.birth_reaction.clone())?;
                dict.set_item("distribution", event.distribution.clone())?;
                dict.set_item("energy_dist", event.energy_dist.clone())?;
                list.append(dict)?;
            }
        }
        Ok(list.into())
    }
}

impl From<TrackStorage> for PyTracks {
    fn from(storage: TrackStorage) -> Self {
        PyTracks { inner: storage }
    }
}
