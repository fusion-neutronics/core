use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use yamc::track::{ParticleTrack, ParticleType, TrackEvent, TrackEventType, TrackStorage};

fn particle_name(p: ParticleType) -> &'static str {
    match p {
        ParticleType::Neutron => "neutron",
        ParticleType::Photon => "photon",
    }
}

/// One event as a flat dict row (the `TrackEvent.to_dict()` layout).
fn event_dict<'py>(py: Python<'py>, e: &TrackEvent) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("particle_id", e.particle_id)?;
    d.set_item("parent_id", e.parent_id)?;
    d.set_item("generation", e.generation)?;
    d.set_item("particle_type", particle_name(e.particle_type))?;
    d.set_item("batch", e.batch)?;
    d.set_item("history", e.history)?;
    d.set_item("event_type", e.event_type.to_string())?;
    d.set_item("x", e.position[0])?;
    d.set_item("y", e.position[1])?;
    d.set_item("z", e.position[2])?;
    d.set_item("u", e.direction[0])?;
    d.set_item("v", e.direction[1])?;
    d.set_item("w", e.direction[2])?;
    d.set_item("energy_in", e.energy_in)?;
    d.set_item("energy_out", e.energy_out)?;
    d.set_item("weight", e.weight)?;
    d.set_item("cell_id", e.cell_id)?;
    d.set_item("reaction_mt", e.reaction_mt)?;
    d.set_item("nuclide", e.nuclide.clone())?;
    d.set_item("birth_reaction", e.birth_reaction.clone())?;
    d.set_item("distribution", e.distribution.clone())?;
    d.set_item("energy_dist", e.energy_dist.clone())?;
    Ok(d)
}

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

    /// Particle species, "neutron" or "photon".
    #[getter]
    fn particle_type(&self) -> &'static str {
        particle_name(self.inner.particle_type)
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
    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        event_dict(py, &self.inner)
    }
}

impl From<TrackEvent> for PyTrackEvent {
    fn from(event: TrackEvent) -> Self {
        PyTrackEvent { inner: event }
    }
}

/// One particle's track: the ordered list of `TrackEvent` steps it went
/// through from its birth, with its particle type ("neutron" or "photon"),
/// history, particle ID (unique within the history; 0 is the source
/// particle), parent ID, and generation. Access
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

    /// Particle species, "neutron" or "photon".
    #[getter]
    fn particle_type(&self) -> &'static str {
        particle_name(self.inner.particle_type)
    }

    /// Global index of the source history this particle belongs to.
    #[getter]
    fn history(&self) -> usize {
        self.inner.history
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
            "ParticleTrack(history={}, particle_id={}, particle_type={}, generation={}, events={})",
            self.inner.history,
            self.inner.particle_id,
            particle_name(self.inner.particle_type),
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

/// All particle tracks recorded by a simulation, one per transported
/// particle (source and secondaries alike), ordered by history then particle
/// ID. Access them via `.tracks`, the total step count via `total_events()`,
/// `to_dataframe_records()` to flatten every event into dict rows for a
/// pandas DataFrame, and `to_vtkhdf()` to view them in ParaView.
/// `len(tracks)` is the number of particle tracks.
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
            "Tracks(particles={}, total_events={})",
            self.inner.tracks.len(),
            self.inner.total_events()
        )
    }

    /// Convert all events to a flat list of dicts for DataFrame creation
    ///
    /// Examples:
    ///     df = pd.DataFrame(tracks.to_dataframe_records())
    ///     photons = df[df.particle_type == "photon"]
    fn to_dataframe_records<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let list = PyList::empty(py);
        for e in self.inner.tracks.iter().flat_map(|t| &t.events) {
            list.append(event_dict(py, e)?)?;
        }
        Ok(list)
    }

    /// Flatten the tracks into polylines, one per particle track.
    ///
    /// Returns a dict of flat lists: ``points`` (``[x, y, z]`` per event, cm)
    /// and ``offsets`` (start of each line in ``points`` plus the total, so
    /// line ``j`` is ``points[offsets[j]:offsets[j + 1]]``); per point
    /// ``energy`` (eV, after the event), ``weight``, ``event_type`` and
    /// ``cell_id`` (-1 if unknown); per line ``particle_type``,
    /// ``particle_id``, ``parent_id`` (-1 for source particles),
    /// ``generation`` and ``history``. ``event_type`` and ``particle_type``
    /// are integer codes indexing ``event_type_names`` and
    /// ``particle_type_names``.
    fn polylines<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let p = self.inner.polylines();
        let d = PyDict::new(py);
        d.set_item("points", p.points)?;
        d.set_item("offsets", p.offsets)?;
        d.set_item("energy", p.energy)?;
        d.set_item("weight", p.weight)?;
        d.set_item("event_type", p.event_type)?;
        d.set_item("cell_id", p.cell_id)?;
        d.set_item("particle_type", p.particle_type)?;
        d.set_item("particle_id", p.particle_id)?;
        d.set_item("parent_id", p.parent_id)?;
        d.set_item("generation", p.generation)?;
        d.set_item("history", p.history)?;
        d.set_item(
            "event_type_names",
            TrackEventType::ALL.map(|t| t.to_string()),
        )?;
        d.set_item("particle_type_names", ["neutron", "photon"])?;
        Ok(d)
    }
}

impl From<TrackStorage> for PyTracks {
    fn from(storage: TrackStorage) -> Self {
        PyTracks { inner: storage }
    }
}
