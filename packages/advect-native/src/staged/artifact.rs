//! Direct Python entry point for runtime-owned graph artifacts.

use advect_runtime::GraphStore as RuntimeGraphStore;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::staged::GraphStore;

/// Parse and transactionally validate canonical graph JSON.
#[pyfunction]
pub(crate) fn deserialize_graph_json(encoded: &str) -> PyResult<GraphStore> {
    RuntimeGraphStore::from_json(encoded)
        .map(GraphStore::from_runtime)
        .map_err(|error| PyValueError::new_err(error.into_message()))
}
