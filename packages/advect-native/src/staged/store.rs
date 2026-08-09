//! Thin Python wrapper over the immutable `advect-runtime` graph store.

use std::sync::Arc;

use advect_runtime::{GraphArtifact, GraphStore as RuntimeGraphStore, NodeId};
use pyo3::exceptions::{PyKeyError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use crate::staged::GraphNode;

type ConstantParts = (String, String, Vec<usize>, Py<PyBytes>, String);

/// Python handle for one runtime-owned immutable graph.
#[derive(Debug)]
#[pyclass(module = "advect._native_core")]
pub(crate) struct GraphStore {
    inner: Arc<RuntimeGraphStore>,
}

impl GraphStore {
    pub(crate) fn from_runtime(store: RuntimeGraphStore) -> Self {
        Self {
            inner: Arc::new(store),
        }
    }

    pub(crate) fn inner_arc(&self) -> Arc<RuntimeGraphStore> {
        Arc::clone(&self.inner)
    }
}

#[pymethods]
impl GraphStore {
    fn __repr__(&self) -> String {
        format!(
            "GraphStore(nodes={}, inputs={:?}, outputs={:?}, array_api={:?})",
            self.inner.node_count(),
            self.inner.inputs(),
            self.inner.outputs(),
            self.inner.required_array_api_version(),
        )
    }

    fn get_node(&self, node_id: NodeId) -> PyResult<GraphNode> {
        self.inner
            .get_node(node_id)
            .map(GraphNode::from_record)
            .map_err(|error| PyKeyError::new_err(error.to_string()))
    }

    fn node_ids(&self) -> Vec<NodeId> {
        self.inner.topological_order()
    }

    #[getter]
    fn node_count(&self) -> usize {
        self.inner.node_count()
    }

    #[getter]
    fn inputs(&self) -> Vec<NodeId> {
        self.inner.inputs().to_vec()
    }

    #[getter]
    fn outputs(&self) -> Vec<NodeId> {
        self.inner.outputs().to_vec()
    }

    #[getter]
    fn required_array_api_version(&self) -> &str {
        self.inner.required_array_api_version()
    }

    /// Canonical portable graph JSON owned by `advect-runtime`.
    fn _to_json(&self) -> PyResult<String> {
        GraphArtifact::store_to_json(&self.inner)
            .map_err(|error| PyValueError::new_err(error.into_message()))
    }

    fn _constant_parts(&self, py: Python<'_>, node_id: NodeId) -> PyResult<ConstantParts> {
        let constant = self.inner.constants().get(&node_id).ok_or_else(|| {
            PyKeyError::new_err(format!("constant node %{node_id} has no payload"))
        })?;
        Ok((
            constant.kind().name().to_owned(),
            constant.dtype().name().to_owned(),
            constant.shape().to_vec(),
            PyBytes::new(py, constant.data()).unbind(),
            constant.digest().to_owned(),
        ))
    }

    fn constant_ids(&self) -> Vec<NodeId> {
        self.inner.constants().keys().copied().collect()
    }
}
