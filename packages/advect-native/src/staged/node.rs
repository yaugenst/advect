//! Python-facing conversion and snapshots for runtime-owned node metadata.

use advect_runtime::{NodeId, NodeRecord};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::staged::conversion::attr::attr_map_to_python;
use crate::staged::conversion::dtype::dtype_to_python;

/// Immutable Python-facing snapshot of one canonical graph node.
#[derive(Debug)]
#[pyclass(module = "advect._native_core", frozen)]
pub(crate) struct GraphNode(pub(crate) NodeRecord);

#[pymethods]
impl GraphNode {
    #[getter]
    fn id(&self) -> NodeId {
        self.0.id
    }

    #[getter]
    fn op(&self) -> &str {
        &self.0.op
    }

    #[getter]
    fn schema_version(&self) -> u32 {
        self.0.schema_version
    }

    #[getter]
    fn inputs(&self) -> Vec<NodeId> {
        self.0.inputs.clone()
    }

    #[getter]
    fn attrs(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        attr_map_to_python(py, self.0.metadata.attrs())
    }

    #[getter]
    fn shape(&self) -> Vec<usize> {
        self.0.metadata.shape().to_vec()
    }

    #[getter]
    fn dtype(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        dtype_to_python(self.0.metadata.dtype(), py)
    }

    #[getter]
    fn name(&self) -> Option<String> {
        self.0.metadata.name().map(str::to_owned)
    }

    #[getter]
    fn num_outputs(&self) -> usize {
        self.0.metadata.num_outputs()
    }

    #[getter]
    fn output_shapes(&self) -> Option<Vec<Vec<usize>>> {
        self.0.metadata.output_shapes()
    }

    #[getter]
    fn output_dtypes(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let Some(dtypes) = self.0.metadata.output_dtypes() else {
            return Ok(py.None());
        };
        let values = dtypes
            .iter()
            .map(|dtype| dtype_to_python(dtype, py))
            .collect::<PyResult<Vec<_>>>()?;
        Ok(PyList::new(py, values)?.into_any().unbind())
    }

    #[getter]
    fn source_location(&self) -> Option<String> {
        self.0.metadata.source_location().map(str::to_owned)
    }
}
