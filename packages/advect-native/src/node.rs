//! Python-facing conversion and snapshots for runtime-owned node metadata.

#[path = "attr_value.rs"]
pub(crate) mod attr_value;
#[path = "dtype.rs"]
pub(crate) mod dtype;

pub(crate) use advect_runtime::NodeMetadata;
use advect_runtime::{GraphError, NodeId};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use self::attr_value::{AttrMapCache, attr_map_to_python};
use self::dtype::{DTypeCache, dtype_to_python};
#[derive(Debug)]
pub(crate) struct PreparedNode {
    op: String,
    inputs: Vec<NodeId>,
    metadata: NodeMetadata,
}

#[derive(Debug)]
pub(crate) struct NodeSpec<'py> {
    pub(crate) op: String,
    pub(crate) inputs: Vec<NodeId>,
    pub(crate) attrs: &'py Bound<'py, PyDict>,
    pub(crate) shape: Vec<usize>,
    pub(crate) dtype: &'py Bound<'py, PyAny>,
    pub(crate) name: Option<String>,
    pub(crate) num_outputs: usize,
    pub(crate) output_shapes: Option<Vec<Vec<usize>>>,
    pub(crate) output_dtypes: Option<&'py Bound<'py, PyAny>>,
    pub(crate) source_location: Option<String>,
}

impl PreparedNode {
    pub(crate) fn from_spec(
        py: Python<'_>,
        spec: NodeSpec<'_>,
        dtype_cache: &mut DTypeCache,
        attr_cache: &mut AttrMapCache,
    ) -> PyResult<Self> {
        if spec.op.is_empty() {
            return Err(PyValueError::new_err("node op must not be empty"));
        }
        let attrs = attr_cache.resolve(py, spec.attrs)?;
        let dtype = dtype_cache.resolve(py, spec.dtype)?;
        let output_dtypes = spec
            .output_dtypes
            .map(|values| dtype_cache.resolve_sequence(py, values))
            .transpose()?;
        let metadata = NodeMetadata::new(
            attrs,
            spec.shape,
            dtype,
            spec.name,
            spec.num_outputs,
            spec.output_shapes,
            output_dtypes,
            spec.source_location,
        )
        .map_err(|error| metadata_error(&spec.op, &error))?;
        Ok(Self {
            op: spec.op,
            inputs: spec.inputs,
            metadata,
        })
    }

    pub(crate) fn into_parts(self) -> (String, Vec<NodeId>, NodeMetadata) {
        (self.op, self.inputs, self.metadata)
    }
}

fn metadata_error(op: &str, error: &GraphError) -> PyErr {
    let message = error.message();
    if message == "node num_outputs must be at least 1" {
        return PyValueError::new_err(format!("Op '{op}' must have num_outputs >= 1 (got 0)"));
    }
    if message == "single-output node must not declare output_shapes/output_dtypes" {
        return PyValueError::new_err(format!(
            "Op '{op}' is single-output; output_shapes/output_dtypes must be None"
        ));
    }
    if message == "multi-output node is missing output_shapes/output_dtypes" {
        return PyValueError::new_err(format!(
            "Op '{op}' has multiple outputs but output_shapes/output_dtypes are missing"
        ));
    }
    PyValueError::new_err(format!("Op '{op}' has invalid metadata: {message}"))
}

/// Immutable Python-facing snapshot of one canonical graph node.
#[derive(Debug)]
#[pyclass(module = "advect._native_core", frozen)]
pub(crate) struct GraphNode {
    id: NodeId,
    op: String,
    schema_version: u32,
    inputs: Vec<NodeId>,
    metadata: NodeMetadata,
}

impl GraphNode {
    pub(crate) fn from_record(record: advect_runtime::NodeRecord) -> Self {
        Self {
            id: record.id,
            op: record.op,
            schema_version: record.schema_version,
            inputs: record.inputs,
            metadata: record.metadata,
        }
    }
}

#[pymethods]
impl GraphNode {
    #[getter]
    fn id(&self) -> NodeId {
        self.id
    }

    #[getter]
    fn op(&self) -> &str {
        &self.op
    }

    #[getter]
    fn schema_version(&self) -> u32 {
        self.schema_version
    }

    #[getter]
    fn inputs(&self) -> Vec<NodeId> {
        self.inputs.clone()
    }

    #[getter]
    fn attrs(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        attr_map_to_python(py, self.metadata.attrs())
    }

    #[getter]
    fn shape(&self) -> Vec<usize> {
        self.metadata.shape().to_vec()
    }

    #[getter]
    fn dtype(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        dtype_to_python(self.metadata.dtype(), py)
    }

    #[getter]
    fn name(&self) -> Option<String> {
        self.metadata.name().map(str::to_owned)
    }

    #[getter]
    fn num_outputs(&self) -> usize {
        self.metadata.num_outputs()
    }

    #[getter]
    fn output_shapes(&self) -> Option<Vec<Vec<usize>>> {
        self.metadata.output_shapes()
    }

    #[getter]
    fn output_dtypes(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let Some(dtypes) = self.metadata.output_dtypes() else {
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
        self.metadata.source_location().map(str::to_owned)
    }
}
