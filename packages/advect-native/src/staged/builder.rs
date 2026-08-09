//! Thin Python wrapper over the runtime graph builder.

use advect_runtime::{
    ConstantKind, GRAPH_FORMAT_VERSION, GraphBuilder as RuntimeGraphBuilder,
    LATEST_ARRAY_API_VERSION, NodeFlags, NodeId, NumericDType, OptimizationReport,
    PortableConstant, optimize,
};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};

use crate::staged::GraphStore;
use crate::staged::conversion::attr::AttrMapCache;
use crate::staged::conversion::dtype::DTypeCache;
use crate::staged::node::{NodeSpec, PreparedNode};

/// Python construction handle around one runtime-owned builder.
#[derive(Debug)]
#[pyclass(module = "advect._native_core")]
pub(crate) struct GraphBuilder {
    inner: Option<RuntimeGraphBuilder>,
    dtype_cache: DTypeCache,
    attr_cache: AttrMapCache,
}

impl GraphBuilder {
    fn require_inner_mut(&mut self) -> PyResult<&mut RuntimeGraphBuilder> {
        self.inner
            .as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("GraphBuilder has already finished"))
    }

    fn prepare_node(&mut self, py: Python<'_>, spec: NodeSpec<'_>) -> PyResult<PreparedNode> {
        PreparedNode::from_spec(py, spec, &mut self.dtype_cache, &mut self.attr_cache)
    }

    fn append_prepared(
        &mut self,
        prepared: PreparedNode,
        schema_version: u32,
        flags: NodeFlags,
    ) -> PyResult<NodeId> {
        let (op, inputs, metadata) = prepared.into_parts();
        self.require_inner_mut()?
            .append_operation(&op, schema_version, &inputs, flags, metadata)
            .map_err(graph_error)
    }
}

#[pymethods]
impl GraphBuilder {
    #[new]
    #[pyo3(signature = (
        version = GRAPH_FORMAT_VERSION,
        *,
        required_array_api_version = LATEST_ARRAY_API_VERSION
    ))]
    fn new(version: &str, required_array_api_version: &str) -> PyResult<Self> {
        Ok(Self {
            inner: Some(
                RuntimeGraphBuilder::new_for_array_api(version, required_array_api_version)
                    .map_err(graph_error)?,
            ),
            dtype_cache: DTypeCache::default(),
            attr_cache: AttrMapCache::default(),
        })
    }

    #[pyo3(signature = (
        op, inputs, attrs, shape, dtype, *, schema_version=1, name=None, num_outputs=1,
        output_shapes=None, output_dtypes=None, source_location=None
    ))]
    #[allow(clippy::needless_pass_by_value)]
    fn append_node(
        &mut self,
        py: Python<'_>,
        op: String,
        inputs: Vec<NodeId>,
        attrs: &Bound<'_, PyDict>,
        shape: Vec<usize>,
        dtype: &Bound<'_, PyAny>,
        schema_version: u32,
        name: Option<String>,
        num_outputs: usize,
        output_shapes: Option<Vec<Vec<usize>>>,
        output_dtypes: Option<&Bound<'_, PyAny>>,
        source_location: Option<String>,
    ) -> PyResult<NodeId> {
        let prepared = self.prepare_node(
            py,
            NodeSpec {
                op,
                inputs,
                attrs,
                shape,
                dtype,
                name,
                num_outputs,
                output_shapes,
                output_dtypes,
                source_location,
            },
        )?;
        self.append_prepared(prepared, schema_version, NodeFlags::NONE)
    }

    #[pyo3(signature = (shape, dtype, *, name=None))]
    fn append_input_node(
        &mut self,
        py: Python<'_>,
        shape: Vec<usize>,
        dtype: &Bound<'_, PyAny>,
        name: Option<String>,
    ) -> PyResult<NodeId> {
        let attrs = PyDict::new(py);
        let prepared = self.prepare_node(
            py,
            NodeSpec {
                op: "advect.input".to_owned(),
                inputs: Vec::new(),
                attrs: &attrs,
                shape,
                dtype,
                name,
                num_outputs: 1,
                output_shapes: None,
                output_dtypes: None,
                source_location: None,
            },
        )?;
        let (_, _, metadata) = prepared.into_parts();
        self.require_inner_mut()?
            .append_input(metadata)
            .map_err(graph_error)
    }

    #[pyo3(signature = (data, shape, dtype, *, kind))]
    fn append_constant(
        &mut self,
        py: Python<'_>,
        data: &Bound<'_, PyBytes>,
        shape: Vec<usize>,
        dtype: &Bound<'_, PyAny>,
        kind: &str,
    ) -> PyResult<(NodeId, String)> {
        let kind = kind
            .parse::<ConstantKind>()
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        let dtype_name = self.dtype_cache.resolve(py, dtype)?.name().to_owned();
        let numeric_dtype = dtype_name
            .parse::<NumericDType>()
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        let constant =
            PortableConstant::new(kind, numeric_dtype, shape.clone(), data.as_bytes().to_vec())
                .map_err(|error| PyValueError::new_err(error.into_message()))?;
        let digest = constant.digest().to_owned();
        let attrs = PyDict::new(py);
        let prepared = self.prepare_node(
            py,
            NodeSpec {
                op: "advect.const".to_owned(),
                inputs: Vec::new(),
                attrs: &attrs,
                shape,
                dtype,
                name: None,
                num_outputs: 1,
                output_shapes: None,
                output_dtypes: None,
                source_location: None,
            },
        )?;
        let (_, _, metadata) = prepared.into_parts();
        let node_id = self
            .require_inner_mut()?
            .append_constant(metadata, constant)
            .map_err(graph_error)?;
        Ok((node_id, digest))
    }

    fn append_output(&mut self, node_id: NodeId) -> PyResult<()> {
        self.require_inner_mut()?
            .append_output(node_id)
            .map_err(graph_error)
    }

    fn finish(&mut self, py: Python<'_>) -> PyResult<FinishResult> {
        let builder = self
            .inner
            .take()
            .ok_or_else(|| PyRuntimeError::new_err("GraphBuilder has already finished"))?;
        let unoptimized = builder.finish_unoptimized().map_err(graph_error)?;
        let trace = unoptimized
            .topological_order()
            .into_iter()
            .map(|node_id| {
                let record = unoptimized.get_node(node_id).map_err(graph_error)?;
                Ok((
                    node_id,
                    record.op.clone(),
                    record.inputs.clone(),
                    record.metadata.name().map(str::to_owned),
                ))
            })
            .collect::<PyResult<TraceSnapshot>>()?;
        let outcome = optimize(unoptimized).map_err(graph_error)?;
        let report = report_to_python(py, &outcome.report)?;
        Ok((
            GraphStore::from_runtime(outcome.store),
            outcome.old_to_new,
            report,
            trace,
        ))
    }
}

/// Pre-optimization tape rows handed to Python: (id, op, inputs, name).
type TraceSnapshot = Vec<(NodeId, String, Vec<NodeId>, Option<String>)>;

/// Full `finish` payload: optimized store, ID remap, report, and the raw tape.
type FinishResult = (GraphStore, Vec<Option<NodeId>>, Py<PyDict>, TraceSnapshot);

fn report_to_python(py: Python<'_>, report: &OptimizationReport) -> PyResult<Py<PyDict>> {
    let passes = PyList::empty(py);
    for pass in &report.passes {
        let payload = PyDict::new(py);
        payload.set_item("name", pass.name)?;
        payload.set_item("nodes_before", pass.nodes_before)?;
        payload.set_item("nodes_after", pass.nodes_after)?;
        payload.set_item("removed_nodes", pass.removed_nodes())?;
        payload.set_item("rewritten_nodes", pass.rewritten_nodes)?;
        passes.append(payload)?;
    }
    let payload = PyDict::new(py);
    payload.set_item("nodes_before", report.nodes_before)?;
    payload.set_item("nodes_after", report.nodes_after)?;
    payload.set_item("rewritten_nodes", report.rewritten_nodes)?;
    payload.set_item("passes", passes)?;
    Ok(payload.unbind())
}

fn graph_error(error: advect_runtime::GraphError) -> PyErr {
    PyValueError::new_err(error.into_message())
}
