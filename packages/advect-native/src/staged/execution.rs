//! `PyO3` host adapter for the runtime-owned execution plan.

use advect_runtime::{
    ExecutionError, Host, LinkedExecutionPlan, LinkedOperation, NodeId, Operand, OutputOwnership,
    PortableConstant, ValueSpec,
};
use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyComplex, PyFloat, PyInt, PyString, PyTuple};
use std::collections::BTreeMap;

use crate::staged::GraphStore;
use crate::staged::conversion::attr::attr_map_to_python;
use crate::staged::conversion::dtype::DTypeCache;

#[derive(Debug)]
struct LinkedEvaluator {
    callable: Py<PyAny>,
}

/// Immutable runtime schedule plus linked Python implementations.
#[derive(Debug)]
#[pyclass(module = "advect._native_core", frozen)]
pub(crate) struct GraphExecutionPlan {
    linked: LinkedExecutionPlan<LinkedEvaluator>,
}

struct PythonHost<'py> {
    py: Python<'py>,
    binder: Option<Py<PyAny>>,
    constants: BTreeMap<NodeId, Py<PyAny>>,
    context: Py<PyAny>,
    dtype_cache: DTypeCache,
}

impl<'py> PythonHost<'py> {
    fn for_link(py: Python<'py>, binder: Py<PyAny>) -> Self {
        Self {
            py,
            binder: Some(binder),
            constants: BTreeMap::new(),
            context: py.None(),
            dtype_cache: DTypeCache::default(),
        }
    }

    fn for_execution(
        py: Python<'py>,
        constant_ids: impl IntoIterator<Item = NodeId>,
        constants: Vec<Py<PyAny>>,
        context: Option<Py<PyAny>>,
    ) -> Self {
        Self {
            py,
            binder: None,
            constants: constant_ids.into_iter().zip(constants).collect(),
            context: context.unwrap_or_else(|| py.None()),
            dtype_cache: DTypeCache::default(),
        }
    }
}

impl Host for PythonHost<'_> {
    type Value = Py<PyAny>;
    type LinkedOp = LinkedEvaluator;
    type Error = PyErr;

    fn link(
        &mut self,
        op: &str,
        _schema_version: u32,
        attrs: &advect_runtime::AttrMap,
        _outputs: &[ValueSpec],
    ) -> Result<LinkedOperation<Self::LinkedOp>, Self::Error> {
        let binder = self
            .binder
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("staged binder is unavailable"))?
            .bind(self.py);
        let attrs = attr_map_to_python(self.py, attrs)?;
        let evaluator = binder.call1((op, attrs.bind(self.py)))?;
        if !evaluator.is_callable() {
            return Err(PyTypeError::new_err(format!(
                "staged evaluator binder returned a non-callable for {op:?}"
            )));
        }
        let donation_positions: Vec<usize> = optional_attr(
            &evaluator,
            intern!(self.py, "__advect_donation_positions__"),
        )?;
        let alias_positions: Vec<usize> =
            optional_attr(&evaluator, intern!(self.py, "__advect_alias_positions__"))?;
        if alias_positions.len() > 1 {
            return Err(PyValueError::new_err(format!(
                "staged evaluator for {op:?} declares more than one alias source"
            )));
        }
        let owns_output: bool =
            optional_attr(&evaluator, intern!(self.py, "__advect_owned_output__"))?;
        if owns_output && !alias_positions.is_empty() {
            return Err(PyValueError::new_err(format!(
                "staged evaluator for {op:?} cannot declare both owned and aliased output"
            )));
        }
        let output_ownership = match alias_positions.first().copied() {
            Some(position) => OutputOwnership::Alias(position),
            None if owns_output => OutputOwnership::Owned,
            None => OutputOwnership::Unknown,
        };
        Ok(LinkedOperation::new(
            LinkedEvaluator {
                callable: evaluator.unbind(),
            },
            donation_positions,
            output_ownership,
        ))
    }

    fn materialize_constant(
        &mut self,
        node_id: NodeId,
        _constant: &PortableConstant,
    ) -> Result<Self::Value, Self::Error> {
        self.constants.remove(&node_id).ok_or_else(|| {
            PyRuntimeError::new_err(format!(
                "staged graph constant value for node %{node_id} is unavailable"
            ))
        })
    }

    fn retain_value(&mut self, value: &Self::Value) -> Result<Self::Value, Self::Error> {
        Ok(value.clone_ref(self.py))
    }

    fn evaluate(
        &mut self,
        operation: &Self::LinkedOp,
        operands: Vec<Operand<'_, Self::Value>>,
    ) -> Result<Self::Value, Self::Error> {
        let values = PyTuple::new(
            self.py,
            operands.iter().map(|operand| operand.value().bind(self.py)),
        )?;
        let donation_position = operands.iter().find_map(|operand| match operand {
            Operand::Borrowed(_) => None,
            Operand::Donated { position, .. } => Some(*position),
        });
        Ok(operation
            .callable
            .bind(self.py)
            .call1((values, self.context.bind(self.py), donation_position))?
            .unbind())
    }

    fn validate_value(
        &mut self,
        value: &Self::Value,
        outputs: &[ValueSpec],
    ) -> Result<(), Self::Error> {
        validate_python_outputs(value.bind(self.py), outputs, &mut self.dtype_cache)
    }
}

fn validate_python_outputs(
    value: &Bound<'_, PyAny>,
    outputs: &[ValueSpec],
    dtype_cache: &mut DTypeCache,
) -> PyResult<()> {
    match outputs {
        [] => Err(PyRuntimeError::new_err(
            "staged node has no declared output specifications",
        )),
        [output] => validate_python_value(value, output, dtype_cache),
        _ => {
            let tuple = value.cast::<PyTuple>().map_err(|_| {
                let type_name = value
                    .get_type()
                    .qualname()
                    .map_or_else(|_| "value".to_owned(), |name| name.to_string());
                PyTypeError::new_err(format!(
                    "staged operation declared {} outputs but returned {type_name}",
                    outputs.len()
                ))
            })?;
            if tuple.len() != outputs.len() {
                return Err(PyValueError::new_err(format!(
                    "staged operation declared {} outputs but returned {}",
                    outputs.len(),
                    tuple.len()
                )));
            }
            for (value, output) in tuple.iter().zip(outputs) {
                validate_python_value(&value, output, dtype_cache)?;
            }
            Ok(())
        }
    }
}

fn validate_python_value(
    value: &Bound<'_, PyAny>,
    result: &ValueSpec,
    dtype_cache: &mut DTypeCache,
) -> PyResult<()> {
    if let Some(kind) = python_scalar_kind(value)? {
        let dtype = result.dtype().name();
        if result.shape().is_empty() && scalar_kind_matches_dtype(kind, dtype) {
            return Ok(());
        }
        return Err(PyValueError::new_err(format!(
            "Staged operation declared shape={}, dtype={}; produced Python {kind}",
            format_shape(result.shape()),
            dtype,
        )));
    }

    // Runs for every node on every call, so format the type name only on error.
    let not_an_array = |_| {
        let type_name = value
            .get_type()
            .qualname()
            .map_or_else(|_| "value".to_owned(), |name| name.to_string());
        PyRuntimeError::new_err(format!(
            "staged operation returned {type_name}, not an array value"
        ))
    };
    let py = value.py();
    let shape = value.getattr(intern!(py, "shape")).map_err(not_an_array)?;
    let dtype = value.getattr(intern!(py, "dtype")).map_err(not_an_array)?;
    let actual_shape = shape
        .extract::<Vec<usize>>()
        .map_err(|_| PyRuntimeError::new_err("staged operation returned an invalid array shape"))?;
    let invalid_dtype =
        |_| PyRuntimeError::new_err("staged operation returned an invalid array dtype");
    if actual_shape != result.shape()
        || !dtype_cache
            .matches(py, &dtype, result.dtype())
            .map_err(invalid_dtype)?
    {
        let actual_dtype = dtype_cache.resolve(py, &dtype).map_err(invalid_dtype)?;
        return Err(PyValueError::new_err(format!(
            "Staged operation declared shape={}, dtype={}; produced shape={}, dtype={}",
            format_shape(result.shape()),
            result.dtype().name(),
            format_shape(&actual_shape),
            actual_dtype.name(),
        )));
    }
    Ok(())
}

/// Classify Python scalars the way `StagedProgram` infers call specs: an
/// instance of a built-in scalar type, subclasses such as `IntEnum` members
/// included, is a Python scalar unless it has `shape` and `dtype`. Array
/// scalars such as `numpy.float64` subclass `float` but have both, so they
/// take the strict shape and dtype path.
fn python_scalar_kind(value: &Bound<'_, PyAny>) -> PyResult<Option<&'static str>> {
    let (kind, exact) = if value.is_instance_of::<PyBool>() {
        // `bool` cannot be subclassed.
        ("bool", true)
    } else if value.is_instance_of::<PyInt>() {
        ("int", value.is_exact_instance_of::<PyInt>())
    } else if value.is_instance_of::<PyFloat>() {
        ("float", value.is_exact_instance_of::<PyFloat>())
    } else if value.is_instance_of::<PyComplex>() {
        ("complex", value.is_exact_instance_of::<PyComplex>())
    } else {
        return Ok(None);
    };
    if exact {
        return Ok(Some(kind));
    }
    let py = value.py();
    let array_scalar =
        value.hasattr(intern!(py, "shape"))? && value.hasattr(intern!(py, "dtype"))?;
    Ok((!array_scalar).then_some(kind))
}

fn scalar_kind_matches_dtype(kind: &str, dtype: &str) -> bool {
    match kind {
        "bool" => dtype == "bool",
        "int" => dtype.starts_with("int") || dtype.starts_with("uint"),
        "float" => dtype.starts_with("float"),
        "complex" => dtype.starts_with("complex"),
        _ => false,
    }
}

/// Link every staged operation exactly once through the Python provider.
#[pyfunction]
#[expect(
    clippy::needless_pass_by_value,
    reason = "PyO3 extracts owned arguments at the Python boundary"
)]
pub(crate) fn build_graph_execution_plan(
    py: Python<'_>,
    store: PyRef<'_, GraphStore>,
    binder: Py<PyAny>,
) -> PyResult<GraphExecutionPlan> {
    let store = store.inner_arc();
    let mut host = PythonHost::for_link(py, binder);
    let linked = LinkedExecutionPlan::from_store(store, &mut host)
        .map_err(|error| execution_error(py, error, "binding"))?;
    Ok(GraphExecutionPlan { linked })
}

/// Execute once with invocation-local values while Rust owns release timing.
#[pyfunction]
#[pyo3(signature = (plan, inputs, constants, context=None))]
#[expect(
    clippy::needless_pass_by_value,
    reason = "PyO3 extracts owned arguments at the Python boundary"
)]
pub(crate) fn execute_graph(
    py: Python<'_>,
    plan: PyRef<'_, GraphExecutionPlan>,
    inputs: Vec<Py<PyAny>>,
    constants: Vec<Py<PyAny>>,
    context: Option<Py<PyAny>>,
) -> PyResult<Vec<Py<PyAny>>> {
    if constants.len() != plan.linked.constant_count() {
        return Err(PyValueError::new_err(format!(
            "staged graph expects {} constants but received {}",
            plan.linked.constant_count(),
            constants.len()
        )));
    }
    let constant_ids = plan.linked.constant_ids();
    let mut host = PythonHost::for_execution(py, constant_ids, constants, context);
    plan.linked
        .execute(&mut host, inputs)
        .map_err(|error| execution_error(py, error, "executing"))
}

fn execution_error(py: Python<'_>, error: ExecutionError<PyErr>, action: &str) -> PyErr {
    match error {
        ExecutionError::Runtime(message) => PyRuntimeError::new_err(message),
        ExecutionError::Host {
            node_id,
            op,
            source,
        } => {
            let _ = source.add_note(
                py,
                format!("while {action} staged operation '{op}' at node %{node_id}"),
            );
            source
        }
    }
}

/// Read one optional evaluator annotation, defaulting when it is absent.
fn optional_attr<'py, T>(evaluator: &Bound<'py, PyAny>, name: &Bound<'py, PyString>) -> PyResult<T>
where
    T: FromPyObjectOwned<'py> + Default,
{
    match evaluator.getattr_opt(name)? {
        Some(value) => value.extract().map_err(Into::into),
        None => Ok(T::default()),
    }
}

fn format_shape(shape: &[usize]) -> String {
    match shape {
        [] => "()".to_owned(),
        [dimension] => format!("({dimension},)"),
        _ => format!(
            "({})",
            shape
                .iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}
