//! Structural real-linearity analysis for traceable JVP rules.

use std::collections::HashSet;

use pyo3::basic::CompareOp;
use pyo3::exceptions::{PyAttributeError, PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;

use super::layout::{self, OperandLayout};
use super::lifecycle::DynamicTape;
use advect_runtime::NodeId;

const WHERE_ARITY: usize = 3;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum LinearityKind {
    Zero,
    Constant,
    Linear,
    Nonlinear,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct Linearity {
    kind: LinearityKind,
    tangent_dependent: bool,
    reason: Option<String>,
}

impl Linearity {
    const fn zero() -> Self {
        Self {
            kind: LinearityKind::Zero,
            tangent_dependent: false,
            reason: None,
        }
    }

    const fn constant() -> Self {
        Self {
            kind: LinearityKind::Constant,
            tangent_dependent: false,
            reason: None,
        }
    }

    const fn linear() -> Self {
        Self {
            kind: LinearityKind::Linear,
            tangent_dependent: true,
            reason: None,
        }
    }

    fn nonlinear(reason: impl Into<String>, inputs: &[Self]) -> Self {
        Self {
            kind: LinearityKind::Nonlinear,
            tangent_dependent: inputs.iter().any(|value| value.tangent_dependent),
            reason: Some(reason.into()),
        }
    }
}

pub(super) fn analyze_real_linearity(
    py: Python<'_>,
    tape: &DynamicTape,
    tangent_input_ids: &[NodeId],
    primitive_name: &str,
) -> PyResult<Vec<NodeId>> {
    tape.require_available()?;
    let tangent_inputs = validate_tangent_inputs(tape, tangent_input_ids)?;
    let mut states = Vec::with_capacity(tape.arena.node_count());

    for node_index in 0..tape.arena.node_count() {
        let node_id = NodeId::try_from(node_index)
            .map_err(|_| PyRuntimeError::new_err("dynamic tape node ID overflowed"))?;
        let (_index, node) = tape.require_node(node_id)?;
        let op = tape.arena.op_name(node.op()).ok_or_else(|| {
            PyRuntimeError::new_err("dynamic tape contains an invalid operation ID")
        })?;

        let mut state = if op == "advect.input" {
            if tangent_inputs.contains(&node_id) {
                Linearity::linear()
            } else {
                Linearity::constant()
            }
        } else if op == "advect.const" {
            if node_value_is_zero(py, tape, node_index)? {
                Linearity::zero()
            } else {
                Linearity::constant()
            }
        } else {
            let parents = tape
                .arena
                .parents(node)
                .ok_or_else(|| PyRuntimeError::new_err("dynamic tape edge range is invalid"))?
                .to_vec();
            let inputs = operand_linearity(py, tape, node_index, &parents, &states)?;
            classify_node(op, &inputs)
        };

        if op != "advect.input"
            && op != "advect.const"
            && state.kind == LinearityKind::Constant
            && node_value_is_zero(py, tape, node_index)?
        {
            state = Linearity::zero();
        }
        states.push(state);
    }

    for &output_id in &tape.outputs {
        let output_index = node_index(output_id)?;
        let state = states.get(output_index).ok_or_else(|| {
            PyRuntimeError::new_err("dynamic tape output linearity state is unavailable")
        })?;
        if matches!(state.kind, LinearityKind::Zero | LinearityKind::Linear) {
            continue;
        }
        let (_index, output) = tape.require_node(output_id)?;
        let op = tape.arena.op_name(output.op()).ok_or_else(|| {
            PyRuntimeError::new_err("dynamic tape contains an invalid operation ID")
        })?;
        let detail = if state.kind == LinearityKind::Constant {
            "returns a tangent-independent nonzero offset"
        } else {
            state
                .reason
                .as_deref()
                .unwrap_or("is nonlinear in its tangent inputs")
        };
        return Err(PyValueError::new_err(format!(
            "JVP rule for '{primitive_name}' is not real-linear: {detail} \
             at '{op}' (tape value %{output_id})."
        )));
    }

    states
        .iter()
        .enumerate()
        .filter(|(_node_index, state)| state.tangent_dependent)
        .map(|(node_index, _state)| {
            NodeId::try_from(node_index)
                .map_err(|_| PyRuntimeError::new_err("dynamic tape node ID overflowed"))
        })
        .collect()
}

fn validate_tangent_inputs(
    tape: &DynamicTape,
    tangent_input_ids: &[NodeId],
) -> PyResult<HashSet<NodeId>> {
    let mut tangent_inputs = HashSet::with_capacity(tangent_input_ids.len());
    for &node_id in tangent_input_ids {
        let (_index, node) = tape.require_node(node_id)?;
        if !node.flags().is_input() {
            return Err(PyValueError::new_err(format!(
                "real-linearity tangent node %{node_id} is not an input"
            )));
        }
        tangent_inputs.insert(node_id);
    }
    Ok(tangent_inputs)
}

fn operand_linearity(
    py: Python<'_>,
    tape: &DynamicTape,
    current_index: usize,
    parents: &[NodeId],
    states: &[Linearity],
) -> PyResult<Vec<Linearity>> {
    let (parent_positions, literal_range) = layout::snapshot_layout(
        &tape.operand_layouts,
        &tape.operand_positions,
        current_index,
        parents.len(),
    )?;
    let operand_count = match *tape.operand_layouts.get(current_index).ok_or_else(|| {
        PyRuntimeError::new_err("dynamic tape operand layout arena is inconsistent")
    })? {
        OperandLayout::ParentsOnly => parents.len(),
        OperandLayout::Mixed { operand_count, .. } => usize::try_from(operand_count)
            .map_err(|_| PyRuntimeError::new_err("operand count is out of range"))?,
    };
    let mut inputs: Vec<Option<Linearity>> = vec![None; operand_count];

    for (&parent, &position) in parents.iter().zip(&parent_positions) {
        let parent_state = states
            .get(node_index(parent)?)
            .ok_or_else(|| PyRuntimeError::new_err("parent linearity state is unavailable"))?
            .clone();
        *inputs
            .get_mut(position)
            .ok_or_else(|| PyRuntimeError::new_err("operand position is unavailable"))? =
            Some(parent_state);
    }

    if let Some((literal_start, literal_count)) = literal_range {
        let literal_end = literal_start
            .checked_add(literal_count)
            .ok_or_else(|| PyRuntimeError::new_err("literal range overflowed"))?;
        let mut literals = tape
            .literals
            .get(literal_start..literal_end)
            .ok_or_else(|| PyRuntimeError::new_err("literal range is invalid"))?
            .iter();
        for input in &mut inputs {
            if input.is_some() {
                continue;
            }
            let literal = literals
                .next()
                .ok_or_else(|| PyRuntimeError::new_err("literal layout is inconsistent"))?
                .as_ref()
                .ok_or_else(|| {
                    PyRuntimeError::new_err(
                        "literal payload was released before linearity analysis",
                    )
                })?;
            *input = Some(if value_is_zero(py, literal.bind(py))? {
                Linearity::zero()
            } else {
                Linearity::constant()
            });
        }
        if literals.next().is_some() {
            return Err(PyRuntimeError::new_err(
                "literal layout retained unused values",
            ));
        }
    }

    inputs
        .into_iter()
        .map(|input| {
            input.ok_or_else(|| PyRuntimeError::new_err("operand linearity slot is uninitialized"))
        })
        .collect()
}

fn classify_node(op: &str, inputs: &[Linearity]) -> Linearity {
    let suffix = op.rsplit('.').next().unwrap_or(op);
    match suffix {
        "add" | "sub" | "subtract" => combine_add(inputs),
        "cross" | "dot" | "einsum" | "inner" | "kron" | "matmul" | "mul" | "multiply" | "outer"
        | "tensordot" => combine_product(inputs),
        "divide" | "true_divide" | "truediv" => classify_division(inputs),
        "take" | "take_along_axis" => classify_gather(inputs),
        "solve" => classify_solve(inputs),
        "astype" | "atleast_1d" | "atleast_2d" | "atleast_3d" | "broadcast_to" | "conj"
        | "conjugate" | "copy" | "cumsum" | "diag" | "diagonal" | "diff" | "expand_dims"
        | "fft" | "fft2" | "fftshift" | "fftn" | "flatten" | "flip" | "fliplr" | "flipud"
        | "getoutput" | "getitem" | "ifft" | "ifft2" | "ifftshift" | "ifftn" | "imag" | "irfft"
        | "irfft2" | "irfftn" | "mean" | "moveaxis" | "negative" | "neg" | "pad" | "positive"
        | "pos" | "ravel" | "real" | "repeat" | "reshape" | "rfft" | "rfft2" | "rfftn" | "roll"
        | "rot90" | "squeeze" | "sum" | "swapaxes" | "tile" | "trace" | "transpose" | "tril"
        | "triu"
            if inputs.len() == 1 =>
        {
            inputs.first().cloned().unwrap_or_else(Linearity::constant)
        }
        "concatenate" | "concat" | "index_update" | "stack" | "vstack" | "hstack" => {
            combine_add(inputs)
        }
        "where" if inputs.len() == WHERE_ARITY => classify_where(inputs),
        "zeros" | "zeros_like" => Linearity::zero(),
        _ if inputs
            .iter()
            .all(|value| matches!(value.kind, LinearityKind::Zero | LinearityKind::Constant)) =>
        {
            Linearity::constant()
        }
        _ => Linearity::nonlinear(
            format!("uses unsupported tangent-dependent operation '{op}'"),
            inputs,
        ),
    }
}

fn classify_gather(inputs: &[Linearity]) -> Linearity {
    let [values, indices] = inputs else {
        return Linearity::nonlinear("gather has unexpected arity", inputs);
    };
    if matches!(
        indices.kind,
        LinearityKind::Linear | LinearityKind::Nonlinear
    ) {
        return Linearity::nonlinear("uses tangent-dependent gather indices", inputs);
    }
    values.clone()
}

fn combine_add(inputs: &[Linearity]) -> Linearity {
    if inputs
        .iter()
        .any(|value| value.kind == LinearityKind::Nonlinear)
    {
        return Linearity::nonlinear("contains a nonlinear operand", inputs);
    }
    let has_constant = inputs
        .iter()
        .any(|value| value.kind == LinearityKind::Constant);
    let has_linear = inputs
        .iter()
        .any(|value| value.kind == LinearityKind::Linear);
    if has_constant && has_linear {
        Linearity::nonlinear("adds a tangent-independent offset", inputs)
    } else if has_linear {
        Linearity::linear()
    } else if has_constant {
        Linearity::constant()
    } else {
        Linearity::zero()
    }
}

fn combine_product(inputs: &[Linearity]) -> Linearity {
    if inputs.iter().any(|value| value.kind == LinearityKind::Zero) {
        return Linearity::zero();
    }
    if inputs
        .iter()
        .any(|value| value.kind == LinearityKind::Nonlinear)
    {
        return Linearity::nonlinear("contains a nonlinear operand", inputs);
    }
    let linear_count = inputs
        .iter()
        .filter(|value| value.kind == LinearityKind::Linear)
        .count();
    match linear_count {
        0 => Linearity::constant(),
        1 => Linearity::linear(),
        _ => Linearity::nonlinear("multiplies tangent-dependent operands", inputs),
    }
}

fn classify_division(inputs: &[Linearity]) -> Linearity {
    let [numerator, denominator] = inputs else {
        return Linearity::nonlinear("division has unexpected arity", inputs);
    };
    if matches!(
        denominator.kind,
        LinearityKind::Linear | LinearityKind::Nonlinear
    ) {
        return Linearity::nonlinear("uses a tangent-dependent denominator", inputs);
    }
    if denominator.kind == LinearityKind::Zero {
        return Linearity::nonlinear("divides by a structural zero", inputs);
    }
    numerator.clone()
}

fn classify_where(inputs: &[Linearity]) -> Linearity {
    let [mask, on_true, on_false] = inputs else {
        return Linearity::nonlinear("where has unexpected arity", inputs);
    };
    if matches!(mask.kind, LinearityKind::Linear | LinearityKind::Nonlinear) {
        return Linearity::nonlinear("uses a tangent-dependent selection mask", inputs);
    }
    combine_add(&[on_true.clone(), on_false.clone()])
}

fn classify_solve(inputs: &[Linearity]) -> Linearity {
    let [matrix, right_hand_side] = inputs else {
        return Linearity::nonlinear("solve has unexpected arity", inputs);
    };
    if matches!(
        matrix.kind,
        LinearityKind::Linear | LinearityKind::Nonlinear
    ) {
        return Linearity::nonlinear("uses a tangent-dependent solve matrix", inputs);
    }
    right_hand_side.clone()
}

fn node_value_is_zero(py: Python<'_>, tape: &DynamicTape, node_index: usize) -> PyResult<bool> {
    let value = tape
        .values
        .get(node_index)
        .and_then(Option::as_ref)
        .ok_or_else(|| PyRuntimeError::new_err("dynamic tape node value is unavailable"))?;
    value_is_zero(py, value.bind(py))
}

fn value_is_zero(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<bool> {
    match value_is_zero_inner(py, value) {
        Ok(is_zero) => Ok(is_zero),
        Err(error)
            if error.is_instance_of::<PyTypeError>(py)
                || error.is_instance_of::<PyValueError>(py) =>
        {
            Ok(false)
        }
        Err(error) => Err(error),
    }
}

fn value_is_zero_inner(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<bool> {
    if value.hasattr("node_id")? {
        return Ok(false);
    }
    let comparison = value.rich_compare(0, CompareOp::Eq)?;
    match comparison.getattr("all") {
        Ok(all_method) if all_method.is_callable() => all_method.call0()?.is_truthy(),
        Ok(_all_attribute) => comparison.is_truthy(),
        Err(error) if error.is_instance_of::<PyAttributeError>(py) => comparison.is_truthy(),
        Err(error) => Err(error),
    }
}

fn node_index(node_id: NodeId) -> PyResult<usize> {
    usize::try_from(node_id)
        .map_err(|_| PyValueError::new_err("dynamic tape node ID is out of range"))
}
