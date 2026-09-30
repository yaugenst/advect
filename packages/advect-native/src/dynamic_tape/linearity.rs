//! Structural real-linearity analysis for traceable JVP rules.

use std::collections::HashSet;

use pyo3::basic::CompareOp;
use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::intern;
use pyo3::prelude::*;

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

    for (node_index, &node) in tape.arena.nodes().iter().enumerate() {
        let op = tape.arena.op_name(node.op()).ok_or_else(|| {
            PyRuntimeError::new_err("dynamic tape contains an invalid operation ID")
        })?;

        let mut state = if op == "advect.input" {
            if tangent_inputs.contains(&node_index) {
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
        let (output_index, output) = tape.require_node(output_id)?;
        let state = states.get(output_index).ok_or_else(|| {
            PyRuntimeError::new_err("dynamic tape output linearity state is unavailable")
        })?;
        if matches!(state.kind, LinearityKind::Zero | LinearityKind::Linear) {
            continue;
        }
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
) -> PyResult<HashSet<usize>> {
    let mut tangent_inputs = HashSet::with_capacity(tangent_input_ids.len());
    for &node_id in tangent_input_ids {
        let (index, node) = tape.require_node(node_id)?;
        if !node.flags().is_input() {
            return Err(PyValueError::new_err(format!(
                "real-linearity tangent node %{node_id} is not an input"
            )));
        }
        tangent_inputs.insert(index);
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
    let layout = tape.operands(current_index, parents.len())?;
    let parent_states = parents
        .iter()
        .map(|&parent| {
            usize::try_from(parent)
                .ok()
                .and_then(|parent_index| states.get(parent_index))
                .cloned()
                .ok_or_else(|| PyRuntimeError::new_err("parent linearity state is unavailable"))
        })
        .collect::<PyResult<Vec<_>>>()?;
    let literal_states = tape
        .literal_payloads(&layout)?
        .iter()
        .map(|literal| {
            let literal = literal.as_ref().ok_or_else(|| {
                PyRuntimeError::new_err("literal payload was released before linearity analysis")
            })?;
            Ok(if value_is_zero(py, literal.bind(py))? {
                Linearity::zero()
            } else {
                Linearity::constant()
            })
        })
        .collect::<PyResult<Vec<_>>>()?;
    layout.interleave(parent_states, literal_states)
}

fn classify_node(op: &str, inputs: &[Linearity]) -> Linearity {
    // Only canonical built-in IDs have known linearity. Custom primitives are
    // opaque even when their names end like a linear built-in.
    match op {
        "advect.index_update"
        | "array.add"
        | "array.concatenate"
        | "array.stack"
        | "array.subtract" => combine_add(inputs),
        "array.cross" | "array.matmul" | "array.multiply" | "array.outer" | "array.tensordot"
        | "array_ext.dot" | "array_ext.einsum" | "array_ext.inner" | "array_ext.kron" => {
            combine_product(inputs)
        }
        "array.divide" | "array_ext.true_divide" => classify_division(inputs),
        "array.take" | "array.take_along_axis" => classify_gather(inputs),
        "array_ext.linalg.solve" => classify_solve(inputs),
        "advect.copy"
        | "advect.getitem"
        | "advect.getoutput"
        | "array.astype"
        | "array.atleast_1d"
        | "array.atleast_2d"
        | "array.atleast_3d"
        | "array.broadcast_to"
        | "array.conjugate"
        | "array.cumsum"
        | "array.diagonal"
        | "array.diff"
        | "array.expand_dims"
        | "array.flip"
        | "array.imag"
        | "array.mean"
        | "array.moveaxis"
        | "array.negative"
        | "array.positive"
        | "array.real"
        | "array.repeat"
        | "array.reshape"
        | "array.roll"
        | "array.squeeze"
        | "array.sum"
        | "array.swapaxes"
        | "array.tile"
        | "array.trace"
        | "array.transpose"
        | "array.tril"
        | "array.triu"
        | "array_ext.diag"
        | "array_ext.fft.fft"
        | "array_ext.fft.fft2"
        | "array_ext.fft.fftn"
        | "array_ext.fft.fftshift"
        | "array_ext.fft.ifft"
        | "array_ext.fft.ifft2"
        | "array_ext.fft.ifftn"
        | "array_ext.fft.ifftshift"
        | "array_ext.fft.irfft"
        | "array_ext.fft.irfft2"
        | "array_ext.fft.irfftn"
        | "array_ext.fft.rfft"
        | "array_ext.fft.rfft2"
        | "array_ext.fft.rfftn"
        | "array_ext.fliplr"
        | "array_ext.flipud"
        | "array_ext.pad"
        | "array_ext.ravel"
        | "array_ext.rot90"
            if inputs.len() == 1 =>
        {
            inputs.first().cloned().unwrap_or_else(Linearity::constant)
        }
        "array.where" if inputs.len() == WHERE_ARITY => classify_where(inputs),
        "array.zeros" | "array.zeros_like" => Linearity::zero(),
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
    // A denominator that is zero at this point is still tangent-independent:
    // the quotient scales the numerator to IEEE infinities, as forward mode does.
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
        .node(node_index)?
        .value
        .as_ref()
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
    if value.hasattr(intern!(py, "node_id"))? {
        return Ok(false);
    }
    let comparison = value.rich_compare(0, CompareOp::Eq)?;
    match comparison.getattr_opt(intern!(py, "all"))? {
        Some(all_method) if all_method.is_callable() => all_method.call0()?.is_truthy(),
        _ => comparison.is_truthy(),
    }
}

#[cfg(test)]
mod tests {
    use super::LinearityKind::{Constant, Linear, Nonlinear, Zero};
    use super::{Linearity, LinearityKind, classify_node};

    const KINDS: [LinearityKind; 4] = [Zero, Constant, Linear, Nonlinear];

    /// Operation, operand kinds, result kind, and a fragment of its reason.
    const CASES: &[(&str, &[LinearityKind], LinearityKind, &str)] = &[
        ("array.add", &[Linear, Linear], Linear, ""),
        ("array.add", &[Zero, Linear], Linear, ""),
        ("array.add", &[Constant, Zero], Constant, ""),
        ("array.add", &[Zero, Zero], Zero, ""),
        ("array.subtract", &[Linear, Constant], Nonlinear, "offset"),
        (
            "array.stack",
            &[Nonlinear, Zero, Linear],
            Nonlinear,
            "nonlinear",
        ),
        ("array.multiply", &[Zero, Nonlinear], Zero, ""),
        ("array.matmul", &[Constant, Linear], Linear, ""),
        ("array.multiply", &[Constant, Constant], Constant, ""),
        ("array.outer", &[Linear, Linear], Nonlinear, "multiplies"),
        (
            "array.multiply",
            &[Linear, Nonlinear],
            Nonlinear,
            "nonlinear",
        ),
        ("array.divide", &[Linear, Constant], Linear, ""),
        (
            "array.divide",
            &[Constant, Linear],
            Nonlinear,
            "denominator",
        ),
        // A constant denominator that is zero here still scales linearly.
        ("array.divide", &[Linear, Zero], Linear, ""),
        ("array.divide", &[Linear], Nonlinear, "arity"),
        ("array.where", &[Constant, Linear, Zero], Linear, ""),
        (
            "array.where",
            &[Constant, Linear, Constant],
            Nonlinear,
            "offset",
        ),
        ("array.where", &[Linear, Zero, Zero], Nonlinear, "mask"),
        ("array.take", &[Linear, Constant], Linear, ""),
        ("array.take", &[Constant, Linear], Nonlinear, "indices"),
        ("array_ext.linalg.solve", &[Constant, Linear], Linear, ""),
        (
            "array_ext.linalg.solve",
            &[Linear, Constant],
            Nonlinear,
            "matrix",
        ),
        ("array.zeros_like", &[Nonlinear], Zero, ""),
        ("array.negative", &[Linear], Linear, ""),
        (
            "array.negative",
            &[Linear, Linear],
            Nonlinear,
            "unsupported",
        ),
        ("array.sin", &[Zero, Constant], Constant, ""),
        ("array.sin", &[Linear], Nonlinear, "'array.sin'"),
        ("custom.array.negative", &[Linear], Nonlinear, "unsupported"),
    ];

    fn state(kind: LinearityKind) -> Linearity {
        match kind {
            Zero => Linearity::zero(),
            Constant => Linearity::constant(),
            Linear => Linearity::linear(),
            Nonlinear => Linearity::nonlinear("operand", &[Linearity::linear()]),
        }
    }

    fn classify(op: &str, kinds: &[LinearityKind]) -> Linearity {
        let inputs = kinds.iter().copied().map(state).collect::<Vec<_>>();
        classify_node(op, &inputs)
    }

    #[test]
    fn classification_table_pins_each_rule_and_reason() {
        for &(op, kinds, expected, reason) in CASES {
            let result = classify(op, kinds);
            let context = format!("{op} over {kinds:?}: {result:?}");
            let dependent_input = kinds.iter().any(|kind| matches!(kind, Linear | Nonlinear));
            assert_eq!(result.kind, expected, "{context}");
            assert_eq!(
                result.tangent_dependent,
                expected == Linear || (expected == Nonlinear && dependent_input),
                "{context}"
            );
            assert_eq!(result.reason.is_some(), expected == Nonlinear, "{context}");
            assert!(
                result.reason.unwrap_or_default().contains(reason),
                "{context}"
            );
        }
    }

    #[test]
    fn additive_and_product_rules_ignore_operand_order() {
        for op in ["array.add", "array.multiply"] {
            for arity in [2, 3] {
                for index in 0..KINDS.len().pow(arity) {
                    let kinds = (0..arity)
                        .scan(index, |rest, _| {
                            let kind = KINDS.get(*rest % KINDS.len()).copied();
                            *rest /= KINDS.len();
                            kind
                        })
                        .collect::<Vec<_>>();
                    let expected = classify(op, &kinds).kind;
                    let mut permuted = kinds.clone();
                    for _ in 0..arity {
                        permuted.rotate_left(1);
                        assert_eq!(classify(op, &permuted).kind, expected, "{op} {kinds:?}");
                    }
                    permuted.reverse();
                    assert_eq!(classify(op, &permuted).kind, expected, "{op} {kinds:?}");
                }
            }
        }
    }
}
