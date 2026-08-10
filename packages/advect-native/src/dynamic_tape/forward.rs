//! Native forward-mode traversal for a concrete dynamic tape.

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyTuple;

use super::MAX_MULTI_SEEDS;
use super::lifecycle::{
    DynamicTape, TraversalKind, clone_required_slot, finish_traversal, snapshot_operands,
};
use advect_runtime::NodeId;

#[derive(Debug)]
struct Invocation {
    node_id: NodeId,
    op_name: String,
    callback: Py<PyAny>,
    output: Py<PyAny>,
    operands: Py<PyTuple>,
    attrs: Py<PyAny>,
    source_location: Option<String>,
}

type TangentSlots = Vec<Option<Py<PyAny>>>;
type PreparedInvocation = Option<(Invocation, Vec<Option<TangentSlots>>)>;

/// Apply one forward JVP over a frozen concrete tape.
#[pyfunction(signature = (
    tape, tangent_seeds, requested_outputs, *, consume=false
))]
pub(crate) fn dynamic_jvp(
    py: Python<'_>,
    tape: Py<DynamicTape>,
    tangent_seeds: Vec<(NodeId, Py<PyAny>)>,
    requested_outputs: Vec<NodeId>,
    consume: bool,
) -> PyResult<Vec<Option<Py<PyAny>>>> {
    let tape = tape.into_bound(py);
    {
        let mut state = tape.try_borrow_mut()?;
        state.begin_traversal(TraversalKind::Forward)?;
    }
    let result = forward_many_inner(py, &tape, vec![tangent_seeds], requested_outputs).and_then(
        |mut results| {
            results
                .pop()
                .ok_or_else(|| PyRuntimeError::new_err("dynamic JVP result is unavailable"))
        },
    );
    finish_traversal(py, &tape, consume, result)
}

/// Apply several forward JVPs in one arena traversal.
#[pyfunction]
pub(crate) fn dynamic_jvp_many(
    py: Python<'_>,
    tape: Py<DynamicTape>,
    tangent_seed_sets: Vec<Vec<(NodeId, Py<PyAny>)>>,
    requested_outputs: Vec<NodeId>,
) -> PyResult<Vec<Vec<Option<Py<PyAny>>>>> {
    if tangent_seed_sets.len() > MAX_MULTI_SEEDS {
        return Err(PyValueError::new_err(format!(
            "dynamic JVP supports at most {MAX_MULTI_SEEDS} seeds per traversal"
        )));
    }
    let tape = tape.into_bound(py);
    {
        let mut state = tape.try_borrow_mut()?;
        state.begin_traversal(TraversalKind::Forward)?;
    }
    let result = forward_many_inner(py, &tape, tangent_seed_sets, requested_outputs);
    finish_traversal(py, &tape, false, result)
}

fn forward_many_inner(
    py: Python<'_>,
    tape: &Bound<'_, DynamicTape>,
    tangent_seed_sets: Vec<Vec<(NodeId, Py<PyAny>)>>,
    requested_outputs: Vec<NodeId>,
) -> PyResult<Vec<Vec<Option<Py<PyAny>>>>> {
    let node_count = tape.try_borrow()?.arena.node_count();
    let requested_output_indices = validate_requested_outputs(tape, requested_outputs)?;
    if tangent_seed_sets.is_empty() {
        return Ok(Vec::new());
    }
    let mut tangent_tables = Vec::with_capacity(tangent_seed_sets.len());
    for tangent_seeds in tangent_seed_sets {
        let mut tangents: Vec<Option<Py<PyAny>>> =
            std::iter::repeat_with(|| None).take(node_count).collect();
        seed_inputs(tape, &mut tangents, tangent_seeds)?;
        tangent_tables.push(tangents);
    }

    for node_index in 0..node_count {
        let node_id = NodeId::try_from(node_index)
            .map_err(|_| PyRuntimeError::new_err("dynamic JVP node ID overflowed"))?;
        let node = tape
            .try_borrow()?
            .arena
            .node(node_id)
            .ok_or_else(|| PyRuntimeError::new_err("dynamic JVP node is unavailable"))?;
        if node.flags().is_input() {
            continue;
        }
        let prepared =
            prepare_invocation(py, tape, node_id, node_index, tangent_tables.as_slice())?;
        let Some((invocation, tangent_sets)) = prepared else {
            continue;
        };
        for (tangents, tangent_set) in tangent_tables.iter_mut().zip(tangent_sets) {
            let Some(tangent_set) = tangent_set else {
                continue;
            };
            let tangent = execute_callback(py, &invocation, &tangent_set)?;
            if !tangent.bind(py).is_none() {
                *tangents
                    .get_mut(node_index)
                    .ok_or_else(|| PyRuntimeError::new_err("dynamic JVP slot is unavailable"))? =
                    Some(tangent);
            }
        }
    }

    let mut results: Vec<Vec<Option<Py<PyAny>>>> = tangent_tables
        .iter()
        .map(|_tangents| Vec::with_capacity(requested_output_indices.len()))
        .collect();
    for index in requested_output_indices {
        for (result, tangents) in results.iter_mut().zip(tangent_tables.iter_mut()) {
            result.push(
                tangents
                    .get_mut(index)
                    .ok_or_else(|| PyRuntimeError::new_err("requested JVP slot is unavailable"))?
                    .take(),
            );
        }
    }
    Ok(results)
}

fn seed_inputs(
    tape: &Bound<'_, DynamicTape>,
    tangents: &mut [Option<Py<PyAny>>],
    seeds: Vec<(NodeId, Py<PyAny>)>,
) -> PyResult<()> {
    let state = tape.try_borrow()?;
    let mut seeded = vec![false; state.arena.node_count()];
    for (node_id, tangent) in seeds {
        let (index, _node) = state.require_node(node_id)?;
        if !state.inputs.contains(&node_id) {
            return Err(PyValueError::new_err(format!(
                "dynamic JVP seed node %{node_id} is not a tape input"
            )));
        }
        if std::mem::replace(
            seeded
                .get_mut(index)
                .ok_or_else(|| PyRuntimeError::new_err("JVP seed marker is unavailable"))?,
            true,
        ) {
            return Err(PyValueError::new_err(format!(
                "dynamic JVP repeats input seed %{node_id}"
            )));
        }
        *tangents
            .get_mut(index)
            .ok_or_else(|| PyRuntimeError::new_err("JVP seed slot is unavailable"))? =
            Some(tangent);
    }
    Ok(())
}

fn validate_requested_outputs(
    tape: &Bound<'_, DynamicTape>,
    requested_outputs: Vec<NodeId>,
) -> PyResult<Vec<usize>> {
    let state = tape.try_borrow()?;
    let mut indices = Vec::with_capacity(requested_outputs.len());
    for node_id in requested_outputs {
        let (index, _node) = state.require_node(node_id)?;
        if !state.outputs.contains(&node_id) {
            return Err(PyValueError::new_err(format!(
                "dynamic JVP requested node %{node_id}, which is not a marked output"
            )));
        }
        indices.push(index);
    }
    Ok(indices)
}

fn prepare_invocation(
    py: Python<'_>,
    tape: &Bound<'_, DynamicTape>,
    node_id: NodeId,
    node_index: usize,
    tangent_tables: &[TangentSlots],
) -> PyResult<PreparedInvocation> {
    let state = tape.try_borrow()?;
    let node = state
        .arena
        .node(node_id)
        .ok_or_else(|| PyRuntimeError::new_err("dynamic JVP node is unavailable"))?;
    let operands = snapshot_operands(py, &state, node_index, node, true)?;
    let operand_count = operands.operands.len();
    let mut tangent_sets = Vec::with_capacity(tangent_tables.len());
    let mut any_tangent = false;
    for tangent_slots in tangent_tables {
        let mut tangents: Vec<Option<Py<PyAny>>> = std::iter::repeat_with(|| None)
            .take(operand_count)
            .collect();
        let mut lane_is_active = false;
        for (&parent, &position) in operands
            .parents
            .iter()
            .zip(operands.parent_positions.iter())
        {
            let parent_index = usize::try_from(parent)
                .map_err(|_| PyRuntimeError::new_err("dynamic JVP parent ID is out of range"))?;
            if let Some(tangent) = tangent_slots.get(parent_index).and_then(Option::as_ref) {
                *tangents.get_mut(position).ok_or_else(|| {
                    PyRuntimeError::new_err("JVP operand position is unavailable")
                })? = Some(tangent.clone_ref(py));
                lane_is_active = true;
                any_tangent = true;
            }
        }
        tangent_sets.push(lane_is_active.then_some(tangents));
    }
    if !any_tangent {
        return Ok(None);
    }

    let op_name = state
        .arena
        .op_name(node.op())
        .ok_or_else(|| PyRuntimeError::new_err("dynamic tape has an invalid operation ID"))?
        .to_owned();
    let callback = state
        .jvp_bindings
        .get(usize::from(node.op()))
        .and_then(Option::as_ref)
        .ok_or_else(|| {
            PyRuntimeError::new_err(format!("dynamic operation '{op_name}' has no JVP binding"))
        })?
        .clone_ref(py);
    let metadata = state
        .metadata
        .get(node_index)
        .ok_or_else(|| PyRuntimeError::new_err("dynamic JVP metadata is unavailable"))?;
    let invocation = Invocation {
        node_id,
        op_name,
        callback,
        output: clone_required_slot(py, &state.values, node_index, "output", node_id)?,
        operands: PyTuple::new(py, operands.operands.iter().map(|value| value.bind(py)))?.unbind(),
        attrs: state
            .attrs
            .get(node_index)
            .and_then(Option::as_ref)
            .map_or_else(|| py.None(), |value| value.clone_ref(py)),
        source_location: metadata.source_location.clone(),
    };
    Ok(Some((invocation, tangent_sets)))
}

fn execute_callback(
    py: Python<'_>,
    invocation: &Invocation,
    tangents: &[Option<Py<PyAny>>],
) -> PyResult<Py<PyAny>> {
    let tangents = PyTuple::new(
        py,
        tangents.iter().map(|tangent| {
            tangent
                .as_ref()
                .map_or_else(|| py.None(), |value| value.clone_ref(py))
        }),
    )?;
    invocation
        .callback
        .bind(py)
        .call1((
            invocation.output.bind(py),
            invocation.operands.bind(py),
            tangents,
            invocation.attrs.bind(py),
            invocation.source_location.as_deref(),
        ))
        .inspect_err(|error| {
            let _ = error.add_note(
                py,
                format!(
                    "while executing JVP for '{}' at dynamic node %{}",
                    invocation.op_name, invocation.node_id
                ),
            );
        })
        .map(Bound::unbind)
}
