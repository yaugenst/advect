//! Native reverse-mode traversal for a concrete dynamic tape.

use pyo3::exceptions::{PyAttributeError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyTuple;

use super::MAX_MULTI_SEEDS;
use super::layout::OperandSnapshot;
use super::lifecycle::{
    DynamicTape, ReverseNeeds, TraversalKind, clone_required_slot, close_and_drop_reverse_payloads,
    finish_traversal, snapshot_operands,
};
use advect_runtime::{NodeCore, NodeId};

const BATCH_VJP_ATTR: &str = "__advect_vjp_many__";
type BatchedContributions = Vec<(usize, Vec<Py<PyAny>>)>;

#[derive(Debug)]
struct Invocation {
    node_id: NodeId,
    op_name: String,
    callback: Py<PyAny>,
    output: Py<PyAny>,
    operands: Py<PyTuple>,
    operand_count: usize,
    parents: Vec<NodeId>,
    parent_positions: Vec<usize>,
    parent_active: Vec<bool>,
    active_positions: Py<PyTuple>,
    parent_specs: Py<PyTuple>,
    attrs: Py<PyAny>,
    residual: Py<PyAny>,
    source_location: Option<String>,
}

/// Apply one reverse VJP over a frozen concrete tape.
#[pyfunction(signature = (
    tape, output_cotangents, requested_inputs, *, consume=false
))]
pub(crate) fn dynamic_vjp(
    py: Python<'_>,
    tape: Py<DynamicTape>,
    output_cotangents: Vec<(NodeId, Py<PyAny>)>,
    requested_inputs: Vec<NodeId>,
    consume: bool,
) -> PyResult<Vec<Option<Py<PyAny>>>> {
    let tape = tape.into_bound(py);
    {
        let mut state = tape.try_borrow_mut()?;
        state.begin_traversal(TraversalKind::Reverse)?;
    }
    if consume {
        let pruned = tape.try_borrow_mut()?.prune_zero_reverse_payloads();
        let prune_result = pruned.and_then(|retired| close_and_drop_reverse_payloads(py, retired));
        if let Err(error) = prune_result {
            return finish_traversal(py, &tape, true, Err(error));
        }
    }
    let result = reverse_inner(py, &tape, output_cotangents, requested_inputs, consume);
    finish_traversal(py, &tape, consume, result)
}

/// Apply several reverse VJPs in one arena traversal.
#[pyfunction]
pub(crate) fn dynamic_vjp_many(
    py: Python<'_>,
    tape: Py<DynamicTape>,
    output_cotangent_sets: Vec<Vec<(NodeId, Py<PyAny>)>>,
    requested_inputs: Vec<NodeId>,
) -> PyResult<Vec<Vec<Option<Py<PyAny>>>>> {
    if output_cotangent_sets.len() > MAX_MULTI_SEEDS {
        return Err(PyValueError::new_err(format!(
            "dynamic VJP supports at most {MAX_MULTI_SEEDS} seeds per traversal"
        )));
    }
    let tape = tape.into_bound(py);
    {
        let mut state = tape.try_borrow_mut()?;
        state.begin_traversal(TraversalKind::Reverse)?;
    }
    let result = reverse_many_inner(py, &tape, output_cotangent_sets, requested_inputs);
    finish_traversal(py, &tape, false, result)
}

fn reverse_inner(
    py: Python<'_>,
    tape: &Bound<'_, DynamicTape>,
    output_cotangents: Vec<(NodeId, Py<PyAny>)>,
    requested_inputs: Vec<NodeId>,
    consume: bool,
) -> PyResult<Vec<Option<Py<PyAny>>>> {
    let node_count = tape.try_borrow()?.arena.node_count();
    let mut cotangents: Vec<Option<Py<PyAny>>> =
        std::iter::repeat_with(|| None).take(node_count).collect();
    validate_requested_inputs(tape, &requested_inputs)?;
    seed_outputs(py, tape, &mut cotangents, output_cotangents)?;

    for node_index in (0..node_count).rev() {
        let node_id = NodeId::try_from(node_index)
            .map_err(|_| PyRuntimeError::new_err("dynamic VJP node ID overflowed"))?;
        let node = tape
            .try_borrow()?
            .arena
            .node(node_id)
            .ok_or_else(|| PyRuntimeError::new_err("dynamic VJP node is unavailable"))?;
        if node.flags().is_input() {
            continue;
        }
        let cotangent = cotangents
            .get_mut(node_index)
            .ok_or_else(|| PyRuntimeError::new_err("dynamic cotangent slot is unavailable"))?
            .take();
        if !node.flags().is_active() {
            continue;
        }
        let Some(cotangent) = cotangent else {
            if consume {
                retire_invocation_payloads(py, tape, node_index)?;
            }
            continue;
        };

        let invocation = prepare_invocation(py, tape, node_id, node_index, node)?;
        let contributions = execute_callback(py, &invocation, &cotangent)?;
        commit_contributions(py, &mut cotangents, &invocation, &contributions)?;
        if consume {
            drop(contributions);
            drop(invocation);
            retire_invocation_payloads(py, tape, node_index)?;
        }
    }

    requested_inputs
        .into_iter()
        .map(|node_id| {
            let index = node_index(node_id, node_count)?;
            Ok(cotangents
                .get_mut(index)
                .ok_or_else(|| PyRuntimeError::new_err("requested cotangent slot is unavailable"))?
                .take())
        })
        .collect()
}

fn reverse_many_inner(
    py: Python<'_>,
    tape: &Bound<'_, DynamicTape>,
    output_cotangent_sets: Vec<Vec<(NodeId, Py<PyAny>)>>,
    requested_inputs: Vec<NodeId>,
) -> PyResult<Vec<Vec<Option<Py<PyAny>>>>> {
    let node_count = tape.try_borrow()?.arena.node_count();
    validate_requested_inputs(tape, &requested_inputs)?;
    if output_cotangent_sets.is_empty() {
        return Ok(Vec::new());
    }
    let mut cotangent_tables = Vec::with_capacity(output_cotangent_sets.len());
    for output_cotangents in output_cotangent_sets {
        let mut cotangents: Vec<Option<Py<PyAny>>> =
            std::iter::repeat_with(|| None).take(node_count).collect();
        seed_outputs(py, tape, &mut cotangents, output_cotangents)?;
        cotangent_tables.push(cotangents);
    }

    for node_index in (0..node_count).rev() {
        let node_id = NodeId::try_from(node_index)
            .map_err(|_| PyRuntimeError::new_err("dynamic VJP node ID overflowed"))?;
        let node = tape
            .try_borrow()?
            .arena
            .node(node_id)
            .ok_or_else(|| PyRuntimeError::new_err("dynamic VJP node is unavailable"))?;
        if node.flags().is_input() {
            continue;
        }
        let node_cotangents = cotangent_tables
            .iter_mut()
            .map(|cotangents| {
                cotangents
                    .get_mut(node_index)
                    .ok_or_else(|| PyRuntimeError::new_err("dynamic cotangent slot is unavailable"))
                    .map(Option::take)
            })
            .collect::<PyResult<Vec<_>>>()?;
        if !node.flags().is_active() || node_cotangents.iter().all(Option::is_none) {
            continue;
        }

        let invocation = prepare_invocation(py, tape, node_id, node_index, node)?;
        if let Some(contribution_sets) =
            execute_batched_callback(py, &invocation, &node_cotangents)?
        {
            for (seed_index, contributions) in contribution_sets {
                let cotangents = cotangent_tables.get_mut(seed_index).ok_or_else(|| {
                    PyRuntimeError::new_err("batched VJP seed index is unavailable")
                })?;
                commit_contributions(py, cotangents, &invocation, &contributions)?;
            }
        } else {
            for (cotangents, cotangent) in cotangent_tables.iter_mut().zip(node_cotangents) {
                let Some(cotangent) = cotangent else {
                    continue;
                };
                let contributions = execute_callback(py, &invocation, &cotangent)?;
                commit_contributions(py, cotangents, &invocation, &contributions)?;
            }
        }
    }

    let mut results: Vec<Vec<Option<Py<PyAny>>>> = cotangent_tables
        .iter()
        .map(|_cotangents| Vec::with_capacity(requested_inputs.len()))
        .collect();
    for node_id in requested_inputs {
        let index = node_index(node_id, node_count)?;
        for (result, cotangents) in results.iter_mut().zip(cotangent_tables.iter_mut()) {
            result.push(
                cotangents
                    .get_mut(index)
                    .ok_or_else(|| {
                        PyRuntimeError::new_err("requested cotangent slot is unavailable")
                    })?
                    .take(),
            );
        }
    }
    Ok(results)
}

fn validate_requested_inputs(
    tape: &Bound<'_, DynamicTape>,
    requested_inputs: &[NodeId],
) -> PyResult<()> {
    let state = tape.try_borrow()?;
    for &node_id in requested_inputs {
        state.require_node(node_id)?;
        if !state.inputs.contains(&node_id) {
            return Err(PyValueError::new_err(format!(
                "dynamic VJP requested node %{node_id}, which is not a tape input"
            )));
        }
    }
    Ok(())
}

fn seed_outputs(
    py: Python<'_>,
    tape: &Bound<'_, DynamicTape>,
    cotangents: &mut [Option<Py<PyAny>>],
    output_cotangents: Vec<(NodeId, Py<PyAny>)>,
) -> PyResult<()> {
    let state = tape.try_borrow()?;
    let mut seeded = vec![false; state.arena.node_count()];
    for (node_id, cotangent) in output_cotangents {
        let (index, _node) = state.require_node(node_id)?;
        if !state.outputs.contains(&node_id) {
            return Err(PyValueError::new_err(format!(
                "dynamic VJP seed node %{node_id} is not a marked output"
            )));
        }
        let seen = seeded
            .get_mut(index)
            .ok_or_else(|| PyRuntimeError::new_err("VJP seed marker is unavailable"))?;
        if *seen {
            return Err(PyValueError::new_err(format!(
                "dynamic VJP repeats output seed %{node_id}"
            )));
        }
        *seen = true;
        if !cotangent.bind(py).is_none() {
            *cotangents
                .get_mut(index)
                .ok_or_else(|| PyRuntimeError::new_err("VJP output slot is unavailable"))? =
                Some(cotangent);
        }
    }
    Ok(())
}

fn prepare_invocation(
    py: Python<'_>,
    tape: &Bound<'_, DynamicTape>,
    node_id: NodeId,
    node_index: usize,
    node: NodeCore,
) -> PyResult<Invocation> {
    let state = tape.try_borrow()?;
    let op_name = state
        .arena
        .op_name(node.op())
        .ok_or_else(|| PyRuntimeError::new_err("dynamic tape has an invalid operation ID"))?
        .to_owned();
    let callback = state
        .vjp_bindings
        .get(usize::from(node.op()))
        .and_then(Option::as_ref)
        .ok_or_else(|| {
            PyRuntimeError::new_err(format!("dynamic operation '{op_name}' has no VJP binding"))
        })?
        .clone_ref(py);
    let needs = state
        .reverse_needs
        .get(usize::from(node.op()))
        .copied()
        .flatten()
        .ok_or_else(|| {
            PyRuntimeError::new_err(format!(
                "dynamic operation '{op_name}' has no reverse retention contract"
            ))
        })?;
    let metadata = state
        .metadata
        .get(node_index)
        .ok_or_else(|| PyRuntimeError::new_err("dynamic VJP metadata is unavailable"))?;
    let OperandSnapshot {
        parents,
        parent_positions,
        parent_active,
        active_positions,
        operands,
        parent_specs,
    } = snapshot_operands(py, &state, node_index, node, needs.primals)?;
    let operand_count = operands.len();
    let operands = PyTuple::new(py, operands.iter().map(|value| value.bind(py)))?.unbind();
    let active_positions = PyTuple::new(py, active_positions.iter().copied())?.unbind();
    let parent_specs = PyTuple::new(
        py,
        parent_specs
            .iter()
            .map(|spec| match spec {
                Some((shape, dtype)) => PyTuple::new(
                    py,
                    [
                        PyTuple::new(py, shape.iter().copied())?.into_any(),
                        dtype.bind(py).clone(),
                    ],
                )
                .map(|value| value.into_any().unbind()),
                None => Ok(py.None()),
            })
            .collect::<PyResult<Vec<_>>>()?,
    )?
    .unbind();
    let residual_slot = snapshot_residual(py, &state, node_index, needs)?;
    let residual = match residual_slot.as_ref() {
        Some(slot) => slot.bind(py).getattr("payload")?.unbind(),
        None => py.None(),
    };
    Ok(Invocation {
        node_id,
        op_name,
        callback,
        output: if needs.output {
            clone_required_slot(py, &state.values, node_index, "output", node_id)?
        } else {
            py.None()
        },
        operands,
        operand_count,
        parents,
        parent_positions,
        parent_active,
        active_positions,
        parent_specs,
        attrs: state
            .attrs
            .get(node_index)
            .and_then(Option::as_ref)
            .map_or_else(|| py.None(), |value| value.clone_ref(py)),
        residual,
        source_location: metadata.source_location.clone(),
    })
}

fn snapshot_residual(
    py: Python<'_>,
    state: &DynamicTape,
    node_index: usize,
    needs: ReverseNeeds,
) -> PyResult<Option<Py<PyAny>>> {
    if !needs.residual {
        return Ok(None);
    }
    state
        .residuals
        .get(node_index)
        .and_then(Option::as_ref)
        .map(|value| Some(value.clone_ref(py)))
        .ok_or_else(|| PyRuntimeError::new_err("dynamic reverse residual payload is unavailable"))
}

fn retire_invocation_payloads(
    py: Python<'_>,
    tape: &Bound<'_, DynamicTape>,
    node_index: usize,
) -> PyResult<()> {
    let retired = tape
        .try_borrow_mut()?
        .retire_node_reverse_payloads(node_index)?;
    close_and_drop_reverse_payloads(py, retired)
}

fn execute_callback(
    py: Python<'_>,
    invocation: &Invocation,
    cotangent: &Py<PyAny>,
) -> PyResult<Vec<Py<PyAny>>> {
    let result = invocation
        .callback
        .bind(py)
        .call1((
            invocation.output.bind(py),
            invocation.operands.bind(py),
            cotangent.bind(py),
            invocation.attrs.bind(py),
            invocation.active_positions.bind(py),
            invocation.residual.bind(py),
            invocation.parent_specs.bind(py),
            invocation.source_location.as_deref(),
        ))
        .inspect_err(|error| {
            let _ = error.add_note(
                py,
                format!(
                    "while executing VJP for '{}' at dynamic node %{}",
                    invocation.op_name, invocation.node_id
                ),
            );
        })?;
    let contributions = result.extract::<Vec<Py<PyAny>>>().map_err(|error| {
        PyValueError::new_err(format!(
            "VJP for '{}' at dynamic node %{} must return a sequence: {error}",
            invocation.op_name, invocation.node_id
        ))
    })?;
    if contributions.len() != invocation.operand_count {
        return Err(PyValueError::new_err(format!(
            "VJP for '{}' at dynamic node %{} returned {} contributions for {} operands",
            invocation.op_name,
            invocation.node_id,
            contributions.len(),
            invocation.operand_count
        )));
    }
    Ok(contributions)
}

fn execute_batched_callback(
    py: Python<'_>,
    invocation: &Invocation,
    cotangents: &[Option<Py<PyAny>>],
) -> PyResult<Option<BatchedContributions>> {
    let callback = match invocation.callback.bind(py).getattr(BATCH_VJP_ATTR) {
        Ok(callback) => callback,
        Err(error) if error.is_instance_of::<PyAttributeError>(py) => return Ok(None),
        Err(error) => return Err(error),
    };
    let active_cotangents = cotangents
        .iter()
        .enumerate()
        .filter_map(|(seed_index, cotangent)| {
            cotangent
                .as_ref()
                .map(|cotangent| (seed_index, cotangent.clone_ref(py)))
        })
        .collect::<Vec<_>>();
    let cotangent_tuple = PyTuple::new(
        py,
        active_cotangents
            .iter()
            .map(|(_seed_index, cotangent)| cotangent.bind(py)),
    )?;
    let result = callback
        .call1((
            invocation.output.bind(py),
            invocation.operands.bind(py),
            cotangent_tuple,
            invocation.attrs.bind(py),
            invocation.active_positions.bind(py),
            invocation.residual.bind(py),
            invocation.parent_specs.bind(py),
            invocation.source_location.as_deref(),
        ))
        .inspect_err(|error| {
            let _ = error.add_note(
                py,
                format!(
                    "while executing batched VJP for '{}' at dynamic node %{}",
                    invocation.op_name, invocation.node_id
                ),
            );
        })?;
    let contribution_sets = result.extract::<Vec<Vec<Py<PyAny>>>>().map_err(|error| {
        PyValueError::new_err(format!(
            "Batched VJP for '{}' at dynamic node %{} must return a sequence of sequences: {error}",
            invocation.op_name, invocation.node_id
        ))
    })?;
    if contribution_sets.len() != active_cotangents.len() {
        return Err(PyValueError::new_err(format!(
            "Batched VJP for '{}' at dynamic node %{} returned {} contribution sets for {} seeds",
            invocation.op_name,
            invocation.node_id,
            contribution_sets.len(),
            active_cotangents.len()
        )));
    }
    for contributions in &contribution_sets {
        if contributions.len() != invocation.operand_count {
            return Err(PyValueError::new_err(format!(
                "Batched VJP for '{}' at dynamic node %{} returned {} contributions for {} operands",
                invocation.op_name,
                invocation.node_id,
                contributions.len(),
                invocation.operand_count
            )));
        }
    }
    Ok(Some(
        active_cotangents
            .into_iter()
            .map(|(seed_index, _cotangent)| seed_index)
            .zip(contribution_sets)
            .collect(),
    ))
}

fn commit_contributions(
    py: Python<'_>,
    cotangents: &mut [Option<Py<PyAny>>],
    invocation: &Invocation,
    contributions: &[Py<PyAny>],
) -> PyResult<()> {
    for parent_index in 0..invocation.parents.len() {
        if !invocation
            .parent_active
            .get(parent_index)
            .copied()
            .ok_or_else(|| PyRuntimeError::new_err("parent activity is unavailable"))?
        {
            continue;
        }
        let parent_id = *invocation
            .parents
            .get(parent_index)
            .ok_or_else(|| PyRuntimeError::new_err("parent ID is unavailable"))?;
        let parent_slot = usize::try_from(parent_id)
            .map_err(|_| PyRuntimeError::new_err("parent ID is out of range"))?;
        let position = *invocation
            .parent_positions
            .get(parent_index)
            .ok_or_else(|| PyRuntimeError::new_err("parent position is unavailable"))?;
        let contribution = contributions
            .get(position)
            .ok_or_else(|| PyRuntimeError::new_err("contribution position is unavailable"))?;
        accumulate_slot(
            py,
            cotangents
                .get_mut(parent_slot)
                .ok_or_else(|| PyRuntimeError::new_err("parent cotangent slot is unavailable"))?,
            contribution.bind(py),
        )?;
    }
    Ok(())
}

fn accumulate_slot(
    py: Python<'_>,
    slot: &mut Option<Py<PyAny>>,
    contribution: &Bound<'_, PyAny>,
) -> PyResult<()> {
    if contribution.is_none() {
        return Ok(());
    }
    let Some(existing) = slot.as_ref() else {
        *slot = Some(contribution.clone().unbind());
        return Ok(());
    };
    *slot = Some(add_cotangents(py, existing.bind(py), contribution)?);
    Ok(())
}

fn add_cotangents(
    py: Python<'_>,
    existing: &Bound<'_, PyAny>,
    contribution: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    if existing.is_none() {
        return Ok(contribution.clone().unbind());
    }
    if contribution.is_none() {
        return Ok(existing.clone().unbind());
    }
    let existing_tuple = existing.cast::<PyTuple>().ok();
    let contribution_tuple = contribution.cast::<PyTuple>().ok();
    match (existing_tuple, contribution_tuple) {
        (Some(existing_items), Some(contribution_items)) => {
            if existing_items.len() != contribution_items.len() {
                return Err(PyValueError::new_err(format!(
                    "cannot add cotangent tuples with lengths {} and {}",
                    existing_items.len(),
                    contribution_items.len()
                )));
            }
            let combined = existing_items
                .iter()
                .zip(contribution_items.iter())
                .map(|(left, right)| add_cotangents(py, &left, &right))
                .collect::<PyResult<Vec<_>>>()?;
            PyTuple::new(py, combined).map(|items| items.into_any().unbind())
        }
        (Some(_), None) | (None, Some(_)) => Err(PyValueError::new_err(
            "cannot add tuple and non-tuple cotangents",
        )),
        (None, None) => existing.add(contribution).map(Bound::unbind),
    }
}

fn node_index(node_id: NodeId, node_count: usize) -> PyResult<usize> {
    let index = usize::try_from(node_id)
        .map_err(|_| PyValueError::new_err("dynamic node ID is out of range"))?;
    if index >= node_count {
        return Err(PyValueError::new_err(format!(
            "dynamic node %{node_id} does not exist"
        )));
    }
    Ok(index)
}
