//! Native forward-mode traversal for a concrete dynamic tape.

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyTuple;

use super::lifecycle::{DynamicTape, TraversalKind, snapshot_operands};
use super::traversal::{
    Invocation, Lane, Seeds, run_traversal, seed_lanes, single_lane, take_requested,
};
use advect_runtime::NodeId;

/// Apply one forward JVP over a frozen concrete tape.
#[pyfunction(signature = (
    tape, tangent_seeds, requested_outputs, *, consume=false
))]
pub(crate) fn dynamic_jvp(
    py: Python<'_>,
    tape: Py<DynamicTape>,
    tangent_seeds: Seeds,
    requested_outputs: Vec<NodeId>,
    consume: bool,
) -> PyResult<Lane> {
    run_traversal(py, tape, TraversalKind::Forward, 1, consume, |tape| {
        forward(py, tape, vec![tangent_seeds], requested_outputs).and_then(single_lane)
    })
}

/// Apply several forward JVPs in one arena traversal.
#[pyfunction]
pub(crate) fn dynamic_jvp_many(
    py: Python<'_>,
    tape: Py<DynamicTape>,
    tangent_seed_sets: Vec<Seeds>,
    requested_outputs: Vec<NodeId>,
) -> PyResult<Vec<Lane>> {
    let lane_count = tangent_seed_sets.len();
    run_traversal(
        py,
        tape,
        TraversalKind::Forward,
        lane_count,
        false,
        |tape| forward(py, tape, tangent_seed_sets, requested_outputs),
    )
}

fn forward(
    py: Python<'_>,
    tape: &Bound<'_, DynamicTape>,
    tangent_seed_sets: Vec<Seeds>,
    requested_outputs: Vec<NodeId>,
) -> PyResult<Vec<Lane>> {
    let (mut lanes, requested) = seed_lanes(
        py,
        &*tape.try_borrow()?,
        TraversalKind::Forward,
        tangent_seed_sets,
        requested_outputs,
    )?;
    if lanes.is_empty() {
        return Ok(Vec::new());
    }
    let node_count = tape.try_borrow()?.arena.node_count();
    for node_index in 0..node_count {
        let Some((invocation, lane_tangents)) = prepare_invocation(py, tape, node_index, &lanes)?
        else {
            continue;
        };
        for (lane, tangents) in lanes.iter_mut().zip(lane_tangents) {
            let Some(tangents) = tangents else {
                continue;
            };
            let tangents = PyTuple::new(
                py,
                tangents
                    .into_iter()
                    .map(|tangent| tangent.unwrap_or_else(|| py.None())),
            )?;
            let callback = invocation.callback.bind(py);
            let tangent = invocation.call(
                callback,
                "JVP",
                (
                    invocation.output.bind(py),
                    invocation.operands.bind(py),
                    tangents,
                    invocation.attrs.bind(py),
                    invocation.source_location.as_deref(),
                ),
            )?;
            if !tangent.is_none() {
                *lane
                    .get_mut(node_index)
                    .ok_or_else(|| PyRuntimeError::new_err("dynamic JVP slot is unavailable"))? =
                    Some(tangent.unbind());
            }
        }
    }
    take_requested(&mut lanes, &requested)
}

/// Bind one operation node that some lane reaches, with each reached lane's
/// operand tangents. Input nodes and tangent-free nodes need no rule call,
/// so their operands are never snapshotted.
fn prepare_invocation(
    py: Python<'_>,
    tape: &Bound<'_, DynamicTape>,
    node_index: usize,
    lanes: &[Lane],
) -> PyResult<Option<(Invocation, Vec<Option<Lane>>)>> {
    let state = tape.try_borrow()?;
    let node = state
        .arena
        .nodes()
        .get(node_index)
        .copied()
        .ok_or_else(|| PyRuntimeError::new_err("dynamic JVP node is unavailable"))?;
    let parents = state
        .arena
        .parents(node)
        .ok_or_else(|| PyRuntimeError::new_err("dynamic tape edge range is invalid"))?;
    if node.flags().is_input()
        || !parents.iter().any(|parent| {
            lanes
                .iter()
                .any(|lane| lane_tangent(lane, parent).is_some())
        })
    {
        return Ok(None);
    }
    let snapshot = snapshot_operands(py, &state, node_index, node, true)?;
    let mut lane_tangents = Vec::with_capacity(lanes.len());
    for lane in lanes {
        let parent_tangents = snapshot
            .parents
            .iter()
            .map(|&parent| lane_tangent(lane, parent).map(|value| value.clone_ref(py)))
            .collect::<Vec<_>>();
        let reached = parent_tangents.iter().any(Option::is_some);
        let tangents = snapshot.layout.interleave(
            parent_tangents,
            snapshot.layout.literals.clone().map(|_| None),
        )?;
        lane_tangents.push(reached.then_some(tangents));
    }
    let mut invocation = Invocation::bind(py, &state, TraversalKind::Forward, node_index, node)?;
    invocation.output = state.required_value(py, node_index, "output", invocation.node_id)?;
    invocation.operands = PyTuple::new(py, snapshot.operands)?.unbind();
    Ok(Some((invocation, lane_tangents)))
}

fn lane_tangent(lane: &Lane, parent: NodeId) -> Option<&Py<PyAny>> {
    usize::try_from(parent)
        .ok()
        .and_then(|parent_index| lane.get(parent_index))
        .and_then(Option::as_ref)
}
