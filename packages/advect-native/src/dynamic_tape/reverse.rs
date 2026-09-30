//! Native reverse-mode traversal for a concrete dynamic tape.

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::PyTuple;

use super::lifecycle::{
    DynamicTape, TraversalKind, close_and_drop_reverse_payloads, node_slot, snapshot_operands,
};
use super::traversal::{
    Invocation, Lane, Seeds, run_traversal, seed_lanes, single_lane, take_requested,
};
use advect_runtime::{NodeCore, NodeId};

const BATCH_VJP_ATTR: &str = "__advect_vjp_many__";

/// One node's VJP call with the payloads only reverse rules receive.
#[derive(Debug)]
struct ReverseInvocation {
    rule: Invocation,
    operand_count: usize,
    parents: Vec<NodeId>,
    parent_positions: Vec<usize>,
    parent_active: Vec<bool>,
    active_positions: Py<PyTuple>,
    parent_specs: Py<PyTuple>,
    residual: Py<PyAny>,
}

/// Apply one reverse VJP over a frozen concrete tape.
#[pyfunction(signature = (
    tape, output_cotangents, requested_inputs, *, consume=false
))]
pub(crate) fn dynamic_vjp(
    py: Python<'_>,
    tape: Py<DynamicTape>,
    output_cotangents: Seeds,
    requested_inputs: Vec<NodeId>,
    consume: bool,
) -> PyResult<Lane> {
    run_traversal(py, tape, TraversalKind::Reverse, 1, consume, |tape| {
        if consume {
            let retired = tape.try_borrow_mut()?.prune_zero_reverse_payloads()?;
            close_and_drop_reverse_payloads(py, retired)?;
        }
        reverse(
            py,
            tape,
            vec![output_cotangents],
            requested_inputs,
            consume,
            false,
        )
        .and_then(single_lane)
    })
}

/// Apply several reverse VJPs in one arena traversal, calling an operation's
/// optional batched VJP once per node for all of its seeds.
#[pyfunction]
pub(crate) fn dynamic_vjp_many(
    py: Python<'_>,
    tape: Py<DynamicTape>,
    output_cotangent_sets: Vec<Seeds>,
    requested_inputs: Vec<NodeId>,
) -> PyResult<Vec<Lane>> {
    let lane_count = output_cotangent_sets.len();
    run_traversal(
        py,
        tape,
        TraversalKind::Reverse,
        lane_count,
        false,
        |tape| {
            reverse(
                py,
                tape,
                output_cotangent_sets,
                requested_inputs,
                false,
                true,
            )
        },
    )
}

/// Sweep the tape backward. A consuming sweep retires each node's payloads
/// right after its last use; a `batched` sweep prefers batched VJPs.
fn reverse(
    py: Python<'_>,
    tape: &Bound<'_, DynamicTape>,
    output_cotangent_sets: Vec<Seeds>,
    requested_inputs: Vec<NodeId>,
    consume: bool,
    batched: bool,
) -> PyResult<Vec<Lane>> {
    let (mut lanes, requested) = seed_lanes(
        py,
        &*tape.try_borrow()?,
        TraversalKind::Reverse,
        output_cotangent_sets,
        requested_inputs,
    )?;
    if lanes.is_empty() {
        return Ok(Vec::new());
    }
    // The optional batched VJP is fixed per operation, so probe it once per
    // operation rather than once per node.
    let batched_vjps = if batched {
        batched_vjp_bindings(py, tape)?
    } else {
        Vec::new()
    };
    let node_count = tape.try_borrow()?.arena.node_count();
    for node_index in (0..node_count).rev() {
        let node = tape
            .try_borrow()?
            .arena
            .nodes()
            .get(node_index)
            .copied()
            .ok_or_else(|| PyRuntimeError::new_err("dynamic VJP node is unavailable"))?;
        if node.flags().is_input() {
            continue;
        }
        let cotangents = lanes
            .iter_mut()
            .map(|lane| {
                lane.get_mut(node_index)
                    .map(Option::take)
                    .ok_or_else(|| PyRuntimeError::new_err("dynamic cotangent slot is unavailable"))
            })
            .collect::<PyResult<Vec<_>>>()?;
        if !node.flags().is_active() {
            continue;
        }
        if cotangents.iter().all(Option::is_none) {
            if consume {
                retire_invocation_payloads(py, tape, node_index)?;
            }
            continue;
        }
        let invocation = prepare_invocation(py, tape, node_index, node)?;
        if let Some(batched_vjp) = batched_vjps
            .get(usize::from(node.op()))
            .and_then(Option::as_ref)
        {
            let contribution_sets =
                execute_batched_callback(py, batched_vjp.bind(py), &invocation, &cotangents)?;
            for (seed_index, contributions) in contribution_sets {
                let lane = lanes.get_mut(seed_index).ok_or_else(|| {
                    PyRuntimeError::new_err("batched VJP seed index is unavailable")
                })?;
                commit_contributions(py, lane, &invocation, &contributions)?;
            }
        } else {
            for (lane, cotangent) in lanes.iter_mut().zip(cotangents) {
                let Some(cotangent) = cotangent else {
                    continue;
                };
                let contributions = execute_callback(py, &invocation, cotangent.bind(py))?;
                commit_contributions(py, lane, &invocation, &contributions)?;
            }
        }
        if consume {
            // Drop the invocation's payload references before retiring them.
            drop(invocation);
            retire_invocation_payloads(py, tape, node_index)?;
        }
    }
    take_requested(&mut lanes, &requested)
}

fn prepare_invocation(
    py: Python<'_>,
    tape: &Bound<'_, DynamicTape>,
    node_index: usize,
    node: NodeCore,
) -> PyResult<ReverseInvocation> {
    let state = tape.try_borrow()?;
    let mut rule = Invocation::bind(py, &state, TraversalKind::Reverse, node_index, node)?;
    let needs = state
        .reverse_needs
        .get(usize::from(node.op()))
        .copied()
        .flatten()
        .ok_or_else(|| {
            PyRuntimeError::new_err(format!(
                "dynamic operation '{}' has no reverse retention contract",
                rule.op_name
            ))
        })?;
    let snapshot = snapshot_operands(py, &state, node_index, node, needs.primals)?;
    let operand_count = snapshot.operands.len();
    let parent_positions = snapshot.layout.parent_positions;
    let active_positions = parent_positions
        .iter()
        .zip(&snapshot.parent_active)
        .filter_map(|(&position, &active)| active.then_some(position))
        .collect::<Vec<_>>();
    let parent_specs = snapshot
        .parent_specs
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
        .collect::<PyResult<Vec<_>>>()?;
    let residual = if needs.residual {
        state
            .node(node_index)?
            .residual
            .as_ref()
            .ok_or_else(|| {
                PyRuntimeError::new_err("dynamic reverse residual payload is unavailable")
            })?
            .bind(py)
            .getattr(intern!(py, "payload"))?
            .unbind()
    } else {
        py.None()
    };
    if needs.output {
        rule.output = state.required_value(py, node_index, "output", rule.node_id)?;
    }
    rule.operands = PyTuple::new(py, snapshot.operands)?.unbind();
    Ok(ReverseInvocation {
        rule,
        operand_count,
        parents: snapshot.parents,
        parent_positions,
        parent_active: snapshot.parent_active,
        active_positions: PyTuple::new(py, active_positions)?.unbind(),
        parent_specs: PyTuple::new(py, parent_specs)?.unbind(),
        residual,
    })
}

impl ReverseInvocation {
    fn call<'py>(
        &self,
        callback: &Bound<'py, PyAny>,
        label: &str,
        cotangents: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = callback.py();
        self.rule.call(
            callback,
            label,
            (
                self.rule.output.bind(py),
                self.rule.operands.bind(py),
                cotangents,
                self.rule.attrs.bind(py),
                self.active_positions.bind(py),
                self.residual.bind(py),
                self.parent_specs.bind(py),
                self.rule.source_location.as_deref(),
            ),
        )
    }

    fn check_contribution_count(&self, label: &str, contributions: &[Py<PyAny>]) -> PyResult<()> {
        if contributions.len() == self.operand_count {
            return Ok(());
        }
        Err(PyValueError::new_err(format!(
            "{label} for '{}' at dynamic node %{} returned {} contributions for {} operands",
            self.rule.op_name,
            self.rule.node_id,
            contributions.len(),
            self.operand_count
        )))
    }
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
    invocation: &ReverseInvocation,
    cotangent: &Bound<'_, PyAny>,
) -> PyResult<Vec<Py<PyAny>>> {
    let result = invocation.call(invocation.rule.callback.bind(py), "VJP", cotangent)?;
    let contributions = result.extract::<Vec<Py<PyAny>>>().map_err(|error| {
        PyValueError::new_err(format!(
            "VJP for '{}' at dynamic node %{} must return a sequence: {error}",
            invocation.rule.op_name, invocation.rule.node_id
        ))
    })?;
    invocation.check_contribution_count("VJP", &contributions)?;
    Ok(contributions)
}

fn batched_vjp_bindings(
    py: Python<'_>,
    tape: &Bound<'_, DynamicTape>,
) -> PyResult<Vec<Option<Py<PyAny>>>> {
    tape.try_borrow()?
        .vjp_bindings
        .iter()
        .map(|binding| {
            let Some(binding) = binding else {
                return Ok(None);
            };
            Ok(binding
                .bind(py)
                .getattr_opt(intern!(py, BATCH_VJP_ATTR))?
                .map(Bound::unbind))
        })
        .collect()
}

/// Call a batched VJP with every seeded lane's cotangent and pair each
/// returned contribution set with its lane.
fn execute_batched_callback(
    py: Python<'_>,
    callback: &Bound<'_, PyAny>,
    invocation: &ReverseInvocation,
    cotangents: &[Option<Py<PyAny>>],
) -> PyResult<Vec<(usize, Vec<Py<PyAny>>)>> {
    let (seed_indices, seeded): (Vec<usize>, Vec<&Py<PyAny>>) = cotangents
        .iter()
        .enumerate()
        .filter_map(|(seed_index, cotangent)| cotangent.as_ref().map(|value| (seed_index, value)))
        .unzip();
    let seeded = PyTuple::new(py, seeded.into_iter().map(|value| value.bind(py)))?;
    let result = invocation.call(callback, "batched VJP", seeded.as_any())?;
    let contribution_sets = result.extract::<Vec<Vec<Py<PyAny>>>>().map_err(|error| {
        PyValueError::new_err(format!(
            "Batched VJP for '{}' at dynamic node %{} must return a sequence of sequences: {error}",
            invocation.rule.op_name, invocation.rule.node_id
        ))
    })?;
    if contribution_sets.len() != seed_indices.len() {
        return Err(PyValueError::new_err(format!(
            "Batched VJP for '{}' at dynamic node %{} returned {} contribution sets for {} seeds",
            invocation.rule.op_name,
            invocation.rule.node_id,
            contribution_sets.len(),
            seed_indices.len()
        )));
    }
    for contributions in &contribution_sets {
        invocation.check_contribution_count("Batched VJP", contributions)?;
    }
    Ok(seed_indices.into_iter().zip(contribution_sets).collect())
}

fn commit_contributions(
    py: Python<'_>,
    cotangents: &mut [Option<Py<PyAny>>],
    invocation: &ReverseInvocation,
    contributions: &[Py<PyAny>],
) -> PyResult<()> {
    let parents = invocation
        .parents
        .iter()
        .zip(&invocation.parent_positions)
        .zip(&invocation.parent_active);
    for ((&parent, &position), &active) in parents {
        if !active {
            continue;
        }
        let contribution = contributions
            .get(position)
            .ok_or_else(|| PyRuntimeError::new_err("contribution position is unavailable"))?;
        accumulate_slot(
            py,
            cotangents
                .get_mut(node_slot(parent)?)
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
        (None, None) => {
            // A concrete Array API array rejects a traced right operand, so
            // a traced contribution goes on the left; addition commutes.
            let snapshot = intern!(py, "_advect_snapshot");
            if contribution.hasattr(snapshot)? && !existing.hasattr(snapshot)? {
                contribution.add(existing).map(Bound::unbind)
            } else {
                existing.add(contribution).map(Bound::unbind)
            }
        }
    }
}
