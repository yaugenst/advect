//! Guarding, seeding, and result handling shared by both traversals.

use advect_runtime::{NodeCore, NodeId};
use pyo3::call::PyCallArgs;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyTuple;

use super::MAX_MULTI_SEEDS;
use super::lifecycle::{DynamicTape, TraversalKind, close_and_drop_retired};

/// One seed lane's values, indexed by tape node.
pub(super) type Lane = Vec<Option<Py<PyAny>>>;
/// Seeds of one lane: `(node, value)` pairs.
pub(super) type Seeds = Vec<(NodeId, Py<PyAny>)>;

impl TraversalKind {
    const fn rule(self) -> &'static str {
        match self {
            Self::Forward => "JVP",
            Self::Reverse => "VJP",
        }
    }

    /// Forward seeds tape inputs and returns marked outputs; reverse swaps them.
    const fn roles(self) -> (Role, Role) {
        match self {
            Self::Forward => (Role::Input, Role::Output),
            Self::Reverse => (Role::Output, Role::Input),
        }
    }
}

/// Tape membership required of seeded and requested nodes.
#[derive(Clone, Copy)]
enum Role {
    Input,
    Output,
}

impl Role {
    const fn name(self) -> &'static str {
        match self {
            Self::Input => "input",
            Self::Output => "output",
        }
    }

    const fn noun(self) -> &'static str {
        match self {
            Self::Input => "tape input",
            Self::Output => "marked output",
        }
    }

    fn holds(self, state: &DynamicTape, node_id: NodeId) -> PyResult<(usize, bool)> {
        let (index, node) = state.require_node(node_id)?;
        let holds = match self {
            Self::Input => node.flags().is_input(),
            Self::Output => state.node(index)?.output,
        };
        Ok((index, holds))
    }
}

/// Run one traversal under the tape's recursion guard, then release the
/// tape's payloads when consuming, whether or not the traversal succeeded.
pub(super) fn run_traversal<T>(
    py: Python<'_>,
    tape: Py<DynamicTape>,
    kind: TraversalKind,
    lane_count: usize,
    consume: bool,
    traverse: impl FnOnce(&Bound<'_, DynamicTape>) -> PyResult<T>,
) -> PyResult<T> {
    if lane_count > MAX_MULTI_SEEDS {
        return Err(PyValueError::new_err(format!(
            "dynamic {} supports at most {MAX_MULTI_SEEDS} seeds per traversal",
            kind.rule()
        )));
    }
    let tape = tape.into_bound(py);
    tape.try_borrow_mut()?.begin_traversal(kind)?;
    let result = traverse(&tape);
    let retired = {
        let mut state = tape.try_borrow_mut()?;
        state.finish_traversal();
        consume.then(|| state.retire_payloads())
    };
    let release_result = retired.map_or(Ok(()), |retired| close_and_drop_retired(py, retired));
    match (result, release_result) {
        (Ok(value), Ok(())) => Ok(value),
        (Ok(_value), Err(release_error)) => Err(release_error),
        (Err(error), Ok(())) => Err(error),
        (Err(error), Err(release_error)) => {
            let _ = error.add_note(
                py,
                format!("DynamicTape payload release also failed: {release_error}"),
            );
            Err(error)
        }
    }
}

/// Validate requested nodes and build one seeded lane per seed set. Reverse
/// lanes skip `None` cotangents.
pub(super) fn seed_lanes(
    py: Python<'_>,
    state: &DynamicTape,
    kind: TraversalKind,
    seed_sets: Vec<Seeds>,
    requested: Vec<NodeId>,
) -> PyResult<(Vec<Lane>, Vec<usize>)> {
    let rule = kind.rule();
    let (seed_role, requested_role) = kind.roles();
    let requested = requested
        .into_iter()
        .map(|node_id| match requested_role.holds(state, node_id)? {
            (index, true) => Ok(index),
            (_, false) => Err(PyValueError::new_err(format!(
                "dynamic {rule} requested node %{node_id}, which is not a {}",
                requested_role.noun()
            ))),
        })
        .collect::<PyResult<Vec<_>>>()?;
    let node_count = state.arena.node_count();
    let mut lanes = Vec::with_capacity(seed_sets.len());
    for seeds in seed_sets {
        let mut lane: Lane = std::iter::repeat_with(|| None).take(node_count).collect();
        let mut seeded = vec![false; node_count];
        for (node_id, value) in seeds {
            let (index, holds) = seed_role.holds(state, node_id)?;
            if !holds {
                return Err(PyValueError::new_err(format!(
                    "dynamic {rule} seed node %{node_id} is not a {}",
                    seed_role.noun()
                )));
            }
            let marker = seeded.get_mut(index).ok_or_else(|| {
                PyRuntimeError::new_err(format!("{rule} seed marker is unavailable"))
            })?;
            if std::mem::replace(marker, true) {
                return Err(PyValueError::new_err(format!(
                    "dynamic {rule} repeats {} seed %{node_id}",
                    seed_role.name()
                )));
            }
            if kind == TraversalKind::Forward || !value.bind(py).is_none() {
                *lane.get_mut(index).ok_or_else(|| {
                    PyRuntimeError::new_err(format!("{rule} seed slot is unavailable"))
                })? = Some(value);
            }
        }
        lanes.push(lane);
    }
    Ok((lanes, requested))
}

/// Take each lane's values at the requested nodes.
pub(super) fn take_requested(lanes: &mut [Lane], requested: &[usize]) -> PyResult<Vec<Lane>> {
    lanes
        .iter_mut()
        .map(|lane| {
            requested
                .iter()
                .map(|&index| {
                    lane.get_mut(index)
                        .map(Option::take)
                        .ok_or_else(|| PyRuntimeError::new_err("requested slot is unavailable"))
                })
                .collect()
        })
        .collect()
}

/// The only lane of a single-seed traversal.
pub(super) fn single_lane(mut lanes: Vec<Lane>) -> PyResult<Lane> {
    lanes
        .pop()
        .ok_or_else(|| PyRuntimeError::new_err("dynamic traversal result is unavailable"))
}

/// One node's rule callback with the payloads every rule receives.
#[derive(Debug)]
pub(super) struct Invocation {
    pub(super) node_id: NodeId,
    pub(super) op_name: String,
    pub(super) callback: Py<PyAny>,
    pub(super) output: Py<PyAny>,
    pub(super) operands: Py<PyTuple>,
    pub(super) attrs: Py<PyAny>,
    pub(super) source_location: Option<String>,
}

impl Invocation {
    /// Bind one node's rule callback and attributes. The output and operands
    /// start as `None` and an empty tuple for the caller to fill.
    pub(super) fn bind(
        py: Python<'_>,
        state: &DynamicTape,
        kind: TraversalKind,
        node_index: usize,
        node: NodeCore,
    ) -> PyResult<Self> {
        let node_id = NodeId::try_from(node_index)
            .map_err(|_| PyRuntimeError::new_err("dynamic node ID overflowed"))?;
        let op_name = state
            .arena
            .op_name(node.op())
            .ok_or_else(|| PyRuntimeError::new_err("dynamic tape has an invalid operation ID"))?
            .to_owned();
        let bindings = match kind {
            TraversalKind::Forward => &state.jvp_bindings,
            TraversalKind::Reverse => &state.vjp_bindings,
        };
        let callback = bindings
            .get(usize::from(node.op()))
            .and_then(Option::as_ref)
            .ok_or_else(|| {
                PyRuntimeError::new_err(format!(
                    "dynamic operation '{op_name}' has no {} binding",
                    kind.rule()
                ))
            })?
            .clone_ref(py);
        let record = state.node(node_index)?;
        Ok(Self {
            node_id,
            op_name,
            callback,
            output: py.None(),
            operands: PyTuple::empty(py).unbind(),
            attrs: record
                .attrs
                .as_ref()
                .map_or_else(|| py.None(), |value| value.clone_ref(py)),
            source_location: record.source_location.clone(),
        })
    }

    /// Call `callback` for this node, noting the rule and node on failure.
    pub(super) fn call<'py>(
        &self,
        callback: &Bound<'py, PyAny>,
        label: &str,
        args: impl PyCallArgs<'py>,
    ) -> PyResult<Bound<'py, PyAny>> {
        callback.call1(args).inspect_err(|error| {
            let _ = error.add_note(
                callback.py(),
                format!(
                    "while executing {label} for '{}' at dynamic node %{}",
                    self.op_name, self.node_id
                ),
            );
        })
    }
}
