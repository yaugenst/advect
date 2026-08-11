//! Dynamic tape state, recording, freezing, traversal, and payload lifecycle.

use std::mem::size_of;

use advect_runtime::{
    DEFAULT_OP_SCHEMA_VERSION, InputRef, NodeCore, NodeFlags, NodeId, RawArena, RawArenaError,
    SchemaVersion,
};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use super::layout::{self, OperandLayout, OperandSnapshot};
use super::linearity;

type DiagnosticSnapshot = Vec<(String, Option<String>, Py<PyAny>)>;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) struct ReverseNeeds {
    pub(super) output: bool,
    pub(super) primals: bool,
    pub(super) residual: bool,
}

#[derive(Debug)]
pub(super) struct DynamicNodeMetadata {
    shape: Vec<usize>,
    dtype: Py<PyAny>,
    name: Option<String>,
    pub(super) source_location: Option<String>,
}

#[derive(Debug)]
pub(super) struct RetiredPayloads {
    metadata: Vec<DynamicNodeMetadata>,
    weak_nodes: Vec<bool>,
    values: Vec<Option<Py<PyAny>>>,
    attrs: Vec<Option<Py<PyAny>>>,
    literals: Vec<Option<Py<PyAny>>>,
    residuals: Vec<Option<Py<PyAny>>>,
    jvp_bindings: Vec<Option<Py<PyAny>>>,
    vjp_bindings: Vec<Option<Py<PyAny>>>,
    reverse_needs: Vec<Option<ReverseNeeds>>,
    reverse_value_uses: Vec<u32>,
}

#[derive(Debug, Default)]
pub(super) struct RetiredReversePayloads {
    values: Vec<Py<PyAny>>,
    attrs: Vec<Py<PyAny>>,
    literals: Vec<Py<PyAny>>,
    residuals: Vec<Py<PyAny>>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum TraversalKind {
    Forward,
    Reverse,
}

impl TraversalKind {
    fn description(self) -> &'static str {
        match self {
            Self::Forward => "forward traversal",
            Self::Reverse => "reverse traversal",
        }
    }
}

/// Native owner for one concrete define-by-run invocation.
#[derive(Debug, Default)]
#[pyclass(module = "advect._native_core")]
pub(crate) struct DynamicTape {
    pub(super) arena: RawArena,
    pub(super) operand_layouts: Vec<OperandLayout>,
    pub(super) operand_positions: Vec<u32>,
    pub(super) metadata: Vec<DynamicNodeMetadata>,
    weak_nodes: Vec<bool>,
    pub(super) values: Vec<Option<Py<PyAny>>>,
    pub(super) attrs: Vec<Option<Py<PyAny>>>,
    pub(super) literals: Vec<Option<Py<PyAny>>>,
    pub(super) residuals: Vec<Option<Py<PyAny>>>,
    pub(super) jvp_bindings: Vec<Option<Py<PyAny>>>,
    pub(super) vjp_bindings: Vec<Option<Py<PyAny>>>,
    pub(super) reverse_needs: Vec<Option<ReverseNeeds>>,
    pub(super) reverse_value_uses: Vec<u32>,
    pub(super) inputs: Vec<NodeId>,
    pub(super) outputs: Vec<NodeId>,
    trace_level: Option<usize>,
    trace_frame_id: Option<usize>,
    sealed: bool,
    consumed: bool,
    reverse_pruned: bool,
    traversal: Option<TraversalKind>,
}

impl DynamicTape {
    pub(super) fn require_recording(&self) -> PyResult<()> {
        if self.consumed {
            return Err(PyRuntimeError::new_err(
                "DynamicTape has released its invocation payloads",
            ));
        }
        if self.sealed {
            return Err(PyRuntimeError::new_err("DynamicTape is frozen"));
        }
        Ok(())
    }

    pub(super) fn require_available(&self) -> PyResult<()> {
        if self.consumed {
            Err(PyRuntimeError::new_err(
                "DynamicTape has released its invocation payloads",
            ))
        } else if !self.sealed {
            Err(PyRuntimeError::new_err(
                "DynamicTape must be frozen before differentiation",
            ))
        } else {
            Ok(())
        }
    }

    pub(super) fn require_node(&self, node_id: NodeId) -> PyResult<(usize, NodeCore)> {
        let index = usize::try_from(node_id)
            .map_err(|_| PyValueError::new_err("dynamic tape node ID is out of range"))?;
        let node = self.arena.node(node_id).ok_or_else(|| {
            PyValueError::new_err(format!("dynamic tape node %{node_id} does not exist"))
        })?;
        Ok((index, node))
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "arguments mirror the dynamic tape node record"
    )]
    fn append_operation(
        &mut self,
        op: &str,
        schema_version: SchemaVersion,
        parents: &[NodeId],
        parent_positions: &[usize],
        literals: Vec<Py<PyAny>>,
        attrs: Option<Py<PyAny>>,
        shape: Vec<usize>,
        dtype: Py<PyAny>,
        name: Option<String>,
        source_location: Option<String>,
        value: Option<Py<PyAny>>,
        residual: Option<Py<PyAny>>,
        input_activity: Option<bool>,
        literal_weak: bool,
    ) -> PyResult<NodeId> {
        self.require_recording()?;
        if op.is_empty() {
            return Err(PyValueError::new_err(
                "dynamic tape operation name must not be empty",
            ));
        }
        let layout_plan =
            layout::validate_operand_layout(parents.len(), parent_positions, literals.len())
                .map_err(PyValueError::new_err)?;
        let is_operation = input_activity.is_none();
        let weak_candidate = is_operation
            && shape.is_empty()
            && (!parents.is_empty() || !literals.is_empty())
            && (literals.is_empty() || literal_weak);
        let mut active_parents = Vec::with_capacity(parents.len());
        let mut all_parents_weak = true;
        for &parent in parents {
            let (index, node) = self.require_node(parent)?;
            active_parents.push(node.flags().is_active());
            if weak_candidate {
                all_parents_weak &= self.weak_nodes.get(index).copied().ok_or_else(|| {
                    PyRuntimeError::new_err("dynamic tape weak-scalar table is inconsistent")
                })?;
            }
        }
        let weak = weak_candidate && all_parents_weak;
        let layout = layout::prepare_layout(
            &layout_plan,
            self.operand_positions.len(),
            self.literals.len(),
            literals.len(),
        )?;
        let flags = if let Some(active) = input_activity {
            NodeFlags::input(active)
        } else {
            NodeFlags::operation(&active_parents)
        };
        let node_id = self
            .arena
            .append(op, schema_version, parents, flags)
            .map_err(raw_arena_error)?;
        layout::commit_layout(
            &mut self.operand_positions,
            &mut self.literals,
            layout_plan,
            literals,
        );
        self.operand_layouts.push(layout);
        self.metadata.push(DynamicNodeMetadata {
            shape,
            dtype,
            name,
            source_location,
        });
        self.weak_nodes.push(weak);
        self.values.push(value);
        self.attrs.push(attrs);
        self.residuals.push(residual);
        if input_activity.is_some() {
            self.inputs.push(node_id);
        }
        Ok(node_id)
    }

    pub(super) fn begin_traversal(&mut self, kind: TraversalKind) -> PyResult<()> {
        self.require_available()?;
        if let Some(active) = self.traversal {
            return Err(PyRuntimeError::new_err(format!(
                "DynamicTape is already executing {}; recursive use of the same tape is unsupported",
                active.description()
            )));
        }
        self.traversal = Some(kind);
        Ok(())
    }

    pub(super) fn finish_traversal(&mut self) {
        self.traversal = None;
    }

    pub(super) fn retire_payloads(&mut self) -> RetiredPayloads {
        self.consumed = true;
        RetiredPayloads {
            metadata: std::mem::take(&mut self.metadata),
            weak_nodes: std::mem::take(&mut self.weak_nodes),
            values: std::mem::take(&mut self.values),
            attrs: std::mem::take(&mut self.attrs),
            literals: std::mem::take(&mut self.literals),
            residuals: std::mem::take(&mut self.residuals),
            jvp_bindings: std::mem::take(&mut self.jvp_bindings),
            vjp_bindings: std::mem::take(&mut self.vjp_bindings),
            reverse_needs: std::mem::take(&mut self.reverse_needs),
            reverse_value_uses: std::mem::take(&mut self.reverse_value_uses),
        }
    }

    fn rebuild_reverse_value_uses(&mut self) -> PyResult<()> {
        let mut uses = vec![0_u32; self.arena.node_count()];
        for (node_index, &node) in self.arena.nodes().iter().enumerate() {
            if node.flags().is_input() || !node.flags().is_active() {
                continue;
            }
            let Some(needs) = self
                .reverse_needs
                .get(usize::from(node.op()))
                .copied()
                .flatten()
            else {
                continue;
            };
            if needs.output {
                increment_reverse_use(&mut uses, node_index)?;
            }
            if needs.primals {
                let parents = self
                    .arena
                    .parents(node)
                    .ok_or_else(|| PyRuntimeError::new_err("dynamic tape edge range is invalid"))?;
                for parent in parents.iter() {
                    let parent_index = usize::try_from(parent).map_err(|_| {
                        PyRuntimeError::new_err("dynamic parent ID is out of range")
                    })?;
                    increment_reverse_use(&mut uses, parent_index)?;
                }
            }
        }
        self.reverse_value_uses = uses;
        Ok(())
    }

    pub(super) fn prune_zero_reverse_payloads(&mut self) -> PyResult<RetiredReversePayloads> {
        if self.reverse_pruned {
            return Ok(RetiredReversePayloads::default());
        }
        self.require_available()?;
        let mut retired = RetiredReversePayloads::default();
        for node_index in 0..self.arena.node_count() {
            if self
                .reverse_value_uses
                .get(node_index)
                .copied()
                .ok_or_else(|| {
                    PyRuntimeError::new_err("dynamic reverse-use table is inconsistent")
                })?
                == 0
                && let Some(value) = self.values.get_mut(node_index).and_then(Option::take)
            {
                retired.values.push(value);
            }

            let node =
                self.arena.nodes().get(node_index).copied().ok_or_else(|| {
                    PyRuntimeError::new_err("dynamic reverse node is unavailable")
                })?;
            let needs = if node.flags().is_input() || !node.flags().is_active() {
                None
            } else {
                self.reverse_needs
                    .get(usize::from(node.op()))
                    .copied()
                    .flatten()
            };
            if needs.is_none()
                && let Some(attrs) = self.attrs.get_mut(node_index).and_then(Option::take)
            {
                retired.attrs.push(attrs);
            }
            if !needs.is_some_and(|item| item.primals) {
                take_node_literals(self, node_index, &mut retired)?;
            }
            if !needs.is_some_and(|item| item.residual)
                && let Some(residual) = self.residuals.get_mut(node_index).and_then(Option::take)
            {
                retired.residuals.push(residual);
            }
        }
        self.reverse_pruned = true;
        Ok(retired)
    }

    pub(super) fn retire_node_reverse_payloads(
        &mut self,
        node_index: usize,
    ) -> PyResult<RetiredReversePayloads> {
        let node = self
            .arena
            .nodes()
            .get(node_index)
            .copied()
            .ok_or_else(|| PyRuntimeError::new_err("dynamic reverse node is unavailable"))?;
        let needs = self
            .reverse_needs
            .get(usize::from(node.op()))
            .copied()
            .flatten()
            .ok_or_else(|| {
                PyRuntimeError::new_err("dynamic operation has no reverse retention contract")
            })?;
        let mut retired = RetiredReversePayloads::default();
        if needs.output {
            decrement_reverse_use(self, node_index, &mut retired)?;
        }
        if needs.primals {
            let parents = self
                .arena
                .parents(node)
                .ok_or_else(|| PyRuntimeError::new_err("dynamic tape edge range is invalid"))?
                .to_vec();
            for parent in parents {
                let parent_index = usize::try_from(parent)
                    .map_err(|_| PyRuntimeError::new_err("dynamic parent ID is out of range"))?;
                decrement_reverse_use(self, parent_index, &mut retired)?;
            }
            take_node_literals(self, node_index, &mut retired)?;
        }
        if let Some(attrs) = self.attrs.get_mut(node_index).and_then(Option::take) {
            retired.attrs.push(attrs);
        }
        if needs.residual {
            let residual = self
                .residuals
                .get_mut(node_index)
                .and_then(Option::take)
                .ok_or_else(|| {
                    PyRuntimeError::new_err("dynamic reverse residual payload is unavailable")
                })?;
            retired.residuals.push(residual);
        }
        Ok(retired)
    }
}

fn increment_reverse_use(uses: &mut [u32], node_index: usize) -> PyResult<()> {
    let slot = uses
        .get_mut(node_index)
        .ok_or_else(|| PyRuntimeError::new_err("dynamic reverse-use slot is unavailable"))?;
    *slot = slot
        .checked_add(1)
        .ok_or_else(|| PyRuntimeError::new_err("dynamic reverse-use count overflowed"))?;
    Ok(())
}

fn decrement_reverse_use(
    state: &mut DynamicTape,
    node_index: usize,
    retired: &mut RetiredReversePayloads,
) -> PyResult<()> {
    let slot = state
        .reverse_value_uses
        .get_mut(node_index)
        .ok_or_else(|| PyRuntimeError::new_err("dynamic reverse-use slot is unavailable"))?;
    *slot = slot
        .checked_sub(1)
        .ok_or_else(|| PyRuntimeError::new_err("dynamic reverse-use count underflowed"))?;
    if *slot == 0 {
        let value = state
            .values
            .get_mut(node_index)
            .and_then(Option::take)
            .ok_or_else(|| {
                PyRuntimeError::new_err("dynamic reverse value was released before its last use")
            })?;
        retired.values.push(value);
    }
    Ok(())
}

fn take_node_literals(
    state: &mut DynamicTape,
    node_index: usize,
    retired: &mut RetiredReversePayloads,
) -> PyResult<()> {
    let parents = state
        .arena
        .nodes()
        .get(node_index)
        .copied()
        .and_then(|node| state.arena.parents(node))
        .ok_or_else(|| PyRuntimeError::new_err("dynamic tape edge range is invalid"))?;
    let parent_count = parents.len();
    let Some((literal_start, literal_count)) = layout::snapshot_layout(
        &state.operand_layouts,
        &state.operand_positions,
        node_index,
        parent_count,
    )?
    .1
    else {
        return Ok(());
    };
    let literal_end = literal_start
        .checked_add(literal_count)
        .ok_or_else(|| PyRuntimeError::new_err("literal range overflowed"))?;
    let slots = state
        .literals
        .get_mut(literal_start..literal_end)
        .ok_or_else(|| PyRuntimeError::new_err("literal range is invalid"))?;
    retired
        .literals
        .extend(slots.iter_mut().filter_map(Option::take));
    Ok(())
}

#[pymethods]
impl DynamicTape {
    #[new]
    fn new() -> Self {
        Self::default()
    }

    #[pyo3(signature = (value, shape, dtype, *, name=None, active=true))]
    fn record_input(
        &mut self,
        value: Py<PyAny>,
        shape: Vec<usize>,
        dtype: Py<PyAny>,
        name: Option<String>,
        active: bool,
    ) -> PyResult<NodeId> {
        self.append_operation(
            "advect.input",
            DEFAULT_OP_SCHEMA_VERSION,
            &[],
            &[],
            Vec::new(),
            None,
            shape,
            dtype,
            name,
            None,
            Some(value),
            None,
            Some(active),
            false,
        )
    }

    #[pyo3(signature = (
        op, inputs, value, attrs, shape, dtype, *,
        schema_version=DEFAULT_OP_SCHEMA_VERSION, name=None, source_location=None
    ))]
    #[expect(
        clippy::needless_pass_by_value,
        reason = "PyO3 extracts owned arguments at the Python boundary"
    )]
    #[expect(
        clippy::too_many_arguments,
        reason = "arguments mirror the public Python tape-recording signature"
    )]
    fn record_operation(
        &mut self,
        op: &str,
        inputs: Vec<NodeId>,
        value: Py<PyAny>,
        attrs: Py<PyAny>,
        shape: Vec<usize>,
        dtype: Py<PyAny>,
        schema_version: SchemaVersion,
        name: Option<String>,
        source_location: Option<String>,
    ) -> PyResult<NodeId> {
        let positions = (0..inputs.len()).collect::<Vec<_>>();
        self.append_operation(
            op,
            schema_version,
            &inputs,
            &positions,
            Vec::new(),
            Some(attrs),
            shape,
            dtype,
            name,
            source_location,
            Some(value),
            None,
            None,
            false,
        )
    }

    #[pyo3(signature = (
        op, inputs, input_positions, literals, value, attrs, shape, dtype, *,
        schema_version=DEFAULT_OP_SCHEMA_VERSION, name=None, source_location=None,
        literal_weak=false
    ))]
    #[expect(
        clippy::needless_pass_by_value,
        reason = "PyO3 extracts owned arguments at the Python boundary"
    )]
    #[expect(
        clippy::too_many_arguments,
        reason = "arguments mirror the public Python literal-recording signature"
    )]
    fn record_operation_with_literals(
        &mut self,
        op: &str,
        inputs: Vec<NodeId>,
        input_positions: Vec<usize>,
        literals: Vec<Py<PyAny>>,
        value: Py<PyAny>,
        attrs: Py<PyAny>,
        shape: Vec<usize>,
        dtype: Py<PyAny>,
        schema_version: SchemaVersion,
        name: Option<String>,
        source_location: Option<String>,
        literal_weak: bool,
    ) -> PyResult<NodeId> {
        self.append_operation(
            op,
            schema_version,
            &inputs,
            &input_positions,
            literals,
            Some(attrs),
            shape,
            dtype,
            name,
            source_location,
            Some(value),
            None,
            None,
            literal_weak,
        )
    }

    fn bind_trace_frame(&mut self, trace_level: usize, trace_frame_id: usize) -> PyResult<()> {
        if self.trace_level.is_some() || self.trace_frame_id.is_some() {
            return Err(PyRuntimeError::new_err(
                "DynamicTape is already bound to a trace frame",
            ));
        }
        self.trace_level = Some(trace_level);
        self.trace_frame_id = Some(trace_frame_id);
        Ok(())
    }

    fn runtime_trace_identity(&self) -> (Option<usize>, Option<usize>) {
        (self.trace_level, self.trace_frame_id)
    }

    fn record_residual(&mut self, node_id: NodeId, residual: Py<PyAny>) -> PyResult<()> {
        let (index, _node) = self.require_node(node_id)?;
        let slot = self.residuals.get_mut(index).ok_or_else(|| {
            PyRuntimeError::new_err("dynamic tape residual arena is inconsistent")
        })?;
        if slot.is_some() {
            return Err(PyRuntimeError::new_err(format!(
                "DynamicTape node %{node_id} already owns a primitive residual"
            )));
        }
        *slot = Some(residual);
        Ok(())
    }

    fn value(&self, py: Python<'_>, node_id: NodeId) -> PyResult<Py<PyAny>> {
        let (index, _node) = self.require_node(node_id)?;
        clone_required_slot(py, &self.values, index, "value", node_id)
    }

    fn values(&self, py: Python<'_>, node_ids: Vec<NodeId>) -> PyResult<Vec<Py<PyAny>>> {
        node_ids
            .into_iter()
            .map(|node_id| self.value(py, node_id))
            .collect()
    }

    fn mark_weak(&mut self, node_id: NodeId) -> PyResult<()> {
        self.require_recording()?;
        let (index, _node) = self.require_node(node_id)?;
        let metadata = self
            .metadata
            .get(index)
            .ok_or_else(|| PyRuntimeError::new_err("dynamic tape metadata is unavailable"))?;
        if !metadata.shape.is_empty() {
            return Err(PyValueError::new_err(
                "only rank-zero dynamic tape values can be weak scalars",
            ));
        }
        *self.weak_nodes.get_mut(index).ok_or_else(|| {
            PyRuntimeError::new_err("dynamic tape weak-scalar slot is unavailable")
        })? = true;
        Ok(())
    }

    fn is_weak(&self, node_id: NodeId) -> PyResult<bool> {
        let (index, _node) = self.require_node(node_id)?;
        self.weak_nodes
            .get(index)
            .copied()
            .ok_or_else(|| PyRuntimeError::new_err("dynamic tape weak-scalar slot is unavailable"))
    }

    fn weak_mask(&self, node_ids: Vec<NodeId>) -> PyResult<Vec<bool>> {
        node_ids
            .into_iter()
            .map(|node_id| self.is_weak(node_id))
            .collect()
    }

    fn mark_output(&mut self, node_id: NodeId) -> PyResult<()> {
        self.require_recording()?;
        self.require_node(node_id)?;
        if self.outputs.contains(&node_id) {
            return Err(PyValueError::new_err(format!(
                "dynamic tape output %{node_id} is already marked"
            )));
        }
        self.outputs.push(node_id);
        Ok(())
    }

    pub(super) fn freeze(
        &mut self,
        py: Python<'_>,
        jvp_bindings: Vec<Py<PyAny>>,
        vjp_bindings: Vec<Py<PyAny>>,
        reverse_needs: Vec<Option<(bool, bool, bool)>>,
    ) -> PyResult<()> {
        self.require_recording()?;
        self.jvp_bindings = normalize_bindings(py, jvp_bindings, self.arena.op_count(), "JVP")?;
        self.vjp_bindings = normalize_bindings(py, vjp_bindings, self.arena.op_count(), "VJP")?;
        self.reverse_needs =
            normalize_reverse_needs(reverse_needs, &self.vjp_bindings, self.arena.op_count())?;
        if self.values.iter().any(Option::is_none) {
            return Err(PyRuntimeError::new_err(
                "DynamicTape cannot freeze with missing concrete values",
            ));
        }
        self.rebuild_reverse_value_uses()?;
        self.sealed = true;
        Ok(())
    }

    fn prune_reverse_payloads(slf: PyRefMut<'_, Self>, py: Python<'_>) -> PyResult<()> {
        if slf.traversal.is_some() {
            return Err(PyRuntimeError::new_err(
                "cannot prune DynamicTape payloads during traversal",
            ));
        }
        let mut slf = slf;
        let retired = slf.prune_zero_reverse_payloads()?;
        drop(slf);
        close_and_drop_reverse_payloads(py, retired)
    }

    fn set_active_nodes(&mut self, node_ids: Vec<NodeId>) -> PyResult<()> {
        self.require_available()?;
        if self.reverse_pruned {
            return Err(PyRuntimeError::new_err(
                "cannot replace DynamicTape activity after reverse payload pruning",
            ));
        }
        if self.traversal.is_some() {
            return Err(PyRuntimeError::new_err(
                "cannot replace DynamicTape activity during traversal",
            ));
        }
        let mut active = vec![false; self.arena.node_count()];
        for node_id in node_ids {
            let (index, _node) = self.require_node(node_id)?;
            *active.get_mut(index).ok_or_else(|| {
                PyRuntimeError::new_err("dynamic tape activity slot is unavailable")
            })? = true;
        }
        self.arena
            .replace_activity(&active)
            .map_err(raw_arena_error)?;
        self.rebuild_reverse_value_uses()
    }

    #[expect(
        clippy::needless_pass_by_value,
        reason = "PyO3 extracts owned arguments at the Python boundary"
    )]
    fn analyze_real_linearity(
        &self,
        py: Python<'_>,
        tangent_input_ids: Vec<NodeId>,
        primitive_name: &str,
    ) -> PyResult<Vec<NodeId>> {
        linearity::analyze_real_linearity(py, self, &tangent_input_ids, primitive_name)
    }

    fn release_payloads(slf: PyRefMut<'_, Self>, py: Python<'_>) -> PyResult<()> {
        if slf.consumed {
            return Ok(());
        }
        if slf.traversal.is_some() {
            return Err(PyRuntimeError::new_err(
                "cannot release DynamicTape payloads during traversal",
            ));
        }
        let mut slf = slf;
        let retired = slf.retire_payloads();
        drop(slf);
        close_and_drop_retired(py, retired)
    }

    fn get_node_name(&self, node_id: NodeId) -> PyResult<Option<String>> {
        let (index, _node) = self.require_node(node_id)?;
        Ok(self
            .metadata
            .get(index)
            .and_then(|metadata| metadata.name.clone()))
    }

    fn _diagnostic_snapshot(&self, py: Python<'_>) -> PyResult<DiagnosticSnapshot> {
        if self.consumed {
            return Err(PyRuntimeError::new_err(
                "DynamicTape has released its invocation payloads",
            ));
        }
        self.arena
            .nodes()
            .iter()
            .enumerate()
            .map(|(index, node)| {
                let node_id = NodeId::try_from(index)
                    .map_err(|_| PyRuntimeError::new_err("dynamic node ID is out of range"))?;
                let op = self
                    .arena
                    .op_schema(node.op())
                    .ok_or_else(|| {
                        PyRuntimeError::new_err("dynamic operation schema is unavailable")
                    })?
                    .name()
                    .to_owned();
                let metadata = self.metadata.get(index).ok_or_else(|| {
                    PyRuntimeError::new_err("dynamic tape metadata is unavailable")
                })?;
                let value = clone_required_slot(py, &self.values, index, "value", node_id)?;
                Ok((op, metadata.source_location.clone(), value))
            })
            .collect()
    }

    #[getter]
    fn node_count(&self) -> usize {
        self.arena.node_count()
    }

    #[getter]
    fn inputs(&self) -> Vec<NodeId> {
        self.inputs.clone()
    }

    #[getter]
    fn op_names(&self) -> Vec<String> {
        self.arena.op_names().map(str::to_owned).collect()
    }

    #[getter]
    fn is_consumed(&self) -> bool {
        self.consumed
    }

    #[expect(
        clippy::too_many_lines,
        reason = "the flat diagnostics table is clearest as one inventory"
    )]
    fn stats(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let result = PyDict::new(py);
        result.set_item("node_count", self.arena.node_count())?;
        result.set_item("edge_count", self.arena.edge_count())?;
        result.set_item("operation_count", self.arena.op_count())?;
        result.set_item("operand_position_count", self.operand_positions.len())?;
        result.set_item(
            "literal_count",
            self.literals.iter().filter(|value| value.is_some()).count(),
        )?;
        result.set_item(
            "retained_value_count",
            self.values.iter().filter(|value| value.is_some()).count(),
        )?;
        result.set_item(
            "retained_attr_count",
            self.attrs.iter().filter(|value| value.is_some()).count(),
        )?;
        result.set_item(
            "residual_count",
            self.residuals.iter().filter(|slot| slot.is_some()).count(),
        )?;
        result.set_item(
            "reverse_value_use_count",
            self.reverse_value_uses
                .iter()
                .map(|&count| usize::try_from(count).unwrap_or(usize::MAX))
                .sum::<usize>(),
        )?;
        result.set_item("reverse_pruned", self.reverse_pruned)?;
        result.set_item("node_core_bytes", size_of::<NodeCore>())?;
        result.set_item("input_ref_bytes", size_of::<InputRef>())?;
        let structural = PyDict::new(py);
        let arena = self.arena.structural_stats();
        let mut native_structural_bytes = 0_usize;
        add_structural_table(
            py,
            &structural,
            "nodes",
            arena.node_len,
            arena.node_capacity,
            arena.node_bytes,
            &mut native_structural_bytes,
        )?;
        add_structural_table(
            py,
            &structural,
            "edges",
            arena.edge_len,
            arena.edge_capacity,
            arena.edge_bytes,
            &mut native_structural_bytes,
        )?;
        add_structural_table(
            py,
            &structural,
            "operation_schemas",
            arena.op_len,
            arena.op_capacity,
            arena.op_schema_bytes,
            &mut native_structural_bytes,
        )?;
        add_structural_table(
            py,
            &structural,
            "operation_index",
            arena.op_len,
            arena.op_index_capacity,
            arena.op_index_bytes,
            &mut native_structural_bytes,
        )?;
        macro_rules! add_vec {
            ($name:literal, $field:expr, $item:ty) => {
                add_structural_table(
                    py,
                    &structural,
                    $name,
                    $field.len(),
                    $field.capacity(),
                    $field.capacity().saturating_mul(size_of::<$item>()),
                    &mut native_structural_bytes,
                )?
            };
        }
        add_vec!("operand_layouts", self.operand_layouts, OperandLayout);
        add_vec!("operand_positions", self.operand_positions, u32);
        add_vec!("metadata", self.metadata, DynamicNodeMetadata);
        add_vec!("weak_nodes", self.weak_nodes, bool);
        add_vec!("values", self.values, Option<Py<PyAny>>);
        add_vec!("attrs", self.attrs, Option<Py<PyAny>>);
        add_vec!("literals", self.literals, Option<Py<PyAny>>);
        add_vec!("residuals", self.residuals, Option<Py<PyAny>>);
        add_vec!("jvp_bindings", self.jvp_bindings, Option<Py<PyAny>>);
        add_vec!("vjp_bindings", self.vjp_bindings, Option<Py<PyAny>>);
        add_vec!("reverse_needs", self.reverse_needs, Option<ReverseNeeds>);
        add_vec!("reverse_value_uses", self.reverse_value_uses, u32);
        add_vec!("inputs", self.inputs, NodeId);
        add_vec!("outputs", self.outputs, NodeId);
        result.set_item("native_structural", structural)?;
        result.set_item("native_structural_bytes", native_structural_bytes)?;
        result.set_item("frozen", self.sealed)?;
        result.set_item("consumed", self.consumed)?;
        Ok(result.unbind())
    }
}

fn add_structural_table(
    py: Python<'_>,
    tables: &Bound<'_, PyDict>,
    name: &str,
    len: usize,
    capacity: usize,
    bytes: usize,
    total: &mut usize,
) -> PyResult<()> {
    let entry = PyDict::new(py);
    entry.set_item("len", len)?;
    entry.set_item("capacity", capacity)?;
    entry.set_item("bytes", bytes)?;
    tables.set_item(name, entry)?;
    *total = total.saturating_add(bytes);
    Ok(())
}

#[expect(
    clippy::too_many_lines,
    reason = "one pass preserves the validated positional operand layout"
)]
pub(super) fn snapshot_operands(
    py: Python<'_>,
    state: &DynamicTape,
    node_index: usize,
    node: NodeCore,
    include_values: bool,
) -> PyResult<OperandSnapshot> {
    let parents = state
        .arena
        .parents(node)
        .ok_or_else(|| PyRuntimeError::new_err("dynamic tape edge range is invalid"))?
        .to_vec();
    let (parent_positions, literal_range) = layout::snapshot_layout(
        &state.operand_layouts,
        &state.operand_positions,
        node_index,
        parents.len(),
    )?;
    let operand_count = match *state.operand_layouts.get(node_index).ok_or_else(|| {
        PyRuntimeError::new_err("dynamic tape operand layout arena is inconsistent")
    })? {
        OperandLayout::ParentsOnly => parents.len(),
        OperandLayout::Mixed { operand_count, .. } => usize::try_from(operand_count)
            .map_err(|_| PyRuntimeError::new_err("operand count is out of range"))?,
    };
    let mut operands: Vec<Option<Py<PyAny>>> = std::iter::repeat_with(|| None)
        .take(operand_count)
        .collect();
    let mut parent_specs: Vec<Option<(Vec<usize>, Py<PyAny>)>> = std::iter::repeat_with(|| None)
        .take(operand_count)
        .collect();
    let mut parent_active = Vec::with_capacity(parents.len());
    let mut active_positions = Vec::with_capacity(parents.len());

    for (parent_index, (&parent, &position)) in
        parents.iter().zip(parent_positions.iter()).enumerate()
    {
        let parent_slot = usize::try_from(parent)
            .map_err(|_| PyRuntimeError::new_err("dynamic parent ID is out of range"))?;
        *operands
            .get_mut(position)
            .ok_or_else(|| PyRuntimeError::new_err("operand position is unavailable"))? =
            Some(if include_values {
                let primal = clone_required_slot(
                    py,
                    &state.values,
                    parent_slot,
                    "primal",
                    NodeId::try_from(node_index)
                        .map_err(|_| PyRuntimeError::new_err("dynamic node ID overflowed"))?,
                )?;
                let is_weak = state.weak_nodes.get(parent_slot).copied().ok_or_else(|| {
                    PyRuntimeError::new_err("dynamic tape weak-scalar table is inconsistent")
                })?;
                if is_weak && !primal.bind(py).hasattr("_advect_snapshot")? {
                    weak_scalar_primal(py, primal)?
                } else {
                    primal
                }
            } else {
                py.None()
            });
        let metadata = state.metadata.get(parent_slot).ok_or_else(|| {
            PyRuntimeError::new_err("dynamic tape parent metadata is inconsistent")
        })?;
        *parent_specs
            .get_mut(position)
            .ok_or_else(|| PyRuntimeError::new_err("parent-spec position is unavailable"))? =
            Some((metadata.shape.clone(), metadata.dtype.clone_ref(py)));
        let is_active = if parents.len() <= 2 {
            node.flags()
                .inline_parent_is_active(parent_index)
                .ok_or_else(|| PyRuntimeError::new_err("inline parent activity is unavailable"))?
        } else {
            state
                .arena
                .node(parent)
                .ok_or_else(|| PyRuntimeError::new_err("dynamic parent node is unavailable"))?
                .flags()
                .is_active()
        };
        parent_active.push(is_active);
        if is_active {
            active_positions.push(position);
        }
    }

    if let Some((literal_start, literal_count)) = literal_range {
        let literal_end = literal_start
            .checked_add(literal_count)
            .ok_or_else(|| PyRuntimeError::new_err("literal range overflowed"))?;
        let mut literals = state
            .literals
            .get(literal_start..literal_end)
            .ok_or_else(|| PyRuntimeError::new_err("literal range is invalid"))?
            .iter();
        for operand in &mut operands {
            if operand.is_none() {
                let literal = literals
                    .next()
                    .ok_or_else(|| PyRuntimeError::new_err("literal layout is inconsistent"))?;
                *operand = Some(if include_values {
                    literal
                        .as_ref()
                        .ok_or_else(|| {
                            PyRuntimeError::new_err(
                                "dynamic tape is missing a required literal payload",
                            )
                        })?
                        .clone_ref(py)
                } else {
                    py.None()
                });
            }
        }
        if literals.next().is_some() {
            return Err(PyRuntimeError::new_err(
                "literal layout retained unused values",
            ));
        }
    }

    Ok(OperandSnapshot {
        parents,
        parent_positions,
        parent_active,
        active_positions,
        operands: operands
            .into_iter()
            .map(|operand| {
                operand.ok_or_else(|| PyRuntimeError::new_err("operand slot is uninitialized"))
            })
            .collect::<PyResult<Vec<_>>>()?,
        parent_specs,
    })
}

fn weak_scalar_primal(py: Python<'_>, primal: Py<PyAny>) -> PyResult<Py<PyAny>> {
    let value = primal.bind(py);
    if value.hasattr("item")? {
        return Ok(value.call_method0("item")?.unbind());
    }
    let dtype = value
        .getattr("dtype")
        .and_then(|dtype| dtype.str())
        .and_then(|dtype| dtype.to_str().map(str::to_owned))
        .unwrap_or_default()
        .to_lowercase();
    let method = if dtype.contains("bool") {
        "__bool__"
    } else if dtype.contains("complex") {
        "__complex__"
    } else if dtype.contains("float") {
        "__float__"
    } else if dtype.contains("int") {
        "__int__"
    } else {
        return Ok(primal);
    };
    Ok(value.call_method0(method)?.unbind())
}

pub(super) fn clone_required_slot(
    py: Python<'_>,
    slots: &[Option<Py<PyAny>>],
    slot: usize,
    role: &str,
    owner: NodeId,
) -> PyResult<Py<PyAny>> {
    slots
        .get(slot)
        .and_then(Option::as_ref)
        .map(|value| value.clone_ref(py))
        .ok_or_else(|| {
            PyRuntimeError::new_err(format!(
                "dynamic tape is missing {role} payload for node %{owner}"
            ))
        })
}

pub(super) fn close_and_drop_retired(py: Python<'_>, retired: RetiredPayloads) -> PyResult<()> {
    let RetiredPayloads {
        metadata,
        weak_nodes,
        values,
        attrs,
        literals,
        residuals,
        jvp_bindings,
        vjp_bindings,
        reverse_needs,
        reverse_value_uses,
    } = retired;
    let close_result = close_residuals(py, residuals);
    drop((
        metadata,
        weak_nodes,
        values,
        attrs,
        literals,
        jvp_bindings,
        vjp_bindings,
        reverse_needs,
        reverse_value_uses,
    ));
    close_result
}

pub(super) fn close_and_drop_reverse_payloads(
    py: Python<'_>,
    retired: RetiredReversePayloads,
) -> PyResult<()> {
    let RetiredReversePayloads {
        values,
        attrs,
        literals,
        residuals,
    } = retired;
    let close_result = close_residuals(py, residuals.into_iter().map(Some).collect());
    drop((values, attrs, literals));
    close_result
}

pub(super) fn finish_traversal<T>(
    py: Python<'_>,
    tape: &Bound<'_, DynamicTape>,
    consume: bool,
    result: PyResult<T>,
) -> PyResult<T> {
    let retired = {
        let mut state = tape.try_borrow_mut()?;
        state.finish_traversal();
        consume.then(|| state.retire_payloads())
    };
    let release_result = match retired {
        Some(retired) => close_and_drop_retired(py, retired),
        None => Ok(()),
    };
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

fn close_residuals(py: Python<'_>, residuals: Vec<Option<Py<PyAny>>>) -> PyResult<()> {
    let mut first_error = None;
    for residual in residuals.into_iter().flatten() {
        if let Err(error) = residual.bind(py).call_method0("close")
            && first_error.is_none()
        {
            first_error = Some(error);
        }
    }
    match first_error {
        Some(error) => Err(error),
        None => Ok(()),
    }
}

fn normalize_bindings(
    py: Python<'_>,
    bindings: Vec<Py<PyAny>>,
    expected: usize,
    kind: &str,
) -> PyResult<Vec<Option<Py<PyAny>>>> {
    if bindings.len() != expected {
        return Err(PyValueError::new_err(format!(
            "DynamicTape {kind} binding count {} does not match operation count {expected}",
            bindings.len()
        )));
    }
    bindings
        .into_iter()
        .enumerate()
        .map(|(op_id, binding)| {
            let bound = binding.bind(py);
            if bound.is_none() {
                Ok(None)
            } else if bound.is_callable() {
                Ok(Some(binding))
            } else {
                Err(PyValueError::new_err(format!(
                    "DynamicTape {kind} binding {op_id} must be callable or None"
                )))
            }
        })
        .collect()
}

fn normalize_reverse_needs(
    needs: Vec<Option<(bool, bool, bool)>>,
    bindings: &[Option<Py<PyAny>>],
    expected: usize,
) -> PyResult<Vec<Option<ReverseNeeds>>> {
    if needs.len() != expected {
        return Err(PyValueError::new_err(format!(
            "DynamicTape reverse-needs count {} does not match operation count {expected}",
            needs.len()
        )));
    }
    needs
        .into_iter()
        .zip(bindings)
        .enumerate()
        .map(
            |(op_id, (needs, binding))| match (needs, binding.is_some()) {
                (None, false) => Ok(None),
                (Some((output, primals, residual)), true) => Ok(Some(ReverseNeeds {
                    output,
                    primals,
                    residual,
                })),
                (None, true) => Err(PyValueError::new_err(format!(
                    "DynamicTape VJP binding {op_id} is missing reverse-needs metadata"
                ))),
                (Some(_), false) => Err(PyValueError::new_err(format!(
                    "DynamicTape reverse-needs metadata {op_id} has no VJP binding"
                ))),
            },
        )
        .collect()
}

fn raw_arena_error(error: RawArenaError) -> PyErr {
    PyValueError::new_err(error.into_message())
}
