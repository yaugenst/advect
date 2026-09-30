//! Dynamic tape state, recording, freezing, traversal, and payload lifecycle.

use std::mem::size_of;

use advect_runtime::{
    DEFAULT_OP_SCHEMA_VERSION, InputRef, NodeCore, NodeFlags, NodeId, Parents, RawArena,
    RawArenaError, SchemaVersion,
};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyComplex, PyDict, PyFloat, PyInt};

use super::layout::{OperandLayout, OperandSnapshot, Operands};
use super::linearity;

type DiagnosticSnapshot = Vec<(String, Option<String>, Py<PyAny>)>;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) struct ReverseNeeds {
    pub(super) output: bool,
    pub(super) primals: bool,
    pub(super) residual: bool,
}

/// Invocation-local record of one tape node, aligned with its arena node.
#[derive(Debug)]
pub(super) struct DynamicNode {
    layout: OperandLayout,
    shape: Vec<usize>,
    dtype: Py<PyAny>,
    name: Option<String>,
    pub(super) source_location: Option<String>,
    weak: bool,
    pub(super) output: bool,
    pub(super) value: Option<Py<PyAny>>,
    pub(super) attrs: Option<Py<PyAny>>,
    pub(super) residual: Option<Py<PyAny>>,
    reverse_uses: u32,
}

#[derive(Debug)]
pub(super) struct RetiredPayloads {
    nodes: Vec<DynamicNode>,
    literals: Vec<Option<Py<PyAny>>>,
    jvp_bindings: Vec<Option<Py<PyAny>>>,
    vjp_bindings: Vec<Option<Py<PyAny>>>,
    reverse_needs: Vec<Option<ReverseNeeds>>,
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
    nodes: Vec<DynamicNode>,
    operand_positions: Vec<u32>,
    literals: Vec<Option<Py<PyAny>>>,
    pub(super) jvp_bindings: Vec<Option<Py<PyAny>>>,
    pub(super) vjp_bindings: Vec<Option<Py<PyAny>>>,
    pub(super) reverse_needs: Vec<Option<ReverseNeeds>>,
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
            return Err(payloads_released());
        }
        if self.sealed {
            return Err(PyRuntimeError::new_err("DynamicTape is frozen"));
        }
        Ok(())
    }

    pub(super) fn require_available(&self) -> PyResult<()> {
        if self.consumed {
            Err(payloads_released())
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

    /// Invocation-local record of one node.
    pub(super) fn node(&self, node_index: usize) -> PyResult<&DynamicNode> {
        let consumed = self.consumed;
        self.nodes
            .get(node_index)
            .ok_or_else(|| node_record_unavailable(consumed))
    }

    fn node_mut(&mut self, node_index: usize) -> PyResult<&mut DynamicNode> {
        let consumed = self.consumed;
        self.nodes
            .get_mut(node_index)
            .ok_or_else(|| node_record_unavailable(consumed))
    }

    /// Clone one retained value, naming its role and consumer on failure.
    pub(super) fn required_value(
        &self,
        py: Python<'_>,
        node_index: usize,
        role: &str,
        owner: NodeId,
    ) -> PyResult<Py<PyAny>> {
        self.nodes
            .get(node_index)
            .and_then(|node| node.value.as_ref())
            .map(|value| value.clone_ref(py))
            .ok_or_else(|| {
                PyRuntimeError::new_err(format!(
                    "dynamic tape is missing {role} payload for node %{owner}"
                ))
            })
    }

    /// Decode one node's operand layout.
    pub(super) fn operands(&self, node_index: usize, parent_count: usize) -> PyResult<Operands> {
        self.node(node_index)?
            .layout
            .decode(&self.operand_positions, parent_count)
    }

    /// Literal payloads in one decoded layout, in operand order.
    pub(super) fn literal_payloads(&self, layout: &Operands) -> PyResult<&[Option<Py<PyAny>>]> {
        self.literals
            .get(layout.literals.clone())
            .ok_or_else(|| PyRuntimeError::new_err("literal range is invalid"))
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
        parent_positions: Option<&[usize]>,
        literals: Vec<Py<PyAny>>,
        attrs: Option<Py<PyAny>>,
        shape: Vec<usize>,
        dtype: Py<PyAny>,
        name: Option<String>,
        source_location: Option<String>,
        value: Py<PyAny>,
        input_activity: Option<bool>,
        weak: bool,
    ) -> PyResult<NodeId> {
        self.require_recording()?;
        if op.is_empty() {
            return Err(PyValueError::new_err(
                "dynamic tape operation name must not be empty",
            ));
        }
        if weak && !shape.is_empty() {
            return Err(weak_rank_error());
        }
        let (layout, positions) = OperandLayout::validate(
            parent_positions,
            parents.len(),
            literals.len(),
            self.operand_positions.len(),
            self.literals.len(),
        )
        .map_err(PyValueError::new_err)?;
        let mut any_parent_active = false;
        for &parent in parents {
            let (_index, node) = self.require_node(parent)?;
            any_parent_active |= node.flags().is_active();
        }
        let flags = if let Some(active) = input_activity {
            NodeFlags::input(active)
        } else {
            NodeFlags::operation(any_parent_active)
        };
        let node_id = self
            .arena
            .append(op, schema_version, parents, flags)
            .map_err(raw_arena_error)?;
        self.operand_positions.extend(positions);
        self.literals.extend(literals.into_iter().map(Some));
        self.nodes.push(DynamicNode {
            layout,
            shape,
            dtype,
            name,
            source_location,
            weak,
            output: false,
            value: Some(value),
            attrs,
            residual: None,
            reverse_uses: 0,
        });
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
            nodes: std::mem::take(&mut self.nodes),
            literals: std::mem::take(&mut self.literals),
            jvp_bindings: std::mem::take(&mut self.jvp_bindings),
            vjp_bindings: std::mem::take(&mut self.vjp_bindings),
            reverse_needs: std::mem::take(&mut self.reverse_needs),
        }
    }

    fn rebuild_reverse_value_uses(&mut self) -> PyResult<()> {
        for node in &mut self.nodes {
            node.reverse_uses = 0;
        }
        for (node_index, &core) in self.arena.nodes().iter().enumerate() {
            let Some(needs) = active_reverse_needs(&self.reverse_needs, core) else {
                continue;
            };
            if needs.output {
                increment_reverse_use(&mut self.nodes, node_index)?;
            }
            if needs.primals {
                for parent in parents(&self.arena, core)?.iter() {
                    increment_reverse_use(&mut self.nodes, node_slot(parent)?)?;
                }
            }
        }
        Ok(())
    }

    pub(super) fn prune_zero_reverse_payloads(&mut self) -> PyResult<RetiredReversePayloads> {
        if self.reverse_pruned {
            return Ok(RetiredReversePayloads::default());
        }
        self.require_available()?;
        let mut retired = RetiredReversePayloads::default();
        for (&core, node) in self.arena.nodes().iter().zip(&mut self.nodes) {
            let needs = active_reverse_needs(&self.reverse_needs, core);
            if node.reverse_uses == 0
                && let Some(value) = node.value.take()
            {
                retired.values.push(value);
            }
            if needs.is_none()
                && let Some(attrs) = node.attrs.take()
            {
                retired.attrs.push(attrs);
            }
            if !needs.is_some_and(|item| item.primals) {
                take_literals(&mut self.literals, node.layout, &mut retired.literals)?;
            }
            if !needs.is_some_and(|item| item.residual)
                && let Some(residual) = node.residual.take()
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
        let core = self
            .arena
            .nodes()
            .get(node_index)
            .copied()
            .ok_or_else(|| PyRuntimeError::new_err("dynamic reverse node is unavailable"))?;
        let mut retired = RetiredReversePayloads::default();
        // An operation without a VJP binding retains nothing for reverse mode:
        // pruning already retired its attrs, literals and residual.
        let Some(needs) = self
            .reverse_needs
            .get(usize::from(core.op()))
            .copied()
            .flatten()
        else {
            return Ok(retired);
        };
        if needs.output {
            decrement_reverse_use(&mut self.nodes, node_index, &mut retired)?;
        }
        if needs.primals {
            for parent in parents(&self.arena, core)?.iter() {
                decrement_reverse_use(&mut self.nodes, node_slot(parent)?, &mut retired)?;
            }
            let layout = self.node(node_index)?.layout;
            take_literals(&mut self.literals, layout, &mut retired.literals)?;
        }
        let node = self.node_mut(node_index)?;
        if let Some(attrs) = node.attrs.take() {
            retired.attrs.push(attrs);
        }
        if needs.residual {
            let residual = node.residual.take().ok_or_else(|| {
                PyRuntimeError::new_err("dynamic reverse residual payload is unavailable")
            })?;
            retired.residuals.push(residual);
        }
        Ok(retired)
    }
}

/// Reverse retention contract of an active operation node.
fn active_reverse_needs(
    reverse_needs: &[Option<ReverseNeeds>],
    core: NodeCore,
) -> Option<ReverseNeeds> {
    if core.flags().is_input() || !core.flags().is_active() {
        return None;
    }
    reverse_needs.get(usize::from(core.op())).copied().flatten()
}

pub(super) fn parents(arena: &RawArena, core: NodeCore) -> PyResult<Parents<'_>> {
    arena
        .parents(core)
        .ok_or_else(|| PyRuntimeError::new_err("dynamic tape edge range is invalid"))
}

pub(super) fn node_slot(node_id: NodeId) -> PyResult<usize> {
    usize::try_from(node_id)
        .map_err(|_| PyRuntimeError::new_err("dynamic parent ID is out of range"))
}

fn increment_reverse_use(nodes: &mut [DynamicNode], node_index: usize) -> PyResult<()> {
    let node = nodes
        .get_mut(node_index)
        .ok_or_else(|| PyRuntimeError::new_err("dynamic reverse-use slot is unavailable"))?;
    node.reverse_uses = node
        .reverse_uses
        .checked_add(1)
        .ok_or_else(|| PyRuntimeError::new_err("dynamic reverse-use count overflowed"))?;
    Ok(())
}

fn decrement_reverse_use(
    nodes: &mut [DynamicNode],
    node_index: usize,
    retired: &mut RetiredReversePayloads,
) -> PyResult<()> {
    let node = nodes
        .get_mut(node_index)
        .ok_or_else(|| PyRuntimeError::new_err("dynamic reverse-use slot is unavailable"))?;
    node.reverse_uses = node
        .reverse_uses
        .checked_sub(1)
        .ok_or_else(|| PyRuntimeError::new_err("dynamic reverse-use count underflowed"))?;
    if node.reverse_uses == 0 {
        let value = node.value.take().ok_or_else(|| {
            PyRuntimeError::new_err("dynamic reverse value was released before its last use")
        })?;
        retired.values.push(value);
    }
    Ok(())
}

fn take_literals(
    literals: &mut [Option<Py<PyAny>>],
    layout: OperandLayout,
    retired: &mut Vec<Py<PyAny>>,
) -> PyResult<()> {
    let literals = literals
        .get_mut(layout.literal_slots()?)
        .ok_or_else(|| PyRuntimeError::new_err("literal range is invalid"))?;
    retired.extend(literals.iter_mut().filter_map(Option::take));
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
            None,
            Vec::new(),
            None,
            shape,
            dtype,
            name,
            None,
            value,
            Some(active),
            false,
        )
    }

    /// Record one operation. Parents fill operand positions in order unless
    /// `input_positions` places them among `literals`, which fill the rest.
    /// `weak` marks a rank-zero result as a weak scalar (NEP 50); the frontend
    /// that computed it decides the category.
    #[pyo3(signature = (
        op, inputs, value, attrs, shape, dtype, *,
        input_positions=None, literals=Vec::new(), weak=false,
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
        input_positions: Option<Vec<usize>>,
        literals: Vec<Py<PyAny>>,
        weak: bool,
        schema_version: SchemaVersion,
        name: Option<String>,
        source_location: Option<String>,
    ) -> PyResult<NodeId> {
        self.append_operation(
            op,
            schema_version,
            &inputs,
            input_positions.as_deref(),
            literals,
            Some(attrs),
            shape,
            dtype,
            name,
            source_location,
            value,
            None,
            weak,
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
        let slot = &mut self.node_mut(index)?.residual;
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
        self.required_value(py, index, "value", node_id)
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
        let node = self.node_mut(index)?;
        if !node.shape.is_empty() {
            return Err(weak_rank_error());
        }
        node.weak = true;
        Ok(())
    }

    fn is_weak(&self, node_id: NodeId) -> PyResult<bool> {
        let (index, _node) = self.require_node(node_id)?;
        Ok(self.node(index)?.weak)
    }

    fn node_is_active(&self, node_id: NodeId) -> PyResult<bool> {
        let (_index, node) = self.require_node(node_id)?;
        Ok(node.flags().is_active())
    }

    fn weak_mask(&self, node_ids: Vec<NodeId>) -> PyResult<Vec<bool>> {
        node_ids
            .into_iter()
            .map(|node_id| self.is_weak(node_id))
            .collect()
    }

    fn mark_output(&mut self, node_id: NodeId) -> PyResult<()> {
        self.require_recording()?;
        let (index, _node) = self.require_node(node_id)?;
        if std::mem::replace(&mut self.node_mut(index)?.output, true) {
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
        Ok(self.nodes.get(index).and_then(|node| node.name.clone()))
    }

    fn _diagnostic_snapshot(&self, py: Python<'_>) -> PyResult<DiagnosticSnapshot> {
        if self.consumed {
            return Err(payloads_released());
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
                let value = self.required_value(py, index, "value", node_id)?;
                Ok((op, self.node(index)?.source_location.clone(), value))
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

    fn stats(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let result = PyDict::new(py);
        let retained = |payload: fn(&DynamicNode) -> bool| {
            self.nodes.iter().filter(|node| payload(node)).count()
        };
        result.set_item("node_count", self.arena.node_count())?;
        result.set_item("edge_count", self.arena.edge_count())?;
        result.set_item("operation_count", self.arena.op_count())?;
        result.set_item("operand_position_count", self.operand_positions.len())?;
        result.set_item("literal_count", self.literals.iter().flatten().count())?;
        result.set_item(
            "retained_value_count",
            retained(|node| node.value.is_some()),
        )?;
        result.set_item("retained_attr_count", retained(|node| node.attrs.is_some()))?;
        result.set_item("residual_count", retained(|node| node.residual.is_some()))?;
        result.set_item(
            "reverse_value_use_count",
            self.nodes
                .iter()
                .map(|node| usize::try_from(node.reverse_uses).unwrap_or(usize::MAX))
                .sum::<usize>(),
        )?;
        result.set_item("reverse_pruned", self.reverse_pruned)?;
        result.set_item("node_core_bytes", size_of::<NodeCore>())?;
        result.set_item("input_ref_bytes", size_of::<InputRef>())?;
        let arena = self.arena.structural_stats();
        let tables = [
            (
                "nodes",
                (arena.node_len, arena.node_capacity, arena.node_bytes),
            ),
            (
                "edges",
                (arena.edge_len, arena.edge_capacity, arena.edge_bytes),
            ),
            (
                "operation_schemas",
                (arena.op_len, arena.op_capacity, arena.op_schema_bytes),
            ),
            (
                "operation_index",
                (arena.op_len, arena.op_index_capacity, arena.op_index_bytes),
            ),
            ("node_records", vec_table(&self.nodes)),
            ("operand_positions", vec_table(&self.operand_positions)),
            ("literals", vec_table(&self.literals)),
            ("jvp_bindings", vec_table(&self.jvp_bindings)),
            ("vjp_bindings", vec_table(&self.vjp_bindings)),
            ("reverse_needs", vec_table(&self.reverse_needs)),
            ("inputs", vec_table(&self.inputs)),
            ("outputs", vec_table(&self.outputs)),
        ];
        let structural = PyDict::new(py);
        let mut native_structural_bytes = 0_usize;
        for (name, (len, capacity, bytes)) in tables {
            let entry = PyDict::new(py);
            entry.set_item("len", len)?;
            entry.set_item("capacity", capacity)?;
            entry.set_item("bytes", bytes)?;
            structural.set_item(name, entry)?;
            native_structural_bytes = native_structural_bytes.saturating_add(bytes);
        }
        result.set_item("native_structural", structural)?;
        result.set_item("native_structural_bytes", native_structural_bytes)?;
        result.set_item("frozen", self.sealed)?;
        result.set_item("consumed", self.consumed)?;
        Ok(result.unbind())
    }
}

/// Length, capacity and shallow reserved bytes of one side table.
fn vec_table<T>(table: &Vec<T>) -> (usize, usize, usize) {
    let capacity = table.capacity();
    (
        table.len(),
        capacity,
        capacity.saturating_mul(size_of::<T>()),
    )
}

pub(super) fn snapshot_operands(
    py: Python<'_>,
    state: &DynamicTape,
    node_index: usize,
    node: NodeCore,
    include_values: bool,
) -> PyResult<OperandSnapshot> {
    let owner = NodeId::try_from(node_index)
        .map_err(|_| PyRuntimeError::new_err("dynamic node ID overflowed"))?;
    let parents = state
        .arena
        .parents(node)
        .ok_or_else(|| PyRuntimeError::new_err("dynamic tape edge range is invalid"))?
        .to_vec();
    let layout = state.operands(node_index, parents.len())?;
    let mut primals = Vec::with_capacity(parents.len());
    let mut specs = Vec::with_capacity(parents.len());
    let mut parent_active = Vec::with_capacity(parents.len());
    for &parent in &parents {
        let (parent_index, parent_core) = state.require_node(parent)?;
        let parent_node = state.node(parent_index)?;
        primals.push(if include_values {
            let primal = state.required_value(py, parent_index, "primal", owner)?;
            if parent_node.weak {
                weak_scalar_primal(py, primal)?
            } else {
                primal
            }
        } else {
            py.None()
        });
        specs.push(Some((
            parent_node.shape.clone(),
            parent_node.dtype.clone_ref(py),
        )));
        parent_active.push(parent_core.flags().is_active());
    }
    let literals = state
        .literal_payloads(&layout)?
        .iter()
        .map(|literal| {
            if !include_values {
                return Ok(py.None());
            }
            literal
                .as_ref()
                .map(|value| value.clone_ref(py))
                .ok_or_else(|| {
                    PyRuntimeError::new_err("dynamic tape is missing a required literal payload")
                })
        })
        .collect::<PyResult<Vec<_>>>()?;
    let operands = layout.interleave(primals, literals)?;
    let parent_specs = layout.interleave(specs, layout.literals.clone().map(|_| None))?;
    Ok(OperandSnapshot {
        parents,
        layout,
        parent_active,
        operands,
        parent_specs,
    })
}

/// Present a weak operand as a Python scalar. Python scalars and values that
/// snapshot themselves pass through unchanged.
fn weak_scalar_primal(py: Python<'_>, primal: Py<PyAny>) -> PyResult<Py<PyAny>> {
    let value = primal.bind(py);
    if value.is_exact_instance_of::<PyFloat>()
        || value.is_exact_instance_of::<PyInt>()
        || value.is_exact_instance_of::<PyBool>()
        || value.is_exact_instance_of::<PyComplex>()
        || value.hasattr(intern!(py, "_advect_snapshot"))?
    {
        return Ok(primal);
    }
    if value.hasattr(intern!(py, "item"))? {
        return Ok(value.call_method0(intern!(py, "item"))?.unbind());
    }
    let dtype = value
        .getattr(intern!(py, "dtype"))
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

/// Close retired residuals, then drop the remaining payloads.
pub(super) fn close_and_drop_retired(py: Python<'_>, retired: RetiredPayloads) -> PyResult<()> {
    let RetiredPayloads {
        mut nodes,
        literals,
        jvp_bindings,
        vjp_bindings,
        reverse_needs,
    } = retired;
    let close_result =
        close_residuals(py, nodes.iter_mut().filter_map(|node| node.residual.take()));
    drop((nodes, literals, jvp_bindings, vjp_bindings, reverse_needs));
    close_result
}

/// Close retired residuals, then drop the remaining payloads.
pub(super) fn close_and_drop_reverse_payloads(
    py: Python<'_>,
    retired: RetiredReversePayloads,
) -> PyResult<()> {
    close_residuals(py, retired.residuals)
}

fn close_residuals(py: Python<'_>, residuals: impl IntoIterator<Item = Py<PyAny>>) -> PyResult<()> {
    let mut first_error = None;
    for residual in residuals {
        if let Err(error) = residual.bind(py).call_method0(intern!(py, "close"))
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

fn weak_rank_error() -> PyErr {
    PyValueError::new_err("only rank-zero dynamic tape values can be weak scalars")
}

fn payloads_released() -> PyErr {
    PyRuntimeError::new_err("DynamicTape has released its invocation payloads")
}

/// Releasing payloads drops every node record, so name that cause.
fn node_record_unavailable(consumed: bool) -> PyErr {
    if consumed {
        payloads_released()
    } else {
        PyRuntimeError::new_err("dynamic tape node record is unavailable")
    }
}

fn raw_arena_error(error: RawArenaError) -> PyErr {
    PyValueError::new_err(error.into_message())
}
