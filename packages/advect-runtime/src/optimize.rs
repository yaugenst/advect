//! Fixed conservative cleanup for durable staged graphs.

use std::collections::{BTreeMap, HashMap};

use crate::{
    AttrMap, AttrValue, GraphError, GraphStore, NodeCore, NodeId, NodeMetadata, PortableConstant,
    RawArena, SchemaVersion,
};

const TRANSPOSE: &str = "array.transpose";

/// Metrics for one fixed cleanup pass.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PassReport {
    /// Pass name.
    pub name: &'static str,
    /// Active node count before the pass.
    pub nodes_before: usize,
    /// Active node count after the pass.
    pub nodes_after: usize,
    /// Nodes rewritten or removed by the pass.
    pub rewritten_nodes: usize,
}

impl PassReport {
    /// Number of nodes removed.
    #[must_use]
    pub const fn removed_nodes(&self) -> usize {
        self.nodes_before.saturating_sub(self.nodes_after)
    }
}

/// Aggregate fixed-cleanup metrics.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OptimizationReport {
    /// Source graph node count.
    pub nodes_before: usize,
    /// Result graph node count.
    pub nodes_after: usize,
    /// Aggregate rewrite count.
    pub rewritten_nodes: usize,
    /// Required pass reports in execution order.
    pub passes: Vec<PassReport>,
}

/// Result of the fixed cleanup pipeline.
#[derive(Debug)]
pub struct OptimizationOutcome {
    /// Optimized immutable graph.
    pub store: GraphStore,
    /// Mapping from source node IDs to optimized IDs.
    pub old_to_new: Vec<Option<NodeId>>,
    /// Cleanup diagnostics.
    pub report: OptimizationReport,
}

#[derive(Clone, Copy, Debug)]
enum Pass {
    Dce,
    Simplify,
    Cse,
}

impl Pass {
    const ORDER: [Self; 3] = [Self::Dce, Self::Simplify, Self::Cse];

    const fn name(self) -> &'static str {
        match self {
            Self::Dce => "dce",
            Self::Simplify => "simplify",
            Self::Cse => "cse",
        }
    }

    fn run(self, graph: &mut WorkingGraph) -> Result<usize, GraphError> {
        match self {
            Self::Dce => graph.prune_unreachable(),
            Self::Simplify => simplify(graph),
            Self::Cse => cse(graph),
        }
    }
}

/// Run DCE, conservative simplification, and CSE exactly once.
pub fn optimize(store: GraphStore) -> Result<OptimizationOutcome, GraphError> {
    let nodes_before = store.node_count();
    let mut graph = WorkingGraph::new(store)?;
    let mut passes = Vec::with_capacity(Pass::ORDER.len());
    let mut rewritten_nodes = 0_usize;
    for pass in Pass::ORDER {
        let pass_nodes_before = graph.node_count();
        let pass_rewrites = pass.run(&mut graph)?;
        rewritten_nodes = rewritten_nodes
            .checked_add(pass_rewrites)
            .ok_or_else(|| GraphError::new("optimizer rewrite count overflowed"))?;
        passes.push(PassReport {
            name: pass.name(),
            nodes_before: pass_nodes_before,
            nodes_after: graph.node_count(),
            rewritten_nodes: pass_rewrites,
        });
    }
    let nodes_after = graph.node_count();
    let (store, old_to_new) = graph.finish()?;
    Ok(OptimizationOutcome {
        store,
        old_to_new,
        report: OptimizationReport {
            nodes_before,
            nodes_after,
            rewritten_nodes,
            passes,
        },
    })
}

struct WorkingGraph {
    store: GraphStore,
    active: Vec<bool>,
    active_count: usize,
    aliases: Vec<Option<NodeId>>,
    output_mask: Vec<bool>,
}

impl WorkingGraph {
    fn new(store: GraphStore) -> Result<Self, GraphError> {
        let node_count = store.node_count();
        let aliases = (0..node_count)
            .map(|index| node_id(index).map(Some))
            .collect::<Result<Vec<_>, _>>()?;
        let mut output_mask = vec![false; node_count];
        for &output_id in store.outputs() {
            let slot = output_mask
                .get_mut(node_index(output_id)?)
                .ok_or_else(|| GraphError::at_node(output_id, "graph output does not exist"))?;
            *slot = true;
        }
        Ok(Self {
            store,
            active: vec![true; node_count],
            active_count: node_count,
            aliases,
            output_mask,
        })
    }

    const fn node_count(&self) -> usize {
        self.active_count
    }

    fn node_ids(&self) -> impl Iterator<Item = NodeId> + '_ {
        self.active
            .iter()
            .enumerate()
            .filter_map(|(index, &active)| active.then(|| NodeId::try_from(index).ok()).flatten())
    }

    fn node(&self, node_id: NodeId) -> Result<NodeCore, GraphError> {
        self.store
            .arena()
            .node(node_id)
            .ok_or_else(|| GraphError::at_node(node_id, "optimizer node does not exist"))
    }

    fn metadata(&self, node_id: NodeId) -> Result<&NodeMetadata, GraphError> {
        self.store
            .metadata()
            .get(node_index(node_id)?)
            .ok_or_else(|| GraphError::at_node(node_id, "optimizer metadata is unavailable"))
    }

    fn op(&self, node_id: NodeId) -> Result<&str, GraphError> {
        let node = self.node(node_id)?;
        self.store
            .arena()
            .op_name(node.op())
            .ok_or_else(|| GraphError::at_node(node_id, "optimizer operation ID is invalid"))
    }

    fn schema_version(&self, node_id: NodeId) -> Result<SchemaVersion, GraphError> {
        let node = self.node(node_id)?;
        self.store
            .arena()
            .op_schema(node.op())
            .map(crate::OpSchema::schema_version)
            .ok_or_else(|| GraphError::at_node(node_id, "optimizer operation schema is invalid"))
    }

    fn parents(&self, node_id: NodeId) -> Result<Vec<NodeId>, GraphError> {
        let node = self.node(node_id)?;
        self.store
            .arena()
            .parents(node)
            .map(crate::Parents::to_vec)
            .ok_or_else(|| GraphError::at_node(node_id, "optimizer parent range is invalid"))
    }

    fn resolved_parents(&self, node_id: NodeId) -> Result<Vec<NodeId>, GraphError> {
        self.parents(node_id)?
            .into_iter()
            .map(|parent_id| {
                self.resolve(parent_id).ok_or_else(|| {
                    GraphError::at_node(
                        node_id,
                        format!("active node references removed parent %{parent_id}"),
                    )
                })
            })
            .collect()
    }

    fn is_output(&self, node_id: NodeId) -> bool {
        node_index(node_id)
            .ok()
            .and_then(|index| self.output_mask.get(index))
            .copied()
            .unwrap_or(false)
    }

    fn resolve(&self, mut node_id: NodeId) -> Option<NodeId> {
        for _ in 0..=self.aliases.len() {
            let index = node_index(node_id).ok()?;
            let next = self.aliases.get(index).copied().flatten()?;
            if next == node_id {
                return self
                    .active
                    .get(index)
                    .copied()
                    .unwrap_or(false)
                    .then_some(node_id);
            }
            node_id = next;
        }
        None
    }

    fn alias_node(&mut self, node_id: NodeId, replacement_id: NodeId) -> Result<(), GraphError> {
        let replacement_id = self
            .resolve(replacement_id)
            .ok_or_else(|| GraphError::at_node(replacement_id, "replacement is unavailable"))?;
        if node_id == replacement_id {
            return Ok(());
        }
        let index = node_index(node_id)?;
        let active = self
            .active
            .get_mut(index)
            .ok_or_else(|| GraphError::at_node(node_id, "activity slot is unavailable"))?;
        if !*active {
            return Err(GraphError::at_node(node_id, "node is already inactive"));
        }
        *active = false;
        self.active_count = self
            .active_count
            .checked_sub(1)
            .ok_or_else(|| GraphError::new("optimizer active-node count underflowed"))?;
        *self
            .aliases
            .get_mut(index)
            .ok_or_else(|| GraphError::at_node(node_id, "alias slot is unavailable"))? =
            Some(replacement_id);
        Ok(())
    }

    fn prune_unreachable(&mut self) -> Result<usize, GraphError> {
        let before = self.node_count();
        let reachable = self.reachable_mask()?;
        for index in 0..self.active.len() {
            if self.active.get(index).copied().unwrap_or(false)
                && !reachable.get(index).copied().unwrap_or(false)
            {
                *self
                    .active
                    .get_mut(index)
                    .ok_or_else(|| GraphError::new("optimizer activity slot is unavailable"))? =
                    false;
                self.active_count = self
                    .active_count
                    .checked_sub(1)
                    .ok_or_else(|| GraphError::new("optimizer active-node count underflowed"))?;
                *self
                    .aliases
                    .get_mut(index)
                    .ok_or_else(|| GraphError::new("optimizer alias slot is unavailable"))? = None;
            }
        }
        Ok(before.saturating_sub(self.node_count()))
    }

    fn reachable_mask(&self) -> Result<Vec<bool>, GraphError> {
        let mut reachable = vec![false; self.active.len()];
        let mut pending =
            Vec::with_capacity(self.store.inputs().len() + self.store.outputs().len());
        pending.extend(self.store.inputs().iter().copied());
        pending.extend(self.store.outputs().iter().copied());
        for node_id in self.node_ids() {
            if !is_known_pure(self.op(node_id)?) {
                pending.push(node_id);
            }
        }
        while let Some(raw_id) = pending.pop() {
            let Some(node_id) = self.resolve(raw_id) else {
                continue;
            };
            let seen = reachable
                .get_mut(node_index(node_id)?)
                .ok_or_else(|| GraphError::at_node(node_id, "reachability slot is unavailable"))?;
            if *seen {
                continue;
            }
            *seen = true;
            pending.extend(self.parents(node_id)?);
        }
        Ok(reachable)
    }

    fn effect_ancestors(&self) -> Result<Vec<bool>, GraphError> {
        let mut observed = vec![false; self.active.len()];
        let mut pending = Vec::new();
        for node_id in self.node_ids() {
            if !is_known_pure(self.op(node_id)?) {
                pending.push(node_id);
            }
        }
        while let Some(raw_id) = pending.pop() {
            let Some(node_id) = self.resolve(raw_id) else {
                continue;
            };
            let seen = observed
                .get_mut(node_index(node_id)?)
                .ok_or_else(|| GraphError::at_node(node_id, "effect slot is unavailable"))?;
            if *seen {
                continue;
            }
            *seen = true;
            pending.extend(self.parents(node_id)?);
        }
        Ok(observed)
    }

    fn finish(self) -> Result<(GraphStore, Vec<Option<NodeId>>), GraphError> {
        let old_to_new = self.old_to_new()?;
        let unchanged = self.active_count == self.active.len()
            && old_to_new
                .iter()
                .enumerate()
                .all(|(index, mapped)| *mapped == NodeId::try_from(index).ok());
        if unchanged {
            return Ok((self.store, old_to_new));
        }
        materialize(self.store, &self.active, &old_to_new)
    }

    fn old_to_new(&self) -> Result<Vec<Option<NodeId>>, GraphError> {
        let mut active_to_new = vec![None; self.active.len()];
        let mut next_id = 0_usize;
        for (index, &active) in self.active.iter().enumerate() {
            if active {
                *active_to_new
                    .get_mut(index)
                    .ok_or_else(|| GraphError::new("optimizer dense-ID slot is unavailable"))? =
                    Some(node_id(next_id)?);
                next_id = next_id
                    .checked_add(1)
                    .ok_or_else(|| GraphError::new("optimizer node count overflowed"))?;
            }
        }
        (0..self.active.len())
            .map(|index| {
                let original_id = node_id(index)?;
                Ok(self
                    .resolve(original_id)
                    .and_then(|resolved| node_index(resolved).ok())
                    .and_then(|resolved| active_to_new.get(resolved).copied().flatten()))
            })
            .collect()
    }
}

fn simplify(graph: &mut WorkingGraph) -> Result<usize, GraphError> {
    let effect_ancestors = graph.effect_ancestors()?;
    let node_ids = graph.node_ids().collect::<Vec<_>>();
    let mut rewrites = Vec::new();
    for node_id in node_ids {
        if graph.is_output(node_id)
            || effect_ancestors
                .get(node_index(node_id)?)
                .copied()
                .unwrap_or(true)
            || graph.op(node_id)? != TRANSPOSE
        {
            continue;
        }
        if let Some(replacement_id) = transpose_replacement(graph, node_id)? {
            rewrites.push((node_id, replacement_id));
        }
    }
    for &(node_id, replacement_id) in &rewrites {
        graph.alias_node(node_id, replacement_id)?;
    }
    if !rewrites.is_empty() {
        let _ = graph.prune_unreachable()?;
    }
    Ok(rewrites.len())
}

fn transpose_replacement(
    graph: &WorkingGraph,
    outer_id: NodeId,
) -> Result<Option<NodeId>, GraphError> {
    let outer_inputs = graph.resolved_parents(outer_id)?;
    let [inner_id] = outer_inputs.as_slice() else {
        return Ok(None);
    };
    if graph.op(*inner_id)? != TRANSPOSE
        || graph.schema_version(outer_id)? != graph.schema_version(*inner_id)?
    {
        return Ok(None);
    }
    let inner_inputs = graph.resolved_parents(*inner_id)?;
    let [replacement_id] = inner_inputs.as_slice() else {
        return Ok(None);
    };
    let outer_metadata = graph.metadata(outer_id)?;
    let inner_metadata = graph.metadata(*inner_id)?;
    let replacement_metadata = graph.metadata(*replacement_id)?;
    if transpose_backend(outer_metadata.attrs()) != transpose_backend(inner_metadata.attrs())
        || !safe_transpose_attrs(outer_metadata.attrs())
        || !safe_transpose_attrs(inner_metadata.attrs())
        || outer_metadata.value_spec() != replacement_metadata.value_spec()
        || outer_metadata.outputs() != replacement_metadata.outputs()
    {
        return Ok(None);
    }
    let ndim = replacement_metadata.shape().len();
    let Some(first) = normalized_axes(inner_metadata.attrs(), ndim) else {
        return Ok(None);
    };
    let Some(second) = normalized_axes(outer_metadata.attrs(), ndim) else {
        return Ok(None);
    };
    let composed = second
        .iter()
        .map(|&axis| first.get(axis).copied())
        .collect::<Option<Vec<_>>>();
    Ok(composed
        .filter(|axes| axes.iter().copied().eq(0..ndim))
        .map(|_| *replacement_id))
}

fn safe_transpose_attrs(attrs: &AttrMap) -> bool {
    attrs
        .keys()
        .all(|key| matches!(key.as_str(), "axes" | "_advect_backend"))
        && matches!(
            attrs.get("_advect_backend"),
            None | Some(AttrValue::String(_))
        )
}

fn transpose_backend(attrs: &AttrMap) -> Option<&str> {
    match attrs.get("_advect_backend") {
        Some(AttrValue::String(backend)) => Some(backend),
        _ => None,
    }
}

fn normalized_axes(attrs: &AttrMap, ndim: usize) -> Option<Vec<usize>> {
    let values = match attrs.get("axes") {
        None | Some(AttrValue::Null) => return Some((0..ndim).rev().collect()),
        Some(AttrValue::List(values) | AttrValue::Tuple(values)) => values,
        _ => return None,
    };
    if values.len() != ndim {
        return None;
    }
    let ndim_i64 = i64::try_from(ndim).ok()?;
    let mut result = Vec::with_capacity(ndim);
    for value in values {
        let AttrValue::Integer(axis) = value else {
            return None;
        };
        let normalized = if *axis < 0 {
            axis.checked_add(ndim_i64)?
        } else {
            *axis
        };
        result.push(usize::try_from(normalized).ok()?);
    }
    let mut sorted = result.clone();
    sorted.sort_unstable();
    sorted.iter().copied().eq(0..ndim).then_some(result)
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct CseKey {
    op: String,
    schema_version: SchemaVersion,
    parents: Vec<NodeId>,
    flags: (bool, bool, Option<bool>, Option<bool>),
    metadata: NodeMetadata,
}

fn cse(graph: &mut WorkingGraph) -> Result<usize, GraphError> {
    let effect_ancestors = graph.effect_ancestors()?;
    let node_ids = graph.node_ids().collect::<Vec<_>>();
    let mut expressions = HashMap::<CseKey, NodeId>::new();
    let mut rewrites = Vec::new();
    for node_id in node_ids {
        let op = graph.op(node_id)?;
        if !is_known_pure(op) {
            expressions.clear();
            continue;
        }
        let index = node_index(node_id)?;
        if graph.is_output(node_id)
            || effect_ancestors.get(index).copied().unwrap_or(true)
            || !is_cse_candidate(op)
        {
            continue;
        }
        let node = graph.node(node_id)?;
        let flags = node.flags();
        let key = CseKey {
            op: op.to_owned(),
            schema_version: graph.schema_version(node_id)?,
            parents: graph.resolved_parents(node_id)?,
            flags: (
                flags.is_input(),
                flags.is_active(),
                flags.inline_parent_is_active(0),
                flags.inline_parent_is_active(1),
            ),
            metadata: graph.metadata(node_id)?.clone(),
        };
        if let Some(&canonical_id) = expressions.get(&key) {
            rewrites.push((node_id, canonical_id));
        } else {
            expressions.insert(key, node_id);
        }
    }
    for &(node_id, canonical_id) in &rewrites {
        graph.alias_node(node_id, canonical_id)?;
    }
    Ok(rewrites.len())
}

fn is_cse_candidate(op: &str) -> bool {
    !matches!(
        op,
        "advect.input" | "advect.const" | "advect.copy" | "array.empty_like"
    )
}

fn is_known_pure(op: &str) -> bool {
    op.starts_with("array.")
        || op.starts_with("array_ext.")
        || matches!(
            op,
            "advect.input"
                | "advect.const"
                | "advect.copy"
                | "advect.getitem"
                | "advect.getoutput"
                | "advect.index_update"
        )
}

fn materialize(
    store: GraphStore,
    active: &[bool],
    old_to_new: &[Option<NodeId>],
) -> Result<(GraphStore, Vec<Option<NodeId>>), GraphError> {
    let (
        version,
        required_array_api_version,
        source_arena,
        source_metadata,
        source_inputs,
        source_outputs,
        source_constants,
    ) = store.into_parts();
    let mut arena = RawArena::default();
    preserve_op_table(&source_arena, &mut arena)?;
    let mut metadata = source_metadata
        .into_iter()
        .map(Some)
        .collect::<Vec<Option<NodeMetadata>>>();
    let mut retained_metadata = Vec::with_capacity(active.iter().filter(|&&keep| keep).count());
    for (index, &keep) in active.iter().enumerate() {
        if !keep {
            continue;
        }
        let old_id = node_id(index)?;
        let source_node = source_arena
            .node(old_id)
            .ok_or_else(|| GraphError::at_node(old_id, "optimizer source node is unavailable"))?;
        let new_parents = source_arena
            .parents(source_node)
            .ok_or_else(|| GraphError::at_node(old_id, "optimizer parent range is invalid"))?
            .iter()
            .map(|parent_id| {
                old_to_new
                    .get(node_index(parent_id)?)
                    .copied()
                    .flatten()
                    .ok_or_else(|| {
                        GraphError::at_node(
                            old_id,
                            format!("cannot materialize parent %{parent_id}"),
                        )
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        arena
            .append_op(source_node.op(), &new_parents, source_node.flags())
            .map_err(|error| GraphError::new(error.into_message()))?;
        retained_metadata.push(
            metadata
                .get_mut(index)
                .and_then(Option::take)
                .ok_or_else(|| GraphError::at_node(old_id, "optimizer metadata is unavailable"))?,
        );
    }
    let inputs = remap_endpoints(&source_inputs, old_to_new, "input")?;
    let outputs = remap_endpoints(&source_outputs, old_to_new, "output")?;
    let constants = remap_constants(source_constants, old_to_new);
    let store = GraphStore::from_parts(
        version,
        required_array_api_version,
        arena,
        retained_metadata,
        inputs,
        outputs,
        constants,
    )?;
    Ok((store, old_to_new.to_vec()))
}

fn preserve_op_table(source: &RawArena, destination: &mut RawArena) -> Result<(), GraphError> {
    for raw_op_id in 0..source.op_count() {
        let op_id = u16::try_from(raw_op_id)
            .map_err(|_| GraphError::new("optimizer operation ID overflowed"))?;
        let schema = source
            .op_schema(op_id)
            .ok_or_else(|| GraphError::new("optimizer operation table is inconsistent"))?;
        let inserted = destination
            .intern_op(schema.name(), schema.schema_version())
            .map_err(|error| GraphError::new(error.into_message()))?;
        if inserted != op_id {
            return Err(GraphError::new(
                "optimizer changed operation table ordering",
            ));
        }
    }
    Ok(())
}

fn remap_endpoints(
    endpoints: &[NodeId],
    old_to_new: &[Option<NodeId>],
    label: &str,
) -> Result<Vec<NodeId>, GraphError> {
    endpoints
        .iter()
        .map(|&node_id| {
            old_to_new
                .get(node_index(node_id)?)
                .copied()
                .flatten()
                .ok_or_else(|| {
                    GraphError::at_node(node_id, format!("optimizer removed graph {label}"))
                })
        })
        .collect()
}

fn remap_constants(
    constants: BTreeMap<NodeId, PortableConstant>,
    old_to_new: &[Option<NodeId>],
) -> BTreeMap<NodeId, PortableConstant> {
    constants
        .into_iter()
        .filter_map(|(old_id, constant)| {
            node_index(old_id)
                .ok()
                .and_then(|index| old_to_new.get(index).copied().flatten())
                .map(|new_id| (new_id, constant))
        })
        .collect()
}

fn node_id(index: usize) -> Result<NodeId, GraphError> {
    NodeId::try_from(index).map_err(|_| GraphError::new("optimizer node ID exceeded its range"))
}

fn node_index(node_id: NodeId) -> Result<usize, GraphError> {
    usize::try_from(node_id)
        .map_err(|_| GraphError::at_node(node_id, "node ID exceeds the host index range"))
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::{AttrMap, DTypeDescriptor, GRAPH_FORMAT_VERSION, GraphBuilder, NodeFlags};

    fn metadata() -> NodeMetadata {
        NodeMetadata::new(
            AttrMap::new(),
            vec![2],
            DTypeDescriptor::from_name("float32").unwrap(),
            None,
            1,
            None,
            None,
            None,
        )
        .unwrap()
    }

    #[test]
    fn fixed_pipeline_runs_three_passes_and_cses() {
        let mut builder = GraphBuilder::new(GRAPH_FORMAT_VERSION).unwrap();
        let input = builder.append_input(metadata()).unwrap();
        let first = builder
            .append_operation("array.sin", 1, &[input], NodeFlags::NONE, metadata())
            .unwrap();
        let duplicate = builder
            .append_operation("array.sin", 1, &[input], NodeFlags::NONE, metadata())
            .unwrap();
        let dead = builder
            .append_operation("array.cos", 1, &[input], NodeFlags::NONE, metadata())
            .unwrap();
        let output = builder
            .append_operation(
                "array.add",
                1,
                &[first, duplicate],
                NodeFlags::NONE,
                metadata(),
            )
            .unwrap();
        builder.append_output(output).unwrap();
        let result = builder.finish().unwrap();
        assert_eq!(
            result.old_to_new,
            [Some(0), Some(1), Some(1), None, Some(2)]
        );
        assert_eq!(
            result
                .report
                .passes
                .iter()
                .map(|pass| pass.name)
                .collect::<Vec<_>>(),
            ["dce", "simplify", "cse"]
        );
        assert_eq!(result.store.node_count(), 3);
        assert!(
            result
                .old_to_new
                .get(usize::try_from(dead).unwrap())
                .is_some()
        );
    }
}
