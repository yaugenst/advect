//! Fixed conservative cleanup for durable staged graphs.

use std::collections::hash_map::Entry;
use std::collections::{BTreeMap, HashMap};

use crate::graph::GraphData;
use crate::{
    AttrMap, AttrValue, GraphError, GraphStore, NodeId, NodeMetadata, NodeRef, OpId,
    PortableConstant, RawArena,
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
            aliases,
            output_mask,
        })
    }

    fn node_count(&self) -> usize {
        self.aliases
            .iter()
            .enumerate()
            .filter(|&(index, alias)| NodeId::try_from(index).ok().as_ref() == alias.as_ref())
            .count()
    }

    fn node_ids(&self) -> impl Iterator<Item = NodeId> + '_ {
        self.aliases
            .iter()
            .enumerate()
            .filter_map(|(index, &alias)| {
                let node_id = NodeId::try_from(index).ok()?;
                (alias == Some(node_id)).then_some(node_id)
            })
    }

    fn node(&self, node_id: NodeId) -> Result<NodeRef<'_>, GraphError> {
        self.store.node(node_id)
    }

    fn resolved_parents(&self, node_id: NodeId) -> Result<Vec<NodeId>, GraphError> {
        self.node(node_id)?
            .parents
            .iter()
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
                return Some(node_id);
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
        let alias = self
            .aliases
            .get_mut(index)
            .ok_or_else(|| GraphError::at_node(node_id, "alias slot is unavailable"))?;
        if *alias != Some(node_id) {
            return Err(GraphError::at_node(node_id, "node is already inactive"));
        }
        *alias = Some(replacement_id);
        Ok(())
    }

    fn prune_unreachable(&mut self) -> Result<usize, GraphError> {
        let before = self.node_count();
        let mut roots = self.effect_roots()?;
        roots.extend_from_slice(self.store.inputs());
        roots.extend_from_slice(self.store.outputs());
        let reachable = self.ancestors(roots)?;
        for index in 0..self.aliases.len() {
            let node_id = node_id(index)?;
            if self.aliases.get(index) == Some(&Some(node_id))
                && !reachable.get(index).copied().unwrap_or(false)
            {
                *self
                    .aliases
                    .get_mut(index)
                    .ok_or_else(|| GraphError::new("optimizer alias slot is unavailable"))? = None;
            }
        }
        Ok(before.saturating_sub(self.node_count()))
    }

    /// Active nodes whose effects must be preserved.
    fn effect_roots(&self) -> Result<Vec<NodeId>, GraphError> {
        let mut roots = Vec::new();
        for node_id in self.node_ids() {
            if !is_known_pure(self.node(node_id)?.schema.name()) {
                roots.push(node_id);
            }
        }
        Ok(roots)
    }

    /// Mask of the active nodes reachable from `pending` through parents.
    fn ancestors(&self, mut pending: Vec<NodeId>) -> Result<Vec<bool>, GraphError> {
        let mut seen = vec![false; self.aliases.len()];
        while let Some(raw_id) = pending.pop() {
            let Some(node_id) = self.resolve(raw_id) else {
                continue;
            };
            let slot = seen
                .get_mut(node_index(node_id)?)
                .ok_or_else(|| GraphError::at_node(node_id, "ancestor slot is unavailable"))?;
            if !std::mem::replace(slot, true) {
                pending.extend(self.node(node_id)?.parents.iter());
            }
        }
        Ok(seen)
    }

    fn finish(self) -> Result<(GraphStore, Vec<Option<NodeId>>), GraphError> {
        let old_to_new = self.old_to_new()?;
        let unchanged = old_to_new
            .iter()
            .enumerate()
            .all(|(index, mapped)| *mapped == NodeId::try_from(index).ok());
        if unchanged {
            return Ok((self.store, old_to_new));
        }
        materialize(self.store, &self.aliases, &old_to_new)
    }

    fn old_to_new(&self) -> Result<Vec<Option<NodeId>>, GraphError> {
        let mut active_to_new = vec![None; self.aliases.len()];
        let mut next_id = 0_usize;
        for (index, &alias) in self.aliases.iter().enumerate() {
            if alias == NodeId::try_from(index).ok() {
                *active_to_new
                    .get_mut(index)
                    .ok_or_else(|| GraphError::new("optimizer dense-ID slot is unavailable"))? =
                    Some(node_id(next_id)?);
                next_id = next_id
                    .checked_add(1)
                    .ok_or_else(|| GraphError::new("optimizer node count overflowed"))?;
            }
        }
        (0..self.aliases.len())
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
    let effect_ancestors = graph.ancestors(graph.effect_roots()?)?;
    let node_ids = graph.node_ids().collect::<Vec<_>>();
    let mut rewrites = 0_usize;
    for node_id in node_ids {
        if graph.is_output(node_id)
            || effect_ancestors
                .get(node_index(node_id)?)
                .copied()
                .unwrap_or(true)
            || graph.node(node_id)?.schema.name() != TRANSPOSE
        {
            continue;
        }
        // Alias immediately so later nodes see the cancelled value in
        // topological order.
        if let Some(replacement_id) = transpose_replacement(graph, node_id)? {
            graph.alias_node(node_id, replacement_id)?;
            rewrites += 1;
        }
    }
    if rewrites > 0 {
        let _ = graph.prune_unreachable()?;
    }
    Ok(rewrites)
}

fn transpose_replacement(
    graph: &WorkingGraph,
    outer_id: NodeId,
) -> Result<Option<NodeId>, GraphError> {
    let outer_inputs = graph.resolved_parents(outer_id)?;
    let [inner_id] = outer_inputs.as_slice() else {
        return Ok(None);
    };
    let outer = graph.node(outer_id)?;
    let inner = graph.node(*inner_id)?;
    if inner.schema != outer.schema {
        return Ok(None);
    }
    let inner_inputs = graph.resolved_parents(*inner_id)?;
    let [replacement_id] = inner_inputs.as_slice() else {
        return Ok(None);
    };
    let outer_metadata = outer.metadata;
    let inner_metadata = inner.metadata;
    let replacement_metadata = graph.node(*replacement_id)?.metadata;
    if transpose_backend(outer_metadata.attrs()) != transpose_backend(inner_metadata.attrs())
        || !safe_transpose_attrs(outer_metadata.attrs())
        || !safe_transpose_attrs(inner_metadata.attrs())
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
    /// Within one arena an operation ID fixes both name and schema version.
    op: OpId,
    parents: Vec<NodeId>,
    metadata: NodeMetadata,
}

fn cse(graph: &mut WorkingGraph) -> Result<usize, GraphError> {
    let effect_ancestors = graph.ancestors(graph.effect_roots()?)?;
    let node_ids = graph.node_ids().collect::<Vec<_>>();
    let mut expressions = HashMap::<CseKey, NodeId>::with_capacity(node_ids.len());
    let mut rewrites = 0_usize;
    for node_id in node_ids {
        let node = graph.node(node_id)?;
        let op = node.schema.name();
        if !is_known_pure(op) {
            expressions.clear();
            continue;
        }
        let index = node_index(node_id)?;
        if effect_ancestors.get(index).copied().unwrap_or(true) || !is_cse_candidate(op) {
            continue;
        }
        let parents = graph.resolved_parents(node_id)?;
        let metadata = canonical_transpose_metadata(graph, op, &parents, node.metadata)?
            .unwrap_or_else(|| node.metadata.clone());
        let key = CseKey {
            op: node.op,
            parents,
            metadata,
        };
        // Value numbering in topological order: aliasing immediately lets
        // later keys resolve through this duplicate to its canonical node.
        // Outputs keep their own nodes but can absorb later duplicates.
        match expressions.entry(key) {
            Entry::Occupied(canonical) if !graph.is_output(node_id) => {
                graph.alias_node(node_id, *canonical.get())?;
                rewrites += 1;
            }
            Entry::Occupied(_) => {}
            Entry::Vacant(slot) => {
                slot.insert(node_id);
            }
        }
    }
    Ok(rewrites)
}

/// Spell a transpose's axes as one explicit permutation, so omitted, null and
/// explicit spellings of the same permutation share a CSE key.
fn canonical_transpose_metadata(
    graph: &WorkingGraph,
    op: &str,
    parents: &[NodeId],
    metadata: &NodeMetadata,
) -> Result<Option<NodeMetadata>, GraphError> {
    let (TRANSPOSE, [parent_id]) = (op, parents) else {
        return Ok(None);
    };
    let attrs = metadata.attrs();
    if !safe_transpose_attrs(attrs) {
        return Ok(None);
    }
    let ndim = graph.node(*parent_id)?.metadata.shape().len();
    let Some(axes) = normalized_axes(attrs, ndim) else {
        return Ok(None);
    };
    let mut canonical = attrs.clone();
    canonical.insert(
        "axes".to_owned(),
        AttrValue::Tuple(
            axes.into_iter()
                .map(|axis| i64::try_from(axis).map(AttrValue::Integer))
                .collect::<Result<_, _>>()
                .map_err(|_| GraphError::new("transpose axis exceeded its range"))?,
        ),
    );
    Ok(Some(metadata.with_attrs(canonical)))
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
                | "advect.scatter_add"
        )
}

fn materialize(
    store: GraphStore,
    aliases: &[Option<NodeId>],
    old_to_new: &[Option<NodeId>],
) -> Result<(GraphStore, Vec<Option<NodeId>>), GraphError> {
    let source = store.into_data();
    let source_arena = source.arena;
    let mut arena = RawArena::default();
    let mut metadata = source
        .metadata
        .into_iter()
        .map(Some)
        .collect::<Vec<Option<NodeMetadata>>>();
    let mut retained_metadata = Vec::with_capacity(
        aliases
            .iter()
            .enumerate()
            .filter(|&(index, alias)| NodeId::try_from(index).ok().as_ref() == alias.as_ref())
            .count(),
    );
    for (index, &alias) in aliases.iter().enumerate() {
        if alias != NodeId::try_from(index).ok() {
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
        let schema = source_arena.op_schema(source_node.op()).ok_or_else(|| {
            GraphError::at_node(old_id, "optimizer operation schema is unavailable")
        })?;
        arena
            .append(
                schema.name(),
                schema.schema_version(),
                &new_parents,
                source_node.flags(),
            )
            .map_err(|error| GraphError::new(error.into_message()))?;
        retained_metadata.push(
            metadata
                .get_mut(index)
                .and_then(Option::take)
                .ok_or_else(|| GraphError::at_node(old_id, "optimizer metadata is unavailable"))?,
        );
    }
    let store = GraphStore::new(GraphData {
        required_array_api_version: source.required_array_api_version,
        arena,
        metadata: retained_metadata,
        inputs: remap_endpoints(&source.inputs, old_to_new, "input")?,
        outputs: remap_endpoints(&source.outputs, old_to_new, "output")?,
        constants: remap_constants(source.constants, old_to_new),
    })?;
    Ok((store, old_to_new.to_vec()))
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
#[expect(
    clippy::unwrap_used,
    reason = "test setup unwraps values whose absence should fail the test"
)]
mod tests {
    use super::*;
    use crate::GraphBuilder;
    use crate::test_support;

    fn metadata() -> NodeMetadata {
        test_support::metadata(vec![2], "float32", AttrMap::new())
    }

    #[test]
    fn fixed_pipeline_runs_three_passes_and_cses() {
        let mut builder = GraphBuilder::new();
        let input = builder.append_input(metadata()).unwrap();
        let first = builder
            .append_operation("array.sin", 1, &[input], metadata())
            .unwrap();
        let duplicate = builder
            .append_operation("array.sin", 1, &[input], metadata())
            .unwrap();
        builder
            .append_operation("array.cos", 1, &[input], metadata())
            .unwrap();
        let output = builder
            .append_operation("array.add", 1, &[first, duplicate], metadata())
            .unwrap();
        builder.append_output(output).unwrap();
        let result = builder.finish().unwrap();
        assert_eq!(
            result.old_to_new,
            [Some(0), Some(1), Some(1), None, Some(2)]
        );
        assert_eq!(result.store.node_count(), 3);
        let report = &result.report;
        assert_eq!(
            (
                report.nodes_before,
                report.nodes_after,
                report.rewritten_nodes
            ),
            (5, 3, 2)
        );
        assert_eq!(
            report
                .passes
                .iter()
                .map(|pass| (
                    pass.name,
                    pass.nodes_before,
                    pass.nodes_after,
                    pass.removed_nodes(),
                    pass.rewritten_nodes
                ))
                .collect::<Vec<_>>(),
            [
                ("dce", 5, 4, 1, 1),
                ("simplify", 4, 4, 0, 0),
                ("cse", 4, 3, 1, 1)
            ]
        );
    }

    #[test]
    fn scatter_add_is_eliminated_and_merged_as_a_pure_operation() {
        let mut builder = GraphBuilder::new();
        let values = builder.append_input(metadata()).unwrap();
        let indices = builder.append_input(metadata()).unwrap();
        let mut scatter = |parents: &[NodeId]| {
            builder
                .append_operation("advect.scatter_add", 1, parents, metadata())
                .unwrap()
        };
        let (first, duplicate) = (scatter(&[values, indices]), scatter(&[values, indices]));
        scatter(&[duplicate, indices]);
        let output = builder
            .append_operation("array.add", 1, &[first, duplicate], metadata())
            .unwrap();
        builder.append_output(output).unwrap();

        let result = builder.finish().unwrap();

        assert_eq!(
            result.old_to_new,
            [Some(0), Some(1), Some(2), Some(2), None, Some(3)]
        );
    }

    fn assert_optimization_is_idempotent(store: GraphStore) {
        let node_count = store.node_count();
        let report = optimize(store).unwrap().report;
        assert_eq!(
            (report.nodes_before, report.nodes_after),
            (node_count, node_count)
        );
        assert_eq!(report.rewritten_nodes, 0);
    }

    #[test]
    fn cse_merges_cascaded_duplicates_in_one_pass() {
        let mut builder = GraphBuilder::new();
        let input = builder.append_input(metadata()).unwrap();
        let mut branches = Vec::new();
        for _ in 0..2 {
            let sin = builder
                .append_operation("array.sin", 1, &[input], metadata())
                .unwrap();
            branches.push(
                builder
                    .append_operation("array.cos", 1, &[sin], metadata())
                    .unwrap(),
            );
        }
        let output = builder
            .append_operation("array.add", 1, &branches, metadata())
            .unwrap();
        builder.append_output(output).unwrap();

        let result = builder.finish().unwrap();

        assert_eq!(result.store.node_count(), 4);
        assert_eq!(
            result.old_to_new,
            [Some(0), Some(1), Some(2), Some(1), Some(2), Some(3)]
        );
        assert_optimization_is_idempotent(result.store);
    }

    #[test]
    fn cse_merges_duplicates_into_earlier_outputs_but_keeps_outputs_distinct() {
        let mut builder = GraphBuilder::new();
        let input = builder.append_input(metadata()).unwrap();
        let mut sin = || {
            builder
                .append_operation("array.sin", 1, &[input], metadata())
                .unwrap()
        };
        let (first, second, internal) = (sin(), sin(), sin());
        let output = builder
            .append_operation("array.cos", 1, &[internal], metadata())
            .unwrap();
        for node in [first, second, output] {
            builder.append_output(node).unwrap();
        }

        let result = builder.finish().unwrap();

        assert_eq!(
            result.old_to_new,
            [Some(0), Some(1), Some(2), Some(1), Some(3)]
        );
        assert_eq!(result.store.outputs(), [1, 2, 3]);
        assert_optimization_is_idempotent(result.store);
    }

    fn transpose_metadata(shape: Vec<usize>, axes: Option<[i64; 3]>) -> NodeMetadata {
        let mut attrs = AttrMap::new();
        if let Some(axes) = axes {
            attrs.insert(
                "axes".to_owned(),
                AttrValue::Tuple(axes.into_iter().map(AttrValue::Integer).collect()),
            );
        }
        test_support::metadata(shape, "float32", attrs)
    }

    #[test]
    fn simplify_cancels_transposes_exposed_by_an_earlier_cancellation() {
        const SWAP: [i64; 3] = [1, 0, 2];
        let mut builder = GraphBuilder::new();
        let input = builder
            .append_input(transpose_metadata(vec![2, 3, 4], None))
            .unwrap();
        let mut transpose = |parent, shape, axes| {
            builder
                .append_operation(
                    TRANSPOSE,
                    1,
                    &[parent],
                    transpose_metadata(shape, Some(axes)),
                )
                .unwrap()
        };
        let swapped = transpose(input, vec![3, 2, 4], SWAP);
        let rotated = transpose(swapped, vec![2, 4, 3], [1, 2, 0]);
        // Cancels `rotated`, which exposes `swapped` as the parent of `restored`.
        let unrotated = transpose(rotated, vec![3, 2, 4], [2, 0, 1]);
        let restored = transpose(unrotated, vec![2, 3, 4], SWAP);
        let output = builder
            .append_operation(
                "array.sin",
                1,
                &[restored],
                transpose_metadata(vec![2, 3, 4], None),
            )
            .unwrap();
        builder.append_output(output).unwrap();

        let result = builder.finish().unwrap();

        assert_eq!(result.store.node_count(), 2);
        assert_eq!(result.store.outputs(), [1]);
        assert_optimization_is_idempotent(result.store);
    }

    #[test]
    fn simplify_and_cse_recover_a_transpose_cancellation_hidden_by_an_output() {
        let mut builder = GraphBuilder::new();
        let input = builder
            .append_input(transpose_metadata(vec![2, 3], None))
            .unwrap();
        let mut transpose = |parent, shape: [usize; 2]| {
            builder
                .append_operation(
                    TRANSPOSE,
                    1,
                    &[parent],
                    transpose_metadata(shape.into(), None),
                )
                .unwrap()
        };
        let first = transpose(input, [3, 2]);
        let cancelled = transpose(first, [2, 3]);
        let duplicate = transpose(cancelled, [3, 2]);
        let last = transpose(duplicate, [2, 3]);
        builder.append_output(first).unwrap();
        builder.append_output(last).unwrap();

        let result = builder.finish().unwrap();

        // `cancelled` resolves to the input, so `duplicate` repeats `first`,
        // an output that CSE keeps and merges `duplicate` into.
        assert_eq!(
            result.old_to_new,
            [Some(0), Some(1), Some(0), Some(1), Some(2)]
        );
        assert_optimization_is_idempotent(result.store);
    }

    #[test]
    fn cse_merges_transposes_that_spell_one_permutation_differently() {
        let square = |attrs| test_support::metadata(vec![3, 3], "float32", attrs);
        let mut builder = GraphBuilder::new();
        let input = builder.append_input(square(AttrMap::new())).unwrap();
        let swapped = builder
            .append_operation(
                TRANSPOSE,
                1,
                &[input],
                square(attrs([("axes", axes(&[1, 0]))])),
            )
            .unwrap();
        // Four reversals of `swapped`: simplify cancels two pairs and leaves
        // two reversals of the input, which repeat `swapped` with axes omitted.
        let mut value = swapped;
        for _ in 0..4 {
            value = builder
                .append_operation(TRANSPOSE, 1, &[value], square(AttrMap::new()))
                .unwrap();
        }
        let output = builder
            .append_operation("array.sin", 1, &[value], square(AttrMap::new()))
            .unwrap();
        builder.append_output(swapped).unwrap();
        builder.append_output(output).unwrap();

        let result = builder.finish().unwrap();

        assert_eq!(result.store.node_count(), 3);
        assert_eq!(result.store.outputs(), [1, 2]);
        assert_eq!(result.store.node(2).unwrap().parents.to_vec(), [1]);
        assert_optimization_is_idempotent(result.store);
    }

    #[test]
    fn a_transpose_of_another_operation_is_kept() {
        // Omitted axes read as a reversal; only the schema check keeps this
        // reversal from cancelling against `array.sin` as if it were one.
        let cube = || test_support::metadata(vec![2, 2, 2], "float32", AttrMap::new());
        let mut builder = GraphBuilder::new();
        let input = builder.append_input(cube()).unwrap();
        let sin = builder
            .append_operation("array.sin", 1, &[input], cube())
            .unwrap();
        let reversed = builder
            .append_operation(TRANSPOSE, 1, &[sin], cube())
            .unwrap();
        let output = builder
            .append_operation("array.cos", 1, &[reversed], cube())
            .unwrap();
        builder.append_output(output).unwrap();

        let result = builder.finish().unwrap();

        assert_eq!(result.old_to_new, [Some(0), Some(1), Some(2), Some(3)]);
    }

    /// How the graph uses a transpose pair `input -> inner -> outer`.
    #[derive(Clone, Copy, Debug)]
    enum PairUse {
        /// `outer -> array.sin -> output`.
        Pure,
        /// `outer` is itself a graph output.
        Output,
        /// `outer -> custom.effect -> output`.
        Effect,
        /// Like `Pure`, and `inner` is a graph output too.
        SharedInner,
    }

    /// Optimize `input -> inner -> outer` over a 2x2x2 input, so every axis
    /// order has the same shape, and return where `inner` and `outer` went.
    /// The input is always `%0`.
    fn optimize_transpose_pair(
        inner: AttrMap,
        outer: AttrMap,
        outer_dtype: &str,
        pair_use: PairUse,
    ) -> (Option<NodeId>, Option<NodeId>) {
        let cube = |dtype, attrs| test_support::metadata(vec![2, 2, 2], dtype, attrs);
        let mut builder = GraphBuilder::new();
        let input = builder
            .append_input(cube("float32", AttrMap::new()))
            .unwrap();
        let inner_id = builder
            .append_operation(TRANSPOSE, 1, &[input], cube("float32", inner))
            .unwrap();
        let outer_id = builder
            .append_operation(TRANSPOSE, 1, &[inner_id], cube(outer_dtype, outer))
            .unwrap();
        let consumer = match pair_use {
            PairUse::Output => None,
            PairUse::Effect => Some("custom.effect"),
            PairUse::Pure | PairUse::SharedInner => Some("array.sin"),
        };
        let output = consumer.map_or(outer_id, |op| {
            builder
                .append_operation(op, 1, &[outer_id], cube(outer_dtype, AttrMap::new()))
                .unwrap()
        });
        builder.append_output(output).unwrap();
        if matches!(pair_use, PairUse::SharedInner) {
            builder.append_output(inner_id).unwrap();
        }
        let old_to_new = builder.finish().unwrap().old_to_new;
        let remapped = |node_id: NodeId| {
            old_to_new
                .get(usize::try_from(node_id).unwrap())
                .copied()
                .flatten()
        };
        (remapped(inner_id), remapped(outer_id))
    }

    fn attrs<const N: usize>(entries: [(&str, AttrValue); N]) -> AttrMap {
        entries
            .into_iter()
            .map(|(key, value)| (key.to_owned(), value))
            .collect()
    }

    fn axes(axes: &[i64]) -> AttrValue {
        AttrValue::Tuple(axes.iter().copied().map(AttrValue::Integer).collect())
    }

    fn backend(name: &str) -> AttrValue {
        AttrValue::String(name.to_owned())
    }

    const ROTATE: [i64; 3] = [1, 2, 0];
    const UNROTATE: [i64; 3] = [2, 0, 1];

    fn rotate() -> AttrMap {
        attrs([("axes", axes(&ROTATE))])
    }

    fn unrotate() -> AttrMap {
        attrs([("axes", axes(&UNROTATE))])
    }

    #[test]
    fn transpose_pairs_that_compose_to_the_identity_cancel() {
        let negative_list = AttrValue::List(vec![
            AttrValue::Integer(-2),
            AttrValue::Integer(-1),
            AttrValue::Integer(-3),
        ]);
        let with_backend = |axes| attrs([("axes", axes), ("_advect_backend", backend("numpy"))]);
        let cases = [
            ("inverse permutations", rotate(), unrotate(), PairUse::Pure),
            (
                "omitted axes",
                AttrMap::new(),
                AttrMap::new(),
                PairUse::Pure,
            ),
            (
                "null and omitted axes",
                attrs([("axes", AttrValue::Null)]),
                AttrMap::new(),
                PairUse::Pure,
            ),
            (
                "negative axes in a list",
                attrs([("axes", negative_list)]),
                unrotate(),
                PairUse::Pure,
            ),
            (
                "same backend",
                with_backend(axes(&ROTATE)),
                with_backend(axes(&UNROTATE)),
                PairUse::Pure,
            ),
            (
                "inner transpose has another use",
                rotate(),
                unrotate(),
                PairUse::SharedInner,
            ),
        ];
        for (label, inner, outer, pair_use) in cases {
            let (inner_id, outer_id) = optimize_transpose_pair(inner, outer, "float32", pair_use);
            assert_eq!(outer_id, Some(0), "{label}");
            let shared = matches!(pair_use, PairUse::SharedInner);
            assert_eq!(inner_id.is_some(), shared, "{label}");
        }
    }

    #[test]
    fn transpose_pairs_are_kept_unless_provably_the_identity() {
        let unrotate_with = |key, value| attrs([("axes", axes(&UNROTATE)), (key, value)]);
        let string_axis = AttrValue::Tuple(vec![
            AttrValue::Integer(2),
            AttrValue::Integer(0),
            AttrValue::String("1".to_owned()),
        ]);
        let cases = [
            (
                "non-inverse permutations",
                rotate(),
                rotate(),
                PairUse::Pure,
            ),
            (
                "different backends",
                attrs([
                    ("axes", axes(&ROTATE)),
                    ("_advect_backend", backend("numpy")),
                ]),
                unrotate_with("_advect_backend", backend("cupy")),
                PairUse::Pure,
            ),
            (
                "backend on one side",
                rotate(),
                unrotate_with("_advect_backend", backend("numpy")),
                PairUse::Pure,
            ),
            (
                "non-string backends",
                attrs([
                    ("axes", axes(&ROTATE)),
                    ("_advect_backend", AttrValue::Integer(1)),
                ]),
                unrotate_with("_advect_backend", AttrValue::Integer(1)),
                PairUse::Pure,
            ),
            (
                "extra inner attribute",
                attrs([("axes", axes(&ROTATE)), ("copy", AttrValue::Bool(true))]),
                unrotate(),
                PairUse::Pure,
            ),
            (
                "extra outer attribute",
                rotate(),
                unrotate_with("copy", AttrValue::Bool(true)),
                PairUse::Pure,
            ),
            (
                "non-integer axis",
                rotate(),
                attrs([("axes", string_axis)]),
                PairUse::Pure,
            ),
            (
                "duplicate axes",
                attrs([("axes", axes(&[1, 1, 0]))]),
                unrotate(),
                PairUse::Pure,
            ),
            (
                "out-of-range axis",
                attrs([("axes", axes(&[1, 2, -4]))]),
                unrotate(),
                PairUse::Pure,
            ),
            (
                "axes of another rank",
                attrs([("axes", axes(&[1, 0]))]),
                attrs([("axes", axes(&[1, 0]))]),
                PairUse::Pure,
            ),
            (
                "outer transpose is an output",
                rotate(),
                unrotate(),
                PairUse::Output,
            ),
            (
                "pair feeds an effect",
                rotate(),
                unrotate(),
                PairUse::Effect,
            ),
        ];
        for (label, inner, outer, pair_use) in cases {
            let (inner_id, outer_id) = optimize_transpose_pair(inner, outer, "float32", pair_use);
            assert_eq!((inner_id, outer_id), (Some(1), Some(2)), "{label}");
        }

        // The input must also carry the outer transpose's exact metadata.
        let (_, outer_id) = optimize_transpose_pair(rotate(), unrotate(), "float64", PairUse::Pure);
        assert_eq!(outer_id, Some(2));
    }
}
