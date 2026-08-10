//! Structural execution planning and one-time host linking.

use std::sync::Arc;

use super::host::{Host, LinkedOperation, OutputOwnership};
use crate::{AttrMap, ExecutionError, GraphStore, NodeId, Parents, ValueSpec};

#[derive(Clone, Copy, Debug)]
pub(super) enum ValueSource {
    Input(usize),
    Constant(NodeId),
    Evaluate,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct NodeView<'a> {
    pub(super) id: NodeId,
    pub(super) op: &'a str,
    pub(super) schema_version: u32,
    pub(super) parents: Parents<'a>,
    pub(super) attrs: &'a AttrMap,
    pub(super) outputs: &'a [ValueSpec],
    pub(super) source: ValueSource,
}

#[derive(Debug)]
pub(super) struct ExecutionPlan {
    pub(super) store: Arc<GraphStore>,
    pub(super) sources: Vec<ValueSource>,
    pub(super) input_count: usize,
    pub(super) remaining_uses: Vec<usize>,
}

impl ExecutionPlan {
    fn from_store<E>(store: Arc<GraphStore>) -> Result<Self, ExecutionError<E>> {
        let node_count = store.node_count();
        let mut sources = vec![ValueSource::Evaluate; node_count];
        for (slot, &node_id) in store.inputs().iter().enumerate() {
            let index = node_index(node_id, node_count, "input")?;
            *sources
                .get_mut(index)
                .ok_or_else(|| ExecutionError::runtime("graph input source is unavailable"))? =
                ValueSource::Input(slot);
        }
        for &node_id in store.constants().keys() {
            let index = node_index(node_id, node_count, "constant")?;
            *sources
                .get_mut(index)
                .ok_or_else(|| ExecutionError::runtime("graph constant source is unavailable"))? =
                ValueSource::Constant(node_id);
        }

        let mut remaining_uses = vec![0_usize; node_count];
        for &node in store.arena().nodes() {
            let parents = store
                .arena()
                .parents(node)
                .ok_or_else(|| ExecutionError::runtime("graph edge range is invalid"))?;
            for parent in parents.iter() {
                increment_use(&mut remaining_uses, parent)?;
            }
        }
        for &output in store.outputs() {
            increment_use(&mut remaining_uses, output)?;
        }
        let input_count = store.inputs().len();
        Ok(Self {
            store,
            sources,
            input_count,
            remaining_uses,
        })
    }

    pub(super) fn node<E>(&self, index: usize) -> Result<NodeView<'_>, ExecutionError<E>> {
        let id = NodeId::try_from(index)
            .map_err(|_| ExecutionError::runtime("graph node ID exceeded its range"))?;
        let node = self
            .store
            .arena()
            .node(id)
            .ok_or_else(|| ExecutionError::runtime("graph node is unavailable"))?;
        let schema = self
            .store
            .arena()
            .op_schema(node.op())
            .ok_or_else(|| ExecutionError::runtime("graph operation ID is invalid"))?;
        let parents = self
            .store
            .arena()
            .parents(node)
            .ok_or_else(|| ExecutionError::runtime("graph edge range is invalid"))?;
        let metadata = self
            .store
            .metadata()
            .get(index)
            .ok_or_else(|| ExecutionError::runtime("graph metadata is unavailable"))?;
        let source = *self
            .sources
            .get(index)
            .ok_or_else(|| ExecutionError::runtime("graph value source is unavailable"))?;
        Ok(NodeView {
            id,
            op: schema.name(),
            schema_version: schema.schema_version(),
            parents,
            attrs: metadata.attrs(),
            outputs: metadata.outputs(),
            source,
        })
    }
}

/// Immutable prelinked plan reused across invocations.
#[derive(Debug)]
pub struct LinkedExecutionPlan<T> {
    pub(super) structure: ExecutionPlan,
    pub(super) bindings: Vec<Option<LinkedOperation<T>>>,
    pub(super) alias_root_sets: Vec<Vec<usize>>,
    pub(super) owned_values: Vec<bool>,
}

impl<T> LinkedExecutionPlan<T> {
    /// Build the dense schedule and bind each operation once through a host.
    pub fn from_store<H>(
        store: Arc<GraphStore>,
        host: &mut H,
    ) -> Result<Self, ExecutionError<H::Error>>
    where
        H: Host<LinkedOp = T>,
    {
        let structure = ExecutionPlan::from_store(store)?;
        let node_count = structure.store.node_count();
        let mut bindings = Vec::with_capacity(node_count);
        for index in 0..node_count {
            let node = structure.node(index)?;
            if !matches!(node.source, ValueSource::Evaluate) {
                bindings.push(None);
                continue;
            }
            let linked = host
                .link(node.op, node.schema_version, node.attrs, node.outputs)
                .map_err(|source| ExecutionError::Host {
                    node_id: node.id,
                    op: node.op.to_owned(),
                    source,
                })?;
            validate_binding(node, &linked)?;
            bindings.push(Some(linked));
        }

        let mut alias_root_sets = (0..node_count).map(|index| vec![index]).collect::<Vec<_>>();
        let mut owned_values = vec![false; node_count];
        for (index, binding) in bindings.iter().enumerate() {
            let Some(binding) = binding else {
                continue;
            };
            let node = structure.node(index)?;
            match binding.output_ownership {
                OutputOwnership::Owned => {
                    *owned_values.get_mut(index).ok_or_else(|| {
                        ExecutionError::runtime("staged ownership slot is unavailable")
                    })? = true;
                }
                OutputOwnership::Alias(position) => {
                    let parent = node.parents.get(position).ok_or_else(|| {
                        ExecutionError::runtime("validated alias position is unavailable")
                    })?;
                    let parent_index = node_index(parent, node_count, "alias source")?;
                    let parent_roots = alias_root_sets
                        .get(parent_index)
                        .ok_or_else(|| {
                            ExecutionError::runtime("staged alias-root set is unavailable")
                        })?
                        .clone();
                    *alias_root_sets.get_mut(index).ok_or_else(|| {
                        ExecutionError::runtime("staged alias slot is unavailable")
                    })? = parent_roots;
                }
                OutputOwnership::Unknown => {
                    let mut roots = vec![index];
                    for parent in node.parents.iter() {
                        let parent_index = node_index(parent, node_count, "unknown alias source")?;
                        roots.extend_from_slice(alias_root_sets.get(parent_index).ok_or_else(
                            || ExecutionError::runtime("staged alias-root set is unavailable"),
                        )?);
                    }
                    roots.sort_unstable();
                    roots.dedup();
                    *alias_root_sets.get_mut(index).ok_or_else(|| {
                        ExecutionError::runtime("staged alias slot is unavailable")
                    })? = roots;
                }
            }
        }
        Ok(Self {
            structure,
            bindings,
            alias_root_sets,
            owned_values,
        })
    }

    /// Number of constants.
    #[must_use]
    pub fn constant_count(&self) -> usize {
        self.structure.store.constants().len()
    }

    /// Portable constant IDs in materialization order.
    pub fn constant_ids(&self) -> impl Iterator<Item = NodeId> + '_ {
        self.structure.store.constants().keys().copied()
    }
}

fn validate_binding<T, E>(
    node: NodeView<'_>,
    binding: &LinkedOperation<T>,
) -> Result<(), ExecutionError<E>> {
    if binding
        .donation_positions
        .iter()
        .any(|&position| position >= node.parents.len())
    {
        return Err(ExecutionError::runtime(format!(
            "linked operation '{}' at node %{} declares an invalid donation position",
            node.op, node.id
        )));
    }
    if let OutputOwnership::Alias(position) = binding.output_ownership
        && position >= node.parents.len()
    {
        return Err(ExecutionError::runtime(format!(
            "linked operation '{}' at node %{} declares an invalid alias position",
            node.op, node.id
        )));
    }
    Ok(())
}

fn increment_use<E>(
    remaining_uses: &mut [usize],
    node_id: NodeId,
) -> Result<(), ExecutionError<E>> {
    let index = node_index(node_id, remaining_uses.len(), "use")?;
    let count = remaining_uses
        .get_mut(index)
        .ok_or_else(|| ExecutionError::runtime("staged use-count slot is unavailable"))?;
    *count = count
        .checked_add(1)
        .ok_or_else(|| ExecutionError::runtime("staged use count overflowed"))?;
    Ok(())
}

pub(super) fn node_index<E>(
    node_id: NodeId,
    node_count: usize,
    role: &str,
) -> Result<usize, ExecutionError<E>> {
    let index = usize::try_from(node_id)
        .map_err(|_| ExecutionError::runtime(format!("graph {role} node ID exceeded its range")))?;
    if index >= node_count {
        return Err(ExecutionError::runtime(format!(
            "graph {role} node %{node_id} does not exist"
        )));
    }
    Ok(index)
}
