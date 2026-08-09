//! Structural execution planning and one-time host linking.

use std::sync::Arc;

use super::host::{Host, LinkedOperation, OutputOwnership};
use crate::{AttrMap, ExecutionError, GraphStore, NodeId, ValueSpec};

#[derive(Clone, Copy, Debug)]
pub(super) enum ValueSource {
    Input(usize),
    Constant(NodeId),
    Evaluate,
}

#[derive(Debug)]
pub(super) struct ExecutionNode {
    pub(super) id: NodeId,
    pub(super) op: String,
    pub(super) schema_version: u32,
    pub(super) parents: Vec<NodeId>,
    pub(super) attrs: AttrMap,
    pub(super) outputs: Vec<ValueSpec>,
    pub(super) source: ValueSource,
}

/// Host-independent dense execution structure.
#[derive(Debug)]
pub struct ExecutionPlan {
    pub(super) store: Arc<GraphStore>,
    pub(super) nodes: Vec<ExecutionNode>,
    pub(super) outputs: Vec<NodeId>,
    pub(super) input_count: usize,
    pub(super) remaining_uses: Vec<usize>,
}

impl ExecutionPlan {
    /// Build and validate a dense structural schedule.
    pub fn from_store(
        store: Arc<GraphStore>,
    ) -> Result<Self, ExecutionError<std::convert::Infallible>> {
        let arena = store.arena();
        if store.metadata().len() != arena.node_count() {
            return Err(ExecutionError::runtime(
                "graph metadata does not match the structural arena",
            ));
        }
        let mut sources = vec![ValueSource::Evaluate; arena.node_count()];
        for (slot, &node_id) in store.inputs().iter().enumerate() {
            let index = node_index(node_id, arena.node_count(), "input")?;
            *sources.get_mut(index).ok_or_else(|| {
                ExecutionError::runtime("graph input source slot is unavailable")
            })? = ValueSource::Input(slot);
        }
        for &node_id in store.constants().keys() {
            let index = node_index(node_id, arena.node_count(), "constant")?;
            let source = sources.get_mut(index).ok_or_else(|| {
                ExecutionError::runtime("graph constant source slot is unavailable")
            })?;
            if !matches!(*source, ValueSource::Evaluate) {
                return Err(ExecutionError::runtime(format!(
                    "graph node %{node_id} has conflicting value sources"
                )));
            }
            *source = ValueSource::Constant(node_id);
        }

        let mut nodes = Vec::with_capacity(arena.node_count());
        for (node_index, (node, metadata)) in arena
            .nodes()
            .iter()
            .copied()
            .zip(store.metadata())
            .enumerate()
        {
            let id = NodeId::try_from(node_index)
                .map_err(|_| ExecutionError::runtime("graph node ID exceeded its range"))?;
            let schema = arena.op_schema(node.op()).ok_or_else(|| {
                ExecutionError::runtime("graph operation table contains an invalid ID")
            })?;
            let parents = arena
                .parents(node)
                .ok_or_else(|| ExecutionError::runtime("graph edge range is invalid"))?
                .to_vec();
            nodes.push(ExecutionNode {
                id,
                op: schema.name().to_owned(),
                schema_version: schema.schema_version(),
                parents,
                attrs: metadata.attrs().clone(),
                outputs: metadata.outputs().to_vec(),
                source: *sources.get(node_index).ok_or_else(|| {
                    ExecutionError::runtime("graph value source slot is unavailable")
                })?,
            });
        }
        let mut remaining_uses = vec![0_usize; nodes.len()];
        for node in &nodes {
            for &parent in &node.parents {
                increment_use(&mut remaining_uses, parent)?;
            }
        }
        for &output in store.outputs() {
            increment_use(&mut remaining_uses, output)?;
        }
        let outputs = store.outputs().to_vec();
        let input_count = store.inputs().len();
        Ok(Self {
            store,
            nodes,
            outputs,
            input_count,
            remaining_uses,
        })
    }

    /// Bind each operation exactly once through a host.
    pub fn link<H: Host>(
        self,
        host: &mut H,
    ) -> Result<LinkedExecutionPlan<H::LinkedOp>, ExecutionError<H::Error>> {
        let bindings = self
            .nodes
            .iter()
            .map(|node| {
                if !matches!(node.source, ValueSource::Evaluate) {
                    return Ok(None);
                }
                let linked = host
                    .link(&node.op, node.schema_version, &node.attrs, &node.outputs)
                    .map_err(|source| ExecutionError::Host {
                        node_id: node.id,
                        op: node.op.clone(),
                        source,
                    })?;
                validate_binding(node, &linked)?;
                Ok(Some(linked))
            })
            .collect::<Result<Vec<_>, _>>()?;

        let mut alias_root_sets = (0..self.nodes.len())
            .map(|index| vec![index])
            .collect::<Vec<_>>();
        let mut owned_values = vec![false; self.nodes.len()];
        for (index, (node, binding)) in self.nodes.iter().zip(&bindings).enumerate() {
            let Some(binding) = binding else {
                continue;
            };
            match binding.output_ownership {
                OutputOwnership::Owned => {
                    *alias_root_sets.get_mut(index).ok_or_else(|| {
                        ExecutionError::runtime("staged alias-root slot is unavailable")
                    })? = vec![index];
                    *owned_values.get_mut(index).ok_or_else(|| {
                        ExecutionError::runtime("staged ownership slot is unavailable")
                    })? = true;
                }
                OutputOwnership::Alias(position) => {
                    let parent = node.parents.get(position).copied().ok_or_else(|| {
                        ExecutionError::runtime("validated alias position is unavailable")
                    })?;
                    let parent_index = node_index(parent, self.nodes.len(), "alias source")?;
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
                    for &parent in &node.parents {
                        let parent_index =
                            node_index(parent, self.nodes.len(), "unknown alias source")?;
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
        Ok(LinkedExecutionPlan {
            structure: self,
            bindings,
            alias_root_sets,
            owned_values,
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
    node: &ExecutionNode,
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
