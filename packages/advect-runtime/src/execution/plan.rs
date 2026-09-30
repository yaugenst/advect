//! Structural execution planning and one-time host linking.

use std::sync::Arc;

use super::host::{Host, LinkedOperation, OutputOwnership};
use crate::{ExecutionError, GraphStore, NodeId, NodeRef};

/// How one node obtains its value during execution.
#[derive(Debug)]
pub(super) enum Step<T> {
    Input,
    Constant,
    Evaluate(LinkedOperation<T>),
}

/// Immutable prelinked plan reused across invocations.
#[derive(Debug)]
pub struct LinkedExecutionPlan<T> {
    pub(super) store: Arc<GraphStore>,
    pub(super) steps: Vec<Step<T>>,
    pub(super) remaining_uses: Vec<usize>,
    pub(super) alias_root_sets: Vec<Vec<usize>>,
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
        let node_count = store.node_count();
        let mut steps = Vec::with_capacity(node_count);
        let mut remaining_uses = vec![0_usize; node_count];
        // Each value lists the owned values whose storage it may share. Only
        // owned values are donation candidates, so inputs, constants, and
        // unknown results need no roots of their own.
        let mut alias_root_sets = Vec::with_capacity(node_count);
        for node in store.nodes() {
            let node = node?;
            for parent in node.parents.iter() {
                increment_use(&mut remaining_uses, parent)?;
            }
            let (step, roots) = match node.schema.name() {
                "advect.input" => (Step::Input, Vec::new()),
                "advect.const" => (Step::Constant, Vec::new()),
                op => {
                    let linked = host
                        .link(
                            op,
                            node.schema.schema_version(),
                            node.metadata.attrs(),
                            node.metadata.outputs(),
                        )
                        .map_err(|source| host_error(node, source))?;
                    validate_binding(node, &linked)?;
                    let roots = alias_roots(&alias_root_sets, node, linked.output_ownership)?;
                    (Step::Evaluate(linked), roots)
                }
            };
            steps.push(step);
            alias_root_sets.push(roots);
        }
        for &output in store.outputs() {
            increment_use(&mut remaining_uses, output)?;
        }
        Ok(Self {
            store,
            steps,
            remaining_uses,
            alias_root_sets,
        })
    }

    /// Number of constants.
    #[must_use]
    pub fn constant_count(&self) -> usize {
        self.store.constants().len()
    }

    /// Portable constant IDs in materialization order.
    pub fn constant_ids(&self) -> impl Iterator<Item = NodeId> + '_ {
        self.store.constants().keys().copied()
    }
}

fn alias_roots<E>(
    alias_root_sets: &[Vec<usize>],
    node: NodeRef<'_>,
    ownership: OutputOwnership,
) -> Result<Vec<usize>, ExecutionError<E>> {
    let parent_roots = |parent| {
        alias_root_sets
            .get(slot(parent)?)
            .ok_or_else(|| ExecutionError::runtime("staged alias-root set is unavailable"))
    };
    Ok(match ownership {
        OutputOwnership::Owned => vec![slot(node.id)?],
        OutputOwnership::Alias(position) => {
            let parent = node.parents.get(position).ok_or_else(|| {
                ExecutionError::runtime("validated alias position is unavailable")
            })?;
            parent_roots(parent)?.clone()
        }
        OutputOwnership::Unknown => {
            let mut roots = Vec::new();
            for parent in node.parents.iter() {
                roots.extend_from_slice(parent_roots(parent)?);
            }
            roots.sort_unstable();
            roots.dedup();
            roots
        }
    })
}

fn validate_binding<T, E>(
    node: NodeRef<'_>,
    binding: &LinkedOperation<T>,
) -> Result<(), ExecutionError<E>> {
    let invalid = |role| {
        Err(ExecutionError::runtime(format!(
            "linked operation '{}' at node %{} declares an invalid {role} position",
            node.schema.name(),
            node.id
        )))
    };
    if binding
        .donation_positions
        .iter()
        .any(|&position| position >= node.parents.len())
    {
        return invalid("donation");
    }
    if let OutputOwnership::Alias(position) = binding.output_ownership
        && position >= node.parents.len()
    {
        return invalid("alias");
    }
    Ok(())
}

fn increment_use<E>(
    remaining_uses: &mut [usize],
    node_id: NodeId,
) -> Result<(), ExecutionError<E>> {
    let count = remaining_uses
        .get_mut(slot(node_id)?)
        .ok_or_else(|| ExecutionError::runtime("staged use-count slot is unavailable"))?;
    *count = count
        .checked_add(1)
        .ok_or_else(|| ExecutionError::runtime("staged use count overflowed"))?;
    Ok(())
}

/// Dense storage slot of one node.
pub(super) fn slot<E>(node_id: NodeId) -> Result<usize, ExecutionError<E>> {
    usize::try_from(node_id)
        .map_err(|_| ExecutionError::runtime(format!("graph node %{node_id} exceeded its range")))
}

/// Attribute a host failure to the node that caused it.
pub(super) fn host_error<E>(node: NodeRef<'_>, source: E) -> ExecutionError<E> {
    ExecutionError::Host {
        node_id: node.id,
        op: node.schema.name().to_owned(),
        source,
    }
}
