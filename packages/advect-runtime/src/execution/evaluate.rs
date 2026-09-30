//! Invocation-local evaluation, value lifetimes, aliasing, and donation.

use super::host::{Host, LinkedOperation, Operand};
use super::plan::{LinkedExecutionPlan, Step, host_error, slot};
use crate::{ExecutionError, NodeRef};

impl<T> LinkedExecutionPlan<T> {
    /// Execute once with invocation-local dense storage.
    pub fn execute<H>(
        &self,
        host: &mut H,
        inputs: Vec<H::Value>,
    ) -> Result<Vec<H::Value>, ExecutionError<H::Error>>
    where
        H: Host<LinkedOp = T>,
    {
        let input_ids = self.store.inputs();
        if inputs.len() != input_ids.len() {
            return Err(ExecutionError::runtime(format!(
                "staged graph expects {} inputs but received {}",
                input_ids.len(),
                inputs.len()
            )));
        }
        let node_count = self.steps.len();
        let mut values: Vec<Option<H::Value>> = (0..node_count).map(|_| None).collect();
        for (value, &node_id) in inputs.into_iter().zip(input_ids) {
            *values.get_mut(slot(node_id)?).ok_or_else(|| {
                ExecutionError::runtime("staged graph input slot is unavailable")
            })? = Some(value);
        }
        let mut remaining_uses = self.remaining_uses.clone();
        let mut live_aliases = vec![0_usize; node_count];

        for ((current_index, step), node) in self.steps.iter().enumerate().zip(self.store.nodes()) {
            let node = node?;
            let value = match step {
                Step::Input => values
                    .get_mut(current_index)
                    .and_then(Option::take)
                    .ok_or_else(|| {
                        ExecutionError::runtime(format!(
                            "staged graph input %{} is unavailable",
                            node.id
                        ))
                    })?,
                Step::Constant => {
                    let constant = self.store.constants().get(&node.id).ok_or_else(|| {
                        ExecutionError::runtime(format!(
                            "staged graph constant %{} is unavailable",
                            node.id
                        ))
                    })?;
                    host.materialize_constant(node.id, constant)
                        .map_err(|source| host_error(node, source))?
                }
                Step::Evaluate(binding) => {
                    let donated = take_donation(
                        self,
                        binding,
                        node,
                        &mut values,
                        &remaining_uses,
                        &mut live_aliases,
                    )?;
                    let operands = collect_operands(&values, node, donated)?;
                    host.evaluate(&binding.implementation, operands)
                        .map_err(|source| host_error(node, source))?
                }
            };
            host.validate_value(&value, node.metadata.outputs())
                .map_err(|source| host_error(node, source))?;

            *values
                .get_mut(current_index)
                .ok_or_else(|| ExecutionError::runtime("staged value slot is unavailable"))? =
                Some(value);
            shift_live_aliases(
                &self.alias_root_sets,
                &mut live_aliases,
                current_index,
                true,
            )?;

            for parent in node.parents.iter() {
                let parent_index = slot(parent)?;
                let remaining = remaining_uses.get_mut(parent_index).ok_or_else(|| {
                    ExecutionError::runtime("staged use-count slot is unavailable")
                })?;
                *remaining = remaining.checked_sub(1).ok_or_else(|| {
                    ExecutionError::runtime(format!(
                        "staged value %{parent} was consumed more often than planned"
                    ))
                })?;
                if *remaining == 0 {
                    release_value(
                        &mut values,
                        &self.alias_root_sets,
                        &mut live_aliases,
                        parent_index,
                    )?;
                }
            }
            if remaining_uses.get(current_index) == Some(&0) {
                release_value(
                    &mut values,
                    &self.alias_root_sets,
                    &mut live_aliases,
                    current_index,
                )?;
            }
        }

        self.collect_outputs(host, &mut values, &mut remaining_uses)
    }

    fn collect_outputs<H: Host<LinkedOp = T>>(
        &self,
        host: &mut H,
        values: &mut [Option<H::Value>],
        remaining_uses: &mut [usize],
    ) -> Result<Vec<H::Value>, ExecutionError<H::Error>> {
        self.store
            .outputs()
            .iter()
            .map(|&node_id| {
                let index = slot(node_id)?;
                let remaining = remaining_uses
                    .get_mut(index)
                    .ok_or_else(|| ExecutionError::runtime("staged output count is unavailable"))?;
                *remaining = remaining
                    .checked_sub(1)
                    .ok_or_else(|| ExecutionError::runtime("staged output count underflowed"))?;
                let stored = values.get_mut(index).ok_or_else(|| {
                    ExecutionError::runtime("staged graph output slot is unavailable")
                })?;
                let missing = || {
                    ExecutionError::runtime(format!(
                        "staged graph output %{node_id} has no computed value"
                    ))
                };
                if *remaining == 0 {
                    return stored.take().ok_or_else(missing);
                }
                let value = stored.as_ref().ok_or_else(missing)?;
                let node = self.store.node(node_id)?;
                host.retain_value(value)
                    .map_err(|source| host_error(node, source))
            })
            .collect()
    }
}

/// Take the first sole-owned, last-use operand the binding accepts for reuse.
fn take_donation<T, V, E>(
    plan: &LinkedExecutionPlan<T>,
    binding: &LinkedOperation<T>,
    node: NodeRef<'_>,
    values: &mut [Option<V>],
    remaining_uses: &[usize],
    live_aliases: &mut [usize],
) -> Result<Option<(usize, V)>, ExecutionError<E>> {
    for &position in &binding.donation_positions {
        let parent = node.parents.get(position).ok_or_else(|| {
            ExecutionError::runtime("validated staged donation position is unavailable")
        })?;
        let parent_index = slot(parent)?;
        // Only owned values are alias roots, each of its own storage, so a live
        // count of one means an owned value no other live value can share.
        if remaining_uses.get(parent_index) != Some(&1)
            || live_aliases.get(parent_index) != Some(&1)
        {
            continue;
        }
        let parent_outputs = plan.store.node(parent)?.metadata.outputs();
        if parent_outputs.len() != 1 || parent_outputs != node.metadata.outputs() {
            continue;
        }
        let value = values
            .get_mut(parent_index)
            .and_then(Option::take)
            .ok_or_else(|| {
                ExecutionError::runtime(format!(
                    "staged donation source %{parent} has no live value"
                ))
            })?;
        shift_live_aliases(&plan.alias_root_sets, live_aliases, parent_index, false)?;
        return Ok(Some((position, value)));
    }
    Ok(None)
}

fn collect_operands<'a, V, E>(
    values: &'a [Option<V>],
    node: NodeRef<'_>,
    mut donated: Option<(usize, V)>,
) -> Result<Vec<Operand<'a, V>>, ExecutionError<E>> {
    let mut operands = Vec::with_capacity(node.parents.len());
    for (position, parent) in node.parents.iter().enumerate() {
        if let Some((_, value)) =
            donated.take_if(|(donated_position, _)| *donated_position == position)
        {
            operands.push(Operand::Donated { position, value });
            continue;
        }
        let value = values
            .get(slot(parent)?)
            .and_then(Option::as_ref)
            .ok_or_else(|| {
                ExecutionError::runtime(format!(
                    "staged operation '{}' at node %{} is missing parent value %{parent}",
                    node.schema.name(),
                    node.id
                ))
            })?;
        operands.push(Operand::Borrowed(value));
    }
    Ok(operands)
}

fn release_value<V, E>(
    values: &mut [Option<V>],
    alias_root_sets: &[Vec<usize>],
    live_aliases: &mut [usize],
    value_index: usize,
) -> Result<(), ExecutionError<E>> {
    let slot = values
        .get_mut(value_index)
        .ok_or_else(|| ExecutionError::runtime("staged value slot is unavailable"))?;
    if slot.take().is_none() {
        return Ok(());
    }
    shift_live_aliases(alias_root_sets, live_aliases, value_index, false)
}

/// Count one value into (`live`) or out of the live aliases of its roots.
fn shift_live_aliases<E>(
    alias_root_sets: &[Vec<usize>],
    live_aliases: &mut [usize],
    value_index: usize,
    live: bool,
) -> Result<(), ExecutionError<E>> {
    let alias_roots = alias_root_sets
        .get(value_index)
        .ok_or_else(|| ExecutionError::runtime("staged alias-root set is unavailable"))?;
    for &alias_root in alias_roots {
        let live_count = live_aliases
            .get_mut(alias_root)
            .ok_or_else(|| ExecutionError::runtime("staged live-alias slot is unavailable"))?;
        *live_count = if live {
            live_count.checked_add(1)
        } else {
            live_count.checked_sub(1)
        }
        .ok_or_else(|| {
            ExecutionError::runtime(if live {
                "staged live-alias count overflowed"
            } else {
                "staged live-alias count underflowed"
            })
        })?;
    }
    Ok(())
}
