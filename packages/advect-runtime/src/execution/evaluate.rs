//! Invocation-local evaluation, value lifetimes, aliasing, and donation.

use super::host::{Host, LinkedOperation, Operand};
use super::plan::{ExecutionPlan, LinkedExecutionPlan, NodeView, ValueSource, node_index};
use crate::ExecutionError;

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
        if inputs.len() != self.structure.input_count {
            return Err(ExecutionError::runtime(format!(
                "staged graph expects {} inputs but received {}",
                self.structure.input_count,
                inputs.len()
            )));
        }
        let mut inputs = inputs.into_iter().map(Some).collect::<Vec<_>>();
        let node_count = self.structure.store.node_count();
        let mut values: Vec<Option<H::Value>> = (0..node_count).map(|_| None).collect();
        let mut remaining_uses = self.structure.remaining_uses.clone();
        let mut live_aliases = vec![0_usize; node_count];

        for current_index in 0..node_count {
            let node = self.structure.node(current_index)?;
            let value = match node.source {
                ValueSource::Input(slot) => {
                    inputs.get_mut(slot).and_then(Option::take).ok_or_else(|| {
                        ExecutionError::runtime(format!(
                            "staged graph input slot {slot} is unavailable"
                        ))
                    })?
                }
                ValueSource::Constant(node_id) => {
                    let constant =
                        self.structure
                            .store
                            .constants()
                            .get(&node_id)
                            .ok_or_else(|| {
                                ExecutionError::runtime(format!(
                                    "staged graph constant %{node_id} is unavailable"
                                ))
                            })?;
                    host.materialize_constant(node_id, constant)
                        .map_err(|source| ExecutionError::Host {
                            node_id,
                            op: node.op.to_owned(),
                            source,
                        })?
                }
                ValueSource::Evaluate => {
                    let binding = self
                        .bindings
                        .get(current_index)
                        .and_then(Option::as_ref)
                        .ok_or_else(|| {
                            ExecutionError::runtime(format!(
                                "staged graph is missing a linked operation for '{}' at node %{}",
                                node.op, node.id
                            ))
                        })?;
                    let donation_position = select_donation_position(
                        self,
                        &remaining_uses,
                        &live_aliases,
                        node,
                        binding,
                    )?;
                    let donated = donation_position
                        .map(|position| {
                            take_donor(
                                &mut values,
                                &self.alias_root_sets,
                                &mut live_aliases,
                                node,
                                position,
                            )
                            .map(|value| (position, value))
                        })
                        .transpose()?;
                    let operands = collect_operands(&values, node, donated)?;
                    host.evaluate(&binding.implementation, operands)
                        .map_err(|source| ExecutionError::Host {
                            node_id: node.id,
                            op: node.op.to_owned(),
                            source,
                        })?
                }
            };
            host.validate_value(&value, node.outputs)
                .map_err(|source| ExecutionError::Host {
                    node_id: node.id,
                    op: node.op.to_owned(),
                    source,
                })?;

            *values
                .get_mut(current_index)
                .ok_or_else(|| ExecutionError::runtime("staged value slot is unavailable"))? =
                Some(value);
            increment_live_aliases(&self.alias_root_sets, &mut live_aliases, current_index)?;

            for parent in node.parents.iter() {
                let parent_index = node_index(parent, node_count, "parent use")?;
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
            if remaining_uses
                .get(current_index)
                .copied()
                .ok_or_else(|| ExecutionError::runtime("staged use-count slot is unavailable"))?
                == 0
            {
                release_value(
                    &mut values,
                    &self.alias_root_sets,
                    &mut live_aliases,
                    current_index,
                )?;
            }
        }

        collect_outputs(host, &self.structure, &mut values, &mut remaining_uses)
    }
}

fn collect_outputs<H: Host>(
    host: &mut H,
    plan: &ExecutionPlan,
    values: &mut [Option<H::Value>],
    remaining_uses: &mut [usize],
) -> Result<Vec<H::Value>, ExecutionError<H::Error>> {
    plan.store
        .outputs()
        .iter()
        .map(|&node_id| {
            let index = node_index(node_id, values.len(), "output")?;
            let remaining = remaining_uses
                .get_mut(index)
                .ok_or_else(|| ExecutionError::runtime("staged output count is unavailable"))?;
            *remaining = remaining
                .checked_sub(1)
                .ok_or_else(|| ExecutionError::runtime("staged output count underflowed"))?;
            let slot = values.get_mut(index).ok_or_else(|| {
                ExecutionError::runtime("staged graph output slot is unavailable")
            })?;
            if *remaining == 0 {
                return slot.take().ok_or_else(|| {
                    ExecutionError::runtime(format!(
                        "staged graph output %{node_id} has no computed value"
                    ))
                });
            }
            let value = slot.as_ref().ok_or_else(|| {
                ExecutionError::runtime(format!(
                    "staged graph output %{node_id} has no computed value"
                ))
            })?;
            let node = plan.node(index)?;
            host.retain_value(value)
                .map_err(|source| ExecutionError::Host {
                    node_id,
                    op: node.op.to_owned(),
                    source,
                })
        })
        .collect()
}

fn select_donation_position<T, E>(
    plan: &LinkedExecutionPlan<T>,
    remaining_uses: &[usize],
    live_aliases: &[usize],
    node: NodeView<'_>,
    binding: &LinkedOperation<T>,
) -> Result<Option<usize>, ExecutionError<E>> {
    for &position in &binding.donation_positions {
        let parent = node.parents.get(position).ok_or_else(|| {
            ExecutionError::runtime("validated staged donation position is unavailable")
        })?;
        let parent_index = node_index(parent, plan.structure.store.node_count(), "donation")?;
        let parent_node = plan.structure.node(parent_index)?;
        let alias_roots = plan
            .alias_root_sets
            .get(parent_index)
            .ok_or_else(|| ExecutionError::runtime("staged alias-root set is unavailable"))?;
        if remaining_uses.get(parent_index) == Some(&1)
            && plan.owned_values.get(parent_index) == Some(&true)
            && alias_roots
                .iter()
                .all(|&alias_root| live_aliases.get(alias_root) == Some(&1))
            && parent_node.outputs.len() == 1
            && parent_node.outputs == node.outputs
        {
            return Ok(Some(position));
        }
    }
    Ok(None)
}

fn take_donor<V, E>(
    values: &mut [Option<V>],
    alias_root_sets: &[Vec<usize>],
    live_aliases: &mut [usize],
    node: NodeView<'_>,
    position: usize,
) -> Result<V, ExecutionError<E>> {
    let parent = node.parents.get(position).ok_or_else(|| {
        ExecutionError::runtime("validated staged donation position is unavailable")
    })?;
    let parent_index = node_index(parent, values.len(), "donation")?;
    let value = values
        .get_mut(parent_index)
        .and_then(Option::take)
        .ok_or_else(|| {
            ExecutionError::runtime(format!(
                "staged donation source %{parent} has no live value"
            ))
        })?;
    decrement_live_aliases(alias_root_sets, live_aliases, parent_index)?;
    Ok(value)
}

fn collect_operands<'a, V, E>(
    values: &'a [Option<V>],
    node: NodeView<'_>,
    mut donated: Option<(usize, V)>,
) -> Result<Vec<Operand<'a, V>>, ExecutionError<E>> {
    let mut operands = Vec::with_capacity(node.parents.len());
    for (position, parent) in node.parents.iter().enumerate() {
        if donated
            .as_ref()
            .is_some_and(|(donated_position, _)| *donated_position == position)
        {
            let (_, value) = donated
                .take()
                .ok_or_else(|| ExecutionError::runtime("staged donated operand is unavailable"))?;
            operands.push(Operand::Donated { position, value });
            continue;
        }
        let value = values
            .get(node_index(parent, values.len(), "parent")?)
            .and_then(Option::as_ref)
            .ok_or_else(|| {
                ExecutionError::runtime(format!(
                    "staged operation '{}' at node %{} is missing parent value %{parent}",
                    node.op, node.id
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
    decrement_live_aliases(alias_root_sets, live_aliases, value_index)
}

fn increment_live_aliases<E>(
    alias_root_sets: &[Vec<usize>],
    live_aliases: &mut [usize],
    value_index: usize,
) -> Result<(), ExecutionError<E>> {
    let alias_roots = alias_root_sets
        .get(value_index)
        .ok_or_else(|| ExecutionError::runtime("staged alias-root set is unavailable"))?;
    for &alias_root in alias_roots {
        let live_count = live_aliases
            .get_mut(alias_root)
            .ok_or_else(|| ExecutionError::runtime("staged live-alias slot is unavailable"))?;
        *live_count = live_count
            .checked_add(1)
            .ok_or_else(|| ExecutionError::runtime("staged live-alias count overflowed"))?;
    }
    Ok(())
}

fn decrement_live_aliases<E>(
    alias_root_sets: &[Vec<usize>],
    live_aliases: &mut [usize],
    value_index: usize,
) -> Result<(), ExecutionError<E>> {
    let alias_roots = alias_root_sets
        .get(value_index)
        .ok_or_else(|| ExecutionError::runtime("staged alias-root set is unavailable"))?;
    for &alias_root in alias_roots {
        let live_count = live_aliases
            .get_mut(alias_root)
            .ok_or_else(|| ExecutionError::runtime("staged live-alias slot is unavailable"))?;
        *live_count = live_count
            .checked_sub(1)
            .ok_or_else(|| ExecutionError::runtime("staged live-alias count underflowed"))?;
    }
    Ok(())
}
