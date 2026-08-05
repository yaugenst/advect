//! Host-driven execution with runtime-owned value lifetimes.

use std::sync::Arc;

use crate::{AttrMap, ExecutionError, GraphStore, NodeId, PortableConstant, ValueSpec};

#[derive(Clone, Copy, Debug)]
enum ValueSource {
    Input(usize),
    Constant(NodeId),
    Evaluate,
}

#[derive(Debug)]
struct ExecutionNode {
    id: NodeId,
    op: String,
    schema_version: u32,
    parents: Vec<NodeId>,
    attrs: AttrMap,
    outputs: Vec<ValueSpec>,
    source: ValueSource,
}

/// Host declaration for one operation output.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OutputOwnership {
    /// The host guarantees fresh, internally owned compatible storage.
    Owned,
    /// The result aliases the operand at this position.
    Alias(usize),
    /// Ownership and aliasing are unknown.
    Unknown,
}

/// One operation linked to a host implementation and its storage contracts.
#[derive(Debug)]
pub struct LinkedOperation<T> {
    implementation: T,
    donation_positions: Vec<usize>,
    output_ownership: OutputOwnership,
}

impl<T> LinkedOperation<T> {
    /// Construct a linked operation.
    #[must_use]
    pub fn new(
        implementation: T,
        donation_positions: Vec<usize>,
        output_ownership: OutputOwnership,
    ) -> Self {
        Self {
            implementation,
            donation_positions,
            output_ownership,
        }
    }
}

/// One ordered evaluation operand.
#[derive(Debug)]
pub enum Operand<'a, V> {
    /// Borrowed live value.
    Borrowed(&'a V),
    /// Last-use owned value offered for physical reuse.
    Donated {
        /// Original operand position.
        position: usize,
        /// Owned host value.
        value: V,
    },
}

impl<V> Operand<'_, V> {
    /// Borrow the underlying value independently of ownership.
    #[must_use]
    pub const fn value(&self) -> &V {
        match self {
            Self::Borrowed(value) => value,
            Self::Donated { value, .. } => value,
        }
    }
}

/// Numerical host for one linked graph.
pub trait Host {
    /// Opaque runtime value.
    type Value;
    /// Host-specific linked operation.
    type LinkedOp;
    /// Host-specific failure.
    type Error;

    /// Link one operation schema and its closed attributes.
    fn link(
        &mut self,
        op: &str,
        schema_version: u32,
        attrs: &AttrMap,
        outputs: &[ValueSpec],
    ) -> Result<LinkedOperation<Self::LinkedOp>, Self::Error>;

    /// Materialize one portable constant for the current host/device.
    fn materialize_constant(
        &mut self,
        node_id: NodeId,
        constant: &PortableConstant,
    ) -> Result<Self::Value, Self::Error>;

    /// Retain another host handle to a value returned more than once.
    fn retain_value(&mut self, value: &Self::Value) -> Result<Self::Value, Self::Error>;

    /// Evaluate one linked operation over ordered operands.
    fn evaluate(
        &mut self,
        operation: &Self::LinkedOp,
        operands: Vec<Operand<'_, Self::Value>>,
    ) -> Result<Self::Value, Self::Error>;

    /// Validate a host value against closed result metadata.
    fn validate_value(
        &mut self,
        value: &Self::Value,
        outputs: &[ValueSpec],
    ) -> Result<(), Self::Error>;
}

/// Host-independent dense execution structure.
#[derive(Debug)]
pub struct ExecutionPlan {
    store: Arc<GraphStore>,
    nodes: Vec<ExecutionNode>,
    outputs: Vec<NodeId>,
    input_count: usize,
    remaining_uses: Vec<usize>,
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
    structure: ExecutionPlan,
    bindings: Vec<Option<LinkedOperation<T>>>,
    alias_root_sets: Vec<Vec<usize>>,
    owned_values: Vec<bool>,
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
        let mut values: Vec<Option<H::Value>> =
            (0..self.structure.nodes.len()).map(|_| None).collect();
        let mut remaining_uses = self.structure.remaining_uses.clone();
        let mut live_aliases = vec![0_usize; self.structure.nodes.len()];

        for node in &self.structure.nodes {
            let current_index = node_index(node.id, self.structure.nodes.len(), "value")?;
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
                            op: node.op.clone(),
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
                            op: node.op.clone(),
                            source,
                        })?
                }
            };
            host.validate_value(&value, &node.outputs)
                .map_err(|source| ExecutionError::Host {
                    node_id: node.id,
                    op: node.op.clone(),
                    source,
                })?;

            *values
                .get_mut(current_index)
                .ok_or_else(|| ExecutionError::runtime("staged value slot is unavailable"))? =
                Some(value);
            increment_live_aliases(&self.alias_root_sets, &mut live_aliases, current_index)?;

            for &parent in &node.parents {
                let parent_index = node_index(parent, self.structure.nodes.len(), "parent use")?;
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

        collect_outputs(
            host,
            &self.structure.nodes,
            &self.structure.outputs,
            &mut values,
            &mut remaining_uses,
        )
    }
}

fn collect_outputs<H: Host>(
    host: &mut H,
    nodes: &[ExecutionNode],
    outputs: &[NodeId],
    values: &mut [Option<H::Value>],
    remaining_uses: &mut [usize],
) -> Result<Vec<H::Value>, ExecutionError<H::Error>> {
    outputs
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
            let node = nodes.get(index).ok_or_else(|| {
                ExecutionError::runtime("staged graph output node is unavailable")
            })?;
            host.retain_value(value)
                .map_err(|source| ExecutionError::Host {
                    node_id,
                    op: node.op.clone(),
                    source,
                })
        })
        .collect()
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

fn select_donation_position<T, E>(
    plan: &LinkedExecutionPlan<T>,
    remaining_uses: &[usize],
    live_aliases: &[usize],
    node: &ExecutionNode,
    binding: &LinkedOperation<T>,
) -> Result<Option<usize>, ExecutionError<E>> {
    for &position in &binding.donation_positions {
        let parent = node.parents.get(position).copied().ok_or_else(|| {
            ExecutionError::runtime("validated staged donation position is unavailable")
        })?;
        let parent_index = node_index(parent, plan.structure.nodes.len(), "donation")?;
        let parent_node = plan
            .structure
            .nodes
            .get(parent_index)
            .ok_or_else(|| ExecutionError::runtime("staged donation source is unavailable"))?;
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
    node: &ExecutionNode,
    position: usize,
) -> Result<V, ExecutionError<E>> {
    let parent = node.parents.get(position).copied().ok_or_else(|| {
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
    node: &ExecutionNode,
    mut donated: Option<(usize, V)>,
) -> Result<Vec<Operand<'a, V>>, ExecutionError<E>> {
    let mut operands = Vec::with_capacity(node.parents.len());
    for (position, &parent) in node.parents.iter().enumerate() {
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

fn node_index<E>(
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

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use std::cell::RefCell;
    use std::collections::BTreeMap;
    use std::fmt::{self, Display, Formatter};
    use std::rc::Rc;

    use super::*;
    use crate::{
        AttrMap, ConstantKind, DTypeDescriptor, GRAPH_FORMAT_VERSION, GraphBuilder, GraphError,
        NodeFlags, NodeMetadata, NumericDType,
    };

    #[derive(Clone, Debug)]
    struct Tracked {
        value: f64,
        id: usize,
        events: Rc<RefCell<Vec<String>>>,
    }

    impl Drop for Tracked {
        fn drop(&mut self) {
            self.events.borrow_mut().push(format!("drop:{}", self.id));
        }
    }

    #[derive(Debug)]
    struct HostError(String);

    impl Display for HostError {
        fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
            formatter.write_str(&self.0)
        }
    }

    #[derive(Debug)]
    enum Op {
        Add,
        Copy,
        Fail,
        MaybeAlias,
    }

    struct ScalarHost {
        events: Rc<RefCell<Vec<String>>>,
        next_id: usize,
        donate: bool,
    }

    impl ScalarHost {
        fn value(&mut self, value: f64) -> Tracked {
            let id = self.next_id;
            self.next_id += 1;
            Tracked {
                value,
                id,
                events: Rc::clone(&self.events),
            }
        }
    }

    impl Host for ScalarHost {
        type Value = Tracked;
        type LinkedOp = Op;
        type Error = HostError;

        fn link(
            &mut self,
            op: &str,
            _schema_version: u32,
            _attrs: &AttrMap,
            _outputs: &[ValueSpec],
        ) -> Result<LinkedOperation<Self::LinkedOp>, Self::Error> {
            match op {
                "array.add" => Ok(LinkedOperation::new(
                    Op::Add,
                    vec![],
                    OutputOwnership::Owned,
                )),
                "advect.copy" => Ok(LinkedOperation::new(
                    Op::Copy,
                    self.donate.then_some(0).into_iter().collect(),
                    OutputOwnership::Owned,
                )),
                "advect.fail" => Ok(LinkedOperation::new(
                    Op::Fail,
                    vec![],
                    OutputOwnership::Owned,
                )),
                "advect.maybe_alias" => Ok(LinkedOperation::new(
                    Op::MaybeAlias,
                    vec![],
                    OutputOwnership::Unknown,
                )),
                _ => Err(HostError(format!("unsupported op {op}"))),
            }
        }

        fn materialize_constant(
            &mut self,
            _node_id: NodeId,
            constant: &PortableConstant,
        ) -> Result<Self::Value, Self::Error> {
            let bytes: [u8; 8] = constant
                .data()
                .try_into()
                .map_err(|_| HostError("invalid scalar bytes".to_owned()))?;
            Ok(self.value(f64::from_le_bytes(bytes)))
        }

        fn retain_value(&mut self, value: &Self::Value) -> Result<Self::Value, Self::Error> {
            self.events
                .borrow_mut()
                .push(format!("retain:{}", value.id));
            Ok(self.value(value.value))
        }

        fn evaluate(
            &mut self,
            operation: &Self::LinkedOp,
            operands: Vec<Operand<'_, Self::Value>>,
        ) -> Result<Self::Value, Self::Error> {
            match operation {
                Op::Add => Ok(self.value(operands.iter().map(|item| item.value().value).sum())),
                Op::Copy => {
                    let mut operands = operands.into_iter();
                    match operands.next() {
                        Some(Operand::Donated { value, .. }) => {
                            self.events
                                .borrow_mut()
                                .push(format!("donate:{}", value.id));
                            Ok(value)
                        }
                        Some(Operand::Borrowed(value)) => Ok(self.value(value.value)),
                        None => Err(HostError("copy requires an operand".to_owned())),
                    }
                }
                Op::MaybeAlias => {
                    let value = operands
                        .first()
                        .ok_or_else(|| HostError("maybe-alias requires an operand".to_owned()))?
                        .value()
                        .value;
                    Ok(self.value(value))
                }
                Op::Fail => Err(HostError("intentional host failure".to_owned())),
            }
        }

        fn validate_value(
            &mut self,
            value: &Self::Value,
            outputs: &[ValueSpec],
        ) -> Result<(), Self::Error> {
            let [result] = outputs else {
                return Err(HostError(
                    "scalar host requires one scalar output".to_owned(),
                ));
            };
            if !result.shape().is_empty() {
                return Err(HostError(
                    "scalar host requires one scalar output".to_owned(),
                ));
            }
            if !value.value.is_finite() {
                return Err(HostError(
                    "scalar host rejects non-finite values".to_owned(),
                ));
            }
            Ok(())
        }
    }

    fn metadata() -> NodeMetadata {
        NodeMetadata::new(
            BTreeMap::new(),
            vec![],
            DTypeDescriptor::from_name("float64").unwrap(),
            None,
            1,
            None,
            None,
            None,
        )
        .unwrap()
    }

    fn graph() -> Result<GraphStore, GraphError> {
        let mut builder = GraphBuilder::new(GRAPH_FORMAT_VERSION)?;
        let input = builder.append_input(metadata())?;
        let constant = PortableConstant::new(
            ConstantKind::Scalar,
            NumericDType::Float64,
            vec![],
            2.0_f64.to_le_bytes().to_vec(),
        )
        .map_err(|error| GraphError::new(error.into_message()))?;
        let constant = builder.append_constant(metadata(), constant)?;
        let sum = builder.append_operation(
            "array.add",
            1,
            &[input, constant],
            NodeFlags::NONE,
            metadata(),
        )?;
        let copied =
            builder.append_operation("advect.copy", 1, &[sum], NodeFlags::NONE, metadata())?;
        builder.append_output(copied)?;
        Ok(builder.finish()?.store)
    }

    fn unknown_alias_graph() -> Result<GraphStore, GraphError> {
        let mut builder = GraphBuilder::new(GRAPH_FORMAT_VERSION)?;
        let input = builder.append_input(metadata())?;
        let constant = PortableConstant::new(
            ConstantKind::Scalar,
            NumericDType::Float64,
            vec![],
            2.0_f64.to_le_bytes().to_vec(),
        )
        .map_err(|error| GraphError::new(error.into_message()))?;
        let constant = builder.append_constant(metadata(), constant)?;
        let owned = builder.append_operation(
            "array.add",
            1,
            &[input, constant],
            NodeFlags::NONE,
            metadata(),
        )?;
        let maybe_alias = builder.append_operation(
            "advect.maybe_alias",
            1,
            &[owned],
            NodeFlags::NONE,
            metadata(),
        )?;
        let copied =
            builder.append_operation("advect.copy", 1, &[owned], NodeFlags::NONE, metadata())?;
        builder.append_output(maybe_alias)?;
        builder.append_output(copied)?;
        Ok(builder.finish()?.store)
    }

    #[test]
    fn graph_outputs_are_a_flat_vector_in_declared_order() {
        let mut builder = GraphBuilder::new(GRAPH_FORMAT_VERSION).unwrap();
        let input = builder.append_input(metadata()).unwrap();
        let constant = PortableConstant::new(
            ConstantKind::Scalar,
            NumericDType::Float64,
            vec![],
            2.0_f64.to_le_bytes().to_vec(),
        )
        .unwrap();
        let constant = builder.append_constant(metadata(), constant).unwrap();
        let first = builder
            .append_operation(
                "array.add",
                1,
                &[input, constant],
                NodeFlags::NONE,
                metadata(),
            )
            .unwrap();
        let second = builder
            .append_operation(
                "array.add",
                1,
                &[first, constant],
                NodeFlags::NONE,
                metadata(),
            )
            .unwrap();
        builder.append_output(second).unwrap();
        builder.append_output(first).unwrap();
        let graph = builder.finish().unwrap().store;
        let events = Rc::new(RefCell::new(Vec::new()));
        let mut host = ScalarHost {
            events,
            next_id: 0,
            donate: false,
        };
        let plan = ExecutionPlan::from_store(Arc::new(graph))
            .unwrap()
            .link(&mut host)
            .unwrap();

        let input = host.value(3.0);
        let outputs = plan.execute(&mut host, vec![input]).unwrap();

        assert_eq!(
            outputs.iter().map(|value| value.value).collect::<Vec<_>>(),
            vec![7.0, 5.0]
        );
    }

    #[test]
    fn repeated_graph_outputs_retain_an_additional_host_handle() {
        let mut builder = GraphBuilder::new(GRAPH_FORMAT_VERSION).unwrap();
        let input = builder.append_input(metadata()).unwrap();
        builder.append_output(input).unwrap();
        builder.append_output(input).unwrap();
        let graph = builder.finish().unwrap().store;
        assert_eq!(graph.outputs(), [input, input]);

        let events = Rc::new(RefCell::new(Vec::new()));
        let mut host = ScalarHost {
            events: Rc::clone(&events),
            next_id: 0,
            donate: false,
        };
        let plan = ExecutionPlan::from_store(Arc::new(graph))
            .unwrap()
            .link(&mut host)
            .unwrap();
        let input_value = host.value(3.0);
        let input_id = input_value.id;
        let outputs = plan.execute(&mut host, vec![input_value]).unwrap();

        assert_eq!(
            outputs.iter().map(|value| value.value).collect::<Vec<_>>(),
            [3.0, 3.0]
        );
        assert!(
            events
                .borrow()
                .iter()
                .any(|event| event == &format!("retain:{input_id}"))
        );
    }

    #[test]
    fn input_values_are_validated_at_the_host_boundary() {
        let mut builder = GraphBuilder::new(GRAPH_FORMAT_VERSION).unwrap();
        let input = builder.append_input(metadata()).unwrap();
        builder.append_output(input).unwrap();
        let graph = builder.finish().unwrap().store;
        let events = Rc::new(RefCell::new(Vec::new()));
        let mut host = ScalarHost {
            events,
            next_id: 0,
            donate: false,
        };
        let plan = ExecutionPlan::from_store(Arc::new(graph))
            .unwrap()
            .link(&mut host)
            .unwrap();
        let invalid = host.value(f64::NAN);

        let error = plan.execute(&mut host, vec![invalid]).unwrap_err();

        assert_eq!(
            error.to_string(),
            "host failed while executing 'advect.input' at node %0: scalar host rejects non-finite values"
        );
    }

    #[test]
    fn materialized_constants_are_validated_at_the_host_boundary() {
        let mut builder = GraphBuilder::new(GRAPH_FORMAT_VERSION).unwrap();
        let constant = PortableConstant::new(
            ConstantKind::Scalar,
            NumericDType::Float64,
            vec![],
            f64::NAN.to_le_bytes().to_vec(),
        )
        .unwrap();
        let constant = builder.append_constant(metadata(), constant).unwrap();
        builder.append_output(constant).unwrap();
        let graph = builder.finish().unwrap().store;
        let events = Rc::new(RefCell::new(Vec::new()));
        let mut host = ScalarHost {
            events,
            next_id: 0,
            donate: false,
        };
        let plan = ExecutionPlan::from_store(Arc::new(graph))
            .unwrap()
            .link(&mut host)
            .unwrap();

        let error = plan.execute(&mut host, vec![]).unwrap_err();

        assert_eq!(
            error.to_string(),
            "host failed while executing 'advect.const' at node %0: scalar host rejects non-finite values"
        );
    }

    #[test]
    fn values_drop_immediately_after_their_last_use() {
        let graph = graph().unwrap();
        let events = Rc::new(RefCell::new(Vec::new()));
        let mut host = ScalarHost {
            events: Rc::clone(&events),
            next_id: 0,
            donate: false,
        };
        let plan = ExecutionPlan::from_store(Arc::new(graph))
            .unwrap()
            .link(&mut host)
            .unwrap();

        let input = host.value(3.0);
        let outputs = plan.execute(&mut host, vec![input]).unwrap();

        assert_eq!(events.borrow().as_slice(), ["drop:0", "drop:1", "drop:2"]);
        drop(outputs);
        assert_eq!(
            events.borrow().as_slice(),
            ["drop:0", "drop:1", "drop:2", "drop:3"]
        );
    }

    #[test]
    fn last_use_owned_value_is_offered_for_donation() {
        let graph = graph().unwrap();
        let events = Rc::new(RefCell::new(Vec::new()));
        let mut host = ScalarHost {
            events: Rc::clone(&events),
            next_id: 0,
            donate: true,
        };
        let plan = ExecutionPlan::from_store(Arc::new(graph))
            .unwrap()
            .link(&mut host)
            .unwrap();
        let input = host.value(3.0);
        let outputs = plan.execute(&mut host, vec![input]).unwrap();
        assert_eq!(outputs.first().map(|value| value.value), Some(5.0));
        assert_eq!(events.borrow().as_slice(), ["drop:0", "drop:1", "donate:2"]);
    }

    #[test]
    fn unknown_alias_prevents_donation_while_possible_parent_alias_is_live() {
        let graph = unknown_alias_graph().unwrap();
        let events = Rc::new(RefCell::new(Vec::new()));
        let mut host = ScalarHost {
            events: Rc::clone(&events),
            next_id: 0,
            donate: true,
        };
        let plan = ExecutionPlan::from_store(Arc::new(graph))
            .unwrap()
            .link(&mut host)
            .unwrap();
        let input = host.value(3.0);
        let outputs = plan.execute(&mut host, vec![input]).unwrap();
        assert_eq!(
            outputs.iter().map(|value| value.value).collect::<Vec<_>>(),
            vec![5.0, 5.0]
        );
        assert!(
            !events
                .borrow()
                .iter()
                .any(|event| event.starts_with("donate:"))
        );
    }

    #[test]
    fn host_error_names_node_and_operation() {
        let mut builder = GraphBuilder::new(GRAPH_FORMAT_VERSION).unwrap();
        let input = builder.append_input(metadata()).unwrap();
        let failed = builder
            .append_operation("advect.fail", 1, &[input], NodeFlags::NONE, metadata())
            .unwrap();
        builder.append_output(failed).unwrap();
        let graph = builder.finish().unwrap().store;
        let events = Rc::new(RefCell::new(Vec::new()));
        let mut host = ScalarHost {
            events,
            next_id: 0,
            donate: false,
        };
        let plan = ExecutionPlan::from_store(Arc::new(graph))
            .unwrap()
            .link(&mut host)
            .unwrap();
        let input = host.value(3.0);
        let error = plan.execute(&mut host, vec![input]).unwrap_err();
        assert_eq!(
            error.to_string(),
            "host failed while executing 'advect.fail' at node %1: intentional host failure"
        );
    }
}
