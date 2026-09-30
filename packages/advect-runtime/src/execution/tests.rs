//! Execution-plan behavior tests.

use std::cell::RefCell;
use std::fmt::{self, Display, Formatter};
use std::rc::Rc;
use std::sync::Arc;

use super::*;
use crate::test_support::{self, scalar_constant};
use crate::{AttrMap, GraphBuilder, GraphStore, NodeId, NodeMetadata, PortableConstant, ValueSpec};

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

/// Link `graph` through a fresh scalar host that offers copies for donation
/// when `donate` is set.
fn linked(graph: GraphStore, donate: bool) -> (ScalarHost, LinkedExecutionPlan<Op>) {
    let mut host = ScalarHost {
        events: Rc::new(RefCell::new(Vec::new())),
        next_id: 0,
        donate,
    };
    let plan = LinkedExecutionPlan::from_store(Arc::new(graph), &mut host).unwrap();
    (host, plan)
}

fn metadata() -> NodeMetadata {
    test_support::metadata(vec![], "float64", AttrMap::new())
}

/// Append `input + 2.0` and return the constant and the sum.
fn add_constant(builder: &mut GraphBuilder) -> (NodeId, NodeId) {
    let input = builder.append_input(metadata()).unwrap();
    let constant = builder
        .append_constant(metadata(), scalar_constant(2.0))
        .unwrap();
    let sum = builder
        .append_operation("array.add", 1, &[input, constant], metadata())
        .unwrap();
    (constant, sum)
}

fn graph() -> GraphStore {
    let mut builder = GraphBuilder::new();
    let (_, sum) = add_constant(&mut builder);
    let copied = builder
        .append_operation("advect.copy", 1, &[sum], metadata())
        .unwrap();
    builder.append_output(copied).unwrap();
    builder.finish().unwrap().store
}

fn unknown_alias_graph() -> GraphStore {
    let mut builder = GraphBuilder::new();
    let (_, owned) = add_constant(&mut builder);
    let maybe_alias = builder
        .append_operation("advect.maybe_alias", 1, &[owned], metadata())
        .unwrap();
    let copied = builder
        .append_operation("advect.copy", 1, &[owned], metadata())
        .unwrap();
    builder.append_output(maybe_alias).unwrap();
    builder.append_output(copied).unwrap();
    builder.finish().unwrap().store
}

#[test]
fn graph_outputs_are_a_flat_vector_in_declared_order() {
    let mut builder = GraphBuilder::new();
    let (constant, first) = add_constant(&mut builder);
    let second = builder
        .append_operation("array.add", 1, &[first, constant], metadata())
        .unwrap();
    builder.append_output(second).unwrap();
    builder.append_output(first).unwrap();
    let (mut host, plan) = linked(builder.finish().unwrap().store, false);

    let input = host.value(3.0);
    let outputs = plan.execute(&mut host, vec![input]).unwrap();

    assert_eq!(
        outputs.iter().map(|value| value.value).collect::<Vec<_>>(),
        vec![7.0, 5.0]
    );
}

#[test]
fn repeated_graph_outputs_retain_an_additional_host_handle() {
    let mut builder = GraphBuilder::new();
    let input = builder.append_input(metadata()).unwrap();
    builder.append_output(input).unwrap();
    builder.append_output(input).unwrap();
    let graph = builder.finish().unwrap().store;
    assert_eq!(graph.outputs(), [input, input]);

    let (mut host, plan) = linked(graph, false);
    let input_value = host.value(3.0);
    let input_id = input_value.id;
    let outputs = plan.execute(&mut host, vec![input_value]).unwrap();

    assert_eq!(
        outputs.iter().map(|value| value.value).collect::<Vec<_>>(),
        [3.0, 3.0]
    );
    assert!(
        host.events
            .borrow()
            .iter()
            .any(|event| event == &format!("retain:{input_id}"))
    );
}

#[test]
fn input_values_are_validated_at_the_host_boundary() {
    let mut builder = GraphBuilder::new();
    let input = builder.append_input(metadata()).unwrap();
    builder.append_output(input).unwrap();
    let (mut host, plan) = linked(builder.finish().unwrap().store, false);
    let invalid = host.value(f64::NAN);

    let error = plan.execute(&mut host, vec![invalid]).unwrap_err();

    assert_eq!(
        error.to_string(),
        "host failed while executing 'advect.input' at node %0: scalar host rejects non-finite values"
    );
}

#[test]
fn materialized_constants_are_validated_at_the_host_boundary() {
    let mut builder = GraphBuilder::new();
    let constant = builder
        .append_constant(metadata(), scalar_constant(f64::NAN))
        .unwrap();
    builder.append_output(constant).unwrap();
    let (mut host, plan) = linked(builder.finish().unwrap().store, false);

    let error = plan.execute(&mut host, vec![]).unwrap_err();

    assert_eq!(
        error.to_string(),
        "host failed while executing 'advect.const' at node %0: scalar host rejects non-finite values"
    );
}

#[test]
fn values_drop_immediately_after_their_last_use() {
    let (mut host, plan) = linked(graph(), false);

    let input = host.value(3.0);
    let outputs = plan.execute(&mut host, vec![input]).unwrap();

    assert_eq!(
        host.events.borrow().as_slice(),
        ["drop:0", "drop:1", "drop:2"]
    );
    drop(outputs);
    assert_eq!(
        host.events.borrow().as_slice(),
        ["drop:0", "drop:1", "drop:2", "drop:3"]
    );
}

#[test]
fn last_use_owned_value_is_offered_for_donation() {
    let (mut host, plan) = linked(graph(), true);
    let input = host.value(3.0);
    let outputs = plan.execute(&mut host, vec![input]).unwrap();
    assert_eq!(outputs.first().map(|value| value.value), Some(5.0));
    assert_eq!(
        host.events.borrow().as_slice(),
        ["drop:0", "drop:1", "donate:2"]
    );
}

#[test]
fn unknown_alias_prevents_donation_while_possible_parent_alias_is_live() {
    let (mut host, plan) = linked(unknown_alias_graph(), true);
    let input = host.value(3.0);
    let outputs = plan.execute(&mut host, vec![input]).unwrap();
    assert_eq!(
        outputs.iter().map(|value| value.value).collect::<Vec<_>>(),
        vec![5.0, 5.0]
    );
    assert!(
        !host
            .events
            .borrow()
            .iter()
            .any(|event| event.starts_with("donate:"))
    );
}

#[test]
fn alias_roots_track_only_owned_storage() {
    let (_, plan) = linked(unknown_alias_graph(), true);

    // Inputs, constants, and unknown results are never donated, so only the
    // owned sum (%2) and copy (%4) are roots; the unknown %3 may alias %2.
    assert_eq!(
        plan.alias_root_sets,
        [vec![], vec![], vec![2], vec![2], vec![4]]
    );
}

/// Host value in an exact storage model: a handle keeps every listed buffer
/// alive, like a view that may reference several arrays. Only a handle
/// returned by an owned operation owns its buffer.
#[derive(Debug)]
struct Handle {
    buffers: Vec<usize>,
    owned: bool,
    live: Rc<RefCell<Vec<usize>>>,
}

impl Drop for Handle {
    fn drop(&mut self) {
        let mut live = self.live.borrow_mut();
        for &buffer in &self.buffers {
            *live.get_mut(buffer).unwrap() -= 1;
        }
    }
}

#[derive(Debug)]
struct StorageOp {
    donation_positions: Vec<usize>,
    ownership: OutputOwnership,
}

/// Owned results allocate a buffer (or reuse a donated one), alias results
/// share their source's buffers, and unknown results share every operand's
/// buffers, so the runtime's alias model is exact for this host.
struct StorageHost {
    live: Rc<RefCell<Vec<usize>>>,
    /// Buffers of borrowed operands that the last evaluation could have
    /// donated if it was their last use.
    donatable: Vec<usize>,
    donations: usize,
}

impl StorageHost {
    fn allocate(&self, owned: bool) -> Handle {
        let mut live = self.live.borrow_mut();
        live.push(0);
        let buffer = live.len() - 1;
        drop(live);
        self.share(vec![buffer], owned)
    }

    fn share(&self, buffers: Vec<usize>, owned: bool) -> Handle {
        let mut live = self.live.borrow_mut();
        for &buffer in &buffers {
            *live.get_mut(buffer).unwrap() += 1;
        }
        drop(live);
        Handle {
            buffers,
            owned,
            live: Rc::clone(&self.live),
        }
    }

    fn is_sole_owner(&self, handle: &Handle) -> bool {
        handle.owned
            && matches!(handle.buffers.as_slice(), [buffer]
                if self.live.borrow().get(*buffer) == Some(&1))
    }

    /// A dead buffer never comes back, so a donatable operand that died
    /// right after its evaluation was at its last use and was not donated.
    fn assert_no_missed_donation(&self) {
        for buffer in &self.donatable {
            assert_ne!(
                self.live.borrow().get(*buffer),
                Some(&0),
                "sole owned last-use operand was not donated"
            );
        }
    }
}

impl Host for StorageHost {
    type Value = Handle;
    type LinkedOp = StorageOp;
    type Error = HostError;

    fn link(
        &mut self,
        op: &str,
        _schema_version: u32,
        _attrs: &AttrMap,
        _outputs: &[ValueSpec],
    ) -> Result<LinkedOperation<Self::LinkedOp>, Self::Error> {
        let (donation_positions, ownership) = match op {
            "test.owned" => (vec![0], OutputOwnership::Owned),
            "test.owned2" => (vec![1, 0], OutputOwnership::Owned),
            "test.alias" => (vec![], OutputOwnership::Alias(0)),
            "test.alias2" => (vec![0], OutputOwnership::Alias(1)),
            "test.unknown" => (vec![0], OutputOwnership::Unknown),
            "test.unknown2" => (vec![], OutputOwnership::Unknown),
            _ => return Err(HostError(format!("unsupported op {op}"))),
        };
        Ok(LinkedOperation::new(
            StorageOp {
                donation_positions: donation_positions.clone(),
                ownership,
            },
            donation_positions,
            ownership,
        ))
    }

    fn materialize_constant(
        &mut self,
        _node_id: NodeId,
        _constant: &PortableConstant,
    ) -> Result<Self::Value, Self::Error> {
        Err(HostError("storage host has no constants".to_owned()))
    }

    fn retain_value(&mut self, value: &Self::Value) -> Result<Self::Value, Self::Error> {
        Ok(self.share(value.buffers.clone(), false))
    }

    fn evaluate(
        &mut self,
        operation: &Self::LinkedOp,
        operands: Vec<Operand<'_, Self::Value>>,
    ) -> Result<Self::Value, Self::Error> {
        self.assert_no_missed_donation();
        self.donatable.clear();
        for &position in &operation.donation_positions {
            let Some(Operand::Borrowed(candidate)) = operands.get(position) else {
                continue;
            };
            let shared = operands
                .iter()
                .filter(|operand| {
                    let buffers = &operand.value().buffers;
                    buffers
                        .iter()
                        .any(|buffer| candidate.buffers.contains(buffer))
                })
                .count();
            if shared == 1 && self.is_sole_owner(candidate) {
                self.donatable.extend_from_slice(&candidate.buffers);
            }
        }
        let buffers = operands
            .iter()
            .map(|operand| operand.value().buffers.clone())
            .collect::<Vec<_>>();
        let mut donated = None;
        for operand in operands {
            if let Operand::Donated { value, .. } = operand {
                assert!(donated.is_none(), "one evaluation received two donations");
                assert!(
                    self.is_sole_owner(&value),
                    "donated value is not the sole owner of its storage"
                );
                donated = Some(value);
            }
        }
        if donated.is_some() {
            // One donation per evaluation, so other eligible operands stay.
            self.donatable.clear();
            self.donations += 1;
        }
        Ok(match operation.ownership {
            OutputOwnership::Owned => donated.unwrap_or_else(|| self.allocate(true)),
            OutputOwnership::Alias(position) => {
                self.share(buffers.get(position).unwrap().clone(), false)
            }
            OutputOwnership::Unknown => {
                let mut all = buffers.concat();
                all.sort_unstable();
                all.dedup();
                self.share(all, false)
            }
        })
    }

    fn validate_value(
        &mut self,
        _value: &Self::Value,
        _outputs: &[ValueSpec],
    ) -> Result<(), Self::Error> {
        Ok(())
    }
}

#[test]
fn random_graphs_donate_exactly_the_sole_owned_last_use_values() {
    const OPS: [(&str, usize); 6] = [
        ("test.owned", 1),
        ("test.owned2", 2),
        ("test.alias", 1),
        ("test.alias2", 2),
        ("test.unknown", 1),
        ("test.unknown2", 2),
    ];
    let mut state = 0x5eed_u64;
    let mut below = |bound: usize| {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        usize::try_from(state >> 33).unwrap() % bound
    };
    let mut donations = 0;
    for _ in 0..500 {
        let mut builder = GraphBuilder::new();
        let mut nodes = vec![
            builder.append_input(metadata()).unwrap(),
            builder.append_input(metadata()).unwrap(),
        ];
        for _ in 0..3 + below(30) {
            let (op, arity) = *OPS.get(below(OPS.len())).unwrap();
            let parents = (0..arity)
                .map(|_| *nodes.get(below(nodes.len())).unwrap())
                .collect::<Vec<_>>();
            nodes.push(
                builder
                    .append_operation(op, 1, &parents, metadata())
                    .unwrap(),
            );
        }
        let output_count = 1 + below(3);
        for _ in 0..output_count {
            builder
                .append_output(*nodes.get(below(nodes.len())).unwrap())
                .unwrap();
        }
        let mut host = StorageHost {
            live: Rc::new(RefCell::new(Vec::new())),
            donatable: Vec::new(),
            donations: 0,
        };
        let graph = builder.finish_unoptimized().unwrap();
        let plan = LinkedExecutionPlan::from_store(Arc::new(graph), &mut host).unwrap();
        let inputs = vec![host.allocate(false), host.allocate(false)];

        let outputs = plan.execute(&mut host, inputs).unwrap();

        host.assert_no_missed_donation();
        assert_eq!(outputs.len(), output_count);
        drop(outputs);
        assert!(host.live.borrow().iter().all(|&count| count == 0));
        donations += host.donations;
    }
    assert!(donations > 300, "only {donations} donations were exercised");
}

#[test]
fn host_error_names_node_and_operation() {
    let mut builder = GraphBuilder::new();
    let input = builder.append_input(metadata()).unwrap();
    let failed = builder
        .append_operation("advect.fail", 1, &[input], metadata())
        .unwrap();
    builder.append_output(failed).unwrap();
    let (mut host, plan) = linked(builder.finish().unwrap().store, false);
    let input = host.value(3.0);
    let error = plan.execute(&mut host, vec![input]).unwrap_err();
    assert_eq!(
        error.to_string(),
        "host failed while executing 'advect.fail' at node %1: intentional host failure"
    );
}
