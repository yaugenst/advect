//! Execution-plan behavior tests.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::fmt::{self, Display, Formatter};
use std::rc::Rc;
use std::sync::Arc;

use super::*;
use crate::{
    AttrMap, ConstantKind, DTypeDescriptor, GraphBuilder, GraphError, GraphStore, NodeFlags,
    NodeId, NodeMetadata, NumericDType, PortableConstant, ValueSpec,
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
    let mut builder = GraphBuilder::new();
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
    let copied = builder.append_operation("advect.copy", 1, &[sum], NodeFlags::NONE, metadata())?;
    builder.append_output(copied)?;
    Ok(builder.finish()?.store)
}

fn unknown_alias_graph() -> Result<GraphStore, GraphError> {
    let mut builder = GraphBuilder::new();
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
    let mut builder = GraphBuilder::new();
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
    let plan = LinkedExecutionPlan::from_store(Arc::new(graph), &mut host).unwrap();

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

    let events = Rc::new(RefCell::new(Vec::new()));
    let mut host = ScalarHost {
        events: Rc::clone(&events),
        next_id: 0,
        donate: false,
    };
    let plan = LinkedExecutionPlan::from_store(Arc::new(graph), &mut host).unwrap();
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
    let mut builder = GraphBuilder::new();
    let input = builder.append_input(metadata()).unwrap();
    builder.append_output(input).unwrap();
    let graph = builder.finish().unwrap().store;
    let events = Rc::new(RefCell::new(Vec::new()));
    let mut host = ScalarHost {
        events,
        next_id: 0,
        donate: false,
    };
    let plan = LinkedExecutionPlan::from_store(Arc::new(graph), &mut host).unwrap();
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
    let plan = LinkedExecutionPlan::from_store(Arc::new(graph), &mut host).unwrap();

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
    let plan = LinkedExecutionPlan::from_store(Arc::new(graph), &mut host).unwrap();

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
    let plan = LinkedExecutionPlan::from_store(Arc::new(graph), &mut host).unwrap();
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
    let plan = LinkedExecutionPlan::from_store(Arc::new(graph), &mut host).unwrap();
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
    let mut builder = GraphBuilder::new();
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
    let plan = LinkedExecutionPlan::from_store(Arc::new(graph), &mut host).unwrap();
    let input = host.value(3.0);
    let error = plan.execute(&mut host, vec![input]).unwrap_err();
    assert_eq!(
        error.to_string(),
        "host failed while executing 'advect.fail' at node %1: intentional host failure"
    );
}
