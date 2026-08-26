//! Cross-language proof for the singular program envelope emitted by Python staging.

use std::sync::Arc;

use advect_runtime::{
    AttrMap, GraphStore, Host, LinkedExecutionPlan, LinkedOperation, NodeId, NumericDType, Operand,
    OutputOwnership, PortableConstant, ValueSpec,
};

const FIXTURE: &str = include_str!("fixtures/python_staged_add_multiply_v2.json");

#[derive(Debug)]
enum Operation {
    Add,
    Multiply,
}

struct VectorHost;

impl Host for VectorHost {
    type Value = Vec<f32>;
    type LinkedOp = Operation;
    type Error = String;

    fn link(
        &mut self,
        op: &str,
        _schema_version: u32,
        _attrs: &AttrMap,
        _outputs: &[ValueSpec],
    ) -> Result<LinkedOperation<Self::LinkedOp>, Self::Error> {
        let operation = match op {
            "array.add" => Operation::Add,
            "array.multiply" => Operation::Multiply,
            _ => return Err(format!("unsupported fixture operation {op:?}")),
        };
        Ok(LinkedOperation::new(
            operation,
            Vec::new(),
            OutputOwnership::Owned,
        ))
    }

    fn materialize_constant(
        &mut self,
        _node_id: NodeId,
        constant: &PortableConstant,
    ) -> Result<Self::Value, Self::Error> {
        if constant.dtype() != NumericDType::Float32 {
            return Err("fixture host only accepts float32 constants".to_owned());
        }
        constant
            .data()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&bytes| Ok(f32::from_le_bytes(bytes)))
            .collect()
    }

    fn retain_value(&mut self, value: &Self::Value) -> Result<Self::Value, Self::Error> {
        Ok(value.clone())
    }

    fn evaluate(
        &mut self,
        operation: &Self::LinkedOp,
        operands: Vec<Operand<'_, Self::Value>>,
    ) -> Result<Self::Value, Self::Error> {
        let [left, right] = operands.as_slice() else {
            return Err("fixture binary operation requires two operands".to_owned());
        };
        if left.value().len() != right.value().len() {
            return Err("fixture binary operands have different lengths".to_owned());
        }
        let apply = match operation {
            Operation::Add => |left: f32, right: f32| left + right,
            Operation::Multiply => |left: f32, right: f32| left * right,
        };
        Ok(left
            .value()
            .iter()
            .copied()
            .zip(right.value().iter().copied())
            .map(|(left, right)| apply(left, right))
            .collect())
    }

    fn validate_value(
        &mut self,
        value: &Self::Value,
        outputs: &[ValueSpec],
    ) -> Result<(), Self::Error> {
        let [result] = outputs else {
            return Err("fixture operation violated its declared value specification".to_owned());
        };
        if result.shape() != [value.len()] || result.dtype().canonical() != "float32" {
            return Err("fixture operation violated its declared value specification".to_owned());
        }
        Ok(())
    }
}

#[test]
#[expect(
    clippy::unwrap_used,
    reason = "the checked-in fixture is required to be complete and valid"
)]
fn python_staged_fixture_round_trips_and_executes_in_pure_rust() {
    let envelope: serde_json::Value = serde_json::from_str(FIXTURE.trim()).unwrap();
    assert_eq!(envelope.get("format").unwrap(), "advect.ssa-program");
    assert_eq!(envelope.get("version").unwrap(), 2);
    let program = envelope.get("program").unwrap();
    let fixture_graph = program.get("graph").unwrap();
    assert_eq!(
        fixture_graph.get("required_array_api_version").unwrap(),
        "2024.12"
    );
    let graph = serde_json::to_string(fixture_graph).unwrap();
    let store = GraphStore::from_json(&graph).unwrap();
    let restored_graph: serde_json::Value =
        serde_json::from_str(&store.to_json().unwrap()).unwrap();
    assert_eq!(&restored_graph, fixture_graph);

    let mut host = VectorHost;
    let plan = LinkedExecutionPlan::from_store(Arc::new(store), &mut host).unwrap();
    let outputs = plan
        .execute(&mut host, vec![vec![3.0, 4.0], vec![2.0, -1.0]])
        .unwrap();

    assert_eq!(outputs, vec![vec![5.0, 3.0], vec![6.0, -4.0]]);
}
