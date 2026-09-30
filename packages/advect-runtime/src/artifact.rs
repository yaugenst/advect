//! Versioned canonical graph artifacts.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize, Serializer};

use crate::graph::{GraphData, validate_array_api_version};
use crate::{
    ArtifactError, AttrMap, DTypeDescriptor, GraphStore, NodeFlags, NodeId, NodeMetadata, NodeRef,
    Parents, PortableConstant, RawArena, ValueSpec,
};

/// Current graph-format version accepted by this runtime.
pub const GRAPH_FORMAT_VERSION: &str = "2.0";
const GRAPH_FORMAT: &str = "advect.graph";
const CORE_OPSET: u32 = 1;
const SEMANTIC_PROFILE: &str = "advect-array-1";
const SEMANTIC_PROFILE_VERSION: u32 = 1;
const PRODUCER: &str = "advect";
const COMPILER_VERSION: u32 = 1;
const OPTIMIZER_VERSION: u32 = 2;

impl GraphStore {
    /// Serialize this validated graph as canonical compact JSON.
    pub fn to_json(&self) -> Result<String, ArtifactError> {
        serde_json::to_string(&GraphWireRef::from_store(self)?)
            .map_err(|error| ArtifactError::new(format!("graph serialization failed: {error}")))
    }

    /// Parse and transactionally validate canonical graph JSON.
    pub fn from_json(encoded: &str) -> Result<Self, ArtifactError> {
        let wire: GraphWire = serde_json::from_str(encoded).map_err(|error| {
            ArtifactError::new(format!("graph deserialization failed: {error}"))
        })?;
        wire.validate_header()?;
        wire.into_store()
    }
}

#[derive(Serialize)]
struct GraphWireRef<'a> {
    format: &'static str,
    version: &'static str,
    core_opset: u32,
    semantic_profile: &'static str,
    semantic_profile_version: u32,
    required_array_api_version: &'a str,
    producer: &'static str,
    compiler_version: u32,
    optimizer_version: u32,
    inputs: &'a [NodeId],
    outputs: &'a [NodeId],
    nodes: Vec<NodeWireRef<'a>>,
    constants: BTreeMap<String, &'a PortableConstant>,
}

impl<'a> GraphWireRef<'a> {
    fn from_store(store: &'a GraphStore) -> Result<Self, ArtifactError> {
        let nodes = store
            .nodes()
            .map(|node| node.map(NodeWireRef::from))
            .collect::<Result<_, _>>()
            .map_err(|error| ArtifactError::new(error.to_string()))?;
        let constants = store
            .constants()
            .iter()
            .map(|(&node_id, constant)| (node_id.to_string(), constant))
            .collect();
        Ok(Self {
            format: GRAPH_FORMAT,
            version: GRAPH_FORMAT_VERSION,
            core_opset: CORE_OPSET,
            semantic_profile: SEMANTIC_PROFILE,
            semantic_profile_version: SEMANTIC_PROFILE_VERSION,
            required_array_api_version: store.required_array_api_version(),
            producer: PRODUCER,
            compiler_version: COMPILER_VERSION,
            optimizer_version: OPTIMIZER_VERSION,
            inputs: store.inputs(),
            outputs: store.outputs(),
            nodes,
            constants,
        })
    }
}

/// Borrowed serialization of one node, field for field with [`NodeWire`].
#[derive(Serialize)]
struct NodeWireRef<'a> {
    id: NodeId,
    op: &'a str,
    schema_version: u32,
    #[serde(serialize_with = "serialize_parents")]
    inputs: Parents<'a>,
    attrs: &'a AttrMap,
    shape: &'a [usize],
    dtype: &'a str,
    num_outputs: usize,
    output_shapes: Option<Vec<&'a [usize]>>,
    output_dtypes: Option<Vec<&'a str>>,
    name: Option<&'a str>,
    source_location: Option<&'a str>,
}

impl<'a> From<NodeRef<'a>> for NodeWireRef<'a> {
    fn from(node: NodeRef<'a>) -> Self {
        let metadata = node.metadata;
        let outputs = metadata.outputs();
        let multi_output = outputs.len() > 1;
        Self {
            id: node.id,
            op: node.schema.name(),
            schema_version: node.schema.schema_version(),
            inputs: node.parents,
            attrs: metadata.attrs(),
            shape: metadata.shape(),
            dtype: metadata.dtype().name(),
            num_outputs: outputs.len(),
            output_shapes: multi_output.then(|| outputs.iter().map(ValueSpec::shape).collect()),
            output_dtypes: multi_output
                .then(|| outputs.iter().map(|output| output.dtype().name()).collect()),
            name: metadata.name(),
            source_location: metadata.source_location(),
        }
    }
}

fn serialize_parents<S: Serializer>(
    parents: &Parents<'_>,
    serializer: S,
) -> Result<S::Ok, S::Error> {
    serializer.collect_seq(parents.iter())
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct GraphWire {
    format: String,
    version: String,
    core_opset: u32,
    semantic_profile: String,
    semantic_profile_version: u32,
    required_array_api_version: String,
    producer: String,
    compiler_version: u32,
    optimizer_version: u32,
    inputs: Vec<NodeId>,
    outputs: Vec<NodeId>,
    nodes: Vec<NodeWire>,
    constants: BTreeMap<String, PortableConstant>,
}

impl GraphWire {
    fn validate_header(&self) -> Result<(), ArtifactError> {
        let checks = [
            (self.format.as_str(), GRAPH_FORMAT, "graph format"),
            (self.version.as_str(), GRAPH_FORMAT_VERSION, "graph version"),
            (
                self.semantic_profile.as_str(),
                SEMANTIC_PROFILE,
                "semantic profile",
            ),
            (self.producer.as_str(), PRODUCER, "graph producer"),
        ];
        for (actual, expected, label) in checks {
            if actual != expected {
                return Err(ArtifactError::new(format!(
                    "Unsupported {label} {actual:?}; expected {expected:?}"
                )));
            }
        }
        let numeric_checks = [
            (self.core_opset, CORE_OPSET, "core opset"),
            (
                self.semantic_profile_version,
                SEMANTIC_PROFILE_VERSION,
                "semantic profile version",
            ),
            (self.compiler_version, COMPILER_VERSION, "compiler version"),
            (
                self.optimizer_version,
                OPTIMIZER_VERSION,
                "optimizer version",
            ),
        ];
        for (actual, expected, label) in numeric_checks {
            if actual != expected {
                return Err(ArtifactError::new(format!(
                    "Unsupported {label} {actual}; expected {expected}"
                )));
            }
        }
        Ok(())
    }

    fn into_store(self) -> Result<GraphStore, ArtifactError> {
        let mut arena = RawArena::default();
        let mut metadata = Vec::with_capacity(self.nodes.len());
        for entry in self.nodes {
            let expected_id = NodeId::try_from(arena.node_count())
                .map_err(|_| ArtifactError::new("graph node ID exceeded its range"))?;
            if entry.id != expected_id {
                return Err(ArtifactError::new(format!(
                    "graph nodes must have dense append-only IDs: expected {expected_id}, got {}",
                    entry.id
                )));
            }
            let flags = if entry.op == "advect.input" {
                NodeFlags::input(false)
            } else {
                NodeFlags::NONE
            };
            arena
                .append(&entry.op, entry.schema_version, &entry.inputs, flags)
                .map_err(|error| ArtifactError::new(error.into_message()))?;
            metadata.push(entry.into_metadata()?);
        }
        // Only the canonical decimal spelling is accepted, so distinct keys
        // name distinct nodes.
        let constants = self
            .constants
            .into_iter()
            .map(|(raw_id, constant)| {
                raw_id
                    .parse::<NodeId>()
                    .ok()
                    .filter(|node_id| node_id.to_string() == raw_id)
                    .map(|node_id| (node_id, constant))
                    .ok_or_else(|| {
                        ArtifactError::new(format!(
                            "graph constant key {raw_id:?} is not a canonical node ID"
                        ))
                    })
            })
            .collect::<Result<_, _>>()?;
        validate_array_api_version(&self.required_array_api_version)
            .and_then(|required_array_api_version| {
                GraphStore::new(GraphData {
                    required_array_api_version,
                    arena,
                    metadata,
                    inputs: self.inputs,
                    outputs: self.outputs,
                    constants,
                })
            })
            .map_err(|error| ArtifactError::new(error.to_string()))
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct NodeWire {
    id: NodeId,
    op: String,
    schema_version: u32,
    inputs: Vec<NodeId>,
    attrs: AttrMap,
    shape: Vec<usize>,
    dtype: String,
    num_outputs: usize,
    output_shapes: Option<Vec<Vec<usize>>>,
    output_dtypes: Option<Vec<String>>,
    name: Option<String>,
    source_location: Option<String>,
}

impl NodeWire {
    fn into_metadata(self) -> Result<NodeMetadata, ArtifactError> {
        let parse_dtype = |name: &str| {
            DTypeDescriptor::from_name(name)
                .map_err(|error| ArtifactError::new(error.into_message()))
        };
        let dtype = parse_dtype(&self.dtype)?;
        let output_dtypes = self
            .output_dtypes
            .map(|dtypes| dtypes.iter().map(|name| parse_dtype(name)).collect())
            .transpose()?;
        NodeMetadata::new(
            self.attrs,
            self.shape,
            dtype,
            self.name,
            self.num_outputs,
            self.output_shapes,
            output_dtypes,
            self.source_location,
        )
        .map_err(|error| ArtifactError::new(error.to_string()))
    }
}

#[cfg(test)]
#[expect(
    clippy::unwrap_used,
    reason = "test setup unwraps values whose absence should fail the test"
)]
mod tests {
    use super::*;
    use crate::test_support::{self, scalar_constant};
    use crate::{AttrValue, GraphBuilder};
    use serde_json::{Value, json};

    fn scalar(attrs: AttrMap) -> NodeMetadata {
        test_support::metadata(vec![], "float64", attrs)
    }

    fn canonical_graph_json() -> String {
        let mut builder = GraphBuilder::new();
        let input = builder.append_input(scalar(AttrMap::new())).unwrap();
        let constant = builder
            .append_constant(scalar(AttrMap::new()), scalar_constant(2.0))
            .unwrap();
        let mut attrs = AttrMap::new();
        attrs.insert("axis".to_owned(), AttrValue::Integer(0));
        let intermediate = builder
            .append_operation("array.add", 1, &[input, constant], scalar(attrs))
            .unwrap();
        let output = builder
            .append_operation(
                "array.add",
                1,
                &[intermediate, constant],
                scalar(AttrMap::new()),
            )
            .unwrap();
        builder.append_output(output).unwrap();
        builder.finish().unwrap().store.to_json().unwrap()
    }

    /// Copy `payload` with each JSON pointer replaced by its value.
    fn replaced<const N: usize>(payload: &Value, patches: [(&str, Value); N]) -> Value {
        let mut malformed = payload.clone();
        for (pointer, value) in patches {
            *malformed.pointer_mut(pointer).unwrap() = value;
        }
        malformed
    }

    #[test]
    fn graph_round_trip_is_byte_identical() {
        let encoded = canonical_graph_json();
        let restored = GraphStore::from_json(&encoded).unwrap();
        assert_eq!(restored.to_json().unwrap(), encoded);
    }

    /// Require every malformed payload to be rejected with its message.
    fn assert_rejected<const N: usize>(cases: [(&str, Value, &str); N]) {
        for (label, payload, fragment) in cases {
            let message = GraphStore::from_json(&payload.to_string())
                .unwrap_err()
                .into_message();
            assert!(message.contains(fragment), "{label}: {message}");
        }
    }

    /// %0 input, %1 constant, %2 = %0 + %1 with an attribute, %3 = %2 + %1.
    fn canonical_graph_value() -> Value {
        serde_json::from_str(&canonical_graph_json()).unwrap()
    }

    #[test]
    fn malformed_artifact_matrix_rejects_transactionally() {
        let valid = canonical_graph_value();
        let constant = valid.pointer("/constants/1").unwrap().clone();
        assert_rejected([
            (
                "unknown attribute field",
                replaced(
                    &valid,
                    [(
                        "/nodes/2/attrs/axis",
                        json!({"kind": "integer", "value": 0, "extra": true}),
                    )],
                ),
                "string \"extra\", expected \"kind\" or \"value\"",
            ),
            (
                "invalid dtype",
                replaced(&valid, [("/nodes/2/dtype", json!(""))]),
                "dtype descriptor must not be empty",
            ),
            (
                "bad edge",
                replaced(&valid, [("/nodes/2/inputs/0", json!("0"))]),
                "invalid type: string \"0\", expected u32",
            ),
            (
                "forward edge",
                replaced(&valid, [("/nodes/2/inputs/0", json!(3))]),
                "node %2 must reference only earlier nodes; got input %3",
            ),
            (
                "non-dense node ID",
                replaced(&valid, [("/nodes/2/id", json!(5))]),
                "dense append-only IDs: expected 2, got 5",
            ),
            (
                "bad constant digest",
                replaced(&valid, [("/constants/1/digest", json!("0".repeat(64)))]),
                "staged constant digest does not match its contents",
            ),
            (
                "zero-padded constant key",
                replaced(&valid, [("/constants", json!({"01": constant}))]),
                "graph constant key \"01\" is not a canonical node ID",
            ),
            (
                "signed constant key",
                replaced(&valid, [("/constants", json!({"+1": constant}))]),
                "graph constant key \"+1\" is not a canonical node ID",
            ),
            (
                "wrong header",
                replaced(&valid, [("/format", json!("not.advect.graph"))]),
                "Unsupported graph format \"not.advect.graph\"",
            ),
            (
                "wrong version",
                replaced(&valid, [("/version", json!("999.0"))]),
                "Unsupported graph version \"999.0\"",
            ),
            (
                "unsupported required Array API version",
                replaced(&valid, [("/required_array_api_version", json!("2025.12"))]),
                "Unsupported required Array API version \"2025.12\"",
            ),
            (
                "zero operation schema",
                replaced(&valid, [("/nodes/2/schema_version", json!(0))]),
                "arena operation schema version must be at least 1",
            ),
            (
                "mixed operation schema",
                replaced(&valid, [("/nodes/3/schema_version", json!(2))]),
                "arena operation 'array.add' is already schema version 1, not 2",
            ),
        ]);
    }

    #[test]
    fn artifacts_with_inconsistent_graph_roles_reject() {
        let valid = canonical_graph_value();
        let constant = valid.pointer("/constants/1").unwrap().clone();
        assert_rejected([
            (
                "missing output",
                replaced(&valid, [("/outputs/0", json!(99))]),
                "graph output does not exist at node %99",
            ),
            (
                "duplicate input",
                replaced(&valid, [("/inputs", json!([0, 0]))]),
                "graph inputs contain duplicate node IDs",
            ),
            (
                "undeclared input node",
                replaced(&valid, [("/inputs", json!([]))]),
                "input role ownership is inconsistent at node %0",
            ),
            (
                "multi-output input node",
                replaced(
                    &valid,
                    [
                        ("/nodes/0/num_outputs", json!(2)),
                        ("/nodes/0/output_shapes", json!([[], []])),
                        ("/nodes/0/output_dtypes", json!(["float64", "float64"])),
                    ],
                ),
                "declared input is not a single-output schema-1 operand-free advect.input node",
            ),
            (
                "constant for missing node",
                replaced(
                    &valid,
                    [("/constants", json!({"1": constant, "99": constant}))],
                ),
                "graph constant does not exist at node %99",
            ),
            (
                "constant payload on an operation",
                replaced(
                    &valid,
                    [("/constants", json!({"1": constant, "2": constant}))],
                ),
                "constant payload ownership is inconsistent at node %2",
            ),
            (
                "constant node without payload",
                replaced(&valid, [("/constants", json!({}))]),
                "constant payload ownership is inconsistent at node %1",
            ),
            (
                "constant with operands",
                replaced(&valid, [("/nodes/1/inputs", json!([0]))]),
                "advect.const nodes must not have operands at node %1",
            ),
            (
                "constant schema version",
                replaced(&valid, [("/nodes/1/schema_version", json!(2))]),
                "advect.const nodes must use schema version 1 at node %1",
            ),
            (
                "constant shape mismatch",
                replaced(&valid, [("/nodes/1/shape", json!([1]))]),
                "portable constant shape/dtype does not match node metadata at node %1",
            ),
        ]);
    }
}
