//! Versioned canonical graph artifacts.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::{
    ArtifactError, AttrMap, DTypeDescriptor, GraphStore, NodeFlags, NodeId, NodeMetadata,
    PortableConstant, RawArena,
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
    nodes: Vec<NodeWire>,
    constants: BTreeMap<String, &'a PortableConstant>,
}

impl<'a> GraphWireRef<'a> {
    fn from_store(store: &'a GraphStore) -> Result<Self, ArtifactError> {
        let nodes = store
            .topological_order()
            .into_iter()
            .map(|node_id| {
                let record = store
                    .get_node(node_id)
                    .map_err(|error| ArtifactError::new(error.to_string()))?;
                Ok(NodeWire {
                    id: record.id,
                    op: record.op,
                    schema_version: record.schema_version,
                    inputs: record.inputs,
                    attrs: record.metadata.attrs().clone(),
                    shape: record.metadata.shape().to_vec(),
                    dtype: record.metadata.dtype().name().to_owned(),
                    num_outputs: record.metadata.num_outputs(),
                    output_shapes: record.metadata.output_shapes(),
                    output_dtypes: record.metadata.output_dtypes().map(|dtypes| {
                        dtypes
                            .into_iter()
                            .map(|dtype| dtype.name().to_owned())
                            .collect()
                    }),
                    name: record.metadata.name().map(str::to_owned),
                    source_location: record.metadata.source_location().map(str::to_owned),
                })
            })
            .collect::<Result<Vec<_>, ArtifactError>>()?;
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
            let node_metadata = entry.to_metadata()?;
            let flags = if entry.op == "advect.input" {
                NodeFlags::input(false)
            } else {
                NodeFlags::NONE
            };
            let appended = arena
                .append(&entry.op, entry.schema_version, &entry.inputs, flags)
                .map_err(|error| ArtifactError::new(error.into_message()))?;
            if appended != expected_id {
                return Err(ArtifactError::new(
                    "graph node append order is inconsistent",
                ));
            }
            metadata.push(node_metadata);
        }
        let mut constants = BTreeMap::new();
        for (raw_id, constant) in self.constants {
            let node_id = raw_id.parse::<NodeId>().map_err(|_| {
                ArtifactError::new(format!("graph constant key {raw_id:?} is not a node ID"))
            })?;
            if constants.insert(node_id, constant).is_some() {
                return Err(ArtifactError::new(format!(
                    "graph constants contain duplicate normalized node ID {node_id}"
                )));
            }
        }
        GraphStore::from_parts(
            &self.required_array_api_version,
            arena,
            metadata,
            self.inputs,
            self.outputs,
            constants,
        )
        .map_err(|error| ArtifactError::new(error.to_string()))
    }
}

#[derive(Serialize, Deserialize)]
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
    fn to_metadata(&self) -> Result<NodeMetadata, ArtifactError> {
        if self.op.is_empty() {
            return Err(ArtifactError::new("graph operation name must not be empty"));
        }
        if self.schema_version == 0 {
            return Err(ArtifactError::new(
                "graph operation schema version must be at least 1",
            ));
        }
        let dtype = DTypeDescriptor::from_name(&self.dtype)
            .map_err(|error| ArtifactError::new(error.into_message()))?;
        let output_dtypes = self
            .output_dtypes
            .as_ref()
            .map(|dtypes| {
                dtypes
                    .iter()
                    .map(|dtype| {
                        DTypeDescriptor::from_name(dtype)
                            .map_err(|error| ArtifactError::new(error.into_message()))
                    })
                    .collect::<Result<Vec<_>, _>>()
            })
            .transpose()?;
        NodeMetadata::new(
            self.attrs.clone(),
            self.shape.clone(),
            dtype,
            self.name.clone(),
            self.num_outputs,
            self.output_shapes.clone(),
            output_dtypes,
            self.source_location.clone(),
        )
        .map_err(|error| ArtifactError::new(error.to_string()))
    }
}

#[cfg(test)]
#[allow(clippy::indexing_slicing, clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::{AttrValue, ConstantKind, GraphBuilder};
    use serde_json::{Value, json};

    fn metadata(shape: Vec<usize>, dtype: &str) -> NodeMetadata {
        metadata_with_attrs(shape, dtype, AttrMap::new())
    }

    fn metadata_with_attrs(shape: Vec<usize>, dtype: &str, attrs: AttrMap) -> NodeMetadata {
        NodeMetadata::new(
            attrs,
            shape,
            DTypeDescriptor::from_name(dtype).unwrap(),
            None,
            1,
            None,
            None,
            None,
        )
        .unwrap()
    }

    fn canonical_graph_json() -> String {
        let mut builder = GraphBuilder::new();
        let input = builder.append_input(metadata(vec![], "float64")).unwrap();
        let constant = PortableConstant::new(
            ConstantKind::Scalar,
            crate::NumericDType::Float64,
            vec![],
            2.0_f64.to_le_bytes().to_vec(),
        )
        .unwrap();
        let constant = builder
            .append_constant(metadata(vec![], "float64"), constant)
            .unwrap();
        let mut attrs = AttrMap::new();
        attrs.insert("axis".to_owned(), AttrValue::Integer(0));
        let intermediate = builder
            .append_operation(
                "array.add",
                1,
                &[input, constant],
                NodeFlags::NONE,
                metadata_with_attrs(vec![], "float64", attrs),
            )
            .unwrap();
        let output = builder
            .append_operation(
                "array.add",
                1,
                &[intermediate, constant],
                NodeFlags::NONE,
                metadata(vec![], "float64"),
            )
            .unwrap();
        builder.append_output(output).unwrap();
        builder.finish().unwrap().store.to_json().unwrap()
    }

    fn replaced(payload: &Value, pointer: &str, value: Value) -> Value {
        let mut malformed = payload.clone();
        *malformed.pointer_mut(pointer).unwrap() = value;
        malformed
    }

    fn inserted(payload: &Value, pointer: &str, field: &str, value: Value) -> Value {
        let mut malformed = payload.clone();
        malformed
            .pointer_mut(pointer)
            .unwrap()
            .as_object_mut()
            .unwrap()
            .insert(field.to_owned(), value);
        malformed
    }

    #[test]
    fn graph_round_trip_is_byte_identical() {
        let encoded = canonical_graph_json();
        let restored = GraphStore::from_json(&encoded).unwrap();
        assert_eq!(restored.to_json().unwrap(), encoded);
    }

    #[test]
    fn malformed_artifact_matrix_rejects_transactionally() {
        let valid: Value = serde_json::from_str(&canonical_graph_json()).unwrap();
        let cases = [
            (
                "unknown attribute field",
                inserted(&valid, "/nodes/2/attrs/axis", "extra", json!(true)),
            ),
            (
                "invalid dtype",
                replaced(&valid, "/nodes/2/dtype", json!("")),
            ),
            (
                "bad edge",
                replaced(&valid, "/nodes/2/inputs/0", json!("0")),
            ),
            (
                "forward edge",
                replaced(&valid, "/nodes/2/inputs/0", json!(3)),
            ),
            (
                "missing edge",
                replaced(&valid, "/nodes/2/inputs/0", json!(99)),
            ),
            ("missing output", replaced(&valid, "/outputs/0", json!(99))),
            (
                "bad constant digest",
                replaced(&valid, "/constants/1/digest", json!("0".repeat(64))),
            ),
            (
                "constant for missing node",
                inserted(
                    &valid,
                    "/constants",
                    "99",
                    valid.pointer("/constants/1").unwrap().clone(),
                ),
            ),
            (
                "wrong header",
                replaced(&valid, "/format", json!("not.advect.graph")),
            ),
            (
                "wrong version",
                replaced(&valid, "/version", json!("999.0")),
            ),
            (
                "unsupported required Array API version",
                replaced(&valid, "/required_array_api_version", json!("2025.12")),
            ),
            (
                "zero operation schema",
                replaced(&valid, "/nodes/2/schema_version", json!(0)),
            ),
            (
                "mixed operation schema",
                replaced(&valid, "/nodes/3/schema_version", json!(2)),
            ),
        ];

        for (label, payload) in cases {
            assert!(
                GraphStore::from_json(&payload.to_string()).is_err(),
                "{label} unexpectedly loaded"
            );
        }
    }
}
