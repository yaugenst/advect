//! Closed graph construction and immutable storage.

use std::collections::BTreeMap;

use crate::{
    GRAPH_FORMAT_VERSION, GraphError, NodeFlags, NodeId, NodeMetadata, NodeRecord,
    PortableConstant, RawArena, Topology, optimize,
};

/// Array API revisions which may be required by a durable graph.
pub const SUPPORTED_ARRAY_API_VERSIONS: &[&str] = &["2022.12", "2023.12", "2024.12"];
/// Default portable contract for newly constructed graphs.
pub const LATEST_ARRAY_API_VERSION: &str = "2024.12";

fn validate_array_api_version(version: &str) -> Result<(), GraphError> {
    if SUPPORTED_ARRAY_API_VERSIONS.contains(&version) {
        return Ok(());
    }
    Err(GraphError::new(format!(
        "Unsupported required Array API version {version:?}"
    )))
}

/// Mutable append-topological construction state.
#[derive(Debug)]
pub struct GraphBuilder {
    version: String,
    required_array_api_version: String,
    arena: RawArena,
    metadata: Vec<NodeMetadata>,
    inputs: Vec<NodeId>,
    outputs: Vec<NodeId>,
    constants: BTreeMap<NodeId, PortableConstant>,
}

impl GraphBuilder {
    /// Create an empty graph builder.
    pub fn new(version: impl Into<String>) -> Result<Self, GraphError> {
        Self::new_for_array_api(version, LATEST_ARRAY_API_VERSION)
    }

    /// Create an empty builder for one explicit portable Array API contract.
    pub fn new_for_array_api(
        version: impl Into<String>,
        required_array_api_version: impl Into<String>,
    ) -> Result<Self, GraphError> {
        let version = version.into();
        if version != GRAPH_FORMAT_VERSION {
            return Err(GraphError::new(format!(
                "Unsupported graph version {version:?}; expected {GRAPH_FORMAT_VERSION:?}"
            )));
        }
        let required_array_api_version = required_array_api_version.into();
        validate_array_api_version(&required_array_api_version)?;
        Ok(Self {
            version,
            required_array_api_version,
            arena: RawArena::default(),
            metadata: Vec::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            constants: BTreeMap::new(),
        })
    }

    /// Append a graph input atomically.
    pub fn append_input(&mut self, metadata: NodeMetadata) -> Result<NodeId, GraphError> {
        if metadata.num_outputs() != 1 {
            return Err(GraphError::new(
                "advect.input nodes must have exactly one output",
            ));
        }
        let node_id = self.append_raw("advect.input", 1, &[], NodeFlags::input(false), metadata)?;
        self.inputs.push(node_id);
        Ok(node_id)
    }

    /// Append a portable constant atomically.
    pub fn append_constant(
        &mut self,
        metadata: NodeMetadata,
        constant: PortableConstant,
    ) -> Result<NodeId, GraphError> {
        if metadata.num_outputs() != 1 {
            return Err(GraphError::new(
                "advect.const nodes must have exactly one output",
            ));
        }
        let shape_matches = metadata.shape() == constant.shape();
        let dtype_matches = metadata.dtype().canonical() == constant.dtype().name();
        if !shape_matches || !dtype_matches {
            return Err(GraphError::new(
                "portable constant shape/dtype does not match node metadata",
            ));
        }
        let node_id = self.append_raw("advect.const", 1, &[], NodeFlags::NONE, metadata)?;
        self.constants.insert(node_id, constant);
        Ok(node_id)
    }

    /// Append one ordinary operation.
    pub fn append_operation(
        &mut self,
        op: &str,
        schema_version: u32,
        parents: &[NodeId],
        flags: NodeFlags,
        metadata: NodeMetadata,
    ) -> Result<NodeId, GraphError> {
        if matches!(op, "advect.input" | "advect.const") {
            return Err(GraphError::new(format!(
                "{op} must be constructed through its atomic builder operation"
            )));
        }
        self.append_raw(op, schema_version, parents, flags, metadata)
    }

    fn append_raw(
        &mut self,
        op: &str,
        schema_version: u32,
        parents: &[NodeId],
        flags: NodeFlags,
        metadata: NodeMetadata,
    ) -> Result<NodeId, GraphError> {
        let node_id = self
            .arena
            .append(op, schema_version, parents, flags)
            .map_err(|error| GraphError::new(error.into_message()))?;
        self.metadata.push(metadata);
        Ok(node_id)
    }

    /// Declare one graph output.
    pub fn append_output(&mut self, node_id: NodeId) -> Result<(), GraphError> {
        self.require_node(node_id)?;
        self.outputs.push(node_id);
        Ok(())
    }

    /// Finish without the staged cleanup pipeline.
    ///
    /// Callers that need the raw tape (for example to report which traced
    /// nodes the optimizer later removes) can snapshot this store and run
    /// [`optimize`] themselves; [`Self::finish`] composes the two.
    pub fn finish_unoptimized(self) -> Result<GraphStore, GraphError> {
        GraphStore::from_parts(
            self.version,
            self.required_array_api_version,
            self.arena,
            self.metadata,
            self.inputs,
            self.outputs,
            self.constants,
        )
    }

    /// Finish and run Advect's fixed staged cleanup.
    pub fn finish(self) -> Result<crate::OptimizationOutcome, GraphError> {
        optimize(self.finish_unoptimized()?)
    }

    fn require_node(&self, node_id: NodeId) -> Result<(), GraphError> {
        if self.arena.node(node_id).is_none() {
            return Err(GraphError::at_node(node_id, "node does not exist"));
        }
        Ok(())
    }
}

/// Finalized immutable compute graph.
#[derive(Debug)]
pub struct GraphStore {
    version: String,
    required_array_api_version: String,
    arena: RawArena,
    metadata: Vec<NodeMetadata>,
    inputs: Vec<NodeId>,
    outputs: Vec<NodeId>,
    constants: BTreeMap<NodeId, PortableConstant>,
    topology: Topology,
}

pub(crate) type GraphParts = (
    String,
    String,
    RawArena,
    Vec<NodeMetadata>,
    Vec<NodeId>,
    Vec<NodeId>,
    BTreeMap<NodeId, PortableConstant>,
);

impl GraphStore {
    pub(crate) fn from_parts(
        version: String,
        required_array_api_version: String,
        arena: RawArena,
        metadata: Vec<NodeMetadata>,
        inputs: Vec<NodeId>,
        outputs: Vec<NodeId>,
        constants: BTreeMap<NodeId, PortableConstant>,
    ) -> Result<Self, GraphError> {
        if version != GRAPH_FORMAT_VERSION {
            return Err(GraphError::new(format!(
                "Unsupported graph version {version:?}; expected {GRAPH_FORMAT_VERSION:?}"
            )));
        }
        validate_array_api_version(&required_array_api_version)?;
        if metadata.len() != arena.node_count() {
            return Err(GraphError::new(
                "graph metadata does not match the structural arena",
            ));
        }
        let topology = Topology::build(&arena, &inputs, &outputs)
            .map_err(|error| GraphError::new(error.into_message()))?;
        let store = Self {
            version,
            required_array_api_version,
            arena,
            metadata,
            inputs,
            outputs,
            constants,
            topology,
        };
        store.validate()?;
        Ok(store)
    }

    /// Graph-format version string.
    #[must_use]
    pub fn version(&self) -> &str {
        &self.version
    }

    /// Minimum Array API revision required to execute this graph.
    #[must_use]
    pub fn required_array_api_version(&self) -> &str {
        &self.required_array_api_version
    }

    /// Structural arena.
    #[must_use]
    pub const fn arena(&self) -> &RawArena {
        &self.arena
    }

    /// Dense metadata aligned with arena nodes.
    #[must_use]
    pub fn metadata(&self) -> &[NodeMetadata] {
        &self.metadata
    }

    /// Declared graph inputs.
    #[must_use]
    pub fn inputs(&self) -> &[NodeId] {
        &self.inputs
    }

    /// Declared graph outputs.
    #[must_use]
    pub fn outputs(&self) -> &[NodeId] {
        &self.outputs
    }

    /// Portable constants keyed by their constant node.
    #[must_use]
    pub const fn constants(&self) -> &BTreeMap<NodeId, PortableConstant> {
        &self.constants
    }

    /// Number of nodes.
    #[must_use]
    pub const fn node_count(&self) -> usize {
        self.arena.node_count()
    }

    /// Dense append order.
    #[must_use]
    pub fn topological_order(&self) -> Vec<NodeId> {
        self.topology.topological_order()
    }

    /// Inspect one immutable node snapshot.
    pub fn get_node(&self, node_id: NodeId) -> Result<NodeRecord, GraphError> {
        let index = usize::try_from(node_id)
            .map_err(|_| GraphError::at_node(node_id, "node ID exceeds the host index range"))?;
        let metadata = self
            .metadata
            .get(index)
            .ok_or_else(|| GraphError::at_node(node_id, "node does not exist"))?;
        NodeRecord::snapshot(&self.arena, metadata, node_id)
    }

    /// Validate all closed graph invariants.
    pub fn validate(&self) -> Result<(), GraphError> {
        let topology_matches = self.topology.node_count() == self.arena.node_count();
        let metadata_matches = self.metadata.len() == self.arena.node_count();
        if !topology_matches || !metadata_matches {
            return Err(GraphError::new(
                "finalized topology does not match the graph arena",
            ));
        }
        let mut declared_inputs = vec![false; self.arena.node_count()];
        for &input_id in &self.inputs {
            let record = self.get_node(input_id)?;
            if record.op != "advect.input"
                || record.schema_version != 1
                || !record.inputs.is_empty()
                || record.metadata.num_outputs() != 1
            {
                return Err(GraphError::at_node(
                    input_id,
                    "declared input is not a single-output schema-1 operand-free advect.input node",
                ));
            }
            let input_index = usize::try_from(input_id)
                .map_err(|_| GraphError::at_node(input_id, "input ID exceeds host range"))?;
            *declared_inputs
                .get_mut(input_index)
                .ok_or_else(|| GraphError::at_node(input_id, "declared input does not exist"))? =
                true;
        }
        for (node_index, node) in self.arena.nodes().iter().copied().enumerate() {
            let node_id = NodeId::try_from(node_index)
                .map_err(|_| GraphError::new("graph node ID exceeded its range"))?;
            let op = self
                .arena
                .op_name(node.op())
                .ok_or_else(|| GraphError::at_node(node_id, "operation ID is invalid"))?;
            if (op == "advect.input") != declared_inputs.get(node_index).copied().unwrap_or(false) {
                return Err(GraphError::at_node(
                    node_id,
                    "input role ownership is inconsistent",
                ));
            }
            let constant = self.constants.get(&node_id);
            if (op == "advect.const") != constant.is_some() {
                return Err(GraphError::at_node(
                    node_id,
                    "constant payload ownership is inconsistent",
                ));
            }
            if let Some(constant) = constant {
                let parents = self.arena.parents(node).ok_or_else(|| {
                    GraphError::at_node(node_id, "constant parent range is invalid")
                })?;
                if !parents.is_empty() {
                    return Err(GraphError::at_node(
                        node_id,
                        "advect.const nodes must not have operands",
                    ));
                }
                let schema = self.arena.op_schema(node.op()).ok_or_else(|| {
                    GraphError::at_node(node_id, "constant operation schema is invalid")
                })?;
                if schema.schema_version() != 1 {
                    return Err(GraphError::at_node(
                        node_id,
                        "advect.const nodes must use schema version 1",
                    ));
                }
                let metadata = self.metadata.get(node_index).ok_or_else(|| {
                    GraphError::at_node(node_id, "constant metadata is unavailable")
                })?;
                if metadata.num_outputs() != 1 {
                    return Err(GraphError::at_node(
                        node_id,
                        "advect.const nodes must have exactly one output",
                    ));
                }
                let shape_matches = metadata.shape() == constant.shape();
                let dtype_matches = metadata.dtype().canonical() == constant.dtype().name();
                if !shape_matches || !dtype_matches {
                    return Err(GraphError::at_node(
                        node_id,
                        "portable constant shape/dtype does not match node metadata",
                    ));
                }
            }
        }
        Ok(())
    }

    pub(crate) fn into_parts(self) -> GraphParts {
        (
            self.version,
            self.required_array_api_version,
            self.arena,
            self.metadata,
            self.inputs,
            self.outputs,
            self.constants,
        )
    }
}
