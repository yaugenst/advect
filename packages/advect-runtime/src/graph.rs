//! Closed graph construction and immutable storage.

use std::collections::BTreeMap;

use crate::{
    GraphError, NodeFlags, NodeId, NodeMetadata, NodeRecord, NodeRef, PortableConstant, RawArena,
    optimize,
};

/// Array API revisions which may be required by a durable graph.
pub const SUPPORTED_ARRAY_API_VERSIONS: &[&str] = &["2022.12", "2023.12", "2024.12"];
/// Default portable contract for newly constructed graphs.
pub const LATEST_ARRAY_API_VERSION: &str = "2024.12";

pub(crate) fn validate_array_api_version(version: &str) -> Result<&'static str, GraphError> {
    SUPPORTED_ARRAY_API_VERSIONS
        .iter()
        .copied()
        .find(|candidate| *candidate == version)
        .ok_or_else(|| {
            GraphError::new(format!(
                "Unsupported required Array API version {version:?}"
            ))
        })
}

/// Graph state shared by construction and immutable storage.
#[derive(Debug)]
pub(crate) struct GraphData {
    pub(crate) required_array_api_version: &'static str,
    pub(crate) arena: RawArena,
    pub(crate) metadata: Vec<NodeMetadata>,
    pub(crate) inputs: Vec<NodeId>,
    pub(crate) outputs: Vec<NodeId>,
    pub(crate) constants: BTreeMap<NodeId, PortableConstant>,
}

/// Mutable append-topological construction state.
#[derive(Debug)]
pub struct GraphBuilder {
    data: GraphData,
}

impl GraphBuilder {
    /// Create an empty graph builder.
    #[must_use]
    pub fn new() -> Self {
        Self::with_version(LATEST_ARRAY_API_VERSION)
    }

    /// Create an empty builder for one explicit portable Array API contract.
    pub fn new_for_array_api(required_array_api_version: &str) -> Result<Self, GraphError> {
        validate_array_api_version(required_array_api_version).map(Self::with_version)
    }

    fn with_version(required_array_api_version: &'static str) -> Self {
        Self {
            data: GraphData {
                required_array_api_version,
                arena: RawArena::default(),
                metadata: Vec::new(),
                inputs: Vec::new(),
                outputs: Vec::new(),
                constants: BTreeMap::new(),
            },
        }
    }

    /// Append a graph input atomically.
    pub fn append_input(&mut self, metadata: NodeMetadata) -> Result<NodeId, GraphError> {
        if metadata.num_outputs() != 1 {
            return Err(GraphError::new(
                "advect.input nodes must have exactly one output",
            ));
        }
        let node_id = self.append_raw("advect.input", 1, &[], NodeFlags::input(false), metadata)?;
        self.data.inputs.push(node_id);
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
        if !constant_matches(&metadata, &constant) {
            return Err(GraphError::new(
                "portable constant shape/dtype does not match node metadata",
            ));
        }
        let node_id = self.append_raw("advect.const", 1, &[], NodeFlags::NONE, metadata)?;
        self.data.constants.insert(node_id, constant);
        Ok(node_id)
    }

    /// Append one ordinary operation.
    pub fn append_operation(
        &mut self,
        op: &str,
        schema_version: u32,
        parents: &[NodeId],
        metadata: NodeMetadata,
    ) -> Result<NodeId, GraphError> {
        if matches!(op, "advect.input" | "advect.const") {
            return Err(GraphError::new(format!(
                "{op} must be constructed through its atomic builder operation"
            )));
        }
        self.append_raw(op, schema_version, parents, NodeFlags::NONE, metadata)
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
            .data
            .arena
            .append(op, schema_version, parents, flags)
            .map_err(|error| GraphError::new(error.into_message()))?;
        self.data.metadata.push(metadata);
        Ok(node_id)
    }

    /// Declare one graph output.
    pub fn append_output(&mut self, node_id: NodeId) -> Result<(), GraphError> {
        if self.data.arena.node(node_id).is_none() {
            return Err(GraphError::at_node(node_id, "node does not exist"));
        }
        self.data.outputs.push(node_id);
        Ok(())
    }

    /// Finish without the staged cleanup pipeline.
    ///
    /// Callers that need the raw tape (for example to report which traced
    /// nodes the optimizer later removes) can snapshot this store and run
    /// [`optimize`] themselves; [`Self::finish`] composes the two.
    pub fn finish_unoptimized(self) -> Result<GraphStore, GraphError> {
        GraphStore::new(self.data)
    }

    /// Finish and run Advect's fixed staged cleanup.
    pub fn finish(self) -> Result<crate::OptimizationOutcome, GraphError> {
        optimize(self.finish_unoptimized()?)
    }
}

impl Default for GraphBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Finalized immutable compute graph.
#[derive(Debug)]
pub struct GraphStore {
    data: GraphData,
}

impl GraphStore {
    /// Validate construction state into an immutable graph.
    pub(crate) fn new(data: GraphData) -> Result<Self, GraphError> {
        if data.metadata.len() != data.arena.node_count() {
            return Err(GraphError::new(
                "graph metadata does not match the structural arena",
            ));
        }
        let store = Self { data };
        store.validate()?;
        Ok(store)
    }

    pub(crate) fn into_data(self) -> GraphData {
        self.data
    }

    /// Minimum Array API revision required to execute this graph.
    #[must_use]
    pub const fn required_array_api_version(&self) -> &'static str {
        self.data.required_array_api_version
    }

    /// Declared graph inputs.
    #[must_use]
    pub fn inputs(&self) -> &[NodeId] {
        &self.data.inputs
    }

    /// Declared graph outputs.
    #[must_use]
    pub fn outputs(&self) -> &[NodeId] {
        &self.data.outputs
    }

    /// Portable constants keyed by their constant node.
    #[must_use]
    pub const fn constants(&self) -> &BTreeMap<NodeId, PortableConstant> {
        &self.data.constants
    }

    /// Number of nodes.
    #[must_use]
    pub const fn node_count(&self) -> usize {
        self.data.arena.node_count()
    }

    /// Dense append order.
    #[must_use]
    pub fn topological_order(&self) -> Vec<NodeId> {
        (0..self.node_count())
            .map_while(|index| NodeId::try_from(index).ok())
            .collect()
    }

    /// Borrow one node.
    pub fn node(&self, node_id: NodeId) -> Result<NodeRef<'_>, GraphError> {
        let missing = || GraphError::at_node(node_id, "node does not exist");
        let core = self.data.arena.node(node_id).ok_or_else(missing)?;
        let metadata = usize::try_from(node_id)
            .ok()
            .and_then(|index| self.data.metadata.get(index))
            .ok_or_else(missing)?;
        let schema = self
            .data
            .arena
            .op_schema(core.op())
            .ok_or_else(|| GraphError::at_node(node_id, "operation schema is invalid"))?;
        let parents = self
            .data
            .arena
            .parents(core)
            .ok_or_else(|| GraphError::at_node(node_id, "parent range is invalid"))?;
        Ok(NodeRef {
            id: node_id,
            op: core.op(),
            schema,
            parents,
            metadata,
        })
    }

    /// Borrow every node in dense append order.
    pub fn nodes(&self) -> impl Iterator<Item = Result<NodeRef<'_>, GraphError>> {
        (0..self.node_count())
            .map_while(|index| NodeId::try_from(index).ok())
            .map(|node_id| self.node(node_id))
    }

    /// Inspect one immutable node snapshot.
    pub fn get_node(&self, node_id: NodeId) -> Result<NodeRecord, GraphError> {
        self.node(node_id).map(NodeRecord::from)
    }

    /// Validate all closed graph invariants.
    fn validate(&self) -> Result<(), GraphError> {
        let node_count = self.node_count();
        for &output_id in self.outputs() {
            validate_endpoint(node_count, output_id, "output", None)?;
        }
        for &constant_id in self.constants().keys() {
            validate_endpoint(node_count, constant_id, "constant", None)?;
        }
        let mut declared_inputs = vec![false; node_count];
        for &input_id in self.inputs() {
            validate_endpoint(node_count, input_id, "input", Some(&mut declared_inputs))?;
            let node = self.node(input_id)?;
            if node.schema.name() != "advect.input"
                || node.schema.schema_version() != 1
                || !node.parents.is_empty()
                || node.metadata.num_outputs() != 1
            {
                return Err(GraphError::at_node(
                    input_id,
                    "declared input is not a single-output schema-1 operand-free advect.input node",
                ));
            }
        }
        for (node, declared_input) in self.nodes().zip(declared_inputs) {
            let node = node?;
            let op = node.schema.name();
            let error = |message| Err(GraphError::at_node(node.id, message));
            if (op == "advect.input") != declared_input {
                return error("input role ownership is inconsistent");
            }
            let constant = self.constants().get(&node.id);
            if (op == "advect.const") != constant.is_some() {
                return error("constant payload ownership is inconsistent");
            }
            let Some(constant) = constant else {
                continue;
            };
            if !node.parents.is_empty() {
                return error("advect.const nodes must not have operands");
            }
            if node.schema.schema_version() != 1 {
                return error("advect.const nodes must use schema version 1");
            }
            if node.metadata.num_outputs() != 1 {
                return error("advect.const nodes must have exactly one output");
            }
            if !constant_matches(node.metadata, constant) {
                return error("portable constant shape/dtype does not match node metadata");
            }
        }
        Ok(())
    }
}

fn constant_matches(metadata: &NodeMetadata, constant: &PortableConstant) -> bool {
    (metadata.shape(), metadata.dtype().canonical()) == (constant.shape(), constant.dtype().name())
}

fn validate_endpoint(
    node_count: usize,
    node_id: NodeId,
    label: &str,
    seen: Option<&mut [bool]>,
) -> Result<(), GraphError> {
    let missing = || GraphError::at_node(node_id, format!("graph {label} does not exist"));
    let index = usize::try_from(node_id)
        .ok()
        .filter(|&index| index < node_count)
        .ok_or_else(missing)?;
    if let Some(seen) = seen
        && std::mem::replace(seen.get_mut(index).ok_or_else(missing)?, true)
    {
        return Err(GraphError::new(format!(
            "graph {label}s contain duplicate node IDs"
        )));
    }
    Ok(())
}
