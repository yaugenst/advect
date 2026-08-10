//! Compact append-only SSA structure shared by dynamic and staged owners.

use std::collections::HashMap;
use std::fmt::{self, Display, Formatter};
use std::mem::size_of;

/// Dense node identity within one arena.
pub type NodeId = u32;
/// Dense operation identity within one arena.
pub type OpId = u16;
/// Stable schema version attached to an operation name.
pub type SchemaVersion = u32;

/// Default schema version for built-in operations.
pub const DEFAULT_OP_SCHEMA_VERSION: SchemaVersion = 1;

const INPUT_FLAG: u8 = 1 << 0;
const ACTIVE_FLAG: u8 = 1 << 1;
const ACTIVE_PARENT_0_FLAG: u8 = 1 << 2;
const ACTIVE_PARENT_1_FLAG: u8 = 1 << 3;

/// Compact structural inputs for one SSA node.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum InputRef {
    /// No parents.
    None,
    /// One inline parent.
    Unary(NodeId),
    /// Two inline parents.
    Binary(NodeId, NodeId),
    /// A range in the arena's n-ary edge storage.
    Nary {
        /// First edge index.
        start: u32,
        /// Number of edges.
        len: u32,
    },
}

/// Structural node flags shared by arena owners.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct NodeFlags(u8);

impl NodeFlags {
    /// No structural flags.
    pub const NONE: Self = Self(0);

    /// Construct flags for an input node.
    #[must_use]
    pub const fn input(active: bool) -> Self {
        Self(INPUT_FLAG | if active { ACTIVE_FLAG } else { 0 })
    }

    /// Construct flags from parent activity.
    #[must_use]
    pub fn operation(active_parents: &[bool]) -> Self {
        Self::operation_activity(active_parents.iter().any(|&active| active), active_parents)
    }

    /// Construct operation flags with explicit result activity.
    #[must_use]
    pub fn operation_activity(active: bool, active_parents: &[bool]) -> Self {
        let mut bits = if active { ACTIVE_FLAG } else { 0 };
        if active_parents.first().copied().unwrap_or(false) {
            bits |= ACTIVE_PARENT_0_FLAG;
        }
        if active_parents.get(1).copied().unwrap_or(false) {
            bits |= ACTIVE_PARENT_1_FLAG;
        }
        Self(bits)
    }

    /// Whether this is an input node.
    #[must_use]
    pub const fn is_input(self) -> bool {
        self.0 & INPUT_FLAG != 0
    }

    /// Whether the node is active for a dynamic transform.
    #[must_use]
    pub const fn is_active(self) -> bool {
        self.0 & ACTIVE_FLAG != 0
    }

    /// Return inline activity for parent zero or one.
    #[must_use]
    pub const fn inline_parent_is_active(self, position: usize) -> Option<bool> {
        match position {
            0 => Some(self.0 & ACTIVE_PARENT_0_FLAG != 0),
            1 => Some(self.0 & ACTIVE_PARENT_1_FLAG != 0),
            _ => None,
        }
    }
}

/// One compact SSA node. Its dense arena position is its [`NodeId`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NodeCore {
    op: OpId,
    inputs: InputRef,
    flags: NodeFlags,
}

impl NodeCore {
    /// Arena-local operation identity.
    #[must_use]
    pub const fn op(self) -> OpId {
        self.op
    }

    /// Compact parent storage.
    #[must_use]
    pub const fn inputs(self) -> InputRef {
        self.inputs
    }

    /// Structural node flags.
    #[must_use]
    pub const fn flags(self) -> NodeFlags {
        self.flags
    }
}

/// Stable operation identity stored in one arena-local table.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OpSchema {
    name: String,
    schema_version: SchemaVersion,
}

impl OpSchema {
    /// Stable operation name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Stable operation schema version.
    #[must_use]
    pub const fn schema_version(&self) -> SchemaVersion {
        self.schema_version
    }
}

#[derive(Debug, Default)]
struct OpTable {
    schemas: Vec<OpSchema>,
    by_name: HashMap<String, OpId>,
}

impl OpTable {
    fn intern(&mut self, name: &str, schema_version: SchemaVersion) -> Result<OpId, RawArenaError> {
        if name.is_empty() {
            return Err(RawArenaError::new("arena operation name must not be empty"));
        }
        if schema_version == 0 {
            return Err(RawArenaError::new(
                "arena operation schema version must be at least 1",
            ));
        }
        if let Some(&op_id) = self.by_name.get(name) {
            let schema = self.schema(op_id).ok_or_else(|| {
                RawArenaError::new("arena operation table contains an invalid operation ID")
            })?;
            if schema.schema_version != schema_version {
                return Err(RawArenaError::new(format!(
                    "arena operation '{}' is already schema version {}, not {}",
                    schema.name, schema.schema_version, schema_version
                )));
            }
            return Ok(op_id);
        }
        let op_id = OpId::try_from(self.schemas.len())
            .map_err(|_| RawArenaError::new("arena operation table exceeded its ID range"))?;
        let owned = name.to_owned();
        self.schemas.push(OpSchema {
            name: owned.clone(),
            schema_version,
        });
        self.by_name.insert(owned, op_id);
        Ok(op_id)
    }

    fn schema(&self, op_id: OpId) -> Option<&OpSchema> {
        self.schemas.get(usize::from(op_id))
    }

    fn len(&self) -> usize {
        self.schemas.len()
    }

    fn names(&self) -> impl ExactSizeIterator<Item = &str> {
        self.schemas.iter().map(OpSchema::name)
    }
}

/// Borrowed parent IDs for one node.
#[derive(Clone, Copy, Debug)]
pub enum Parents<'a> {
    /// No parents.
    None,
    /// One parent.
    Unary(NodeId),
    /// Two parents.
    Binary(NodeId, NodeId),
    /// A borrowed n-ary parent slice.
    Nary(&'a [NodeId]),
}

impl<'a> Parents<'a> {
    /// Number of parents.
    #[must_use]
    pub const fn len(self) -> usize {
        match self {
            Self::None => 0,
            Self::Unary(_) => 1,
            Self::Binary(_, _) => 2,
            Self::Nary(values) => values.len(),
        }
    }

    /// Whether there are no parents.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.len() == 0
    }

    /// Parent at one operand position.
    #[must_use]
    pub fn get(self, index: usize) -> Option<NodeId> {
        match (self, index) {
            (Self::Unary(parent), 0) => Some(parent),
            (Self::Binary(left, _), 0) => Some(left),
            (Self::Binary(_, right), 1) => Some(right),
            (Self::Nary(values), index) => values.get(index).copied(),
            _ => None,
        }
    }

    /// Iterate over parent IDs.
    #[must_use]
    pub const fn iter(self) -> ParentIter<'a> {
        ParentIter {
            parents: self,
            index: 0,
        }
    }

    /// Copy parent IDs into a vector.
    #[must_use]
    pub fn to_vec(self) -> Vec<NodeId> {
        self.iter().collect()
    }
}

/// Iterator over compact parent storage.
#[derive(Debug)]
pub struct ParentIter<'a> {
    parents: Parents<'a>,
    index: usize,
}

impl Iterator for ParentIter<'_> {
    type Item = NodeId;

    fn next(&mut self) -> Option<Self::Item> {
        let value = self.parents.get(self.index)?;
        self.index += 1;
        Some(value)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.parents.len().saturating_sub(self.index);
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for ParentIter<'_> {}

/// Shallow capacity accounting for one [`RawArena`].
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RawArenaStructuralStats {
    /// Number of live node entries.
    pub node_len: usize,
    /// Node vector capacity.
    pub node_capacity: usize,
    /// Number of live n-ary edges.
    pub edge_len: usize,
    /// Edge vector capacity.
    pub edge_capacity: usize,
    /// Number of interned operation schemas.
    pub op_len: usize,
    /// Operation-schema vector capacity.
    pub op_capacity: usize,
    /// Operation-name index capacity.
    pub op_index_capacity: usize,
    /// Shallow bytes reserved by the node vector.
    pub node_bytes: usize,
    /// Shallow bytes reserved by the edge vector.
    pub edge_bytes: usize,
    /// Shallow bytes reserved by the operation-schema vector.
    pub op_schema_bytes: usize,
    /// Estimated shallow bytes reserved by the operation-name index.
    pub op_index_bytes: usize,
}

/// One compact append-only SSA arena.
#[derive(Debug, Default)]
pub struct RawArena {
    nodes: Vec<NodeCore>,
    edges: Vec<NodeId>,
    ops: OpTable,
}

impl RawArena {
    /// Append a node after validating its operation and parents.
    pub fn append(
        &mut self,
        op_name: &str,
        schema_version: SchemaVersion,
        parents: &[NodeId],
        flags: NodeFlags,
    ) -> Result<NodeId, RawArenaError> {
        let (node_id, inputs) = self.prepare_node(parents)?;
        let op_id = self.ops.intern(op_name, schema_version)?;
        self.commit_node(op_id, parents, flags, inputs);
        Ok(node_id)
    }

    fn commit_node(&mut self, op_id: OpId, parents: &[NodeId], flags: NodeFlags, inputs: InputRef) {
        if matches!(inputs, InputRef::Nary { .. }) {
            self.edges.extend_from_slice(parents);
        }
        self.nodes.push(NodeCore {
            op: op_id,
            inputs,
            flags,
        });
    }

    /// Look up one node.
    #[must_use]
    pub fn node(&self, node_id: NodeId) -> Option<NodeCore> {
        let index = usize::try_from(node_id).ok()?;
        self.nodes.get(index).copied()
    }

    /// Resolve one node's parent representation.
    #[must_use]
    pub fn parents(&self, node: NodeCore) -> Option<Parents<'_>> {
        match node.inputs {
            InputRef::None => Some(Parents::None),
            InputRef::Unary(parent) => Some(Parents::Unary(parent)),
            InputRef::Binary(left, right) => Some(Parents::Binary(left, right)),
            InputRef::Nary { start, len } => {
                let start = usize::try_from(start).ok()?;
                let len = usize::try_from(len).ok()?;
                let end = start.checked_add(len)?;
                self.edges.get(start..end).map(Parents::Nary)
            }
        }
    }

    /// Look up one operation schema.
    #[must_use]
    pub fn op_schema(&self, op_id: OpId) -> Option<&OpSchema> {
        self.ops.schema(op_id)
    }

    /// Look up one operation name.
    #[must_use]
    pub fn op_name(&self, op_id: OpId) -> Option<&str> {
        self.op_schema(op_id).map(OpSchema::name)
    }

    /// Number of nodes.
    #[must_use]
    pub const fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Number of n-ary edges.
    #[must_use]
    pub const fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Number of operation schemas.
    #[must_use]
    pub fn op_count(&self) -> usize {
        self.ops.len()
    }

    /// Iterate over operation names in dense ID order.
    #[must_use]
    pub fn op_names(&self) -> impl ExactSizeIterator<Item = &str> {
        self.ops.names()
    }

    /// Borrow nodes in append order.
    #[must_use]
    pub fn nodes(&self) -> &[NodeCore] {
        &self.nodes
    }

    /// Report shallow structural capacities and bytes.
    #[must_use]
    pub fn structural_stats(&self) -> RawArenaStructuralStats {
        let node_capacity = self.nodes.capacity();
        let edge_capacity = self.edges.capacity();
        let op_capacity = self.ops.schemas.capacity();
        let op_index_capacity = self.ops.by_name.capacity();
        RawArenaStructuralStats {
            node_len: self.nodes.len(),
            node_capacity,
            edge_len: self.edges.len(),
            edge_capacity,
            op_len: self.ops.schemas.len(),
            op_capacity,
            op_index_capacity,
            node_bytes: node_capacity.saturating_mul(size_of::<NodeCore>()),
            edge_bytes: edge_capacity.saturating_mul(size_of::<NodeId>()),
            op_schema_bytes: op_capacity.saturating_mul(size_of::<OpSchema>()),
            // HashMap does not expose its bucket layout. This is a stable
            // shallow lower-bound estimate used only for diagnostics.
            op_index_bytes: op_index_capacity.saturating_mul(size_of::<(String, OpId)>()),
        }
    }

    /// Replace dynamic activity flags without changing topology.
    pub fn replace_activity(&mut self, active: &[bool]) -> Result<(), RawArenaError> {
        if active.len() != self.nodes.len() {
            return Err(RawArenaError::new(format!(
                "arena activity table has {} entries for {} nodes",
                active.len(),
                self.nodes.len()
            )));
        }
        for (node_index, &node_active) in active.iter().enumerate() {
            let node = *self.nodes.get(node_index).ok_or_else(|| {
                RawArenaError::new(format!("arena node %{node_index} is unavailable"))
            })?;
            let flags = if node.flags().is_input() {
                NodeFlags::input(node_active)
            } else {
                let parents = self.parents(node).ok_or_else(|| {
                    RawArenaError::new(format!(
                        "arena node %{node_index} has an invalid parent range"
                    ))
                })?;
                let parent_activity = parents
                    .iter()
                    .map(|parent| {
                        usize::try_from(parent)
                            .ok()
                            .and_then(|index| active.get(index))
                            .copied()
                            .ok_or_else(|| {
                                RawArenaError::new(format!(
                                    "arena node %{node_index} has invalid parent %{parent}"
                                ))
                            })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                NodeFlags::operation_activity(node_active, &parent_activity)
            };
            self.nodes
                .get_mut(node_index)
                .ok_or_else(|| {
                    RawArenaError::new(format!("arena node %{node_index} is unavailable"))
                })?
                .flags = flags;
        }
        Ok(())
    }

    fn prepare_node(&self, parents: &[NodeId]) -> Result<(NodeId, InputRef), RawArenaError> {
        let node_id = NodeId::try_from(self.nodes.len())
            .map_err(|_| RawArenaError::new("arena exceeded its node ID range"))?;
        if let Some(&parent) = parents.iter().find(|&&parent| parent >= node_id) {
            return Err(RawArenaError::new(format!(
                "node %{node_id} must reference only earlier nodes; got input %{parent}"
            )));
        }
        let inputs = match parents {
            [] => InputRef::None,
            [parent] => InputRef::Unary(*parent),
            [left, right] => InputRef::Binary(*left, *right),
            _ => {
                let start = u32::try_from(self.edges.len())
                    .map_err(|_| RawArenaError::new("arena edge range exceeded its index range"))?;
                let len = u32::try_from(parents.len())
                    .map_err(|_| RawArenaError::new("arena node has too many parents"))?;
                start
                    .checked_add(len)
                    .ok_or_else(|| RawArenaError::new("arena edge range overflowed"))?;
                InputRef::Nary { start, len }
            }
        };
        Ok((node_id, inputs))
    }
}

/// Structural arena construction failure.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RawArenaError(String);

impl RawArenaError {
    fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }

    /// Consume the error into its stable diagnostic message.
    #[must_use]
    pub fn into_message(self) -> String {
        self.0
    }
}

impl Display for RawArenaError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for RawArenaError {}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn append_is_dense_and_topological() {
        let mut arena = RawArena::default();
        let input = arena
            .append(
                "advect.input",
                DEFAULT_OP_SCHEMA_VERSION,
                &[],
                NodeFlags::input(false),
            )
            .unwrap();
        let output = arena
            .append(
                "array.sin",
                DEFAULT_OP_SCHEMA_VERSION,
                &[input],
                NodeFlags::NONE,
            )
            .unwrap();
        assert_eq!((input, output), (0, 1));
        assert_eq!(
            arena.parents(arena.node(output).unwrap()).unwrap().to_vec(),
            [input]
        );
        assert!(arena.append("array.cos", 1, &[4], NodeFlags::NONE).is_err());
    }

    #[test]
    fn structural_stats_cover_capacity() {
        let mut arena = RawArena::default();
        arena
            .append("advect.input", 1, &[], NodeFlags::input(false))
            .unwrap();
        let stats = arena.structural_stats();
        assert_eq!(stats.node_len, 1);
        assert!(stats.node_capacity >= stats.node_len);
        assert!(stats.node_bytes >= size_of::<NodeCore>());
    }
}
