//! Validation for append-topological graph arenas.

use std::fmt::{self, Display, Formatter};

use crate::{NodeId, RawArena};

/// Invalid graph topology.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TopologyError(String);

impl TopologyError {
    fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }

    /// Consume the error into its diagnostic message.
    #[must_use]
    pub fn into_message(self) -> String {
        self.0
    }
}

impl Display for TopologyError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for TopologyError {}

/// Dense topology for a finalized append-only graph.
#[derive(Clone, Debug)]
pub struct Topology {
    node_count: usize,
}

impl Topology {
    /// Validate graph endpoints and parent order.
    pub fn build(
        arena: &RawArena,
        graph_inputs: &[NodeId],
        graph_outputs: &[NodeId],
    ) -> Result<Self, TopologyError> {
        validate_endpoints(arena.node_count(), graph_inputs, "input", true)?;
        validate_endpoints(arena.node_count(), graph_outputs, "output", false)?;
        for (node_index, node) in arena.nodes().iter().copied().enumerate() {
            let node_id = NodeId::try_from(node_index)
                .map_err(|_| TopologyError::new("graph node ID exceeded its range"))?;
            let parents = arena
                .parents(node)
                .ok_or_else(|| TopologyError::new("graph node has an invalid edge range"))?;
            for parent_id in parents.iter() {
                let parent_index = usize::try_from(parent_id)
                    .map_err(|_| TopologyError::new("graph parent ID exceeded its range"))?;
                if parent_index >= arena.node_count() {
                    return Err(TopologyError::new(format!(
                        "node %{node_id} references missing input %{parent_id}"
                    )));
                }
                if parent_id >= node_id {
                    return Err(TopologyError::new(format!(
                        "node %{node_id} must reference only earlier nodes; got input %{parent_id}"
                    )));
                }
            }
        }
        Ok(Self {
            node_count: arena.node_count(),
        })
    }

    /// Number of graph nodes.
    #[must_use]
    pub const fn node_count(&self) -> usize {
        self.node_count
    }

    /// Dense append order.
    #[must_use]
    pub fn topological_order(&self) -> Vec<NodeId> {
        (0..self.node_count)
            .map_while(|index| NodeId::try_from(index).ok())
            .collect()
    }
}

fn validate_endpoints(
    node_count: usize,
    node_ids: &[NodeId],
    label: &str,
    require_unique: bool,
) -> Result<(), TopologyError> {
    let mut seen = vec![false; node_count];
    for &node_id in node_ids {
        let node_index = usize::try_from(node_id)
            .map_err(|_| TopologyError::new(format!("graph {label} ID exceeded its range")))?;
        let slot = seen.get_mut(node_index).ok_or_else(|| {
            TopologyError::new(format!("graph {label} %{node_id} does not exist"))
        })?;
        if require_unique && *slot {
            return Err(TopologyError::new(format!(
                "graph {label}s contain duplicate node IDs"
            )));
        }
        *slot = true;
    }
    Ok(())
}
