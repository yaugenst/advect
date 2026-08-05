//! Typed runtime errors with stable graph context.

use std::fmt::{self, Display, Formatter};

use crate::NodeId;

/// Invalid graph construction or invariant.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GraphError {
    message: String,
    node_id: Option<NodeId>,
}

impl GraphError {
    /// Construct a graph-wide error.
    #[must_use]
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            node_id: None,
        }
    }

    /// Construct an error attributed to one node.
    #[must_use]
    pub fn at_node(node_id: NodeId, message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            node_id: Some(node_id),
        }
    }

    /// Optional graph node context.
    #[must_use]
    pub const fn node_id(&self) -> Option<NodeId> {
        self.node_id
    }

    /// Stable diagnostic message.
    #[must_use]
    pub fn message(&self) -> &str {
        &self.message
    }

    /// Consume the error into its diagnostic message.
    #[must_use]
    pub fn into_message(self) -> String {
        self.to_string()
    }
}

impl Display for GraphError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        if let Some(node_id) = self.node_id {
            write!(formatter, "{} at node %{node_id}", self.message)
        } else {
            formatter.write_str(&self.message)
        }
    }
}

impl std::error::Error for GraphError {}

/// Invalid serialized graph artifact.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ArtifactError(String);

impl ArtifactError {
    /// Construct an artifact error.
    #[must_use]
    pub fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }

    /// Consume the error into its stable message.
    #[must_use]
    pub fn into_message(self) -> String {
        self.0
    }
}

impl Display for ArtifactError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for ArtifactError {}

/// Structural or host failure while linking or executing a graph.
#[derive(Debug)]
pub enum ExecutionError<E> {
    /// Invalid structural schedule or invocation.
    Runtime(String),
    /// Host failure attributed to one operation node.
    Host {
        /// Failing node.
        node_id: NodeId,
        /// Stable operation name.
        op: String,
        /// Host-specific source error.
        source: E,
    },
}

impl<E> ExecutionError<E> {
    pub(crate) fn runtime(message: impl Into<String>) -> Self {
        Self::Runtime(message.into())
    }
}

impl<E: Display> Display for ExecutionError<E> {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Runtime(message) => formatter.write_str(message),
            Self::Host {
                node_id,
                op,
                source,
            } => write!(
                formatter,
                "host failed while executing '{op}' at node %{node_id}: {source}"
            ),
        }
    }
}

impl<E: std::error::Error + 'static> std::error::Error for ExecutionError<E> {}
