//! Python-independent SSA graph and lifetime runtime for Advect.

mod arena;
mod artifact;
mod attr;
mod constant;
mod dtype;
mod error;
mod execution;
mod graph;
mod node;
mod optimize;
mod topology;

pub use arena::{
    DEFAULT_OP_SCHEMA_VERSION, InputRef, NodeCore, NodeFlags, NodeId, OpId, OpSchema, Parents,
    RawArena, RawArenaError, RawArenaStructuralStats, SchemaVersion,
};
pub use artifact::{GRAPH_FORMAT_VERSION, GraphArtifact};
pub use attr::{AttrMap, AttrValue, ExactFloat};
pub use constant::{
    CONSTANT_FORMAT, CONSTANT_VERSION, ConstantError, ConstantKind, NumericDType, PortableConstant,
};
pub use dtype::{DTypeDescriptor, DTypeError};
pub use error::{ArtifactError, ExecutionError, GraphError};
pub use execution::{
    ExecutionPlan, Host, LinkedExecutionPlan, LinkedOperation, Operand, OutputOwnership,
};
pub use graph::{GraphBuilder, GraphStore, LATEST_ARRAY_API_VERSION, SUPPORTED_ARRAY_API_VERSIONS};
pub use node::{NodeMetadata, NodeRecord, ValueSpec};
pub use optimize::{OptimizationOutcome, OptimizationReport, PassReport, optimize};
pub use topology::{Topology, TopologyError};
