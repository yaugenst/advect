//! Closed node metadata.

use serde::{Deserialize, Serialize};

use crate::{AttrMap, DTypeDescriptor, GraphError, NodeId, RawArena};

/// Shape and dtype of one flat runtime value.
#[derive(Clone, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
pub struct ValueSpec {
    shape: Vec<usize>,
    dtype: DTypeDescriptor,
}

impl ValueSpec {
    /// Construct a value specification.
    #[must_use]
    pub const fn new(shape: Vec<usize>, dtype: DTypeDescriptor) -> Self {
        Self { shape, dtype }
    }

    /// Shape dimensions.
    #[must_use]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Dtype descriptor.
    #[must_use]
    pub const fn dtype(&self) -> &DTypeDescriptor {
        &self.dtype
    }
}

/// Durable metadata for one graph node.
#[derive(Clone, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
pub struct NodeMetadata {
    attrs: AttrMap,
    value: ValueSpec,
    name: Option<String>,
    outputs: Vec<ValueSpec>,
    source_location: Option<String>,
}

impl NodeMetadata {
    /// Construct and validate node metadata.
    pub fn new(
        attrs: AttrMap,
        shape: Vec<usize>,
        dtype: DTypeDescriptor,
        name: Option<String>,
        num_outputs: usize,
        output_shapes: Option<Vec<Vec<usize>>>,
        output_dtypes: Option<Vec<DTypeDescriptor>>,
        source_location: Option<String>,
    ) -> Result<Self, GraphError> {
        if num_outputs == 0 {
            return Err(GraphError::new("node num_outputs must be at least 1"));
        }
        let value = ValueSpec::new(shape, dtype);
        let outputs = if num_outputs == 1 {
            if output_shapes.is_some() || output_dtypes.is_some() {
                return Err(GraphError::new(
                    "single-output node must not declare output_shapes/output_dtypes",
                ));
            }
            vec![value.clone()]
        } else {
            let shapes = output_shapes.ok_or_else(|| {
                GraphError::new("multi-output node is missing output_shapes/output_dtypes")
            })?;
            let dtypes = output_dtypes.ok_or_else(|| {
                GraphError::new("multi-output node is missing output_shapes/output_dtypes")
            })?;
            if shapes.len() != num_outputs || dtypes.len() != num_outputs {
                return Err(GraphError::new(format!(
                    "node expects {num_outputs} outputs but got {} shapes and {} dtypes",
                    shapes.len(),
                    dtypes.len()
                )));
            }
            let values = shapes
                .into_iter()
                .zip(dtypes)
                .map(|(shape, dtype)| ValueSpec::new(shape, dtype))
                .collect::<Vec<_>>();
            if values.first() != Some(&value) {
                return Err(GraphError::new(
                    "output-0 metadata must match the node shape and dtype",
                ));
            }
            values
        };
        Ok(Self {
            attrs,
            value,
            name,
            outputs,
            source_location,
        })
    }

    /// Closed attributes.
    #[must_use]
    pub const fn attrs(&self) -> &AttrMap {
        &self.attrs
    }

    /// Primary output shape.
    #[must_use]
    pub fn shape(&self) -> &[usize] {
        self.value.shape()
    }

    /// Primary output dtype.
    #[must_use]
    pub const fn dtype(&self) -> &DTypeDescriptor {
        self.value.dtype()
    }

    /// Primary output specification.
    #[must_use]
    pub const fn value_spec(&self) -> &ValueSpec {
        &self.value
    }

    /// Optional user-facing name.
    #[must_use]
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    /// Number of logical outputs.
    #[must_use]
    pub const fn num_outputs(&self) -> usize {
        self.outputs.len()
    }

    /// Every logical output specification.
    #[must_use]
    pub fn outputs(&self) -> &[ValueSpec] {
        &self.outputs
    }

    /// Output shapes for a multi-output node.
    #[must_use]
    pub fn output_shapes(&self) -> Option<Vec<Vec<usize>>> {
        (self.outputs.len() > 1).then(|| {
            self.outputs
                .iter()
                .map(|output| output.shape().to_vec())
                .collect()
        })
    }

    /// Output dtypes for a multi-output node.
    #[must_use]
    pub fn output_dtypes(&self) -> Option<Vec<DTypeDescriptor>> {
        (self.outputs.len() > 1).then(|| {
            self.outputs
                .iter()
                .map(|output| output.dtype().clone())
                .collect()
        })
    }

    /// Optional source location.
    #[must_use]
    pub fn source_location(&self) -> Option<&str> {
        self.source_location.as_deref()
    }
}

/// Immutable snapshot of one graph node.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NodeRecord {
    /// Dense node ID.
    pub id: NodeId,
    /// Stable operation name.
    pub op: String,
    /// Operation schema version.
    pub schema_version: u32,
    /// Parent node IDs.
    pub inputs: Vec<NodeId>,
    /// Closed node metadata.
    pub metadata: NodeMetadata,
}

impl NodeRecord {
    pub(crate) fn snapshot(
        arena: &RawArena,
        metadata: &NodeMetadata,
        node_id: NodeId,
    ) -> Result<Self, GraphError> {
        let node = arena
            .node(node_id)
            .ok_or_else(|| GraphError::at_node(node_id, "node does not exist"))?;
        let schema = arena
            .op_schema(node.op())
            .ok_or_else(|| GraphError::at_node(node_id, "operation schema is invalid"))?;
        let inputs = arena
            .parents(node)
            .ok_or_else(|| GraphError::at_node(node_id, "parent range is invalid"))?;
        Ok(Self {
            id: node_id,
            op: schema.name().to_owned(),
            schema_version: schema.schema_version(),
            inputs: inputs.to_vec(),
            metadata: metadata.clone(),
        })
    }
}
