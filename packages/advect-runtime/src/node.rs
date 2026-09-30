//! Closed node metadata.

use crate::{AttrMap, DTypeDescriptor, GraphError, NodeId, OpId, OpSchema, Parents};

/// Shape and dtype of one flat runtime value.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
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
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct NodeMetadata {
    attrs: AttrMap,
    name: Option<String>,
    outputs: Vec<ValueSpec>,
    source_location: Option<String>,
}

impl NodeMetadata {
    /// Construct and validate node metadata.
    #[expect(
        clippy::too_many_arguments,
        reason = "arguments mirror the durable node metadata record"
    )]
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
            vec![value]
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

    /// The same metadata with already validated replacement attributes.
    pub(crate) fn with_attrs(&self, attrs: AttrMap) -> Self {
        Self {
            attrs,
            ..self.clone()
        }
    }

    /// Primary output shape.
    #[must_use]
    pub fn shape(&self) -> &[usize] {
        self.primary_output().shape()
    }

    /// Primary output dtype.
    #[must_use]
    pub fn dtype(&self) -> &DTypeDescriptor {
        self.primary_output().dtype()
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

    fn primary_output(&self) -> &ValueSpec {
        let Some(value) = self.outputs.first() else {
            unreachable!("validated node metadata always has an output")
        };
        value
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

/// Borrowed view of one graph node.
#[derive(Clone, Copy, Debug)]
pub struct NodeRef<'a> {
    /// Dense node ID.
    pub id: NodeId,
    /// Arena-local operation identity.
    pub op: OpId,
    /// Stable operation name and schema version.
    pub schema: &'a OpSchema,
    /// Parent node IDs.
    pub parents: Parents<'a>,
    /// Closed node metadata.
    pub metadata: &'a NodeMetadata,
}

impl From<NodeRef<'_>> for NodeRecord {
    fn from(node: NodeRef<'_>) -> Self {
        Self {
            id: node.id,
            op: node.schema.name().to_owned(),
            schema_version: node.schema.schema_version(),
            inputs: node.parents.to_vec(),
            metadata: node.metadata.clone(),
        }
    }
}
