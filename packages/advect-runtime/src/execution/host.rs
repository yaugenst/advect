//! Host contracts and host-owned values used by staged execution.

use crate::{AttrMap, NodeId, PortableConstant, ValueSpec};

/// Host declaration for one operation output.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OutputOwnership {
    /// The host guarantees fresh, internally owned compatible storage.
    Owned,
    /// The result aliases the operand at this position.
    Alias(usize),
    /// Ownership and aliasing are unknown.
    Unknown,
}

/// One operation linked to a host implementation and its storage contracts.
#[derive(Debug)]
pub struct LinkedOperation<T> {
    pub(super) implementation: T,
    pub(super) donation_positions: Vec<usize>,
    pub(super) output_ownership: OutputOwnership,
}

impl<T> LinkedOperation<T> {
    /// Construct a linked operation.
    #[must_use]
    pub fn new(
        implementation: T,
        donation_positions: Vec<usize>,
        output_ownership: OutputOwnership,
    ) -> Self {
        Self {
            implementation,
            donation_positions,
            output_ownership,
        }
    }
}

/// One ordered evaluation operand.
#[derive(Debug)]
pub enum Operand<'a, V> {
    /// Borrowed live value.
    Borrowed(&'a V),
    /// Last-use owned value offered for physical reuse.
    Donated {
        /// Original operand position.
        position: usize,
        /// Owned host value.
        value: V,
    },
}

impl<V> Operand<'_, V> {
    /// Borrow the underlying value independently of ownership.
    #[must_use]
    pub const fn value(&self) -> &V {
        match self {
            Self::Borrowed(value) => value,
            Self::Donated { value, .. } => value,
        }
    }
}

/// Numerical host for one linked graph.
pub trait Host {
    /// Opaque runtime value.
    type Value;
    /// Host-specific linked operation.
    type LinkedOp;
    /// Host-specific failure.
    type Error;

    /// Link one operation schema and its closed attributes.
    fn link(
        &mut self,
        op: &str,
        schema_version: u32,
        attrs: &AttrMap,
        outputs: &[ValueSpec],
    ) -> Result<LinkedOperation<Self::LinkedOp>, Self::Error>;

    /// Materialize one portable constant for the current host/device.
    fn materialize_constant(
        &mut self,
        node_id: NodeId,
        constant: &PortableConstant,
    ) -> Result<Self::Value, Self::Error>;

    /// Retain another host handle to a value returned more than once.
    fn retain_value(&mut self, value: &Self::Value) -> Result<Self::Value, Self::Error>;

    /// Evaluate one linked operation over ordered operands.
    fn evaluate(
        &mut self,
        operation: &Self::LinkedOp,
        operands: Vec<Operand<'_, Self::Value>>,
    ) -> Result<Self::Value, Self::Error>;

    /// Validate a host value against closed result metadata.
    fn validate_value(
        &mut self,
        value: &Self::Value,
        outputs: &[ValueSpec],
    ) -> Result<(), Self::Error>;
}
