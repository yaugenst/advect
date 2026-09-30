//! Fixtures shared by the crate's unit tests.

use crate::{AttrMap, ConstantKind, DTypeDescriptor, NodeMetadata, NumericDType, PortableConstant};

/// Single-output node metadata.
pub(crate) fn metadata(shape: Vec<usize>, dtype: &str, attrs: AttrMap) -> NodeMetadata {
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

/// A float64 Python-scalar constant.
pub(crate) fn scalar_constant(value: f64) -> PortableConstant {
    PortableConstant::new(
        ConstantKind::Scalar,
        NumericDType::Float64,
        vec![],
        value.to_le_bytes().to_vec(),
    )
    .unwrap()
}
