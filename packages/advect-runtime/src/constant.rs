//! Portable numeric constants.

use std::fmt::{self, Display, Formatter};
use std::str::FromStr;

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use sha2::{Digest, Sha256};

use crate::ArtifactError;

/// Stable portable constant format name.
pub const CONSTANT_FORMAT: &str = "advect.numeric-constant";
/// Current portable constant format version.
pub const CONSTANT_VERSION: u32 = 2;
const CONSTANT_LAYOUT: &str = "C";
const CONSTANT_BYTE_ORDER: &str = "little";

/// Whether rank-zero data preserves weak scalar or array semantics.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ConstantKind {
    /// Python scalar semantics.
    Scalar,
    /// Array semantics, including rank-zero arrays.
    Array,
}

impl ConstantKind {
    /// Stable artifact spelling.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Scalar => "scalar",
            Self::Array => "array",
        }
    }
}

impl FromStr for ConstantKind {
    type Err = ConstantError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "scalar" => Ok(Self::Scalar),
            "array" => Ok(Self::Array),
            _ => Err(ConstantError::new(format!(
                "unknown staged constant kind {value:?}"
            ))),
        }
    }
}

/// Standard numeric dtype supported by portable constant bytes.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum NumericDType {
    /// Boolean byte.
    Bool,
    /// Signed 8-bit integer.
    Int8,
    /// Signed 16-bit integer.
    Int16,
    /// Signed 32-bit integer.
    Int32,
    /// Signed 64-bit integer.
    Int64,
    /// Unsigned 8-bit integer.
    Uint8,
    /// Unsigned 16-bit integer.
    Uint16,
    /// Unsigned 32-bit integer.
    Uint32,
    /// Unsigned 64-bit integer.
    Uint64,
    /// IEEE binary16.
    Float16,
    /// IEEE binary32.
    Float32,
    /// IEEE binary64.
    Float64,
    /// Two IEEE binary32 components.
    Complex64,
    /// Two IEEE binary64 components.
    Complex128,
}

impl NumericDType {
    /// Canonical logical dtype name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Bool => "bool",
            Self::Int8 => "int8",
            Self::Int16 => "int16",
            Self::Int32 => "int32",
            Self::Int64 => "int64",
            Self::Uint8 => "uint8",
            Self::Uint16 => "uint16",
            Self::Uint32 => "uint32",
            Self::Uint64 => "uint64",
            Self::Float16 => "float16",
            Self::Float32 => "float32",
            Self::Float64 => "float64",
            Self::Complex64 => "complex64",
            Self::Complex128 => "complex128",
        }
    }

    /// Bytes per logical element.
    #[must_use]
    pub const fn item_size(self) -> usize {
        match self {
            Self::Bool | Self::Int8 | Self::Uint8 => 1,
            Self::Int16 | Self::Uint16 | Self::Float16 => 2,
            Self::Int32 | Self::Uint32 | Self::Float32 => 4,
            Self::Int64 | Self::Uint64 | Self::Float64 | Self::Complex64 => 8,
            Self::Complex128 => 16,
        }
    }
}

impl FromStr for NumericDType {
    type Err = ConstantError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "bool" => Ok(Self::Bool),
            "int8" => Ok(Self::Int8),
            "int16" => Ok(Self::Int16),
            "int32" => Ok(Self::Int32),
            "int64" => Ok(Self::Int64),
            "uint8" => Ok(Self::Uint8),
            "uint16" => Ok(Self::Uint16),
            "uint32" => Ok(Self::Uint32),
            "uint64" => Ok(Self::Uint64),
            "float16" => Ok(Self::Float16),
            "float32" => Ok(Self::Float32),
            "float64" => Ok(Self::Float64),
            "complex64" => Ok(Self::Complex64),
            "complex128" => Ok(Self::Complex128),
            _ => Err(ConstantError::new(format!(
                "unsupported staged constant dtype {value:?}"
            ))),
        }
    }
}

/// Closed C-contiguous little-endian numeric constant.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PortableConstant {
    kind: ConstantKind,
    dtype: NumericDType,
    shape: Vec<usize>,
    data: Vec<u8>,
    digest: String,
}

impl PortableConstant {
    /// Construct and validate a portable constant.
    pub fn new(
        kind: ConstantKind,
        dtype: NumericDType,
        shape: Vec<usize>,
        data: Vec<u8>,
    ) -> Result<Self, ArtifactError> {
        validate_shape_and_bytes(kind, dtype, &shape, &data)
            .map_err(|error| ArtifactError::new(error.to_string()))?;
        let digest = digest_body(kind, dtype, &shape, &data)
            .map_err(|error| ArtifactError::new(error.to_string()))?;
        Ok(Self {
            kind,
            dtype,
            shape,
            data,
            digest,
        })
    }

    /// Scalar-versus-array semantics.
    #[must_use]
    pub const fn kind(&self) -> ConstantKind {
        self.kind
    }

    /// Canonical numeric dtype.
    #[must_use]
    pub const fn dtype(&self) -> NumericDType {
        self.dtype
    }

    /// C-order shape.
    #[must_use]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Canonical little-endian C-contiguous bytes.
    #[must_use]
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    /// Lowercase SHA-256 digest over canonical JSON without the digest field.
    #[must_use]
    pub fn digest(&self) -> &str {
        &self.digest
    }

    /// Serialize the exact v2 constant object.
    pub fn to_json(&self) -> Result<String, ArtifactError> {
        serde_json::to_string(self)
            .map_err(|error| ArtifactError::new(format!("constant serialization failed: {error}")))
    }

    /// Parse and validate one exact v2 constant object.
    pub fn from_json(encoded: &str) -> Result<Self, ArtifactError> {
        serde_json::from_str(encoded).map_err(|error| {
            ArtifactError::new(format!("constant deserialization failed: {error}"))
        })
    }
}

#[derive(Serialize)]
struct ConstantWireRef<'a> {
    format: &'static str,
    version: u32,
    kind: &'static str,
    dtype: &'static str,
    shape: &'a [usize],
    layout: &'static str,
    byte_order: &'static str,
    data: String,
    digest: &'a str,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ConstantWire {
    format: String,
    version: u32,
    kind: String,
    dtype: String,
    shape: Vec<usize>,
    layout: String,
    byte_order: String,
    data: String,
    digest: String,
}

impl Serialize for PortableConstant {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        ConstantWireRef {
            format: CONSTANT_FORMAT,
            version: CONSTANT_VERSION,
            kind: self.kind.name(),
            dtype: self.dtype.name(),
            shape: &self.shape,
            layout: CONSTANT_LAYOUT,
            byte_order: CONSTANT_BYTE_ORDER,
            data: crate::hex::encode(&self.data),
            digest: &self.digest,
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for PortableConstant {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = ConstantWire::deserialize(deserializer)?;
        if wire.format != CONSTANT_FORMAT {
            return Err(serde::de::Error::custom(format!(
                "unknown staged constant format {:?}",
                wire.format
            )));
        }
        if wire.version != CONSTANT_VERSION {
            return Err(serde::de::Error::custom(format!(
                "unsupported staged constant version {}",
                wire.version
            )));
        }
        if wire.layout != CONSTANT_LAYOUT {
            return Err(serde::de::Error::custom(format!(
                "unsupported staged constant layout {:?}",
                wire.layout
            )));
        }
        if wire.byte_order != CONSTANT_BYTE_ORDER {
            return Err(serde::de::Error::custom(format!(
                "unsupported staged constant byte order {:?}",
                wire.byte_order
            )));
        }
        let kind = wire.kind.parse().map_err(serde::de::Error::custom)?;
        let dtype = wire.dtype.parse().map_err(serde::de::Error::custom)?;
        let data = crate::hex::decode(&wire.data).map_err(|message| {
            serde::de::Error::custom(format!("staged constant data {message}"))
        })?;
        validate_shape_and_bytes(kind, dtype, &wire.shape, &data)
            .map_err(serde::de::Error::custom)?;
        if wire.digest.len() != 64
            || !wire
                .digest
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            return Err(serde::de::Error::custom(
                "staged constant digest must be lowercase SHA-256 hex",
            ));
        }
        let expected =
            digest_body(kind, dtype, &wire.shape, &data).map_err(serde::de::Error::custom)?;
        if wire.digest != expected {
            return Err(serde::de::Error::custom(
                "staged constant digest does not match its contents",
            ));
        }
        Ok(Self {
            kind,
            dtype,
            shape: wire.shape,
            data,
            digest: wire.digest,
        })
    }
}

fn validate_shape_and_bytes(
    kind: ConstantKind,
    dtype: NumericDType,
    shape: &[usize],
    data: &[u8],
) -> Result<(), ConstantError> {
    if kind == ConstantKind::Scalar && !shape.is_empty() {
        return Err(ConstantError::new(
            "a staged scalar constant must have rank zero",
        ));
    }
    let element_count = shape.iter().try_fold(1_usize, |count, &dimension| {
        count
            .checked_mul(dimension)
            .ok_or_else(|| ConstantError::new("staged constant shape overflows"))
    })?;
    let expected = element_count
        .checked_mul(dtype.item_size())
        .ok_or_else(|| ConstantError::new("staged constant byte count overflows"))?;
    if data.len() != expected {
        return Err(ConstantError::new(format!(
            "staged constant shape/dtype require {expected} bytes; got {}",
            data.len()
        )));
    }
    if dtype == NumericDType::Bool && data.iter().any(|&value| value > 1) {
        return Err(ConstantError::new(
            "staged bool constant bytes must be exactly 0 or 1",
        ));
    }
    Ok(())
}

fn digest_body(
    kind: ConstantKind,
    dtype: NumericDType,
    shape: &[usize],
    data: &[u8],
) -> Result<String, ConstantError> {
    let mut digest = Sha256::new();
    digest.update(br#"{"byte_order":"little","data":""#);
    let mut hex_buffer = [0_u8; 8192];
    for chunk in data.chunks(hex_buffer.len() / 2) {
        for (encoded, &byte) in hex_buffer.chunks_exact_mut(2).zip(chunk) {
            encoded
                .copy_from_slice(&[crate::hex::digit(byte >> 4), crate::hex::digit(byte & 0x0f)]);
        }
        let encoded_len = chunk
            .len()
            .checked_mul(2)
            .ok_or_else(|| ConstantError::new("constant digest byte count overflows"))?;
        let encoded = hex_buffer
            .get(..encoded_len)
            .ok_or_else(|| ConstantError::new("constant digest buffer is inconsistent"))?;
        digest.update(encoded);
    }
    let shape = serde_json::to_string(shape)
        .map_err(|error| ConstantError::new(format!("constant digest encoding failed: {error}")))?;
    let suffix = format!(
        "\",\"dtype\":\"{}\",\"format\":\"{CONSTANT_FORMAT}\",\"kind\":\"{}\",\
         \"layout\":\"{CONSTANT_LAYOUT}\",\"shape\":{shape},\"version\":{CONSTANT_VERSION}}}",
        dtype.name(),
        kind.name(),
    );
    digest.update(suffix.as_bytes());
    Ok(crate::hex::encode(&digest.finalize()))
}

#[derive(Clone, Debug, Eq, PartialEq)]
/// Invalid closed numeric constant representation.
pub struct ConstantError(String);

impl ConstantError {
    fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl Display for ConstantError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for ConstantError {}

#[cfg(test)]
#[expect(
    clippy::indexing_slicing,
    clippy::unwrap_used,
    reason = "tests use direct fixture access and unwrap to fail immediately"
)]
mod tests {
    use super::*;
    use serde_json::Value;

    #[test]
    fn v2_constant_matches_canonical_python_digest_contract() {
        let constant = PortableConstant::new(
            ConstantKind::Array,
            NumericDType::Float32,
            vec![2],
            vec![0, 0, 128, 63, 0, 0, 0, 64],
        )
        .unwrap();
        assert_eq!(
            constant.digest(),
            "076953dbc57aa8dc31166b382da7895310caea4299fbe8f673ba0250a8eb447f"
        );
        let encoded = constant.to_json().unwrap();
        assert_eq!(PortableConstant::from_json(&encoded).unwrap(), constant);
    }

    #[test]
    fn rank_zero_array_remains_distinct_from_scalar() {
        let bytes = 1.0_f64.to_le_bytes().to_vec();
        let scalar = PortableConstant::new(
            ConstantKind::Scalar,
            NumericDType::Float64,
            vec![],
            bytes.clone(),
        )
        .unwrap();
        let array =
            PortableConstant::new(ConstantKind::Array, NumericDType::Float64, vec![], bytes)
                .unwrap();
        assert_ne!(scalar.digest(), array.digest());
    }

    #[test]
    fn digest_corruption_rejects() {
        let constant =
            PortableConstant::new(ConstantKind::Array, NumericDType::Int8, vec![1], vec![7])
                .unwrap();
        let mut payload: Value = serde_json::from_str(&constant.to_json().unwrap()).unwrap();
        payload["data"] = Value::String("08".to_owned());
        assert!(PortableConstant::from_json(&payload.to_string()).is_err());
    }

    #[test]
    fn noncanonical_bool_bytes_reject() {
        assert!(
            PortableConstant::new(ConstantKind::Array, NumericDType::Bool, vec![2], vec![0, 2],)
                .is_err()
        );
    }
}
