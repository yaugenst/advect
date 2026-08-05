//! Closed graph-attribute algebra.

use std::collections::BTreeMap;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// Deterministically ordered attribute mapping.
pub type AttrMap = BTreeMap<String, AttrValue>;

/// A Python-independent, hashable graph attribute value.
#[derive(Clone, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[serde(
    tag = "kind",
    content = "value",
    rename_all = "snake_case",
    deny_unknown_fields
)]
pub enum AttrValue {
    /// Null.
    Null,
    /// Boolean scalar.
    Bool(bool),
    /// Signed 64-bit integer.
    Integer(i64),
    /// Exact IEEE-754 binary64 value.
    Float(ExactFloat),
    /// Unicode string.
    String(String),
    /// Immutable byte string encoded textually in artifacts.
    Bytes(#[serde(with = "hex_bytes")] Vec<u8>),
    /// Mutable-sequence semantics.
    List(Vec<Self>),
    /// Immutable-sequence semantics.
    Tuple(Vec<Self>),
    /// String-keyed mapping.
    Map(AttrMap),
}

/// Hashable IEEE-754 representation preserving signed zero and NaN payloads.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ExactFloat(u64);

impl ExactFloat {
    /// Capture one floating-point value exactly.
    #[must_use]
    pub const fn from_f64(value: f64) -> Self {
        Self(value.to_bits())
    }

    /// Restore the floating-point value.
    #[must_use]
    pub const fn to_f64(self) -> f64 {
        f64::from_bits(self.0)
    }

    /// Raw IEEE-754 bits.
    #[must_use]
    pub const fn bits(self) -> u64 {
        self.0
    }
}

impl Serialize for ExactFloat {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&format!("{:016x}", self.0))
    }
}

impl<'de> Deserialize<'de> for ExactFloat {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let encoded = String::deserialize(deserializer)?;
        if encoded.len() != 16
            || !encoded
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            return Err(serde::de::Error::custom(
                "exact float bits must be 16 lowercase hexadecimal characters",
            ));
        }
        u64::from_str_radix(&encoded, 16)
            .map(Self)
            .map_err(serde::de::Error::custom)
    }
}

mod hex_bytes {
    use serde::{Deserialize, Deserializer, Serializer};

    pub(super) fn serialize<S>(bytes: &[u8], serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&encode(bytes))
    }

    pub(super) fn deserialize<'de, D>(deserializer: D) -> Result<Vec<u8>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let encoded = String::deserialize(deserializer)?;
        decode(&encoded).map_err(serde::de::Error::custom)
    }

    fn encode(bytes: &[u8]) -> String {
        let mut encoded = String::with_capacity(bytes.len().saturating_mul(2));
        for &byte in bytes {
            encoded.push(hex_digit(byte >> 4));
            encoded.push(hex_digit(byte & 0x0f));
        }
        encoded
    }

    fn decode(encoded: &str) -> Result<Vec<u8>, &'static str> {
        if !encoded.len().is_multiple_of(2) {
            return Err("hex byte string must contain an even number of characters");
        }
        encoded
            .as_bytes()
            .chunks_exact(2)
            .map(|pair| {
                let high = digit(*pair.first().ok_or("missing high hex digit")?)?;
                let low = digit(*pair.get(1).ok_or("missing low hex digit")?)?;
                Ok((high << 4) | low)
            })
            .collect()
    }

    fn digit(value: u8) -> Result<u8, &'static str> {
        match value {
            b'0'..=b'9' => Ok(value - b'0'),
            b'a'..=b'f' => Ok(value - b'a' + 10),
            _ => Err("hex byte string must use lowercase hexadecimal"),
        }
    }

    fn hex_digit(value: u8) -> char {
        char::from(if value < 10 {
            b'0' + value
        } else {
            b'a' + (value - 10)
        })
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn exact_float_and_bytes_round_trip() {
        let attrs = AttrValue::Tuple(vec![
            AttrValue::Float(ExactFloat::from_f64(-0.0)),
            AttrValue::Float(ExactFloat::from_f64(f64::from_bits(0x7ff8_0000_0000_0042))),
            AttrValue::Bytes(vec![0, 127, 255]),
        ]);
        let encoded = serde_json::to_string(&attrs).unwrap();
        let decoded: AttrValue = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded, attrs);
        assert!(encoded.contains("\"007fff\""));
    }

    #[test]
    fn uppercase_exact_float_bits_reject() {
        let encoded = r#"{"kind":"float","value":"000000000000000A"}"#;
        assert!(serde_json::from_str::<AttrValue>(encoded).is_err());
    }
}
