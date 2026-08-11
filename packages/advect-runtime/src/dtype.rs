//! Backend-neutral dtype descriptors.

use std::fmt::{self, Display, Formatter};

use serde::{Deserialize, Serialize};

/// Stable, Python-independent dtype identity.
#[derive(Clone, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
pub struct DTypeDescriptor {
    canonical: String,
    display: String,
}

impl DTypeDescriptor {
    /// Construct a descriptor from one logical name.
    pub fn from_name(name: &str) -> Result<Self, DTypeError> {
        let display = display_name(name);
        let canonical = canonical_name(&display);
        Self::from_parts(canonical, display)
    }

    /// Construct a descriptor from explicit stable identity and display text.
    pub fn from_parts(
        canonical: impl Into<String>,
        display: impl Into<String>,
    ) -> Result<Self, DTypeError> {
        let canonical = canonical.into();
        let display = display.into();
        validate(&canonical, &display)?;
        Ok(Self { canonical, display })
    }

    /// Stable equality and artifact identity.
    #[must_use]
    pub fn canonical(&self) -> &str {
        &self.canonical
    }

    /// Backend-neutral display name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.display
    }
}

/// Invalid dtype descriptor.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DTypeError(String);

impl DTypeError {
    /// Consume the error into its message.
    #[must_use]
    pub fn into_message(self) -> String {
        self.0
    }
}

impl Display for DTypeError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for DTypeError {}

fn validate(canonical: &str, display: &str) -> Result<(), DTypeError> {
    if canonical.is_empty() || display.is_empty() {
        return Err(DTypeError("dtype descriptor must not be empty".to_owned()));
    }
    if canonical.contains('\0') || display.contains('\0') {
        return Err(DTypeError(
            "dtype descriptor must not contain NUL characters".to_owned(),
        ));
    }
    Ok(())
}

fn display_name(name: &str) -> String {
    const ROOTS: &[&str] = &["array_api_strict", "cupy", "numpy"];
    let trimmed = name.trim();
    for root in ROOTS {
        if let Some(suffix) = trimmed
            .strip_prefix(root)
            .and_then(|value| value.strip_prefix('.'))
            && !suffix.is_empty()
            && !suffix.contains('.')
        {
            return suffix.to_owned();
        }
    }
    trimmed.to_owned()
}

fn canonical_name(name: &str) -> String {
    match name {
        "bool_" => "bool",
        "byte" => "int8",
        "ubyte" => "uint8",
        "short" => "int16",
        "ushort" => "uint16",
        "int" | "intp" | "long" => "int64",
        "uint" | "uintp" | "ulong" => "uint64",
        "half" => "float16",
        "single" => "float32",
        "double" | "float" => "float64",
        "csingle" => "complex64",
        "cdouble" | "complex" => "complex128",
        _ => name,
    }
    .to_owned()
}

#[cfg(test)]
#[expect(
    clippy::unwrap_used,
    reason = "test setup unwraps values whose absence should fail the test"
)]
mod tests {
    use super::*;

    #[test]
    fn logical_names_do_not_depend_on_host_endian() {
        assert_eq!(
            DTypeDescriptor::from_name("numpy.float32")
                .unwrap()
                .canonical(),
            "float32"
        );
        assert_eq!(
            DTypeDescriptor::from_name(">f8").unwrap().canonical(),
            ">f8"
        );
    }
}
