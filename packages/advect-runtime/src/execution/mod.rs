//! Host-driven execution with runtime-owned value lifetimes.

mod evaluate;
mod host;
mod plan;

pub use host::{Host, LinkedOperation, Operand, OutputOwnership};
pub use plan::LinkedExecutionPlan;

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests;
