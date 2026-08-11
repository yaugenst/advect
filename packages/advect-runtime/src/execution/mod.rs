//! Host-driven execution with runtime-owned value lifetimes.

mod evaluate;
mod host;
mod plan;

pub use host::{Host, LinkedOperation, Operand, OutputOwnership};
pub use plan::LinkedExecutionPlan;

#[cfg(test)]
#[expect(
    clippy::unwrap_used,
    reason = "test setup unwraps values whose absence should fail the test"
)]
mod tests;
