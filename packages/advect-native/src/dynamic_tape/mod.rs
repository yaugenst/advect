//! Invocation-local concrete autodiff recording and traversal.

mod forward;
mod layout;
mod lifecycle;
mod linearity;
mod reverse;

#[cfg(test)]
mod tests;

pub(crate) use forward::{dynamic_jvp, dynamic_jvp_many};
pub(crate) use lifecycle::DynamicTape;
pub(crate) use reverse::{dynamic_vjp, dynamic_vjp_many};

pub(super) const MAX_MULTI_SEEDS: usize = 16;
