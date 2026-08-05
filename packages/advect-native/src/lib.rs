//! Canonical storage for staged graphs.

mod deserialization;
mod dynamic_tape;
mod graph_builder;
mod graph_store;
mod node;
mod staged_execution;

use deserialization::deserialize_graph_json;
use dynamic_tape::{DynamicTape, dynamic_jvp, dynamic_jvp_many, dynamic_vjp, dynamic_vjp_many};
use graph_builder::GraphBuilder;
use graph_store::GraphStore;
use node::GraphNode;
use pyo3::prelude::*;
use staged_execution::{GraphExecutionPlan, build_graph_execution_plan, execute_graph};

#[pymodule]
fn _native_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add(
        "__build_profile__",
        if cfg!(debug_assertions) {
            "debug"
        } else {
            "release"
        },
    )?;
    m.add_class::<GraphNode>()?;
    m.add_class::<GraphBuilder>()?;
    m.add_class::<GraphStore>()?;
    m.add_class::<GraphExecutionPlan>()?;
    m.add_class::<DynamicTape>()?;
    m.add_function(wrap_pyfunction!(deserialize_graph_json, m)?)?;
    m.add_function(wrap_pyfunction!(dynamic_jvp, m)?)?;
    m.add_function(wrap_pyfunction!(dynamic_jvp_many, m)?)?;
    m.add_function(wrap_pyfunction!(dynamic_vjp, m)?)?;
    m.add_function(wrap_pyfunction!(dynamic_vjp_many, m)?)?;
    m.add_function(wrap_pyfunction!(build_graph_execution_plan, m)?)?;
    m.add_function(wrap_pyfunction!(execute_graph, m)?)?;
    Ok(())
}
