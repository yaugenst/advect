//! Required `PyO3` adapter for Advect's native runtime.

mod dynamic_tape;
mod staged;

use dynamic_tape::{DynamicTape, dynamic_jvp, dynamic_jvp_many, dynamic_vjp, dynamic_vjp_many};
use pyo3::prelude::*;
use staged::{
    GraphBuilder, GraphExecutionPlan, GraphNode, GraphStore, build_graph_execution_plan,
    deserialize_graph_json, execute_graph,
};

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
