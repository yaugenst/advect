//! Dynamic tape layout and lifecycle unit tests.

use pyo3::Python;

use super::layout::{LayoutPlan, validate_operand_layout};
use super::lifecycle::DynamicTape;

#[test]
fn tape_without_outputs_can_freeze() {
    Python::initialize();
    Python::attach(|py| {
        let mut tape = DynamicTape::default();

        assert!(tape.freeze(py, Vec::new(), Vec::new(), Vec::new()).is_ok());
        assert!(tape.require_available().is_ok());
        assert!(tape.require_recording().is_err());
    });
}

#[test]
fn parents_only_layout_needs_no_position_side_table() {
    assert_eq!(
        validate_operand_layout(2, &[0, 1], 0),
        Ok(LayoutPlan::ParentsOnly)
    );
}

#[test]
fn mixed_layout_preserves_parent_positions() {
    assert_eq!(
        validate_operand_layout(2, &[0, 2], 1),
        Ok(LayoutPlan::Mixed {
            positions: vec![0, 2],
            operand_count: 3,
        })
    );
}

#[test]
fn layout_rejects_duplicate_and_out_of_range_positions() {
    assert!(validate_operand_layout(2, &[0, 0], 0).is_err_and(|error| error.contains("repeats")));
    assert!(validate_operand_layout(1, &[2], 1).is_err_and(|error| error.contains("outside")));
}
