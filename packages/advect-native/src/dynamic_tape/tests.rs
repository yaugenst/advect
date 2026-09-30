//! Dynamic tape layout and lifecycle unit tests.

use pyo3::Python;

use super::layout::OperandLayout;
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
    for positions in [None, Some([0, 1].as_slice())] {
        assert!(matches!(
            OperandLayout::validate(positions, 2, 0, 5, 7),
            Ok((OperandLayout::ParentsOnly, positions)) if positions.is_empty()
        ));
    }
}

#[test]
fn mixed_layout_preserves_parent_positions() {
    let (layout, positions) = OperandLayout::validate(Some(&[0, 2]), 2, 1, 5, 7).unwrap();
    assert_eq!(positions, [0, 2]);
    let mut arena = vec![9; 5];
    arena.extend(positions);
    let operands = layout.decode(&arena, 2).unwrap();
    assert_eq!(operands.parent_positions, [0, 2]);
    assert_eq!(operands.literals, 7..8);
    assert_eq!(
        operands.interleave(["a", "c"], ["b"]).unwrap(),
        ["a", "b", "c"]
    );
}

#[test]
fn layout_rejects_duplicate_and_out_of_range_positions() {
    assert!(
        OperandLayout::validate(Some(&[0, 0]), 2, 0, 0, 0)
            .is_err_and(|error| error.contains("repeats"))
    );
    assert!(
        OperandLayout::validate(Some(&[2]), 1, 1, 0, 0)
            .is_err_and(|error| error.contains("outside"))
    );
    assert!(
        OperandLayout::validate(None, 1, 1, 0, 0)
            .is_err_and(|error| error.contains("no parent positions"))
    );
}
