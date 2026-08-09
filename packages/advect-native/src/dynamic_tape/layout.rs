//! Compact operand-layout plans and snapshots for dynamic tape nodes.

use advect_runtime::NodeId;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

pub(super) type LayoutSnapshot = (Vec<usize>, Option<(usize, usize)>);

#[derive(Clone, Copy, Debug)]
pub(super) enum OperandLayout {
    ParentsOnly,
    Mixed {
        position_start: u32,
        literal_start: u32,
        operand_count: u32,
        literal_count: u32,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) enum LayoutPlan {
    ParentsOnly,
    Mixed {
        positions: Vec<u32>,
        operand_count: u32,
    },
}

#[derive(Debug)]
pub(super) struct OperandSnapshot {
    pub(super) parents: Vec<NodeId>,
    pub(super) parent_positions: Vec<usize>,
    pub(super) parent_active: Vec<bool>,
    pub(super) active_positions: Vec<usize>,
    pub(super) operands: Vec<Py<PyAny>>,
    pub(super) parent_specs: Vec<Option<(Vec<usize>, Py<PyAny>)>>,
}

pub(super) fn prepare_layout(
    plan: &LayoutPlan,
    position_len: usize,
    literal_len: usize,
    literal_count: usize,
) -> PyResult<OperandLayout> {
    match plan {
        LayoutPlan::ParentsOnly => Ok(OperandLayout::ParentsOnly),
        LayoutPlan::Mixed {
            positions: _,
            operand_count,
        } => {
            let position_start = u32::try_from(position_len).map_err(|_| {
                PyValueError::new_err(
                    "dynamic tape operand-position arena exceeded its index range",
                )
            })?;
            let literal_start = u32::try_from(literal_len).map_err(|_| {
                PyValueError::new_err("dynamic tape literal arena exceeded its index range")
            })?;
            let literal_count = u32::try_from(literal_count)
                .map_err(|_| PyValueError::new_err("dynamic tape node has too many literals"))?;
            Ok(OperandLayout::Mixed {
                position_start,
                literal_start,
                operand_count: *operand_count,
                literal_count,
            })
        }
    }
}

pub(super) fn commit_layout(
    operand_positions: &mut Vec<u32>,
    literal_slots: &mut Vec<Option<Py<PyAny>>>,
    plan: LayoutPlan,
    literals: Vec<Py<PyAny>>,
) {
    if let LayoutPlan::Mixed { positions, .. } = plan {
        operand_positions.extend(positions);
        literal_slots.extend(literals.into_iter().map(Some));
    }
}

pub(super) fn snapshot_layout(
    layouts: &[OperandLayout],
    operand_positions: &[u32],
    node_index: usize,
    parent_count: usize,
) -> PyResult<LayoutSnapshot> {
    match *layouts.get(node_index).ok_or_else(|| {
        PyRuntimeError::new_err("dynamic tape operand layout arena is inconsistent")
    })? {
        OperandLayout::ParentsOnly => Ok(((0..parent_count).collect(), None)),
        OperandLayout::Mixed {
            position_start,
            literal_start,
            operand_count: _,
            literal_count,
        } => {
            let position_start = usize::try_from(position_start)
                .map_err(|_| PyRuntimeError::new_err("operand position start is out of range"))?;
            let position_end = position_start
                .checked_add(parent_count)
                .ok_or_else(|| PyRuntimeError::new_err("operand position range overflowed"))?;
            let positions = operand_positions
                .get(position_start..position_end)
                .ok_or_else(|| PyRuntimeError::new_err("operand position range is invalid"))?
                .iter()
                .map(|&position| {
                    usize::try_from(position)
                        .map_err(|_| PyRuntimeError::new_err("operand position is out of range"))
                })
                .collect::<PyResult<Vec<_>>>()?;
            Ok((
                positions,
                Some((
                    usize::try_from(literal_start)
                        .map_err(|_| PyRuntimeError::new_err("literal start is out of range"))?,
                    usize::try_from(literal_count)
                        .map_err(|_| PyRuntimeError::new_err("literal count is out of range"))?,
                )),
            ))
        }
    }
}

pub(super) fn validate_operand_layout(
    parent_count: usize,
    parent_positions: &[usize],
    literal_count: usize,
) -> Result<LayoutPlan, String> {
    if parent_positions.len() != parent_count {
        return Err(format!(
            "dynamic operand layout has {parent_count} parents but {} parent positions",
            parent_positions.len()
        ));
    }
    let operand_count = parent_count
        .checked_add(literal_count)
        .ok_or_else(|| "dynamic operand count overflowed".to_owned())?;
    let mut occupied = vec![false; operand_count];
    for &position in parent_positions {
        let slot = occupied.get_mut(position).ok_or_else(|| {
            format!("parent position {position} is outside operand arity {operand_count}")
        })?;
        if *slot {
            return Err(format!("operand layout repeats parent position {position}"));
        }
        *slot = true;
    }
    let gaps = occupied.iter().filter(|&&value| !value).count();
    if gaps != literal_count {
        return Err(format!(
            "operand layout has {gaps} literal slots but {literal_count} literals"
        ));
    }
    if literal_count == 0
        && parent_positions
            .iter()
            .copied()
            .eq(0..parent_positions.len())
    {
        return Ok(LayoutPlan::ParentsOnly);
    }
    let positions = parent_positions
        .iter()
        .copied()
        .map(|position| {
            u32::try_from(position)
                .map_err(|_| "operand position exceeded its index range".to_owned())
        })
        .collect::<Result<Vec<_>, _>>()?;
    let operand_count = u32::try_from(operand_count)
        .map_err(|_| "operand count exceeded its index range".to_owned())?;
    Ok(LayoutPlan::Mixed {
        positions,
        operand_count,
    })
}
