//! Compact operand layouts for dynamic tape nodes.

use std::ops::Range;

use advect_runtime::NodeId;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

/// Where one node's operands live. Parents fill operand positions in order,
/// or, when mixed, sit at recorded positions while the node's literals fill
/// the remaining positions in order.
#[derive(Clone, Copy, Debug)]
pub(super) enum OperandLayout {
    ParentsOnly,
    Mixed {
        position_start: u32,
        literal_start: u32,
        literal_count: u32,
    },
}

/// One node's decoded operand layout.
#[derive(Debug)]
pub(super) struct Operands {
    /// Operand position of each parent, in parent order.
    pub(super) parent_positions: Vec<usize>,
    /// Literal arena slots, in operand order.
    pub(super) literals: Range<usize>,
}

#[derive(Debug)]
pub(super) struct OperandSnapshot {
    pub(super) parents: Vec<NodeId>,
    pub(super) layout: Operands,
    pub(super) parent_active: Vec<bool>,
    pub(super) operands: Vec<Py<PyAny>>,
    pub(super) parent_specs: Vec<Option<(Vec<usize>, Py<PyAny>)>>,
}

impl OperandLayout {
    /// Validate one recorded layout against the current arena lengths, and
    /// return it with the parent positions to append to the position arena.
    /// Without explicit positions, parents fill every operand position.
    pub(super) fn validate(
        parent_positions: Option<&[usize]>,
        parent_count: usize,
        literal_count: usize,
        position_start: usize,
        literal_start: usize,
    ) -> Result<(Self, Vec<u32>), String> {
        let parent_positions = match parent_positions {
            None if literal_count == 0 => return Ok((Self::ParentsOnly, Vec::new())),
            None => {
                return Err(format!(
                    "dynamic operand layout has {literal_count} literals but no parent positions"
                ));
            }
            Some(parent_positions) => parent_positions,
        };
        if parent_positions.len() != parent_count {
            return Err(format!(
                "dynamic operand layout has {parent_count} parents but {} parent positions",
                parent_positions.len()
            ));
        }
        if literal_count == 0 && parent_positions.iter().copied().eq(0..parent_count) {
            return Ok((Self::ParentsOnly, Vec::new()));
        }
        let operand_count = parent_count
            .checked_add(literal_count)
            .ok_or_else(|| "dynamic operand count overflowed".to_owned())?;
        let mut occupied = vec![false; operand_count];
        for &position in parent_positions {
            let slot = occupied.get_mut(position).ok_or_else(|| {
                format!("parent position {position} is outside operand arity {operand_count}")
            })?;
            if std::mem::replace(slot, true) {
                return Err(format!("operand layout repeats parent position {position}"));
            }
        }
        let gaps = occupied.iter().filter(|&&value| !value).count();
        if gaps != literal_count {
            return Err(format!(
                "operand layout has {gaps} literal slots but {literal_count} literals"
            ));
        }
        let positions = parent_positions
            .iter()
            .map(|&position| {
                u32::try_from(position)
                    .map_err(|_| "operand position exceeded its index range".to_owned())
            })
            .collect::<Result<Vec<_>, _>>()?;
        let layout = Self::Mixed {
            position_start: u32::try_from(position_start).map_err(|_| {
                "dynamic tape operand-position arena exceeded its index range".to_owned()
            })?,
            literal_start: u32::try_from(literal_start)
                .map_err(|_| "dynamic tape literal arena exceeded its index range".to_owned())?,
            literal_count: u32::try_from(literal_count)
                .map_err(|_| "dynamic tape node has too many literals".to_owned())?,
        };
        Ok((layout, positions))
    }

    /// Literal arena slots owned by this node.
    pub(super) fn literal_slots(self) -> PyResult<Range<usize>> {
        let Self::Mixed {
            literal_start,
            literal_count,
            ..
        } = self
        else {
            return Ok(0..0);
        };
        let start = arena_index(literal_start)?;
        let end = start
            .checked_add(arena_index(literal_count)?)
            .ok_or_else(|| PyRuntimeError::new_err("literal range overflowed"))?;
        Ok(start..end)
    }

    /// Decode parent positions from the position arena.
    pub(super) fn decode(
        self,
        operand_positions: &[u32],
        parent_count: usize,
    ) -> PyResult<Operands> {
        let parent_positions = match self {
            Self::ParentsOnly => (0..parent_count).collect(),
            Self::Mixed { position_start, .. } => {
                let start = arena_index(position_start)?;
                let end = start
                    .checked_add(parent_count)
                    .ok_or_else(|| PyRuntimeError::new_err("operand position range overflowed"))?;
                operand_positions
                    .get(start..end)
                    .ok_or_else(|| PyRuntimeError::new_err("operand position range is invalid"))?
                    .iter()
                    .map(|&position| arena_index(position))
                    .collect::<PyResult<_>>()?
            }
        };
        Ok(Operands {
            parent_positions,
            literals: self.literal_slots()?,
        })
    }
}

impl Operands {
    /// Order per-parent and per-literal items by operand position.
    pub(super) fn interleave<T>(
        &self,
        parents: impl IntoIterator<Item = T>,
        literals: impl IntoIterator<Item = T>,
    ) -> PyResult<Vec<T>> {
        let count = self.parent_positions.len() + self.literals.len();
        if self.parent_positions.iter().copied().eq(0..count) {
            // Parents already fill every position in order.
            return Ok(parents.into_iter().collect());
        }
        let mut slots: Vec<Option<T>> = std::iter::repeat_with(|| None).take(count).collect();
        for (&position, item) in self.parent_positions.iter().zip(parents) {
            *slots
                .get_mut(position)
                .ok_or_else(|| PyRuntimeError::new_err("operand position is unavailable"))? =
                Some(item);
        }
        let mut literals = literals.into_iter();
        for slot in slots.iter_mut().filter(|slot| slot.is_none()) {
            *slot = Some(
                literals
                    .next()
                    .ok_or_else(|| PyRuntimeError::new_err("literal layout is inconsistent"))?,
            );
        }
        if literals.next().is_some() {
            return Err(PyRuntimeError::new_err(
                "literal layout retained unused values",
            ));
        }
        slots
            .into_iter()
            .map(|slot| {
                slot.ok_or_else(|| PyRuntimeError::new_err("operand slot is uninitialized"))
            })
            .collect()
    }
}

fn arena_index(value: u32) -> PyResult<usize> {
    usize::try_from(value)
        .map_err(|_| PyRuntimeError::new_err("operand layout index is out of range"))
}
