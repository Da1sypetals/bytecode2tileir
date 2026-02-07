//! SSA value model.

use crate::cuda_tile_ir::ids::{BlockId, OpId, TypeId, ValueId};

#[derive(Debug, Clone)]
pub enum ValueDef {
    BlockArg { block: BlockId, index: u32 },
    OpResult { op: OpId, index: u32 },
}

#[derive(Debug, Clone)]
pub struct ValueData {
    pub ty: TypeId,
    pub def: ValueDef,
}

impl ValueData {
    pub fn ty(&self) -> TypeId {
        self.ty
    }

    pub fn is_block_arg(&self) -> bool {
        matches!(self.def, ValueDef::BlockArg { .. })
    }

    pub fn is_op_result(&self) -> bool {
        matches!(self.def, ValueDef::OpResult { .. })
    }

    pub fn as_block_arg(&self) -> Option<(BlockId, u32)> {
        match self.def {
            ValueDef::BlockArg { block, index } => Some((block, index)),
            _ => None,
        }
    }

    pub fn as_op_result(&self) -> Option<(OpId, u32)> {
        match self.def {
            ValueDef::OpResult { op, index } => Some((op, index)),
            _ => None,
        }
    }

    pub fn id_debug(_id: ValueId) -> &'static str {
        "%v"
    }
}
