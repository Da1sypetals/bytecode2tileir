//! IR operation node.

use smallvec::SmallVec;

use crate::cuda_tile_ir::debug::Location;
use crate::cuda_tile_ir::ids::{AttrId, OpId, RegionId, ValueId};
use crate::cuda_tile_ir::{OpAttrKey, Opcode};

pub type AttrMap = SmallVec<[(OpAttrKey, AttrId); 8]>;

#[derive(Debug, Clone)]
pub struct Operation {
    pub opcode: Opcode,
    pub operands: SmallVec<[ValueId; 4]>,
    pub results: SmallVec<[ValueId; 2]>,
    pub attrs: AttrMap,
    pub regions: Vec<RegionId>,
    pub loc: Location,
}

impl Operation {
    pub fn id_debug(_id: OpId) -> &'static str {
        "op"
    }
}
