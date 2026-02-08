//! CFG containers: block/region/function/module.

use crate::bytecode::format::ConstId;
use crate::bytecode::funcs::FunctionFlags;
use crate::cuda_tile_ir::debug::Location;
use crate::cuda_tile_ir::ids::{
    AttrId, BlockId, FunctionId, GlobalId, OpId, RegionId, TypeId, ValueId,
};

#[derive(Debug, Clone)]
pub struct Block {
    pub args: Vec<ValueId>,
    pub ops: Vec<OpId>,
}

#[derive(Debug, Clone)]
pub struct Region {
    pub blocks: Vec<BlockId>,
}

#[derive(Debug, Clone)]
pub struct Function {
    pub name: String,
    pub signature: TypeId,
    pub flags: FunctionFlags,
    pub loc: Location,
    pub opt_hints: Option<AttrId>,
    pub body: RegionId,
}

#[derive(Debug, Clone)]
pub struct Global {
    pub name: String,
    pub ty: TypeId,
    pub init: ConstId,
    pub alignment: u64,
}

#[derive(Debug)]
pub struct Module {
    pub name: String,
    pub globals: Vec<GlobalId>,
    pub functions: Vec<FunctionId>,
    pub arena: crate::cuda_tile_ir::arena::IrArena,
    pub consts: crate::cuda_tile_ir::consts::ConstPool,
}
