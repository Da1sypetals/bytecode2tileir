//! IR arena storage.

use crate::cuda_tile_ir::attrs::Attr;
use crate::cuda_tile_ir::cfg::{Block, Function, Global, Region};
use crate::cuda_tile_ir::ids::{AttrId, BlockId, FunctionId, GlobalId, OpId, RegionId, TypeId, ValueId};
use crate::cuda_tile_ir::ir::Operation;
use crate::cuda_tile_ir::types::Type;
use crate::cuda_tile_ir::value::ValueData;

#[derive(Debug, Default)]
pub struct IrArena {
    pub types: Vec<Type>,
    pub attrs: Vec<Attr>,
    pub values: Vec<ValueData>,
    pub ops: Vec<Operation>,
    pub blocks: Vec<Block>,
    pub regions: Vec<Region>,
    pub globals: Vec<Global>,
    pub functions: Vec<Function>,
}

impl IrArena {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn type_(&self, id: TypeId) -> &Type {
        &self.types[id.0 as usize]
    }

    pub fn attr_(&self, id: AttrId) -> &Attr {
        &self.attrs[id.0 as usize]
    }

    pub fn value_(&self, id: ValueId) -> &ValueData {
        &self.values[id.0 as usize]
    }

    pub fn op_(&self, id: OpId) -> &Operation {
        &self.ops[id.0 as usize]
    }

    pub fn block_(&self, id: BlockId) -> &Block {
        &self.blocks[id.0 as usize]
    }

    pub fn region_(&self, id: RegionId) -> &Region {
        &self.regions[id.0 as usize]
    }

    pub fn global_(&self, id: GlobalId) -> &Global {
        &self.globals[id.0 as usize]
    }

    pub fn function_(&self, id: FunctionId) -> &Function {
        &self.functions[id.0 as usize]
    }

    pub fn intern_type(&mut self, ty: Type) -> TypeId {
        let id = TypeId(self.types.len() as u32);
        self.types.push(ty);
        id
    }

    pub fn intern_attr(&mut self, attr: Attr) -> AttrId {
        let id = AttrId(self.attrs.len() as u32);
        self.attrs.push(attr);
        id
    }

    pub fn new_value(&mut self, data: ValueData) -> ValueId {
        let id = ValueId(self.values.len() as u32);
        self.values.push(data);
        id
    }

    pub fn new_op(&mut self, op: Operation) -> OpId {
        let id = OpId(self.ops.len() as u32);
        self.ops.push(op);
        id
    }

    pub fn new_block(&mut self, b: Block) -> BlockId {
        let id = BlockId(self.blocks.len() as u32);
        self.blocks.push(b);
        id
    }

    pub fn new_region(&mut self, r: Region) -> RegionId {
        let id = RegionId(self.regions.len() as u32);
        self.regions.push(r);
        id
    }

    pub fn new_global(&mut self, g: Global) -> GlobalId {
        let id = GlobalId(self.globals.len() as u32);
        self.globals.push(g);
        id
    }

    pub fn new_function(&mut self, f: Function) -> FunctionId {
        let id = FunctionId(self.functions.len() as u32);
        self.functions.push(f);
        id
    }
}
