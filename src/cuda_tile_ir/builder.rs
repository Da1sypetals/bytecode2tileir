//! Module builder around `IrArena`.

use crate::cuda_tile_ir::arena::IrArena;
use crate::cuda_tile_ir::cfg::{Function, Global, Module};
use crate::cuda_tile_ir::consts::ConstPool;
use crate::cuda_tile_ir::ids::{FunctionId, GlobalId};

#[derive(Debug)]
pub struct IrBuilder {
    pub name: String,
    pub arena: IrArena,
    pub globals: Vec<GlobalId>,
    pub functions: Vec<FunctionId>,
    pub consts: ConstPool,
}

impl IrBuilder {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            arena: IrArena::new(),
            globals: Vec::new(),
            functions: Vec::new(),
            consts: ConstPool::empty(),
        }
    }

    pub fn add_global(&mut self, g: Global) -> GlobalId {
        let id = self.arena.new_global(g);
        self.globals.push(id);
        id
    }

    pub fn add_function(&mut self, f: Function) -> FunctionId {
        let id = self.arena.new_function(f);
        self.functions.push(id);
        id
    }

    pub fn build(self) -> Module {
        Module {
            name: self.name,
            globals: self.globals,
            functions: self.functions,
            arena: self.arena,
            consts: self.consts,
        }
    }
}
