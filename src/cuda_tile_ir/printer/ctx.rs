use std::fmt::{self, Display};

use crate::cuda_tile_ir::attrs::Attr;
use crate::cuda_tile_ir::cfg::Module;
use crate::cuda_tile_ir::ids::{TypeId, ValueId};
use crate::cuda_tile_ir::ir::Operation;
use crate::cuda_tile_ir::types::Type;
use crate::cuda_tile_ir::OpAttrKey;

#[derive(Debug, Default, Clone, Copy)]
pub struct PrinterConfig {}

#[derive(Debug, Clone, Copy)]
pub struct PrinterCtx<'m> {
    pub module: &'m Module,
    pub cfg: PrinterConfig,
}

impl<'m> PrinterCtx<'m> {
    pub fn new(module: &'m Module) -> Self {
        Self {
            module,
            cfg: PrinterConfig::default(),
        }
    }

    pub fn ty(&self, id: TypeId) -> &'m Type {
        self.module.arena.type_(id)
    }

    pub fn value_ty(&self, v: ValueId) -> TypeId {
        self.module.arena.value_(v).ty
    }

    pub fn attr(&self, op: &Operation, key: OpAttrKey) -> Option<&'m Attr> {
        op.attrs
            .iter()
            .find_map(|(k, v)| (*k == key).then_some(*v))
            .and_then(|id| self.module.arena.attrs.get(id.0 as usize))
    }

    pub fn slot(&self, v: ValueId) -> Slot {
        Slot(v)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Slot(ValueId);

impl Display for Slot {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "%{}", self.0 .0)
    }
}
