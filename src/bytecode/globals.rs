//! Global section parsing.

use crate::bytecode::error::Result;
use crate::bytecode::format::{ConstId, GlobalId, StrId, TypeId};
use crate::bytecode::reader::{ByteRead, Cursor};

#[derive(Debug, Clone)]
pub struct GlobalDecl {
    pub name: StrId,
    pub ty: TypeId,
    pub init: ConstId,
    pub alignment: u64,
    pub id: GlobalId,
}

pub fn parse_global_table(payload: &[u8]) -> Result<Vec<GlobalDecl>> {
    let mut r = Cursor::new(payload);
    let num = r.read_var_u64()? as usize;
    let mut globals = Vec::with_capacity(num);

    for i in 0..num {
        let name = StrId(r.read_var_u64()? as u32);
        let ty = TypeId(r.read_var_u64()? as u32);
        let init = ConstId(r.read_var_u64()? as u32);
        let alignment = r.read_var_u64()?;
        globals.push(GlobalDecl {
            name,
            ty,
            init,
            alignment,
            id: GlobalId(i as u32),
        });
    }

    Ok(globals)
}
