//! Function table section (declarations + raw body bytes).

use bitflags::bitflags;

use crate::bytecode::attrs::RawAttr;
use crate::bytecode::consts::ConstPool;
use crate::bytecode::error::Result;
use crate::bytecode::format::{FuncId, LocIndex, StrId, TypeId};
use crate::bytecode::reader::{ByteRead, Cursor};
use crate::bytecode::strings::StringTable;
use crate::bytecode::types::TypeTable;

bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct FunctionFlags: u8 {
        const PRIVATE       = 0x01;
        const KERNEL_ENTRY  = 0x02;
        const HAS_OPT_HINTS = 0x04;
    }
}

#[derive(Debug, Clone)]
pub struct FunctionDecl<'a> {
    pub name: StrId,
    pub signature: TypeId,
    pub flags: FunctionFlags,
    pub loc: LocIndex,
    pub opt_hints: Option<RawAttr>,
    pub body: &'a [u8],
    pub id: FuncId,
}

pub fn parse_function_table<'a>(
    payload: &'a [u8],
    strings: &StringTable<'a>,
    types: &mut TypeTable<'a>,
    consts: &ConstPool<'a>,
) -> Result<Vec<FunctionDecl<'a>>> {
    let mut r = Cursor::new(payload);
    let num_funcs = r.read_var_u64()? as usize;
    let mut funcs = Vec::with_capacity(num_funcs);

    for i in 0..num_funcs {
        let name = StrId(r.read_var_u64()? as u32);
        let signature = TypeId(r.read_var_u64()? as u32);
        let flags = FunctionFlags::from_bits_truncate(r.read_u8()?);
        let loc = LocIndex(r.read_var_u64()? as u32);

        let opt_hints = if flags.contains(FunctionFlags::KERNEL_ENTRY)
            && flags.contains(FunctionFlags::HAS_OPT_HINTS)
        {
            Some(crate::bytecode::attrs::parse_self_contained_attr(
                &mut r, strings, types, consts,
            )?)
        } else {
            None
        };

        let body_len = r.read_var_u64()? as usize;
        let body = r.read_bytes(body_len)?;

        funcs.push(FunctionDecl {
            name,
            signature,
            flags,
            loc,
            opt_hints,
            body,
            id: FuncId(i as u32),
        });
    }

    Ok(funcs)
}
