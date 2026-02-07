//! String section table.

use crate::bytecode::error::{BytecodeError, Result};
use crate::bytecode::format::StrId;
use crate::bytecode::reader::{ByteRead, Cursor};
use crate::bytecode::table::OffsetTable;

pub struct StringTable<'a> {
    table: OffsetTable<'a, u32>,
}

impl<'a> StringTable<'a> {
    pub fn parse(payload: &'a [u8]) -> Result<Self> {
        let mut r = Cursor::new(payload);
        let num = r.read_var_u64()? as usize;

        // Offsets are uint32_t[]; writers pad to 4-byte boundary.
        r.align_to(4, 0xCB)?;

        let mut offsets = Vec::with_capacity(num);
        for _ in 0..num {
            offsets.push(r.read_u32_le()?);
        }

        let blob = r.slice_from_pos();
        let table = OffsetTable { blob, offsets };
        table.validate_monotonic("string")?;

        Ok(Self { table })
    }

    pub fn get(&self, id: StrId) -> Result<&'a str> {
        let idx = id.0 as usize;
        let bytes = self.table.slice(idx, "string")?;
        std::str::from_utf8(bytes).map_err(|_| BytecodeError::InvalidUtf8)
    }
}
