//! Constant section table.

use crate::bytecode::error::Result;
use crate::bytecode::format::ConstId;
use crate::bytecode::reader::{ByteRead, Cursor};
use crate::bytecode::table::OffsetTable;

pub struct ConstPool<'a> {
    table: OffsetTable<'a, u64>,
}

impl<'a> ConstPool<'a> {
    pub fn empty() -> Self {
        Self {
            table: OffsetTable {
                blob: &[],
                offsets: Vec::new(),
            },
        }
    }

    pub fn parse(payload: &'a [u8]) -> Result<Self> {
        let mut r = Cursor::new(payload);
        let num = r.read_var_u64()? as usize;

        // Offsets are uint64_t[]; writers pad to 8-byte boundary.
        r.align_to(8, 0xCB)?;

        let mut offsets = Vec::with_capacity(num);
        for _ in 0..num {
            offsets.push(r.read_u64_le()?);
        }

        let blob = r.slice_from_pos();
        let table = OffsetTable { blob, offsets };
        table.validate_monotonic("const")?;

        Ok(Self { table })
    }

    pub fn get(&self, id: ConstId) -> Result<&'a [u8]> {
        let idx = id.0 as usize;
        self.table.slice(idx, "const")
    }

    pub fn len(&self) -> usize {
        self.table.len()
    }

    pub fn offsets(&self) -> &[u64] {
        &self.table.offsets
    }

    pub fn blob(&self) -> &'a [u8] {
        self.table.blob
    }
}
