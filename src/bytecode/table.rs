//! Offset-table helpers used by several sections (strings, consts, types, debug).

use crate::bytecode::error::{BytecodeError, Result};

#[derive(Debug, Clone)]
pub struct OffsetTable<'a, O> {
    pub blob: &'a [u8],
    pub offsets: Vec<O>,
}

impl<'a> OffsetTable<'a, u32> {
    pub fn len(&self) -> usize {
        self.offsets.len()
    }

    pub fn validate_monotonic(&self, table: &'static str) -> Result<()> {
        let mut prev = 0u64;
        for (idx, &off) in self.offsets.iter().enumerate() {
            let off_u64 = off as u64;
            if off_u64 < prev || off_u64 > self.blob.len() as u64 {
                return Err(BytecodeError::CorruptTable {
                    table,
                    idx,
                    offset: off_u64,
                    blob_len: self.blob.len(),
                });
            }
            prev = off_u64;
        }
        Ok(())
    }

    pub fn slice(&self, idx: usize, table: &'static str) -> Result<&'a [u8]> {
        if idx >= self.offsets.len() {
            return Err(BytecodeError::IndexOutOfBounds {
                kind: table,
                index: idx as u64,
                max: self.offsets.len(),
            });
        }

        let start = self.offsets[idx] as usize;
        let end = if idx + 1 < self.offsets.len() {
            self.offsets[idx + 1] as usize
        } else {
            self.blob.len()
        };

        if end < start || end > self.blob.len() {
            return Err(BytecodeError::CorruptTable {
                table,
                idx,
                offset: self.offsets[idx] as u64,
                blob_len: self.blob.len(),
            });
        }

        Ok(&self.blob[start..end])
    }
}

impl<'a> OffsetTable<'a, u64> {
    pub fn len(&self) -> usize {
        self.offsets.len()
    }

    pub fn validate_monotonic(&self, table: &'static str) -> Result<()> {
        let mut prev = 0u64;
        for (idx, &off) in self.offsets.iter().enumerate() {
            if off < prev || off > self.blob.len() as u64 {
                return Err(BytecodeError::CorruptTable {
                    table,
                    idx,
                    offset: off,
                    blob_len: self.blob.len(),
                });
            }
            prev = off;
        }
        Ok(())
    }

    pub fn slice(&self, idx: usize, table: &'static str) -> Result<&'a [u8]> {
        if idx >= self.offsets.len() {
            return Err(BytecodeError::IndexOutOfBounds {
                kind: table,
                index: idx as u64,
                max: self.offsets.len(),
            });
        }

        let start = self.offsets[idx] as usize;
        let end = if idx + 1 < self.offsets.len() {
            self.offsets[idx + 1] as usize
        } else {
            self.blob.len()
        };

        if end < start || end > self.blob.len() {
            return Err(BytecodeError::CorruptTable {
                table,
                idx,
                offset: self.offsets[idx],
                blob_len: self.blob.len(),
            });
        }

        Ok(&self.blob[start..end])
    }
}
