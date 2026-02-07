//! Byte reader primitives (Cursor + varint + alignment).

use crate::bytecode::error::{BytecodeError, Result};

pub trait ByteRead<'a> {
    fn pos(&self) -> usize;
    fn remaining(&self) -> usize;

    fn read_u8(&mut self) -> Result<u8>;
    fn read_u16_le(&mut self) -> Result<u16>;
    fn read_u32_le(&mut self) -> Result<u32>;
    fn read_u64_le(&mut self) -> Result<u64>;

    fn read_var_u64(&mut self) -> Result<u64>;
    fn read_var_i64(&mut self) -> Result<i64>;

    fn read_bytes(&mut self, n: usize) -> Result<&'a [u8]>;

    /// Align to `align` by consuming padding bytes. Verifies each padding byte equals `pad`.
    fn align_to(&mut self, align: usize, pad: u8) -> Result<()>;

    /// Align to `align` by consuming padding bytes without validating their values.
    fn align_to_skip(&mut self, align: usize) -> Result<()> {
        if align < 2 {
            return Ok(());
        }
        let padding = (align - (self.pos() % align)) % align;
        let _ = self.read_bytes(padding)?;
        Ok(())
    }
}

pub struct Cursor<'a> {
    data: &'a [u8],
    offset: usize,
}

impl<'a> Cursor<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, offset: 0 }
    }

    pub fn slice_from_pos(&self) -> &'a [u8] {
        &self.data[self.offset..]
    }

    pub fn read_i32_le(&mut self) -> Result<i32> {
        Ok(self.read_u32_le()? as i32)
    }

    pub fn read_i64_le(&mut self) -> Result<i64> {
        Ok(self.read_u64_le()? as i64)
    }
}

impl<'a> ByteRead<'a> for Cursor<'a> {
    fn pos(&self) -> usize {
        self.offset
    }

    fn remaining(&self) -> usize {
        self.data.len().saturating_sub(self.offset)
    }

    fn read_u8(&mut self) -> Result<u8> {
        if self.offset >= self.data.len() {
            return Err(BytecodeError::UnexpectedEof {
                at: self.offset,
                needed: 1,
                remaining: 0,
            });
        }
        let v = self.data[self.offset];
        self.offset += 1;
        Ok(v)
    }

    fn read_u16_le(&mut self) -> Result<u16> {
        let bytes = self.read_bytes(2)?;
        Ok(u16::from_le_bytes([bytes[0], bytes[1]]))
    }

    fn read_u32_le(&mut self) -> Result<u32> {
        let bytes = self.read_bytes(4)?;
        Ok(u32::from_le_bytes(bytes.try_into().unwrap()))
    }

    fn read_u64_le(&mut self) -> Result<u64> {
        let bytes = self.read_bytes(8)?;
        Ok(u64::from_le_bytes(bytes.try_into().unwrap()))
    }

    fn read_var_u64(&mut self) -> Result<u64> {
        let mut result = 0u64;
        let mut shift = 0u32;
        loop {
            let byte = self.read_u8()?;
            result |= ((byte & 0x7F) as u64) << shift;
            if (byte & 0x80) == 0 {
                break;
            }
            shift += 7;
            if shift > 63 {
                return Err(BytecodeError::ParseError("varint overflow".into()));
            }
        }
        Ok(result)
    }

    fn read_var_i64(&mut self) -> Result<i64> {
        let v = self.read_var_u64()?;
        // zigzag decode
        Ok(((v >> 1) as i64) ^ (-((v & 1) as i64)))
    }

    fn read_bytes(&mut self, n: usize) -> Result<&'a [u8]> {
        if self.offset + n > self.data.len() {
            let remaining = self.data.len().saturating_sub(self.offset);
            return Err(BytecodeError::UnexpectedEof {
                at: self.offset,
                needed: n,
                remaining,
            });
        }
        let slice = &self.data[self.offset..self.offset + n];
        self.offset += n;
        Ok(slice)
    }

    fn align_to(&mut self, align: usize, pad: u8) -> Result<()> {
        if align < 2 {
            return Ok(());
        }
        let padding = (align - (self.offset % align)) % align;
        if padding == 0 {
            return Ok(());
        }

        let start = self.offset;
        let bytes = self.read_bytes(padding)?;
        for (i, &b) in bytes.iter().enumerate() {
            if b != pad {
                return Err(BytecodeError::InvalidPadding {
                    at: start + i,
                    expected: pad,
                    found: b,
                });
            }
        }
        Ok(())
    }
}
