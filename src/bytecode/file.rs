//! Bytecode file reader: header + section index.

use std::collections::HashMap;

use crate::bytecode::error::{BytecodeError, Result};
use crate::bytecode::format::{MAGIC, SECTION_END, Version};
use crate::bytecode::reader::{ByteRead, Cursor};

#[derive(Debug, Clone)]
pub struct SectionRange {
    pub id: u8,
    pub start: usize,
    pub len: usize,
    pub alignment: u32,
}

#[derive(Debug, Default)]
pub struct SectionIndex {
    pub ranges: HashMap<u8, SectionRange>,
}

#[derive(Debug)]
pub struct BytecodeFile<'a> {
    pub data: &'a [u8],
    pub version: Version,
    pub sections: SectionIndex,
}

impl<'a> BytecodeFile<'a> {
    pub fn parse(data: &'a [u8]) -> Result<Self> {
        let mut r = Cursor::new(data);

        let magic = r.read_bytes(MAGIC.len())?;
        if magic != MAGIC {
            return Err(BytecodeError::InvalidMagic);
        }

        let major = r.read_u8()?;
        let minor = r.read_u8()?;
        let tag = r.read_u16_le()?;
        let version = Version { major, minor, tag };

        if major < Version::MIN_SUPPORTED.major
            || (major == Version::MIN_SUPPORTED.major && minor < Version::MIN_SUPPORTED.minor)
        {
            return Err(BytecodeError::UnsupportedVersion { major, minor, tag });
        }

        let mut sections = SectionIndex::default();

        loop {
            if r.remaining() == 0 {
                return Err(BytecodeError::ParseError(
                    "missing end-of-bytecode marker section".into(),
                ));
            }

            let id_and_align = r.read_u8()?;
            let id = id_and_align & 0x7F;
            let has_alignment = (id_and_align & 0x80) != 0;

            if id == SECTION_END {
                if has_alignment {
                    return Err(BytecodeError::ParseError(
                        "end section must not carry alignment".into(),
                    ));
                }
                if r.remaining() != 0 {
                    return Err(BytecodeError::ParseError(
                        "unexpected trailing bytes after end section".into(),
                    ));
                }
                break;
            }

            let length_u64 = r.read_var_u64()?;
            let length = usize::try_from(length_u64)
                .map_err(|_| BytecodeError::ParseError("section length overflow".into()))?;

            let alignment = if has_alignment {
                let a = r.read_var_u64()?;
                if a == 0 || !a.is_power_of_two() {
                    return Err(BytecodeError::InvalidAlignment(a));
                }
                r.align_to(a as usize, 0xCB)?;
                a as u32
            } else {
                1
            };

            if sections.ranges.contains_key(&id) {
                return Err(BytecodeError::DuplicateSection(id));
            }

            let start = r.pos();
            let _payload = r.read_bytes(length)?;
            sections.ranges.insert(
                id,
                SectionRange {
                    id,
                    start,
                    len: length,
                    alignment,
                },
            );
        }

        Ok(Self {
            data,
            version,
            sections,
        })
    }

    pub fn section_bytes(&self, id: u8) -> Option<&'a [u8]> {
        let range = self.sections.ranges.get(&id)?;
        self.data.get(range.start..range.start + range.len)
    }

    pub fn section_range(&self, id: u8) -> Option<&SectionRange> {
        self.sections.ranges.get(&id)
    }
}
