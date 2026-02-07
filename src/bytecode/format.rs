//! Tile IR bytecode format primitives.

pub const MAGIC: [u8; 8] = [0x7F, b'T', b'i', b'l', b'e', b'I', b'R', 0x00];

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Version {
    pub major: u8,
    pub minor: u8,
    pub tag: u16,
}

impl Version {
    pub const MIN_SUPPORTED: Version = Version {
        major: 1,
        minor: 0,
        tag: 0,
    };
    pub const CURRENT: Version = Version {
        major: 13,
        minor: 1,
        tag: 0,
    };
}

pub const SECTION_END: u8 = 0x00;
pub const SECTION_STRINGS: u8 = 0x01;
pub const SECTION_FUNCS: u8 = 0x02;
pub const SECTION_DEBUG: u8 = 0x03;
pub const SECTION_CONSTS: u8 = 0x04;
pub const SECTION_TYPES: u8 = 0x05;
pub const SECTION_GLOBALS: u8 = 0x06;

pub use crate::cuda_tile_ir::ids::GlobalId;
pub use crate::cuda_tile_ir::ids::TypeId;

#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StrId(pub u32);

#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ConstId(pub u32);

#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FuncId(pub u32);

#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DebugId(pub u32);

#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LocIndex(pub u32);
