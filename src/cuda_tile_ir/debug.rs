//! Debug info model + resolved source locations.

use crate::bytecode::format::{DebugId, StrId};
use crate::cuda_tile_ir::ids::TypeId;

#[derive(Debug, Clone)]
pub enum DebugEntry {
    Unknown,
    DICompileUnit {
        language: u8,
        file: DebugId,
        producer: StrId,
        optimized: bool,
        emission_kind: u8,
    },
    DIFile {
        filename: StrId,
        directory: StrId,
    },
    DILexicalBlock {
        line: u64,
        column: u64,
        scope: DebugId,
    },
    DILoc {
        line: u64,
        column: u64,
        scope: DebugId,
        inlined_at: DebugId,
    },
    DISubprogram {
        name: StrId,
        linkage_name: StrId,
        file: DebugId,
        line: u64,
        ty: TypeId,
        scope_line: u64,
        flags: u64,
        unit: DebugId,
    },
    CallSiteLoc {
        callee: DebugId,
        caller: DebugId,
    },
}

#[derive(Debug, Clone, Default)]
pub struct Location {
    pub file: Option<String>,
    pub line: u32,
    pub column: u32,
}
