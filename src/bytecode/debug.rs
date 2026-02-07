//! Debug section parsing + location resolution.

use crate::bytecode::error::{BytecodeError, Result};
use crate::bytecode::format::{DebugId, LocIndex, StrId, TypeId};
use crate::bytecode::reader::{ByteRead, Cursor};
use crate::bytecode::strings::StringTable;
use crate::bytecode::types::TypeTable;
use crate::cuda_tile_ir::debug::{DebugEntry, Location};

pub struct DebugTable<'a> {
    op_index_offsets: Vec<u32>,
    indices: Vec<DebugId>,

    entry_offsets: Vec<u32>,
    entry_blob: &'a [u8],
    cache: Vec<Option<DebugEntry>>,
}

impl<'a> DebugTable<'a> {
    pub fn parse(payload: &'a [u8]) -> Result<Self> {
        let mut r = Cursor::new(payload);

        let di_ops_num = r.read_var_u64()? as usize;
        r.align_to(4, 0xCB)?;
        let mut op_index_offsets = Vec::with_capacity(di_ops_num);
        for _ in 0..di_ops_num {
            op_index_offsets.push(r.read_u32_le()?);
        }

        let di_indices_num = r.read_var_u64()? as usize;
        r.align_to(8, 0xCB)?;
        let mut indices = Vec::with_capacity(di_indices_num);
        for _ in 0..di_indices_num {
            let v = r.read_u64_le()?;
            let v_u32 = u32::try_from(v).map_err(|_| {
                BytecodeError::ParseError("debug index does not fit into u32".into())
            })?;
            indices.push(DebugId(v_u32));
        }

        let di_attr_num = r.read_var_u64()? as usize;
        r.align_to(4, 0xCB)?;
        let mut entry_offsets = Vec::with_capacity(di_attr_num);
        for _ in 0..di_attr_num {
            entry_offsets.push(r.read_u32_le()?);
        }

        let entry_blob = r.slice_from_pos();
        validate_u32_offsets("debug entry", &entry_offsets, entry_blob.len())?;

        Ok(Self {
            op_index_offsets,
            indices,
            entry_offsets,
            entry_blob,
            cache: {
                let mut cache = vec![None; di_attr_num + 1];
                cache[0] = Some(DebugEntry::Unknown);
                cache
            },
        })
    }

    pub fn resolve_location(
        &mut self,
        loc: LocIndex,
        strings: &StringTable<'a>,
        types: &mut TypeTable<'a>,
    ) -> Result<Location> {
        if loc.0 == 0 {
            return Ok(Location::default());
        }

        // `LocIndex` here refers to the function-level debug slot index. The debug
        // section stores, for each function, an offset into the flattened per-op
        // debug attribute id array.
        let func_idx = (loc.0 - 1) as usize;
        if func_idx >= self.op_index_offsets.len() {
            // Some producers may emit location indices that do not have a
            // corresponding debug mapping entry. Treat these as UnknownLoc
            // rather than hard-failing the whole module.
            return Ok(Location::default());
        }

        let indices_off = self.op_index_offsets[func_idx] as usize;
        if indices_off >= self.indices.len() {
            return Ok(Location::default());
        }

        let id = self.indices[indices_off];
        let entry = self.entry(id, strings, types)?.clone();
        self.entry_to_location(&entry, strings, types)
    }

    pub fn resolve_op_location(
        &mut self,
        func_loc: LocIndex,
        op_index: u32,
        strings: &StringTable<'a>,
        types: &mut TypeTable<'a>,
    ) -> Result<Location> {
        if func_loc.0 == 0 {
            return Ok(Location::default());
        }

        let func_idx = (func_loc.0 - 1) as usize;
        if func_idx >= self.op_index_offsets.len() {
            return Ok(Location::default());
        }

        // Per-function index list layout:
        //   indices[base + 0] = function debug attr
        //   indices[base + 1 + op_index] = op debug attr
        let base = self.op_index_offsets[func_idx] as usize;
        let idx = base.saturating_add(1).saturating_add(op_index as usize);
        if idx >= self.indices.len() {
            return Ok(Location::default());
        }

        let id = self.indices[idx];
        let entry = self.entry(id, strings, types)?.clone();
        self.entry_to_location(&entry, strings, types)
    }

    pub fn entry(
        &mut self,
        id: DebugId,
        strings: &StringTable<'a>,
        types: &mut TypeTable<'a>,
    ) -> Result<&DebugEntry> {
        let idx = id.0 as usize;
        if idx >= self.cache.len() {
            // Debug graphs can legitimately contain missing/out-of-range
            // references (e.g., stripped scopes). Fall back to Unknown.
            return Ok(self.cache[0].as_ref().expect("cache[0] must be Unknown"));
        }

        if self.cache[idx].is_none() {
            // DebugId is 1-based; entry_offsets is indexed by (id-1).
            let off_idx = idx - 1;
            let start = self.entry_offsets[off_idx] as usize;
            let end = if off_idx + 1 < self.entry_offsets.len() {
                self.entry_offsets[off_idx + 1] as usize
            } else {
                self.entry_blob.len()
            };

            if end < start || end > self.entry_blob.len() {
                return Err(BytecodeError::CorruptTable {
                    table: "debug entry",
                    idx: off_idx,
                    offset: self.entry_offsets[off_idx] as u64,
                    blob_len: self.entry_blob.len(),
                });
            }

            let bytes = &self.entry_blob[start..end];
            let entry = self.parse_entry(bytes, strings, types)?;
            self.cache[idx] = Some(entry);
        }

        Ok(self.cache[idx].as_ref().unwrap())
    }

    fn parse_entry(
        &mut self,
        bytes: &[u8],
        _strings: &StringTable<'a>,
        _types: &mut TypeTable<'a>,
    ) -> Result<DebugEntry> {
        let mut r = Cursor::new(bytes);
        let tag = r.read_var_u64()? as u8;

        match tag {
            0x00 => Ok(DebugEntry::Unknown),
            0x01 => {
                // cuTile encodes DICompileUnit minimally as just the `DIFile` id.
                // Fill in non-modeled fields with defaults.
                let file = DebugId(read_u32_var(&mut r)?);
                let language = 0;
                let producer = StrId(0);
                let optimized = false;
                let emission_kind = 0;
                Ok(DebugEntry::DICompileUnit {
                    language,
                    file,
                    producer,
                    optimized,
                    emission_kind,
                })
            }
            0x02 => {
                let filename = StrId(read_u32_var(&mut r)?);
                let directory = StrId(read_u32_var(&mut r)?);
                Ok(DebugEntry::DIFile {
                    filename,
                    directory,
                })
            }
            0x03 => {
                // cuTile lexical block: parent_scope, file, line, column.
                let scope = DebugId(read_u32_var(&mut r)?);
                let _file = DebugId(read_u32_var(&mut r)?);
                let line = r.read_var_u64()?;
                let column = r.read_var_u64()?;
                Ok(DebugEntry::DILexicalBlock {
                    line,
                    column,
                    scope,
                })
            }
            0x04 => {
                // cuTile DILoc: scope, filename, line, column.
                let scope = DebugId(read_u32_var(&mut r)?);
                let _filename = StrId(read_u32_var(&mut r)?);
                let line = r.read_var_u64()?;
                let column = r.read_var_u64()?;
                let inlined_at = DebugId(0);
                Ok(DebugEntry::DILoc {
                    line,
                    column,
                    scope,
                    inlined_at,
                })
            }
            0x05 => {
                // cuTile subprogram: file, line, name, linkage_name, compile_unit, scope_line.
                let file = DebugId(read_u32_var(&mut r)?);
                let line = r.read_var_u64()?;
                let name = StrId(read_u32_var(&mut r)?);
                let linkage_name = StrId(read_u32_var(&mut r)?);
                let unit = DebugId(read_u32_var(&mut r)?);
                let scope_line = r.read_var_u64()?;
                let ty = TypeId(0);
                let flags = 0;
                Ok(DebugEntry::DISubprogram {
                    name,
                    linkage_name,
                    file,
                    line,
                    ty,
                    scope_line,
                    flags,
                    unit,
                })
            }
            0x06 => {
                let callee = DebugId(read_u32_var(&mut r)?);
                let caller = DebugId(read_u32_var(&mut r)?);
                Ok(DebugEntry::CallSiteLoc { callee, caller })
            }
            _ => Ok(DebugEntry::Unknown),
        }
    }

    fn entry_to_location(
        &mut self,
        entry: &DebugEntry,
        strings: &StringTable<'a>,
        types: &mut TypeTable<'a>,
    ) -> Result<Location> {
        match entry {
            DebugEntry::DILoc {
                line,
                column,
                scope,
                inlined_at,
            } => {
                if inlined_at.0 != 0 {
                    let inl = self.entry(*inlined_at, strings, types)?.clone();
                    return self.entry_to_location(&inl, strings, types);
                }

                let file = self.resolve_file(*scope, strings, types)?;
                Ok(Location {
                    file,
                    line: (*line).min(u32::MAX as u64) as u32,
                    column: (*column).min(u32::MAX as u64) as u32,
                })
            }
            DebugEntry::CallSiteLoc { callee: _, caller } => {
                let caller_entry = self.entry(*caller, strings, types)?.clone();
                self.entry_to_location(&caller_entry, strings, types)
            }
            _ => Ok(Location::default()),
        }
    }

    fn resolve_file(
        &mut self,
        scope: DebugId,
        strings: &StringTable<'a>,
        types: &mut TypeTable<'a>,
    ) -> Result<Option<String>> {
        if scope.0 == 0 {
            return Ok(None);
        }

        let entry = self.entry(scope, strings, types)?.clone();
        match entry {
            DebugEntry::DIFile {
                filename,
                directory,
            } => {
                let filename = strings.get(filename)?.to_string();
                let directory = strings.get(directory)?.to_string();
                if directory.is_empty() {
                    Ok(Some(filename))
                } else {
                    Ok(Some(format!("{directory}/{filename}")))
                }
            }
            DebugEntry::DISubprogram { file, .. } => self.resolve_file(file, strings, types),
            DebugEntry::DILexicalBlock { scope, .. } => self.resolve_file(scope, strings, types),
            DebugEntry::DICompileUnit { file, .. } => self.resolve_file(file, strings, types),
            DebugEntry::DILoc { scope, .. } => self.resolve_file(scope, strings, types),
            DebugEntry::CallSiteLoc { caller, .. } => self.resolve_file(caller, strings, types),
            DebugEntry::Unknown => Ok(None),
        }
    }
}

fn read_u32_var<'a>(r: &mut impl ByteRead<'a>) -> Result<u32> {
    let v = r.read_var_u64()?;
    u32::try_from(v).map_err(|_| BytecodeError::ParseError("varint does not fit into u32".into()))
}

fn validate_u32_offsets(table: &'static str, offsets: &[u32], blob_len: usize) -> Result<()> {
    let mut prev = 0u64;
    for (idx, &off) in offsets.iter().enumerate() {
        let off_u64 = off as u64;
        if off_u64 < prev || off_u64 > blob_len as u64 {
            return Err(BytecodeError::CorruptTable {
                table,
                idx,
                offset: off_u64,
                blob_len,
            });
        }
        prev = off_u64;
    }
    Ok(())
}
