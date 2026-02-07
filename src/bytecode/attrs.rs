//! Bytecode-level self-contained attributes (tree form).

use crate::bytecode::consts::ConstPool;
use crate::bytecode::error::{BytecodeError, Result};
use crate::bytecode::format::{ConstId, StrId, TypeId};
use crate::bytecode::reader::ByteRead;
use crate::bytecode::strings::StringTable;
use crate::bytecode::tags::AttributeTag;
use crate::bytecode::types::TypeTable;
use crate::cuda_tile_ir::attrs::DenseStorage;
use crate::cuda_tile_ir::enums::FloatKind;
use crate::cuda_tile_ir::types::Type;

#[derive(Debug, Clone)]
pub enum RawAttr {
    Unit,
    Bool(bool),
    Int {
        ty: TypeId,
        value: i64,
    },
    Float {
        kind: FloatKind,
        bits: u64,
    },
    Type(TypeId),
    String(String),
    Array(Vec<RawAttr>),
    DenseElements {
        ty: TypeId,
        storage: DenseStorage,
    },
    Dict(Vec<(String, RawAttr)>),
    OptimizationHints(Vec<(String, RawAttr)>),
    DivBy {
        divisor: u64,
        unsigned_int: bool,
        every: Option<i64>,
        along: Option<i64>,
    },
    SameElements(Vec<i64>),
    Bounded {
        lb: Option<i64>,
        ub: Option<i64>,
    },
    NonNegative,
}

pub fn parse_self_contained_attr<'a, R>(
    r: &mut R,
    strings: &StringTable<'a>,
    types: &mut TypeTable<'a>,
    consts: &ConstPool<'a>,
) -> Result<RawAttr>
where
    R: ByteRead<'a> + ?Sized,
{
    let tag = r.read_u8()?;
    let tag = AttributeTag::try_from(tag).map_err(|_| BytecodeError::InvalidAttrTag(tag))?;

    match tag {
        AttributeTag::Bool => Ok(RawAttr::Bool(r.read_u8()? != 0)),
        AttributeTag::Integer => {
            let ty = TypeId(read_u32_var(r)?);
            let raw = r.read_var_u64()?;

            let bits = match types.get(ty)? {
                Type::Int { width } => width as u32,
                _ => 64,
            };

            let value = if bits < 64 {
                let mask = (1u64 << bits) - 1;
                let v = raw & mask;
                let sign_bit = 1u64 << (bits - 1);
                if (v & sign_bit) != 0 {
                    (v as i64) - ((1u64 << bits) as i64)
                } else {
                    v as i64
                }
            } else {
                raw as i64
            };

            Ok(RawAttr::Int { ty, value })
        }
        AttributeTag::Float => {
            let ty = TypeId(read_u32_var(r)?);
            let kind = match types.get(ty)? {
                Type::Float(k) => k,
                _ => FloatKind::F32,
            };

            // Spec says APFloat bytes; current tileiras bytecode is compatible with this compact encoding.
            let bits = match kind.bit_width() {
                8 => r.read_u8()? as u64,
                _ => r.read_var_i64()? as u64,
            };
            Ok(RawAttr::Float { kind, bits })
        }
        AttributeTag::Type => Ok(RawAttr::Type(TypeId(read_u32_var(r)?))),
        AttributeTag::String => {
            let s = strings.get(StrId(read_u32_var(r)?))?.to_string();
            Ok(RawAttr::String(s))
        }
        AttributeTag::Array => {
            let count = r.read_var_u64()? as usize;
            let mut arr = Vec::with_capacity(count);
            for _ in 0..count {
                arr.push(parse_self_contained_attr(r, strings, types, consts)?);
            }
            Ok(RawAttr::Array(arr))
        }
        AttributeTag::DenseElements => {
            let ty = TypeId(read_u32_var(r)?);
            let v = r.read_var_u64()?;

            let const_len = consts_len(consts);
            if v < const_len as u64 {
                let id = ConstId(u32::try_from(v).map_err(|_| {
                    BytecodeError::ParseError("const index does not fit into u32".into())
                })?);
                Ok(RawAttr::DenseElements {
                    ty,
                    storage: DenseStorage::Const(id),
                })
            } else {
                let num_strings = v as usize;
                let mut ids = Vec::with_capacity(num_strings);
                for _ in 0..num_strings {
                    ids.push(StrId(read_u32_var(r)?));
                }
                Ok(RawAttr::DenseElements {
                    ty,
                    storage: DenseStorage::Strings(ids),
                })
            }
        }
        AttributeTag::DivBy => {
            let divisor = r.read_var_u64()?;
            let flags = r.read_u8()?;
            let unsigned_int = (flags & 0x01) != 0;
            let every = if (flags & 0x02) != 0 {
                Some(r.read_var_i64()?)
            } else {
                None
            };
            let along = if (flags & 0x04) != 0 {
                Some(r.read_var_i64()?)
            } else {
                None
            };
            Ok(RawAttr::DivBy {
                divisor,
                unsigned_int,
                every,
                along,
            })
        }
        AttributeTag::SameElements => Ok(RawAttr::SameElements(read_i64_array(r)?)),
        AttributeTag::Dictionary => {
            let count = r.read_var_u64()? as usize;
            let mut dict = Vec::with_capacity(count);
            for _ in 0..count {
                let key = strings.get(StrId(read_u32_var(r)?))?.to_string();
                let value = parse_self_contained_attr(r, strings, types, consts)?;
                dict.push((key, value));
            }
            Ok(RawAttr::Dict(dict))
        }
        AttributeTag::OptimizationHints => {
            let count = r.read_var_u64()? as usize;
            let mut hints = Vec::with_capacity(count);
            for _ in 0..count {
                let key = strings.get(StrId(read_u32_var(r)?))?.to_string();
                let value = parse_self_contained_attr(r, strings, types, consts)?;
                hints.push((key, value));
            }
            Ok(RawAttr::OptimizationHints(hints))
        }
        AttributeTag::NonNegative => {
            // Some producers encode `NonNegative` as a unit attribute, while others use the same
            // tag value for the `Bounded` assume predicate (lb/ub).
            if r.remaining() == 0 {
                Ok(RawAttr::NonNegative)
            } else {
                let flags = r.read_u8()?;
                let lb = if (flags & 0x01) != 0 {
                    Some(r.read_var_i64()?)
                } else {
                    None
                };
                let ub = if (flags & 0x02) != 0 {
                    Some(r.read_var_i64()?)
                } else {
                    None
                };
                Ok(RawAttr::Bounded { lb, ub })
            }
        }
    }
}

fn read_u32_var<'a, R: ByteRead<'a> + ?Sized>(r: &mut R) -> Result<u32> {
    let v = r.read_var_u64()?;
    u32::try_from(v).map_err(|_| BytecodeError::ParseError("varint does not fit into u32".into()))
}

fn read_i64_array<'a, R: ByteRead<'a> + ?Sized>(r: &mut R) -> Result<Vec<i64>> {
    let count = r.read_var_u64()? as usize;
    let mut v = Vec::with_capacity(count);
    for _ in 0..count {
        v.push(r.read_u64_le()? as i64);
    }
    Ok(v)
}

fn consts_len(consts: &ConstPool<'_>) -> usize {
    // ConstPool is a thin wrapper; treat missing section as length 0.
    // (Concrete length checking happens when resolving ConstId.)
    consts.len()
}
