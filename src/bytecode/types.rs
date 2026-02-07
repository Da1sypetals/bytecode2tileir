//! Type section parsing (lazy cache).

use crate::bytecode::error::{BytecodeError, Result};
use crate::bytecode::format::TypeId;
use crate::bytecode::reader::{ByteRead, Cursor};
use crate::bytecode::table::OffsetTable;
use crate::bytecode::tags::TypeTag;
use crate::cuda_tile_ir::enums::PaddingValue;
use crate::cuda_tile_ir::types::{Dim, IndexType, Shape, Type};

pub struct TypeTable<'a> {
    raw: OffsetTable<'a, u32>,
    cache: Vec<Option<Type>>,
}

impl<'a> TypeTable<'a> {
    pub fn parse(payload: &'a [u8]) -> Result<Self> {
        let mut r = Cursor::new(payload);
        let num = r.read_var_u64()? as usize;

        r.align_to(4, 0xCB)?;

        let mut offsets = Vec::with_capacity(num);
        for _ in 0..num {
            offsets.push(r.read_u32_le()?);
        }

        let blob = r.slice_from_pos();
        let raw = OffsetTable { blob, offsets };
        raw.validate_monotonic("type")?;

        Ok(Self {
            raw,
            cache: vec![None; num],
        })
    }

    pub fn get(&mut self, id: TypeId) -> Result<Type> {
        let idx = id.0 as usize;
        if idx >= self.cache.len() {
            return Err(BytecodeError::IndexOutOfBounds {
                kind: "type",
                index: id.0 as u64,
                max: self.cache.len(),
            });
        }

        if let Some(ty) = &self.cache[idx] {
            return Ok(ty.clone());
        }

        let bytes = self.raw.slice(idx, "type")?;
        let ty = self.parse_type(bytes)?;
        self.cache[idx] = Some(ty.clone());
        Ok(ty)
    }

    pub fn len(&self) -> usize {
        self.cache.len()
    }

    fn parse_type(&mut self, bytes: &[u8]) -> Result<Type> {
        let mut r = Cursor::new(bytes);
        let tag = r.read_u8()?;
        let tag = TypeTag::try_from(tag).map_err(|_| BytecodeError::InvalidTypeTag(tag))?;

        match tag {
            TypeTag::I1 => Ok(Type::Int { width: 1 }),
            TypeTag::I8 => Ok(Type::Int { width: 8 }),
            TypeTag::I16 => Ok(Type::Int { width: 16 }),
            TypeTag::I32 => Ok(Type::Int { width: 32 }),
            TypeTag::I64 => Ok(Type::Int { width: 64 }),
            TypeTag::F16 => Ok(Type::Float(crate::cuda_tile_ir::enums::FloatKind::F16)),
            TypeTag::BF16 => Ok(Type::Float(crate::cuda_tile_ir::enums::FloatKind::BF16)),
            TypeTag::F32 => Ok(Type::Float(crate::cuda_tile_ir::enums::FloatKind::F32)),
            TypeTag::TF32 => Ok(Type::Float(crate::cuda_tile_ir::enums::FloatKind::TF32)),
            TypeTag::F64 => Ok(Type::Float(crate::cuda_tile_ir::enums::FloatKind::F64)),
            TypeTag::F8E4M3FN => Ok(Type::Float(crate::cuda_tile_ir::enums::FloatKind::F8E4M3FN)),
            TypeTag::F8E5M2 => Ok(Type::Float(crate::cuda_tile_ir::enums::FloatKind::F8E5M2)),
            TypeTag::Token => Ok(Type::Token),
            TypeTag::Pointer => {
                let pointee = TypeId(r.read_var_u64()? as u32);
                Ok(Type::Ptr { pointee })
            }
            TypeTag::Tile => {
                let element = TypeId(r.read_var_u64()? as u32);
                let shape = read_shape_i64(&mut r)?;
                Ok(Type::Tile { element, shape })
            }
            TypeTag::TensorView => {
                let element = TypeId(r.read_var_u64()? as u32);
                let shape = read_shape_i64(&mut r)?;
                let strides = read_i64_array(&mut r)?;
                if strides.len() != shape.0.len() {
                    return Err(BytecodeError::ParseError(format!(
                        "tensor_view rank mismatch: shape rank {} but strides len {}",
                        shape.0.len(),
                        strides.len()
                    )));
                }

                // Tile IR bytecode v13.1 may omit `indexTypeTag` even though some specs mention it.
                // Accept both encodings:
                // - If no bytes remain, default to I64.
                // - If exactly one byte remains, interpret as indexTypeTag (I32=0x03, I64=0x04).
                let index = match r.remaining() {
                    0 => IndexType::I64,
                    1 => match r.read_u8()? {
                        0x03 => IndexType::I32,
                        0x04 => IndexType::I64,
                        other => {
                            return Err(BytecodeError::InvalidEnum {
                                name: "IndexTypeTag",
                                value: other as u64,
                            });
                        }
                    },
                    n => {
                        return Err(BytecodeError::ParseError(format!(
                            "unexpected trailing bytes in tensor_view type: {n}"
                        )));
                    }
                };
                Ok(Type::TensorView {
                    element,
                    shape,
                    strides,
                    index,
                })
            }
            TypeTag::PartitionView => {
                let tile_shape = read_i32_array(&mut r)?;
                let view = TypeId(r.read_var_u64()? as u32);
                let dim_map = read_i32_array(&mut r)?;
                let masked_byte = r.read_u8()?;
                let masked = masked_byte != 0;
                let padding_value = if masked {
                    match r.remaining() {
                        0 => None,
                        1 => {
                            let raw = r.read_u8()? as u32;
                            Some(PaddingValue::try_from(raw).map_err(|_| {
                                BytecodeError::InvalidEnum {
                                    name: "PaddingValue",
                                    value: raw as u64,
                                }
                            })?)
                        }
                        n => {
                            return Err(BytecodeError::ParseError(format!(
                                "unexpected trailing bytes in partition_view type: {n}"
                            )));
                        }
                    }
                } else if r.remaining() == 0 {
                    None
                } else {
                    return Err(BytecodeError::ParseError(format!(
                        "unexpected trailing bytes in partition_view type"
                    )));
                };
                Ok(Type::PartitionView {
                    tile_shape,
                    view,
                    dim_map,
                    masked,
                    padding_value,
                })
            }
            TypeTag::Func => {
                let num_params = r.read_var_u64()? as usize;
                let mut params = Vec::with_capacity(num_params);
                for _ in 0..num_params {
                    params.push(TypeId(r.read_var_u64()? as u32));
                }
                let num_results = r.read_var_u64()? as usize;
                let mut results = Vec::with_capacity(num_results);
                for _ in 0..num_results {
                    results.push(TypeId(r.read_var_u64()? as u32));
                }
                Ok(Type::Func { params, results })
            }
        }
    }
}

fn read_i32_array<'a>(r: &mut impl ByteRead<'a>) -> Result<Vec<i32>> {
    let count = r.read_var_u64()? as usize;
    let mut v = Vec::with_capacity(count);
    for _ in 0..count {
        v.push(r.read_u32_le()? as i32);
    }
    Ok(v)
}

fn read_i64_array<'a>(r: &mut impl ByteRead<'a>) -> Result<Vec<i64>> {
    let count = r.read_var_u64()? as usize;
    let mut v = Vec::with_capacity(count);
    for _ in 0..count {
        v.push(r.read_u64_le()? as i64);
    }
    Ok(v)
}

fn read_shape_i64<'a>(r: &mut impl ByteRead<'a>) -> Result<Shape> {
    let dims = read_i64_array(r)?;
    let dims = dims
        .into_iter()
        .map(|d| if d < 0 { Dim::Dynamic } else { Dim::Static(d) })
        .collect();
    Ok(Shape(dims))
}
