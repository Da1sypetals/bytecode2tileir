//! Miscellaneous opcode decoders: constants, globals, views, tokens, debug.

use smallvec::SmallVec;

use crate::bytecode::attrs;
use crate::bytecode::decode_body::BodyDecoder;
use crate::bytecode::error::Result;
use crate::bytecode::format::{ConstId, StrId, TypeId, Version};
use crate::bytecode::reader::ByteRead;
use crate::cuda_tile_ir::attrs::{Attr, DenseStorage};
use crate::cuda_tile_ir::debug::Location;
use crate::cuda_tile_ir::ids::{OpId, RegionId, ValueId};
use crate::cuda_tile_ir::{OpAttrKey, Opcode};

/// Bytecode version 13.2.0 – several opcodes gained new fields at this version.
const V13_2: Version = Version {
    major: 13,
    minor: 2,
    tag: 0,
};

pub fn decode_constant<'a, 'm>(
    d: &mut BodyDecoder<'a, 'm>,
    r: &mut dyn ByteRead<'a>,
    loc: Location,
) -> Result<OpId> {
    let result_ty = TypeId(read_u32_var(r)?);
    let const_idx = ConstId(read_u32_var(r)?);

    let value_attr = d.arena.intern_attr(Attr::DenseElements {
        ty: result_ty,
        storage: DenseStorage::Const(const_idx),
    });

    let mut attrs = crate::cuda_tile_ir::ir::AttrMap::new();
    attrs.push((OpAttrKey::Value, value_attr));

    Ok(d.build_op(
        Opcode::Constant,
        SmallVec::new(),
        std::iter::once(result_ty),
        attrs,
        Vec::<RegionId>::new(),
        loc,
    ))
}

pub fn decode_return<'a, 'm>(
    d: &mut BodyDecoder<'a, 'm>,
    r: &mut dyn ByteRead<'a>,
    loc: Location,
) -> Result<OpId> {
    let _num_results = r.read_var_u64()?;
    let num_operands = r.read_var_u64()? as usize;
    let mut operands = SmallVec::<[ValueId; 4]>::with_capacity(num_operands);
    for _ in 0..num_operands {
        operands.push(d.read_value_from_stream(r)?);
    }

    Ok(d.build_op(
        Opcode::Return,
        operands,
        std::iter::empty(),
        crate::cuda_tile_ir::ir::AttrMap::new(),
        Vec::<RegionId>::new(),
        loc,
    ))
}

pub fn decode_iota<'a, 'm>(
    d: &mut BodyDecoder<'a, 'm>,
    r: &mut dyn ByteRead<'a>,
    loc: Location,
) -> Result<OpId> {
    let result_ty = TypeId(read_u32_var(r)?);
    Ok(d.build_op(
        Opcode::Iota,
        SmallVec::new(),
        std::iter::once(result_ty),
        crate::cuda_tile_ir::ir::AttrMap::new(),
        Vec::<RegionId>::new(),
        loc,
    ))
}

pub fn decode_get_global<'a, 'm>(
    d: &mut BodyDecoder<'a, 'm>,
    r: &mut dyn ByteRead<'a>,
    loc: Location,
) -> Result<OpId> {
    let result_ty = TypeId(read_u32_var(r)?);
    let name = d.ctx.strings.get(StrId(read_u32_var(r)?))?.to_string();
    let name_attr = d.arena.intern_attr(Attr::FlatSymbolRef(name));

    let mut attrs = crate::cuda_tile_ir::ir::AttrMap::new();
    attrs.push((OpAttrKey::GlobalName, name_attr));

    Ok(d.build_op(
        Opcode::GetGlobal,
        SmallVec::new(),
        std::iter::once(result_ty),
        attrs,
        Vec::<RegionId>::new(),
        loc,
    ))
}

pub fn decode_make_tensor_view<'a, 'm>(
    d: &mut BodyDecoder<'a, 'm>,
    r: &mut dyn ByteRead<'a>,
    loc: Location,
) -> Result<OpId> {
    let num_results = r.read_var_u64()? as usize;
    let mut result_tys = Vec::with_capacity(num_results);
    for _ in 0..num_results {
        result_tys.push(TypeId(read_u32_var(r)?));
    }

    let base = d.read_value_from_stream(r)?;

    let num_shape = r.read_var_u64()? as usize;
    let mut operands = SmallVec::<[ValueId; 4]>::with_capacity(1 + num_shape);
    operands.push(base);
    for _ in 0..num_shape {
        operands.push(d.read_value_from_stream(r)?);
    }

    let num_strides = r.read_var_u64()? as usize;
    for _ in 0..num_strides {
        operands.push(d.read_value_from_stream(r)?);
    }

    let seg = d.arena.intern_attr(Attr::DenseI32Array(vec![
        1,
        num_shape as i32,
        num_strides as i32,
    ]));
    let mut attrs = crate::cuda_tile_ir::ir::AttrMap::new();
    attrs.push((OpAttrKey::OperandSegmentSizes, seg));

    Ok(d.build_op(
        Opcode::MakeTensorView,
        operands,
        result_tys,
        attrs,
        Vec::<RegionId>::new(),
        loc,
    ))
}

pub fn decode_make_partition_view<'a, 'm>(
    d: &mut BodyDecoder<'a, 'm>,
    r: &mut dyn ByteRead<'a>,
    loc: Location,
) -> Result<OpId> {
    let result_ty = TypeId(read_u32_var(r)?);
    let tv = d.read_value_from_stream(r)?;

    Ok(d.build_op(
        Opcode::MakePartitionView,
        SmallVec::from_slice(&[tv]),
        std::iter::once(result_ty),
        crate::cuda_tile_ir::ir::AttrMap::new(),
        Vec::<RegionId>::new(),
        loc,
    ))
}

pub fn decode_get_tensor_shape<'a, 'm>(
    d: &mut BodyDecoder<'a, 'm>,
    r: &mut dyn ByteRead<'a>,
    loc: Location,
) -> Result<OpId> {
    let num_results = r.read_var_u64()? as usize;
    let mut result_tys = Vec::with_capacity(num_results);
    for _ in 0..num_results {
        result_tys.push(TypeId(read_u32_var(r)?));
    }
    let src = d.read_value_from_stream(r)?;

    Ok(d.build_op(
        Opcode::GetTensorShape,
        SmallVec::from_slice(&[src]),
        result_tys,
        crate::cuda_tile_ir::ir::AttrMap::new(),
        Vec::<RegionId>::new(),
        loc,
    ))
}

pub fn decode_get_index_space_shape<'a, 'm>(
    d: &mut BodyDecoder<'a, 'm>,
    r: &mut dyn ByteRead<'a>,
    loc: Location,
) -> Result<OpId> {
    let num_results = r.read_var_u64()? as usize;
    let mut result_tys = Vec::with_capacity(num_results);
    for _ in 0..num_results {
        result_tys.push(TypeId(read_u32_var(r)?));
    }
    let src = d.read_value_from_stream(r)?;

    Ok(d.build_op(
        Opcode::GetIndexSpaceShape,
        SmallVec::from_slice(&[src]),
        result_tys,
        crate::cuda_tile_ir::ir::AttrMap::new(),
        Vec::<RegionId>::new(),
        loc,
    ))
}

pub fn decode_get_tile_block_id<'a, 'm>(
    d: &mut BodyDecoder<'a, 'm>,
    r: &mut dyn ByteRead<'a>,
    loc: Location,
) -> Result<OpId> {
    let x = TypeId(read_u32_var(r)?);
    let y = TypeId(read_u32_var(r)?);
    let z = TypeId(read_u32_var(r)?);

    Ok(d.build_op(
        Opcode::GetTileBlockId,
        SmallVec::new(),
        [x, y, z],
        crate::cuda_tile_ir::ir::AttrMap::new(),
        Vec::<RegionId>::new(),
        loc,
    ))
}

pub fn decode_get_num_tile_blocks<'a, 'm>(
    d: &mut BodyDecoder<'a, 'm>,
    r: &mut dyn ByteRead<'a>,
    loc: Location,
) -> Result<OpId> {
    let x = TypeId(read_u32_var(r)?);
    let y = TypeId(read_u32_var(r)?);
    let z = TypeId(read_u32_var(r)?);

    Ok(d.build_op(
        Opcode::GetNumTileBlocks,
        SmallVec::new(),
        [x, y, z],
        crate::cuda_tile_ir::ir::AttrMap::new(),
        Vec::<RegionId>::new(),
        loc,
    ))
}

pub fn decode_make_token<'a, 'm>(
    d: &mut BodyDecoder<'a, 'm>,
    r: &mut dyn ByteRead<'a>,
    loc: Location,
) -> Result<OpId> {
    let result_ty = TypeId(read_u32_var(r)?);
    Ok(d.build_op(
        Opcode::MakeToken,
        SmallVec::new(),
        std::iter::once(result_ty),
        crate::cuda_tile_ir::ir::AttrMap::new(),
        Vec::<RegionId>::new(),
        loc,
    ))
}

pub fn decode_join_tokens<'a, 'm>(
    d: &mut BodyDecoder<'a, 'm>,
    r: &mut dyn ByteRead<'a>,
    loc: Location,
) -> Result<OpId> {
    let num_results = r.read_var_u64()? as usize;
    let mut result_tys = Vec::with_capacity(num_results);
    for _ in 0..num_results {
        result_tys.push(TypeId(read_u32_var(r)?));
    }
    let num_tokens = r.read_var_u64()? as usize;
    let mut operands = SmallVec::<[ValueId; 4]>::with_capacity(num_tokens);
    for _ in 0..num_tokens {
        operands.push(d.read_value_from_stream(r)?);
    }

    Ok(d.build_op(
        Opcode::JoinTokens,
        operands,
        result_tys,
        crate::cuda_tile_ir::ir::AttrMap::new(),
        Vec::<RegionId>::new(),
        loc,
    ))
}

pub fn decode_print<'a, 'm>(
    d: &mut BodyDecoder<'a, 'm>,
    r: &mut dyn ByteRead<'a>,
    loc: Location,
) -> Result<OpId> {
    let num_results = r.read_var_u64()? as usize;
    let mut result_tys = Vec::with_capacity(num_results);
    for _ in 0..num_results {
        result_tys.push(TypeId(read_u32_var(r)?));
    }

    // Since v13.2: read flags (bit 0 = token present)
    let has_token = if d.ctx.version >= V13_2 {
        let flags = r.read_var_u64()?;
        (flags & 1) != 0
    } else {
        false
    };

    let format = d.ctx.strings.get(StrId(read_u32_var(r)?))?.to_string();
    let format_attr = d.arena.intern_attr(Attr::String(format));

    let num_args = r.read_var_u64()? as usize;
    let mut operands =
        SmallVec::<[ValueId; 4]>::with_capacity(num_args + if has_token { 1 } else { 0 });
    for _ in 0..num_args {
        operands.push(d.read_value_from_stream(r)?);
    }

    // Read optional token operand (v13.2)
    if has_token {
        operands.push(d.read_value_from_stream(r)?);
    }

    let mut attrs = crate::cuda_tile_ir::ir::AttrMap::new();
    attrs.push((OpAttrKey::Format, format_attr));

    Ok(d.build_op(
        Opcode::Print,
        operands,
        result_tys,
        attrs,
        Vec::<RegionId>::new(),
        loc,
    ))
}

pub fn decode_assert<'a, 'm>(
    d: &mut BodyDecoder<'a, 'm>,
    r: &mut dyn ByteRead<'a>,
    loc: Location,
) -> Result<OpId> {
    let msg = d.ctx.strings.get(StrId(read_u32_var(r)?))?.to_string();
    let msg_attr = d.arena.intern_attr(Attr::String(msg));
    let cond = d.read_value_from_stream(r)?;

    let mut attrs = crate::cuda_tile_ir::ir::AttrMap::new();
    attrs.push((OpAttrKey::Message, msg_attr));

    Ok(d.build_op(
        Opcode::Assert,
        SmallVec::from_slice(&[cond]),
        std::iter::empty(),
        attrs,
        Vec::<RegionId>::new(),
        loc,
    ))
}

pub fn decode_assume<'a, 'm>(
    d: &mut BodyDecoder<'a, 'm>,
    r: &mut dyn ByteRead<'a>,
    loc: Location,
) -> Result<OpId> {
    let result_ty = TypeId(read_u32_var(r)?);
    let raw_pred =
        attrs::parse_self_contained_attr(r, &d.ctx.strings, &mut d.ctx.types, &d.ctx.consts)?;
    let pred_attr = d.intern_raw_attr(raw_pred);
    let value = d.read_value_from_stream(r)?;

    let mut attrs = crate::cuda_tile_ir::ir::AttrMap::new();
    attrs.push((OpAttrKey::Predicate, pred_attr));

    Ok(d.build_op(
        Opcode::Assume,
        SmallVec::from_slice(&[value]),
        std::iter::once(result_ty),
        attrs,
        Vec::<RegionId>::new(),
        loc,
    ))
}

pub fn decode_entry<'a, 'm>(
    d: &mut BodyDecoder<'a, 'm>,
    _r: &mut dyn ByteRead<'a>,
    loc: Location,
) -> Result<OpId> {
    Ok(d.build_op(
        Opcode::Entry,
        SmallVec::new(),
        std::iter::empty(),
        crate::cuda_tile_ir::ir::AttrMap::new(),
        Vec::<RegionId>::new(),
        loc,
    ))
}

pub fn decode_global<'a, 'm>(
    d: &mut BodyDecoder<'a, 'm>,
    _r: &mut dyn ByteRead<'a>,
    loc: Location,
) -> Result<OpId> {
    Ok(d.build_op(
        Opcode::Global,
        SmallVec::new(),
        std::iter::empty(),
        crate::cuda_tile_ir::ir::AttrMap::new(),
        Vec::<RegionId>::new(),
        loc,
    ))
}

pub fn decode_module<'a, 'm>(
    d: &mut BodyDecoder<'a, 'm>,
    _r: &mut dyn ByteRead<'a>,
    loc: Location,
) -> Result<OpId> {
    Ok(d.build_op(
        Opcode::Module,
        SmallVec::new(),
        std::iter::empty(),
        crate::cuda_tile_ir::ir::AttrMap::new(),
        Vec::<RegionId>::new(),
        loc,
    ))
}

fn read_u32_var<'a>(r: &mut dyn ByteRead<'a>) -> Result<u32> {
    let v = r.read_var_u64()?;
    u32::try_from(v).map_err(|_| {
        crate::bytecode::error::BytecodeError::ParseError("varint does not fit into u32".into())
    })
}
