//! Tile manipulation, reduction, and MMA opcode decoders.

use smallvec::SmallVec;

use crate::bytecode::attrs;
use crate::bytecode::decode_body::BodyDecoder;
use crate::bytecode::error::{BytecodeError, Result};
use crate::bytecode::format::TypeId;
use crate::bytecode::reader::ByteRead;
use crate::cuda_tile_ir::attrs::Attr;
use crate::cuda_tile_ir::debug::Location;
use crate::cuda_tile_ir::enums::Signedness;
use crate::cuda_tile_ir::ids::{OpId, RegionId, ValueId};
use crate::cuda_tile_ir::{OpAttrKey, Opcode};

pub fn decode_broadcast<'a, 'm>(
    d: &mut BodyDecoder<'a, 'm>,
    r: &mut dyn ByteRead<'a>,
    loc: Location,
) -> Result<OpId> {
    let result_ty = TypeId(read_u32_var(r)?);
    let src = d.read_value_from_stream(r)?;
    Ok(d.build_op(
        Opcode::Broadcast,
        SmallVec::from_slice(&[src]),
        std::iter::once(result_ty),
        crate::cuda_tile_ir::ir::AttrMap::new(),
        Vec::<RegionId>::new(),
        loc,
    ))
}

pub fn decode_extract<'a, 'm>(
    d: &mut BodyDecoder<'a, 'm>,
    r: &mut dyn ByteRead<'a>,
    loc: Location,
) -> Result<OpId> {
    let num_results = r.read_var_u64()? as usize;
    let mut result_tys = Vec::with_capacity(num_results);
    for _ in 0..num_results {
        result_tys.push(TypeId(read_u32_var(r)?));
    }

    let num_operands = r.read_var_u64()? as usize;
    let mut operands = SmallVec::<[ValueId; 4]>::with_capacity(num_operands);
    for _ in 0..num_operands {
        operands.push(d.read_value_from_stream(r)?);
    }

    Ok(d.build_op(
        Opcode::Extract,
        operands,
        result_tys,
        crate::cuda_tile_ir::ir::AttrMap::new(),
        Vec::<RegionId>::new(),
        loc,
    ))
}

pub fn decode_cat<'a, 'm>(
    d: &mut BodyDecoder<'a, 'm>,
    r: &mut dyn ByteRead<'a>,
    loc: Location,
) -> Result<OpId> {
    let result_ty = TypeId(read_u32_var(r)?);
    let dim = r.read_var_u64()? as i64;
    let lhs = d.read_value_from_stream(r)?;
    let rhs = d.read_value_from_stream(r)?;

    let i64_ty = d.intern_builtin_int(64);
    let dim_attr = d.arena.intern_attr(Attr::Int {
        ty: i64_ty,
        value: dim,
    });

    let mut attrs = crate::cuda_tile_ir::ir::AttrMap::new();
    attrs.push((OpAttrKey::Dim, dim_attr));

    Ok(d.build_op(
        Opcode::Cat,
        SmallVec::from_slice(&[lhs, rhs]),
        std::iter::once(result_ty),
        attrs,
        Vec::<RegionId>::new(),
        loc,
    ))
}

pub fn decode_permute<'a, 'm>(
    d: &mut BodyDecoder<'a, 'm>,
    r: &mut dyn ByteRead<'a>,
    loc: Location,
) -> Result<OpId> {
    let result_ty = TypeId(read_u32_var(r)?);
    let perm = read_i32_array(r)?;
    let src = d.read_value_from_stream(r)?;

    let perm_attr = d.arena.intern_attr(Attr::DenseI32Array(perm));
    let mut attrs = crate::cuda_tile_ir::ir::AttrMap::new();
    attrs.push((OpAttrKey::Permutation, perm_attr));

    Ok(d.build_op(
        Opcode::Permute,
        SmallVec::from_slice(&[src]),
        std::iter::once(result_ty),
        attrs,
        Vec::<RegionId>::new(),
        loc,
    ))
}

pub fn decode_reshape<'a, 'm>(
    d: &mut BodyDecoder<'a, 'm>,
    r: &mut dyn ByteRead<'a>,
    loc: Location,
) -> Result<OpId> {
    let result_ty = TypeId(read_u32_var(r)?);
    let src = d.read_value_from_stream(r)?;
    Ok(d.build_op(
        Opcode::Reshape,
        SmallVec::from_slice(&[src]),
        std::iter::once(result_ty),
        crate::cuda_tile_ir::ir::AttrMap::new(),
        Vec::<RegionId>::new(),
        loc,
    ))
}

pub fn decode_offset<'a, 'm>(
    d: &mut BodyDecoder<'a, 'm>,
    r: &mut dyn ByteRead<'a>,
    loc: Location,
) -> Result<OpId> {
    let result_ty = TypeId(read_u32_var(r)?);
    let base = d.read_value_from_stream(r)?;
    let off = d.read_value_from_stream(r)?;
    Ok(d.build_op(
        Opcode::Offset,
        SmallVec::from_slice(&[base, off]),
        std::iter::once(result_ty),
        crate::cuda_tile_ir::ir::AttrMap::new(),
        Vec::<RegionId>::new(),
        loc,
    ))
}

pub fn decode_reduce<'a, 'm>(
    d: &mut BodyDecoder<'a, 'm>,
    r: &mut dyn ByteRead<'a>,
    loc: Location,
) -> Result<OpId> {
    let num_results = r.read_var_u64()? as usize;
    let mut result_tys = Vec::with_capacity(num_results);
    for _ in 0..num_results {
        result_tys.push(TypeId(read_u32_var(r)?));
    }

    let dim = r.read_var_u64()? as i64;
    let i64_ty = d.intern_builtin_int(64);
    let dim_attr = d.arena.intern_attr(Attr::Int {
        ty: i64_ty,
        value: dim,
    });

    let num_identities = r.read_var_u64()? as usize;
    let mut identity_ids = Vec::with_capacity(num_identities);
    for _ in 0..num_identities {
        let raw =
            attrs::parse_self_contained_attr(r, &d.ctx.strings, &mut d.ctx.types, &d.ctx.consts)?;
        identity_ids.push(d.intern_raw_attr(raw));
    }
    let identities_attr = d.arena.intern_attr(Attr::Array(identity_ids));

    let num_operands = r.read_var_u64()? as usize;
    let mut operands = SmallVec::<[ValueId; 4]>::with_capacity(num_operands);
    for _ in 0..num_operands {
        operands.push(d.read_value_from_stream(r)?);
    }

    let num_regions = r.read_var_u64()? as usize;
    let mut regions = Vec::with_capacity(num_regions);
    for _ in 0..num_regions {
        regions.push(d.decode_region_from_stream(r)?);
    }

    let mut attrs = crate::cuda_tile_ir::ir::AttrMap::new();
    attrs.push((OpAttrKey::Dim, dim_attr));
    attrs.push((OpAttrKey::Identities, identities_attr));

    Ok(d.build_op(Opcode::Reduce, operands, result_tys, attrs, regions, loc))
}

pub fn decode_scan<'a, 'm>(
    d: &mut BodyDecoder<'a, 'm>,
    r: &mut dyn ByteRead<'a>,
    loc: Location,
) -> Result<OpId> {
    let num_results = r.read_var_u64()? as usize;
    let mut result_tys = Vec::with_capacity(num_results);
    for _ in 0..num_results {
        result_tys.push(TypeId(read_u32_var(r)?));
    }

    let dim = r.read_var_u64()? as i64;
    let reverse = r.read_u8()? != 0;

    let i64_ty = d.intern_builtin_int(64);
    let dim_attr = d.arena.intern_attr(Attr::Int {
        ty: i64_ty,
        value: dim,
    });
    let reverse_attr = d.arena.intern_attr(Attr::Bool(reverse));

    let num_identities = r.read_var_u64()? as usize;
    let mut identity_ids = Vec::with_capacity(num_identities);
    for _ in 0..num_identities {
        let raw =
            attrs::parse_self_contained_attr(r, &d.ctx.strings, &mut d.ctx.types, &d.ctx.consts)?;
        identity_ids.push(d.intern_raw_attr(raw));
    }
    let identities_attr = d.arena.intern_attr(Attr::Array(identity_ids));

    let num_operands = r.read_var_u64()? as usize;
    let mut operands = SmallVec::<[ValueId; 4]>::with_capacity(num_operands);
    for _ in 0..num_operands {
        operands.push(d.read_value_from_stream(r)?);
    }

    let num_regions = r.read_var_u64()? as usize;
    let mut regions = Vec::with_capacity(num_regions);
    for _ in 0..num_regions {
        regions.push(d.decode_region_from_stream(r)?);
    }

    let mut attrs = crate::cuda_tile_ir::ir::AttrMap::new();
    attrs.push((OpAttrKey::Dim, dim_attr));
    attrs.push((OpAttrKey::Reverse, reverse_attr));
    attrs.push((OpAttrKey::Identities, identities_attr));

    Ok(d.build_op(Opcode::Scan, operands, result_tys, attrs, regions, loc))
}

pub fn decode_mmaf<'a, 'm>(
    d: &mut BodyDecoder<'a, 'm>,
    r: &mut dyn ByteRead<'a>,
    loc: Location,
) -> Result<OpId> {
    let result_ty = TypeId(read_u32_var(r)?);
    let lhs = d.read_value_from_stream(r)?;
    let rhs = d.read_value_from_stream(r)?;
    let acc = d.read_value_from_stream(r)?;
    Ok(d.build_op(
        Opcode::MmaF,
        SmallVec::from_slice(&[lhs, rhs, acc]),
        std::iter::once(result_ty),
        crate::cuda_tile_ir::ir::AttrMap::new(),
        Vec::<RegionId>::new(),
        loc,
    ))
}

pub fn decode_mmai<'a, 'm>(
    d: &mut BodyDecoder<'a, 'm>,
    r: &mut dyn ByteRead<'a>,
    loc: Location,
) -> Result<OpId> {
    let result_ty = TypeId(read_u32_var(r)?);
    let sign_lhs = read_enum::<Signedness>(r, "Signedness")?;
    let sign_rhs = read_enum::<Signedness>(r, "Signedness")?;
    let lhs = d.read_value_from_stream(r)?;
    let rhs = d.read_value_from_stream(r)?;
    let acc = d.read_value_from_stream(r)?;

    let lhs_attr = d.arena.intern_attr(Attr::Signedness(sign_lhs));
    let rhs_attr = d.arena.intern_attr(Attr::Signedness(sign_rhs));

    let mut attrs = crate::cuda_tile_ir::ir::AttrMap::new();
    attrs.push((OpAttrKey::SignednessLhs, lhs_attr));
    attrs.push((OpAttrKey::SignednessRhs, rhs_attr));

    Ok(d.build_op(
        Opcode::MmaI,
        SmallVec::from_slice(&[lhs, rhs, acc]),
        std::iter::once(result_ty),
        attrs,
        Vec::<RegionId>::new(),
        loc,
    ))
}

fn read_enum<'a, E>(r: &mut dyn ByteRead<'a>, name: &'static str) -> Result<E>
where
    E: TryFrom<u32, Error = ()>,
{
    let v = r.read_u8()? as u32;
    E::try_from(v).map_err(|_| BytecodeError::InvalidEnum {
        name,
        value: v as u64,
    })
}

fn read_i32_array<'a>(r: &mut dyn ByteRead<'a>) -> Result<Vec<i32>> {
    let count = r.read_var_u64()? as usize;
    let mut v = Vec::with_capacity(count);
    for _ in 0..count {
        v.push(r.read_u32_le()? as i32);
    }
    Ok(v)
}

fn read_u32_var<'a>(r: &mut dyn ByteRead<'a>) -> Result<u32> {
    let v = r.read_var_u64()?;
    u32::try_from(v).map_err(|_| BytecodeError::ParseError("varint does not fit into u32".into()))
}
