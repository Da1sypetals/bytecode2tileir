//! Memory + atomic opcode decoders.

use smallvec::SmallVec;

use crate::bytecode::attrs;
use crate::bytecode::decode_body::BodyDecoder;
use crate::bytecode::error::{BytecodeError, Result};
use crate::bytecode::format::{StrId, TypeId};
use crate::bytecode::reader::ByteRead;
use crate::cuda_tile_ir::attrs::Attr;
use crate::cuda_tile_ir::debug::Location;
use crate::cuda_tile_ir::enums::{AtomicRMWMode, MemoryOrdering, MemoryScope};
use crate::cuda_tile_ir::ids::{AttrId, OpId, RegionId, ValueId};
use crate::cuda_tile_ir::{OpAttrKey, Opcode};

fn read_memory_scope<'a>(r: &mut dyn ByteRead<'a>) -> Result<MemoryScope> {
    read_enum::<MemoryScope>(r, "MemoryScope")
}

pub fn decode_load_ptr_tko<'a, 'm>(
    d: &mut BodyDecoder<'a, 'm>,
    r: &mut dyn ByteRead<'a>,
    loc: Location,
) -> Result<OpId> {
    let result_ty = TypeId(read_u32_var(r)?);
    let token_ty = TypeId(read_u32_var(r)?);
    let flags = r.read_var_u64()? as u32;

    let mem_ord = read_enum::<MemoryOrdering>(r, "MemoryOrdering")?;
    let mem_ord_attr = d.arena.intern_attr(Attr::MemoryOrdering(mem_ord));

    let mut attrs_map = crate::cuda_tile_ir::ir::AttrMap::new();
    attrs_map.push((OpAttrKey::MemoryOrderingSemantics, mem_ord_attr));

    if (flags & 0x1) != 0 {
        let mem_scope = read_memory_scope(r)?;
        let scope_attr = d.arena.intern_attr(Attr::MemoryScope(mem_scope));
        attrs_map.push((OpAttrKey::MemoryScope, scope_attr));
    }
    if (flags & 0x2) != 0 {
        let hints_attr = parse_optimization_hints_untagged(d, r)?;
        attrs_map.push((OpAttrKey::OptimizationHints, hints_attr));
    }

    let mut operands = SmallVec::<[ValueId; 4]>::new();
    operands.push(d.read_value_from_stream(r)?);
    let mut segment_sizes = vec![1i32];

    for bit in [0x4, 0x8, 0x10] {
        if (flags & bit) != 0 {
            operands.push(d.read_value_from_stream(r)?);
            segment_sizes.push(1);
        } else {
            segment_sizes.push(0);
        }
    }

    let seg_attr = d.arena.intern_attr(Attr::DenseI32Array(segment_sizes));
    attrs_map.push((OpAttrKey::OperandSegmentSizes, seg_attr));

    Ok(d.build_op(
        Opcode::LoadPtrTko,
        operands,
        [result_ty, token_ty],
        attrs_map,
        Vec::<RegionId>::new(),
        loc,
    ))
}

pub fn decode_load_view_tko<'a, 'm>(
    d: &mut BodyDecoder<'a, 'm>,
    r: &mut dyn ByteRead<'a>,
    loc: Location,
) -> Result<OpId> {
    let num_results = r.read_var_u64()? as usize;
    let mut result_tys = Vec::with_capacity(num_results);
    for _ in 0..num_results {
        result_tys.push(TypeId(read_u32_var(r)?));
    }

    let flags = r.read_var_u64()? as u32;
    let mem_ord = read_enum::<MemoryOrdering>(r, "MemoryOrdering")?;
    let mem_ord_attr = d.arena.intern_attr(Attr::MemoryOrdering(mem_ord));

    let mut attrs_map = crate::cuda_tile_ir::ir::AttrMap::new();
    attrs_map.push((OpAttrKey::MemoryOrderingSemantics, mem_ord_attr));

    if (flags & 0x1) != 0 {
        let mem_scope = read_memory_scope(r)?;
        let scope_attr = d.arena.intern_attr(Attr::MemoryScope(mem_scope));
        attrs_map.push((OpAttrKey::MemoryScope, scope_attr));
    }
    if (flags & 0x2) != 0 {
        let hints_attr = parse_optimization_hints_untagged(d, r)?;
        attrs_map.push((OpAttrKey::OptimizationHints, hints_attr));
    }

    let mut operands = SmallVec::<[ValueId; 4]>::new();
    operands.push(d.read_value_from_stream(r)?);
    let mut segment_sizes = vec![1i32];

    let num_indices = r.read_var_u64()? as usize;
    for _ in 0..num_indices {
        operands.push(d.read_value_from_stream(r)?);
    }
    segment_sizes.push(num_indices as i32);

    if (flags & 0x4) != 0 {
        operands.push(d.read_value_from_stream(r)?);
        segment_sizes.push(1);
    } else {
        segment_sizes.push(0);
    }

    let seg_attr = d.arena.intern_attr(Attr::DenseI32Array(segment_sizes));
    attrs_map.push((OpAttrKey::OperandSegmentSizes, seg_attr));

    Ok(d.build_op(
        Opcode::LoadViewTko,
        operands,
        result_tys,
        attrs_map,
        Vec::<RegionId>::new(),
        loc,
    ))
}

pub fn decode_store_ptr_tko<'a, 'm>(
    d: &mut BodyDecoder<'a, 'm>,
    r: &mut dyn ByteRead<'a>,
    loc: Location,
) -> Result<OpId> {
    let token_ty = TypeId(read_u32_var(r)?);
    let flags = r.read_var_u64()? as u32;

    let mem_ord = read_enum::<MemoryOrdering>(r, "MemoryOrdering")?;
    let mem_ord_attr = d.arena.intern_attr(Attr::MemoryOrdering(mem_ord));

    let mut attrs_map = crate::cuda_tile_ir::ir::AttrMap::new();
    attrs_map.push((OpAttrKey::MemoryOrderingSemantics, mem_ord_attr));

    if (flags & 0x1) != 0 {
        let mem_scope = read_memory_scope(r)?;
        let scope_attr = d.arena.intern_attr(Attr::MemoryScope(mem_scope));
        attrs_map.push((OpAttrKey::MemoryScope, scope_attr));
    }
    if (flags & 0x2) != 0 {
        let hints_attr = parse_optimization_hints_untagged(d, r)?;
        attrs_map.push((OpAttrKey::OptimizationHints, hints_attr));
    }

    let mut operands = SmallVec::<[ValueId; 4]>::new();
    operands.push(d.read_value_from_stream(r)?); // ptr
    operands.push(d.read_value_from_stream(r)?); // val
    let mut segment_sizes = vec![1i32, 1];

    if (flags & 0x4) != 0 {
        operands.push(d.read_value_from_stream(r)?);
        segment_sizes.push(1);
    } else {
        segment_sizes.push(0);
    }
    if (flags & 0x8) != 0 {
        operands.push(d.read_value_from_stream(r)?);
        segment_sizes.push(1);
    } else {
        segment_sizes.push(0);
    }

    let seg_attr = d.arena.intern_attr(Attr::DenseI32Array(segment_sizes));
    attrs_map.push((OpAttrKey::OperandSegmentSizes, seg_attr));

    Ok(d.build_op(
        Opcode::StorePtrTko,
        operands,
        std::iter::once(token_ty),
        attrs_map,
        Vec::<RegionId>::new(),
        loc,
    ))
}

pub fn decode_store_view_tko<'a, 'm>(
    d: &mut BodyDecoder<'a, 'm>,
    r: &mut dyn ByteRead<'a>,
    loc: Location,
) -> Result<OpId> {
    let num_results = r.read_var_u64()? as usize;
    let mut result_tys = Vec::with_capacity(num_results);
    for _ in 0..num_results {
        result_tys.push(TypeId(read_u32_var(r)?));
    }

    let flags = r.read_var_u64()? as u32;
    let mem_ord = read_enum::<MemoryOrdering>(r, "MemoryOrdering")?;
    let mem_ord_attr = d.arena.intern_attr(Attr::MemoryOrdering(mem_ord));

    let mut attrs_map = crate::cuda_tile_ir::ir::AttrMap::new();
    attrs_map.push((OpAttrKey::MemoryOrderingSemantics, mem_ord_attr));

    if (flags & 0x1) != 0 {
        let mem_scope = read_memory_scope(r)?;
        let scope_attr = d.arena.intern_attr(Attr::MemoryScope(mem_scope));
        attrs_map.push((OpAttrKey::MemoryScope, scope_attr));
    }
    if (flags & 0x2) != 0 {
        let hints_attr = parse_optimization_hints_untagged(d, r)?;
        attrs_map.push((OpAttrKey::OptimizationHints, hints_attr));
    }

    let mut operands = SmallVec::<[ValueId; 4]>::new();
    operands.push(d.read_value_from_stream(r)?); // tile
    operands.push(d.read_value_from_stream(r)?); // view
    let mut segment_sizes = vec![1i32, 1];

    let num_indices = r.read_var_u64()? as usize;
    for _ in 0..num_indices {
        operands.push(d.read_value_from_stream(r)?);
    }
    segment_sizes.push(num_indices as i32);

    if (flags & 0x4) != 0 {
        operands.push(d.read_value_from_stream(r)?);
        segment_sizes.push(1);
    } else {
        segment_sizes.push(0);
    }

    let seg_attr = d.arena.intern_attr(Attr::DenseI32Array(segment_sizes));
    attrs_map.push((OpAttrKey::OperandSegmentSizes, seg_attr));

    Ok(d.build_op(
        Opcode::StoreViewTko,
        operands,
        result_tys,
        attrs_map,
        Vec::<RegionId>::new(),
        loc,
    ))
}

pub fn decode_atomic_cas_tko<'a, 'm>(
    d: &mut BodyDecoder<'a, 'm>,
    r: &mut dyn ByteRead<'a>,
    loc: Location,
) -> Result<OpId> {
    let result_ty = TypeId(read_u32_var(r)?);
    let token_ty = TypeId(read_u32_var(r)?);
    let flags = r.read_var_u64()? as u32;

    let mem_ord = read_enum::<MemoryOrdering>(r, "MemoryOrdering")?;
    let mem_scope = read_memory_scope(r)?;

    let mem_ord_attr = d.arena.intern_attr(Attr::MemoryOrdering(mem_ord));
    let mem_scope_attr = d.arena.intern_attr(Attr::MemoryScope(mem_scope));

    let mut operands = SmallVec::<[ValueId; 4]>::new();
    operands.push(d.read_value_from_stream(r)?); // ptr
    operands.push(d.read_value_from_stream(r)?); // cmp
    operands.push(d.read_value_from_stream(r)?); // val
    let mut segment_sizes = vec![1i32, 1, 1];

    if (flags & 1) != 0 {
        operands.push(d.read_value_from_stream(r)?);
        segment_sizes.push(1);
    } else {
        segment_sizes.push(0);
    }
    if (flags & 2) != 0 {
        operands.push(d.read_value_from_stream(r)?);
        segment_sizes.push(1);
    } else {
        segment_sizes.push(0);
    }

    let seg_attr = d.arena.intern_attr(Attr::DenseI32Array(segment_sizes));

    let mut attrs_map = crate::cuda_tile_ir::ir::AttrMap::new();
    attrs_map.push((OpAttrKey::MemoryOrderingSemantics, mem_ord_attr));
    attrs_map.push((OpAttrKey::MemoryScope, mem_scope_attr));
    attrs_map.push((OpAttrKey::OperandSegmentSizes, seg_attr));

    Ok(d.build_op(
        Opcode::AtomicCASTko,
        operands,
        [result_ty, token_ty],
        attrs_map,
        Vec::<RegionId>::new(),
        loc,
    ))
}

pub fn decode_atomic_rmw_tko<'a, 'm>(
    d: &mut BodyDecoder<'a, 'm>,
    r: &mut dyn ByteRead<'a>,
    loc: Location,
) -> Result<OpId> {
    let result_ty = TypeId(read_u32_var(r)?);
    let token_ty = TypeId(read_u32_var(r)?);
    let flags = r.read_var_u64()? as u32;

    let mem_ord = read_enum::<MemoryOrdering>(r, "MemoryOrdering")?;
    let mem_scope = read_memory_scope(r)?;
    let mode = read_enum::<AtomicRMWMode>(r, "AtomicRMWMode")?;

    let mem_ord_attr = d.arena.intern_attr(Attr::MemoryOrdering(mem_ord));
    let mem_scope_attr = d.arena.intern_attr(Attr::MemoryScope(mem_scope));
    let mode_attr = d.arena.intern_attr(Attr::AtomicRMWMode(mode));

    let mut operands = SmallVec::<[ValueId; 4]>::new();
    operands.push(d.read_value_from_stream(r)?); // ptr
    operands.push(d.read_value_from_stream(r)?); // arg
    let mut segment_sizes = vec![1i32, 1];

    if (flags & 1) != 0 {
        operands.push(d.read_value_from_stream(r)?);
        segment_sizes.push(1);
    } else {
        segment_sizes.push(0);
    }
    if (flags & 2) != 0 {
        operands.push(d.read_value_from_stream(r)?);
        segment_sizes.push(1);
    } else {
        segment_sizes.push(0);
    }

    let seg_attr = d.arena.intern_attr(Attr::DenseI32Array(segment_sizes));

    let mut attrs_map = crate::cuda_tile_ir::ir::AttrMap::new();
    attrs_map.push((OpAttrKey::MemoryOrderingSemantics, mem_ord_attr));
    attrs_map.push((OpAttrKey::MemoryScope, mem_scope_attr));
    attrs_map.push((OpAttrKey::Mode, mode_attr));
    attrs_map.push((OpAttrKey::OperandSegmentSizes, seg_attr));

    Ok(d.build_op(
        Opcode::AtomicRMWTko,
        operands,
        [result_ty, token_ty],
        attrs_map,
        Vec::<RegionId>::new(),
        loc,
    ))
}

fn parse_optimization_hints_untagged<'a, 'm>(
    d: &mut BodyDecoder<'a, 'm>,
    r: &mut dyn ByteRead<'a>,
) -> Result<AttrId> {
    let count = r.read_var_u64()? as usize;
    let mut hints = Vec::with_capacity(count);
    for _ in 0..count {
        let key = d.ctx.strings.get(StrId(read_u32_var(r)?))?.to_string();
        let raw =
            attrs::parse_self_contained_attr(r, &d.ctx.strings, &mut d.ctx.types, &d.ctx.consts)?;
        let value = d.intern_raw_attr(raw);
        hints.push((key, value));
    }
    Ok(d.arena.intern_attr(Attr::OptimizationHints(hints)))
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

fn read_u32_var<'a>(r: &mut dyn ByteRead<'a>) -> Result<u32> {
    let v = r.read_var_u64()?;
    u32::try_from(v).map_err(|_| BytecodeError::ParseError("varint does not fit into u32".into()))
}
