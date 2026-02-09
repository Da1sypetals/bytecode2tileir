//! Control flow opcode decoders (regions/blocks).

use smallvec::SmallVec;

use crate::bytecode::decode_body::BodyDecoder;
use crate::bytecode::error::{BytecodeError, Result};
use crate::bytecode::format::{TypeId, Version};
use crate::bytecode::reader::ByteRead;
use crate::cuda_tile_ir::attrs::Attr;
use crate::cuda_tile_ir::debug::Location;
use crate::cuda_tile_ir::ids::{OpId, RegionId, ValueId};
use crate::cuda_tile_ir::{OpAttrKey, Opcode};

/// Bytecode version 13.2.0 – several opcodes gained new fields at this version.
const V13_2: Version = Version {
    major: 13,
    minor: 2,
    tag: 0,
};

pub fn decode_for<'a, 'm>(
    d: &mut BodyDecoder<'a, 'm>,
    r: &mut dyn ByteRead<'a>,
    loc: Location,
) -> Result<OpId> {
    let num_results = r.read_var_u64()? as usize;
    let mut result_tys = Vec::with_capacity(num_results);
    for _ in 0..num_results {
        result_tys.push(TypeId(read_u32_var(r)?));
    }

    // Since v13.2: read flags (bit 0 = unsignedCmp)
    let _unsigned_cmp = if d.ctx.version >= V13_2 {
        let flags = r.read_var_u64()?;
        (flags & 1) != 0
    } else {
        false
    };

    let num_operands = r.read_var_u64()? as usize;
    if num_operands < 3 {
        return Err(BytecodeError::ParseError(format!(
            "Expected at least 3 operands, got {}",
            num_operands
        )));
    }

    let mut operands = SmallVec::<[ValueId; 4]>::with_capacity(num_operands);
    for _ in 0..num_operands {
        operands.push(d.read_value_from_stream(r)?);
    }

    let num_init = num_operands - 3;
    let segment_sizes = vec![1, 1, 1, num_init as i32];
    let seg_attr = d.arena.intern_attr(Attr::DenseI32Array(segment_sizes));

    let num_regions = r.read_var_u64()? as usize;
    let mut regions = Vec::with_capacity(num_regions);
    for _ in 0..num_regions {
        regions.push(d.decode_region_from_stream(r)?);
    }

    let mut attrs = crate::cuda_tile_ir::ir::AttrMap::new();
    attrs.push((OpAttrKey::OperandSegmentSizes, seg_attr));

    Ok(d.build_op(Opcode::For, operands, result_tys, attrs, regions, loc))
}

pub fn decode_loop<'a, 'm>(
    d: &mut BodyDecoder<'a, 'm>,
    r: &mut dyn ByteRead<'a>,
    loc: Location,
) -> Result<OpId> {
    let num_results = r.read_var_u64()? as usize;
    let mut result_tys = Vec::with_capacity(num_results);
    for _ in 0..num_results {
        result_tys.push(TypeId(read_u32_var(r)?));
    }

    let num_init = r.read_var_u64()? as usize;
    let mut operands = SmallVec::<[ValueId; 4]>::with_capacity(num_init);
    for _ in 0..num_init {
        operands.push(d.read_value_from_stream(r)?);
    }

    let num_regions = r.read_var_u64()? as usize;
    let mut regions = Vec::with_capacity(num_regions);
    for _ in 0..num_regions {
        regions.push(d.decode_region_from_stream(r)?);
    }

    Ok(d.build_op(
        Opcode::Loop,
        operands,
        result_tys,
        crate::cuda_tile_ir::ir::AttrMap::new(),
        regions,
        loc,
    ))
}

pub fn decode_if<'a, 'm>(
    d: &mut BodyDecoder<'a, 'm>,
    r: &mut dyn ByteRead<'a>,
    loc: Location,
) -> Result<OpId> {
    let num_results = r.read_var_u64()? as usize;
    let mut result_tys = Vec::with_capacity(num_results);
    for _ in 0..num_results {
        result_tys.push(TypeId(read_u32_var(r)?));
    }

    let cond = d.read_value_from_stream(r)?;

    let num_regions = r.read_var_u64()? as usize;
    let mut regions = Vec::with_capacity(num_regions);
    for _ in 0..num_regions {
        regions.push(d.decode_region_from_stream(r)?);
    }

    Ok(d.build_op(
        Opcode::If,
        SmallVec::from_slice(&[cond]),
        result_tys,
        crate::cuda_tile_ir::ir::AttrMap::new(),
        regions,
        loc,
    ))
}

pub fn decode_break<'a, 'm>(
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
        Opcode::Break,
        operands,
        std::iter::empty(),
        crate::cuda_tile_ir::ir::AttrMap::new(),
        Vec::<RegionId>::new(),
        loc,
    ))
}

pub fn decode_continue<'a, 'm>(
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
        Opcode::Continue,
        operands,
        std::iter::empty(),
        crate::cuda_tile_ir::ir::AttrMap::new(),
        Vec::<RegionId>::new(),
        loc,
    ))
}

pub fn decode_yield<'a, 'm>(
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
        Opcode::Yield,
        operands,
        std::iter::empty(),
        crate::cuda_tile_ir::ir::AttrMap::new(),
        Vec::<RegionId>::new(),
        loc,
    ))
}

fn read_u32_var<'a>(r: &mut dyn ByteRead<'a>) -> Result<u32> {
    let v = r.read_var_u64()?;
    u32::try_from(v).map_err(|_| BytecodeError::ParseError("varint does not fit into u32".into()))
}
