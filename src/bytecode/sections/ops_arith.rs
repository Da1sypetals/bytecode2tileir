//! Arithmetic / comparison / conversion opcode decoders.

use smallvec::SmallVec;

use crate::bytecode::decode_body::BodyDecoder;
use crate::bytecode::error::{BytecodeError, Result};
use crate::bytecode::format::TypeId;
use crate::bytecode::reader::ByteRead;
use crate::cuda_tile_ir::attrs::Attr;
use crate::cuda_tile_ir::debug::Location;
use crate::cuda_tile_ir::enums::{
    ComparisonOrdering, ComparisonPredicate, IntegerOverflow, RoundingMode, Signedness,
};
use crate::cuda_tile_ir::ids::{OpId, RegionId};
use crate::cuda_tile_ir::{OpAttrKey, Opcode};

macro_rules! bin_operands {
    ($d:expr, $r:expr) => {{
        let lhs = $d.read_value_from_stream($r)?;
        let rhs = $d.read_value_from_stream($r)?;
        SmallVec::from_slice(&[lhs, rhs])
    }};
}

pub fn decode_addf<'a, 'm>(
    d: &mut BodyDecoder<'a, 'm>,
    r: &mut dyn ByteRead<'a>,
    loc: Location,
) -> Result<OpId> {
    decode_float_binary_with_rounding(d, r, Opcode::AddF, loc)
}
pub fn decode_subf<'a, 'm>(
    d: &mut BodyDecoder<'a, 'm>,
    r: &mut dyn ByteRead<'a>,
    loc: Location,
) -> Result<OpId> {
    decode_float_binary_with_rounding(d, r, Opcode::SubF, loc)
}
pub fn decode_mulf<'a, 'm>(
    d: &mut BodyDecoder<'a, 'm>,
    r: &mut dyn ByteRead<'a>,
    loc: Location,
) -> Result<OpId> {
    decode_float_binary_with_rounding(d, r, Opcode::MulF, loc)
}
pub fn decode_divf<'a, 'm>(
    d: &mut BodyDecoder<'a, 'm>,
    r: &mut dyn ByteRead<'a>,
    loc: Location,
) -> Result<OpId> {
    decode_float_binary_with_rounding(d, r, Opcode::DivF, loc)
}

pub fn decode_maxf<'a, 'm>(
    d: &mut BodyDecoder<'a, 'm>,
    r: &mut dyn ByteRead<'a>,
    loc: Location,
) -> Result<OpId> {
    decode_float_binary_with_nan_flush(d, r, Opcode::MaxF, loc)
}
pub fn decode_minf<'a, 'm>(
    d: &mut BodyDecoder<'a, 'm>,
    r: &mut dyn ByteRead<'a>,
    loc: Location,
) -> Result<OpId> {
    decode_float_binary_with_nan_flush(d, r, Opcode::MinF, loc)
}

pub fn decode_remf<'a, 'm>(
    d: &mut BodyDecoder<'a, 'm>,
    r: &mut dyn ByteRead<'a>,
    loc: Location,
) -> Result<OpId> {
    decode_float_binary_simple(d, r, Opcode::RemF, loc)
}
pub fn decode_pow<'a, 'm>(
    d: &mut BodyDecoder<'a, 'm>,
    r: &mut dyn ByteRead<'a>,
    loc: Location,
) -> Result<OpId> {
    decode_float_binary_simple(d, r, Opcode::Pow, loc)
}

pub fn decode_absi<'a, 'm>(
    d: &mut BodyDecoder<'a, 'm>,
    r: &mut dyn ByteRead<'a>,
    loc: Location,
) -> Result<OpId> {
    decode_int_unary_simple(d, r, Opcode::AbsI, loc)
}
pub fn decode_negi<'a, 'm>(
    d: &mut BodyDecoder<'a, 'm>,
    r: &mut dyn ByteRead<'a>,
    loc: Location,
) -> Result<OpId> {
    decode_int_unary_simple(d, r, Opcode::NegI, loc)
}

pub fn decode_addi<'a, 'm>(
    d: &mut BodyDecoder<'a, 'm>,
    r: &mut dyn ByteRead<'a>,
    loc: Location,
) -> Result<OpId> {
    decode_int_binary_with_overflow(d, r, Opcode::AddI, loc)
}
pub fn decode_subi<'a, 'm>(
    d: &mut BodyDecoder<'a, 'm>,
    r: &mut dyn ByteRead<'a>,
    loc: Location,
) -> Result<OpId> {
    decode_int_binary_with_overflow(d, r, Opcode::SubI, loc)
}
pub fn decode_muli<'a, 'm>(
    d: &mut BodyDecoder<'a, 'm>,
    r: &mut dyn ByteRead<'a>,
    loc: Location,
) -> Result<OpId> {
    decode_int_binary_with_overflow(d, r, Opcode::MulI, loc)
}
pub fn decode_shli<'a, 'm>(
    d: &mut BodyDecoder<'a, 'm>,
    r: &mut dyn ByteRead<'a>,
    loc: Location,
) -> Result<OpId> {
    decode_int_binary_with_overflow(d, r, Opcode::ShLI, loc)
}

pub fn decode_divi<'a, 'm>(
    d: &mut BodyDecoder<'a, 'm>,
    r: &mut dyn ByteRead<'a>,
    loc: Location,
) -> Result<OpId> {
    let result_ty = TypeId(read_u32_var(r)?);
    let sign = read_enum::<Signedness>(r, "Signedness")?;
    let rm = read_enum::<RoundingMode>(r, "RoundingMode")?;
    let operands = bin_operands!(d, r);

    let sign_attr = d.arena.intern_attr(Attr::Signedness(sign));
    let rm_attr = d.arena.intern_attr(Attr::RoundingMode(rm));

    let mut attrs = crate::cuda_tile_ir::ir::AttrMap::new();
    attrs.push((OpAttrKey::Signedness, sign_attr));
    attrs.push((OpAttrKey::RoundingMode, rm_attr));

    Ok(d.build_op(
        Opcode::DivI,
        operands,
        std::iter::once(result_ty),
        attrs,
        Vec::<RegionId>::new(),
        loc,
    ))
}

pub fn decode_remi<'a, 'm>(
    d: &mut BodyDecoder<'a, 'm>,
    r: &mut dyn ByteRead<'a>,
    loc: Location,
) -> Result<OpId> {
    decode_int_binary_with_signedness(d, r, Opcode::RemI, loc)
}
pub fn decode_maxi<'a, 'm>(
    d: &mut BodyDecoder<'a, 'm>,
    r: &mut dyn ByteRead<'a>,
    loc: Location,
) -> Result<OpId> {
    decode_int_binary_with_signedness(d, r, Opcode::MaxI, loc)
}
pub fn decode_mini<'a, 'm>(
    d: &mut BodyDecoder<'a, 'm>,
    r: &mut dyn ByteRead<'a>,
    loc: Location,
) -> Result<OpId> {
    decode_int_binary_with_signedness(d, r, Opcode::MinI, loc)
}

pub fn decode_mulhii<'a, 'm>(
    d: &mut BodyDecoder<'a, 'm>,
    r: &mut dyn ByteRead<'a>,
    loc: Location,
) -> Result<OpId> {
    decode_int_binary_simple(d, r, Opcode::MulhiI, loc)
}
pub fn decode_andi<'a, 'm>(
    d: &mut BodyDecoder<'a, 'm>,
    r: &mut dyn ByteRead<'a>,
    loc: Location,
) -> Result<OpId> {
    decode_int_binary_simple(d, r, Opcode::AndI, loc)
}
pub fn decode_ori<'a, 'm>(
    d: &mut BodyDecoder<'a, 'm>,
    r: &mut dyn ByteRead<'a>,
    loc: Location,
) -> Result<OpId> {
    decode_int_binary_simple(d, r, Opcode::OrI, loc)
}
pub fn decode_xori<'a, 'm>(
    d: &mut BodyDecoder<'a, 'm>,
    r: &mut dyn ByteRead<'a>,
    loc: Location,
) -> Result<OpId> {
    decode_int_binary_simple(d, r, Opcode::XOrI, loc)
}

pub fn decode_shri<'a, 'm>(
    d: &mut BodyDecoder<'a, 'm>,
    r: &mut dyn ByteRead<'a>,
    loc: Location,
) -> Result<OpId> {
    decode_int_binary_with_signedness(d, r, Opcode::ShRI, loc)
}

pub fn decode_absf<'a, 'm>(
    d: &mut BodyDecoder<'a, 'm>,
    r: &mut dyn ByteRead<'a>,
    loc: Location,
) -> Result<OpId> {
    decode_float_unary_no_attrs(d, r, Opcode::AbsF, loc)
}
pub fn decode_negf<'a, 'm>(
    d: &mut BodyDecoder<'a, 'm>,
    r: &mut dyn ByteRead<'a>,
    loc: Location,
) -> Result<OpId> {
    decode_float_unary_no_attrs(d, r, Opcode::NegF, loc)
}
pub fn decode_ceil<'a, 'm>(
    d: &mut BodyDecoder<'a, 'm>,
    r: &mut dyn ByteRead<'a>,
    loc: Location,
) -> Result<OpId> {
    decode_float_unary_no_attrs(d, r, Opcode::Ceil, loc)
}
pub fn decode_floor<'a, 'm>(
    d: &mut BodyDecoder<'a, 'm>,
    r: &mut dyn ByteRead<'a>,
    loc: Location,
) -> Result<OpId> {
    decode_float_unary_no_attrs(d, r, Opcode::Floor, loc)
}
pub fn decode_exp<'a, 'm>(
    d: &mut BodyDecoder<'a, 'm>,
    r: &mut dyn ByteRead<'a>,
    loc: Location,
) -> Result<OpId> {
    decode_float_unary_no_attrs(d, r, Opcode::Exp, loc)
}
pub fn decode_log<'a, 'm>(
    d: &mut BodyDecoder<'a, 'm>,
    r: &mut dyn ByteRead<'a>,
    loc: Location,
) -> Result<OpId> {
    decode_float_unary_no_attrs(d, r, Opcode::Log, loc)
}
pub fn decode_log2<'a, 'm>(
    d: &mut BodyDecoder<'a, 'm>,
    r: &mut dyn ByteRead<'a>,
    loc: Location,
) -> Result<OpId> {
    decode_float_unary_no_attrs(d, r, Opcode::Log2, loc)
}
pub fn decode_sin<'a, 'm>(
    d: &mut BodyDecoder<'a, 'm>,
    r: &mut dyn ByteRead<'a>,
    loc: Location,
) -> Result<OpId> {
    decode_float_unary_no_attrs(d, r, Opcode::Sin, loc)
}
pub fn decode_cos<'a, 'm>(
    d: &mut BodyDecoder<'a, 'm>,
    r: &mut dyn ByteRead<'a>,
    loc: Location,
) -> Result<OpId> {
    decode_float_unary_no_attrs(d, r, Opcode::Cos, loc)
}
pub fn decode_tan<'a, 'm>(
    d: &mut BodyDecoder<'a, 'm>,
    r: &mut dyn ByteRead<'a>,
    loc: Location,
) -> Result<OpId> {
    decode_float_unary_no_attrs(d, r, Opcode::Tan, loc)
}
pub fn decode_sinh<'a, 'm>(
    d: &mut BodyDecoder<'a, 'm>,
    r: &mut dyn ByteRead<'a>,
    loc: Location,
) -> Result<OpId> {
    decode_float_unary_no_attrs(d, r, Opcode::SinH, loc)
}
pub fn decode_cosh<'a, 'm>(
    d: &mut BodyDecoder<'a, 'm>,
    r: &mut dyn ByteRead<'a>,
    loc: Location,
) -> Result<OpId> {
    decode_float_unary_no_attrs(d, r, Opcode::CosH, loc)
}
pub fn decode_tanh<'a, 'm>(
    d: &mut BodyDecoder<'a, 'm>,
    r: &mut dyn ByteRead<'a>,
    loc: Location,
) -> Result<OpId> {
    decode_float_unary_no_attrs(d, r, Opcode::TanH, loc)
}

pub fn decode_exp2<'a, 'm>(
    d: &mut BodyDecoder<'a, 'm>,
    r: &mut dyn ByteRead<'a>,
    loc: Location,
) -> Result<OpId> {
    decode_float_unary_with_flush_flag(d, r, Opcode::Exp2, loc)
}
pub fn decode_rsqrt<'a, 'm>(
    d: &mut BodyDecoder<'a, 'm>,
    r: &mut dyn ByteRead<'a>,
    loc: Location,
) -> Result<OpId> {
    decode_float_unary_with_flush_flag(d, r, Opcode::Rsqrt, loc)
}
pub fn decode_sqrt<'a, 'm>(
    d: &mut BodyDecoder<'a, 'm>,
    r: &mut dyn ByteRead<'a>,
    loc: Location,
) -> Result<OpId> {
    let result_ty = TypeId(read_u32_var(r)?);
    let flush = r.read_var_u64()? != 0;
    let rm = read_enum::<RoundingMode>(r, "RoundingMode")?;
    let operand = d.read_value_from_stream(r)?;

    let mut attrs = crate::cuda_tile_ir::ir::AttrMap::new();
    let rm_attr = d.arena.intern_attr(Attr::RoundingMode(rm));
    attrs.push((OpAttrKey::RoundingMode, rm_attr));
    if flush {
        let unit = d.arena.intern_attr(Attr::Unit);
        attrs.push((OpAttrKey::FlushToZero, unit));
    }

    Ok(d.build_op(
        Opcode::Sqrt,
        SmallVec::from_slice(&[operand]),
        std::iter::once(result_ty),
        attrs,
        Vec::<RegionId>::new(),
        loc,
    ))
}

pub fn decode_fma<'a, 'm>(
    d: &mut BodyDecoder<'a, 'm>,
    r: &mut dyn ByteRead<'a>,
    loc: Location,
) -> Result<OpId> {
    let result_ty = TypeId(read_u32_var(r)?);
    let flush = r.read_var_u64()? != 0;
    let rm = read_enum::<RoundingMode>(r, "RoundingMode")?;
    let a = d.read_value_from_stream(r)?;
    let b = d.read_value_from_stream(r)?;
    let c = d.read_value_from_stream(r)?;

    let rm_attr = d.arena.intern_attr(Attr::RoundingMode(rm));
    let mut attrs = crate::cuda_tile_ir::ir::AttrMap::new();
    attrs.push((OpAttrKey::RoundingMode, rm_attr));
    if flush {
        let unit = d.arena.intern_attr(Attr::Unit);
        attrs.push((OpAttrKey::FlushToZero, unit));
    }

    Ok(d.build_op(
        Opcode::Fma,
        SmallVec::from_slice(&[a, b, c]),
        std::iter::once(result_ty),
        attrs,
        Vec::<RegionId>::new(),
        loc,
    ))
}

pub fn decode_select<'a, 'm>(
    d: &mut BodyDecoder<'a, 'm>,
    r: &mut dyn ByteRead<'a>,
    loc: Location,
) -> Result<OpId> {
    let result_ty = TypeId(read_u32_var(r)?);
    let cond = d.read_value_from_stream(r)?;
    let t = d.read_value_from_stream(r)?;
    let f = d.read_value_from_stream(r)?;
    Ok(d.build_op(
        Opcode::Select,
        SmallVec::from_slice(&[cond, t, f]),
        std::iter::once(result_ty),
        crate::cuda_tile_ir::ir::AttrMap::new(),
        Vec::<RegionId>::new(),
        loc,
    ))
}

pub fn decode_cmpf<'a, 'm>(
    d: &mut BodyDecoder<'a, 'm>,
    r: &mut dyn ByteRead<'a>,
    loc: Location,
) -> Result<OpId> {
    let result_ty = TypeId(read_u32_var(r)?);
    let pred = read_enum::<ComparisonPredicate>(r, "ComparisonPredicate")?;
    let ord = read_enum::<ComparisonOrdering>(r, "ComparisonOrdering")?;
    let operands = bin_operands!(d, r);

    let pred_attr = d.arena.intern_attr(Attr::ComparisonPredicate(pred));
    let ord_attr = d.arena.intern_attr(Attr::ComparisonOrdering(ord));
    let mut attrs = crate::cuda_tile_ir::ir::AttrMap::new();
    attrs.push((OpAttrKey::ComparisonPredicate, pred_attr));
    attrs.push((OpAttrKey::ComparisonOrdering, ord_attr));

    Ok(d.build_op(
        Opcode::CmpF,
        operands,
        std::iter::once(result_ty),
        attrs,
        Vec::<RegionId>::new(),
        loc,
    ))
}

pub fn decode_cmpi<'a, 'm>(
    d: &mut BodyDecoder<'a, 'm>,
    r: &mut dyn ByteRead<'a>,
    loc: Location,
) -> Result<OpId> {
    let result_ty = TypeId(read_u32_var(r)?);
    let pred = read_enum::<ComparisonPredicate>(r, "ComparisonPredicate")?;
    let sign = read_enum::<Signedness>(r, "Signedness")?;
    let operands = bin_operands!(d, r);

    let pred_attr = d.arena.intern_attr(Attr::ComparisonPredicate(pred));
    let sign_attr = d.arena.intern_attr(Attr::Signedness(sign));
    let mut attrs = crate::cuda_tile_ir::ir::AttrMap::new();
    attrs.push((OpAttrKey::ComparisonPredicate, pred_attr));
    attrs.push((OpAttrKey::Signedness, sign_attr));

    Ok(d.build_op(
        Opcode::CmpI,
        operands,
        std::iter::once(result_ty),
        attrs,
        Vec::<RegionId>::new(),
        loc,
    ))
}

pub fn decode_ftof<'a, 'm>(
    d: &mut BodyDecoder<'a, 'm>,
    r: &mut dyn ByteRead<'a>,
    loc: Location,
) -> Result<OpId> {
    let result_ty = TypeId(read_u32_var(r)?);
    let rm = read_enum::<RoundingMode>(r, "RoundingMode")?;
    let operand = d.read_value_from_stream(r)?;
    let rm_attr = d.arena.intern_attr(Attr::RoundingMode(rm));
    let mut attrs = crate::cuda_tile_ir::ir::AttrMap::new();
    attrs.push((OpAttrKey::RoundingMode, rm_attr));
    Ok(d.build_op(
        Opcode::FToF,
        SmallVec::from_slice(&[operand]),
        std::iter::once(result_ty),
        attrs,
        Vec::<RegionId>::new(),
        loc,
    ))
}

pub fn decode_ftoi<'a, 'm>(
    d: &mut BodyDecoder<'a, 'm>,
    r: &mut dyn ByteRead<'a>,
    loc: Location,
) -> Result<OpId> {
    let result_ty = TypeId(read_u32_var(r)?);
    let sign = read_enum::<Signedness>(r, "Signedness")?;
    let rm = read_enum::<RoundingMode>(r, "RoundingMode")?;
    let operand = d.read_value_from_stream(r)?;
    let sign_attr = d.arena.intern_attr(Attr::Signedness(sign));
    let rm_attr = d.arena.intern_attr(Attr::RoundingMode(rm));
    let mut attrs = crate::cuda_tile_ir::ir::AttrMap::new();
    attrs.push((OpAttrKey::Signedness, sign_attr));
    attrs.push((OpAttrKey::RoundingMode, rm_attr));
    Ok(d.build_op(
        Opcode::FToI,
        SmallVec::from_slice(&[operand]),
        std::iter::once(result_ty),
        attrs,
        Vec::<RegionId>::new(),
        loc,
    ))
}

pub fn decode_itof<'a, 'm>(
    d: &mut BodyDecoder<'a, 'm>,
    r: &mut dyn ByteRead<'a>,
    loc: Location,
) -> Result<OpId> {
    let result_ty = TypeId(read_u32_var(r)?);
    let sign = read_enum::<Signedness>(r, "Signedness")?;
    let rm = read_enum::<RoundingMode>(r, "RoundingMode")?;
    let operand = d.read_value_from_stream(r)?;
    let sign_attr = d.arena.intern_attr(Attr::Signedness(sign));
    let rm_attr = d.arena.intern_attr(Attr::RoundingMode(rm));
    let mut attrs = crate::cuda_tile_ir::ir::AttrMap::new();
    attrs.push((OpAttrKey::Signedness, sign_attr));
    attrs.push((OpAttrKey::RoundingMode, rm_attr));
    Ok(d.build_op(
        Opcode::IToF,
        SmallVec::from_slice(&[operand]),
        std::iter::once(result_ty),
        attrs,
        Vec::<RegionId>::new(),
        loc,
    ))
}

pub fn decode_exti<'a, 'm>(
    d: &mut BodyDecoder<'a, 'm>,
    r: &mut dyn ByteRead<'a>,
    loc: Location,
) -> Result<OpId> {
    let result_ty = TypeId(read_u32_var(r)?);
    let sign = read_enum::<Signedness>(r, "Signedness")?;
    let operand = d.read_value_from_stream(r)?;
    let sign_attr = d.arena.intern_attr(Attr::Signedness(sign));
    let mut attrs = crate::cuda_tile_ir::ir::AttrMap::new();
    attrs.push((OpAttrKey::Signedness, sign_attr));
    Ok(d.build_op(
        Opcode::ExtI,
        SmallVec::from_slice(&[operand]),
        std::iter::once(result_ty),
        attrs,
        Vec::<RegionId>::new(),
        loc,
    ))
}

pub fn decode_trunci<'a, 'm>(
    d: &mut BodyDecoder<'a, 'm>,
    r: &mut dyn ByteRead<'a>,
    loc: Location,
) -> Result<OpId> {
    let result_ty = TypeId(read_u32_var(r)?);
    let overflow = read_enum::<IntegerOverflow>(r, "IntegerOverflow")?;
    let operand = d.read_value_from_stream(r)?;
    let overflow_attr = d.arena.intern_attr(Attr::IntegerOverflow(overflow));
    let mut attrs = crate::cuda_tile_ir::ir::AttrMap::new();
    attrs.push((OpAttrKey::Overflow, overflow_attr));
    Ok(d.build_op(
        Opcode::TruncI,
        SmallVec::from_slice(&[operand]),
        std::iter::once(result_ty),
        attrs,
        Vec::<RegionId>::new(),
        loc,
    ))
}

pub fn decode_bitcast<'a, 'm>(
    d: &mut BodyDecoder<'a, 'm>,
    r: &mut dyn ByteRead<'a>,
    loc: Location,
) -> Result<OpId> {
    decode_simple_cast(d, r, Opcode::Bitcast, loc)
}
pub fn decode_int_to_ptr<'a, 'm>(
    d: &mut BodyDecoder<'a, 'm>,
    r: &mut dyn ByteRead<'a>,
    loc: Location,
) -> Result<OpId> {
    decode_simple_cast(d, r, Opcode::IntToPtr, loc)
}
pub fn decode_ptr_to_int<'a, 'm>(
    d: &mut BodyDecoder<'a, 'm>,
    r: &mut dyn ByteRead<'a>,
    loc: Location,
) -> Result<OpId> {
    decode_simple_cast(d, r, Opcode::PtrToInt, loc)
}
pub fn decode_ptr_to_ptr<'a, 'm>(
    d: &mut BodyDecoder<'a, 'm>,
    r: &mut dyn ByteRead<'a>,
    loc: Location,
) -> Result<OpId> {
    decode_simple_cast(d, r, Opcode::PtrToPtr, loc)
}

fn decode_simple_cast<'a, 'm>(
    d: &mut BodyDecoder<'a, 'm>,
    r: &mut dyn ByteRead<'a>,
    opcode: Opcode,
    loc: Location,
) -> Result<OpId> {
    let result_ty = TypeId(read_u32_var(r)?);
    let operand = d.read_value_from_stream(r)?;
    Ok(d.build_op(
        opcode,
        SmallVec::from_slice(&[operand]),
        std::iter::once(result_ty),
        crate::cuda_tile_ir::ir::AttrMap::new(),
        Vec::<RegionId>::new(),
        loc,
    ))
}

fn decode_float_binary_with_rounding<'a, 'm>(
    d: &mut BodyDecoder<'a, 'm>,
    r: &mut dyn ByteRead<'a>,
    opcode: Opcode,
    loc: Location,
) -> Result<OpId> {
    let result_ty = TypeId(read_u32_var(r)?);
    let flush = r.read_var_u64()? != 0;
    let rm = read_enum::<RoundingMode>(r, "RoundingMode")?;
    let operands = bin_operands!(d, r);

    let rm_attr = d.arena.intern_attr(Attr::RoundingMode(rm));
    let mut attrs = crate::cuda_tile_ir::ir::AttrMap::new();
    attrs.push((OpAttrKey::RoundingMode, rm_attr));
    if flush {
        let unit = d.arena.intern_attr(Attr::Unit);
        attrs.push((OpAttrKey::FlushToZero, unit));
    }

    Ok(d.build_op(
        opcode,
        operands,
        std::iter::once(result_ty),
        attrs,
        Vec::<RegionId>::new(),
        loc,
    ))
}

fn decode_float_binary_simple<'a, 'm>(
    d: &mut BodyDecoder<'a, 'm>,
    r: &mut dyn ByteRead<'a>,
    opcode: Opcode,
    loc: Location,
) -> Result<OpId> {
    let result_ty = TypeId(read_u32_var(r)?);
    let operands = bin_operands!(d, r);
    Ok(d.build_op(
        opcode,
        operands,
        std::iter::once(result_ty),
        crate::cuda_tile_ir::ir::AttrMap::new(),
        Vec::<RegionId>::new(),
        loc,
    ))
}

fn decode_float_binary_with_nan_flush<'a, 'm>(
    d: &mut BodyDecoder<'a, 'm>,
    r: &mut dyn ByteRead<'a>,
    opcode: Opcode,
    loc: Location,
) -> Result<OpId> {
    let result_ty = TypeId(read_u32_var(r)?);
    let flags = r.read_var_u64()? as u32;
    let operands = bin_operands!(d, r);

    let mut attrs = crate::cuda_tile_ir::ir::AttrMap::new();
    if (flags & 0x1) != 0 {
        let a = d.arena.intern_attr(Attr::Bool(true));
        attrs.push((OpAttrKey::PropagateNan, a));
    }
    if (flags & 0x2) != 0 {
        let a = d.arena.intern_attr(Attr::Unit);
        attrs.push((OpAttrKey::FlushToZero, a));
    }

    Ok(d.build_op(
        opcode,
        operands,
        std::iter::once(result_ty),
        attrs,
        Vec::<RegionId>::new(),
        loc,
    ))
}

fn decode_float_unary_no_attrs<'a, 'm>(
    d: &mut BodyDecoder<'a, 'm>,
    r: &mut dyn ByteRead<'a>,
    opcode: Opcode,
    loc: Location,
) -> Result<OpId> {
    let result_ty = TypeId(read_u32_var(r)?);
    let operand = d.read_value_from_stream(r)?;
    Ok(d.build_op(
        opcode,
        SmallVec::from_slice(&[operand]),
        std::iter::once(result_ty),
        crate::cuda_tile_ir::ir::AttrMap::new(),
        Vec::<RegionId>::new(),
        loc,
    ))
}

fn decode_float_unary_with_flush_flag<'a, 'm>(
    d: &mut BodyDecoder<'a, 'm>,
    r: &mut dyn ByteRead<'a>,
    opcode: Opcode,
    loc: Location,
) -> Result<OpId> {
    let result_ty = TypeId(read_u32_var(r)?);
    let flags = r.read_var_u64()?;
    let operand = d.read_value_from_stream(r)?;

    let mut attrs = crate::cuda_tile_ir::ir::AttrMap::new();
    if (flags & 1) != 0 {
        let unit = d.arena.intern_attr(Attr::Unit);
        attrs.push((OpAttrKey::FlushToZero, unit));
    }

    Ok(d.build_op(
        opcode,
        SmallVec::from_slice(&[operand]),
        std::iter::once(result_ty),
        attrs,
        Vec::<RegionId>::new(),
        loc,
    ))
}

fn decode_int_binary_with_overflow<'a, 'm>(
    d: &mut BodyDecoder<'a, 'm>,
    r: &mut dyn ByteRead<'a>,
    opcode: Opcode,
    loc: Location,
) -> Result<OpId> {
    let result_ty = TypeId(read_u32_var(r)?);
    let overflow = read_enum::<IntegerOverflow>(r, "IntegerOverflow")?;
    let operands = bin_operands!(d, r);

    let overflow_attr = d.arena.intern_attr(Attr::IntegerOverflow(overflow));
    let mut attrs = crate::cuda_tile_ir::ir::AttrMap::new();
    attrs.push((OpAttrKey::Overflow, overflow_attr));

    Ok(d.build_op(
        opcode,
        operands,
        std::iter::once(result_ty),
        attrs,
        Vec::<RegionId>::new(),
        loc,
    ))
}

fn decode_int_binary_with_signedness<'a, 'm>(
    d: &mut BodyDecoder<'a, 'm>,
    r: &mut dyn ByteRead<'a>,
    opcode: Opcode,
    loc: Location,
) -> Result<OpId> {
    let result_ty = TypeId(read_u32_var(r)?);
    let sign = read_enum::<Signedness>(r, "Signedness")?;
    let operands = bin_operands!(d, r);
    let sign_attr = d.arena.intern_attr(Attr::Signedness(sign));

    let mut attrs = crate::cuda_tile_ir::ir::AttrMap::new();
    attrs.push((OpAttrKey::Signedness, sign_attr));

    Ok(d.build_op(
        opcode,
        operands,
        std::iter::once(result_ty),
        attrs,
        Vec::<RegionId>::new(),
        loc,
    ))
}

fn decode_int_binary_simple<'a, 'm>(
    d: &mut BodyDecoder<'a, 'm>,
    r: &mut dyn ByteRead<'a>,
    opcode: Opcode,
    loc: Location,
) -> Result<OpId> {
    let result_ty = TypeId(read_u32_var(r)?);
    let operands = bin_operands!(d, r);
    Ok(d.build_op(
        opcode,
        operands,
        std::iter::once(result_ty),
        crate::cuda_tile_ir::ir::AttrMap::new(),
        Vec::<RegionId>::new(),
        loc,
    ))
}

fn decode_int_unary_simple<'a, 'm>(
    d: &mut BodyDecoder<'a, 'm>,
    r: &mut dyn ByteRead<'a>,
    opcode: Opcode,
    loc: Location,
) -> Result<OpId> {
    let result_ty = TypeId(read_u32_var(r)?);
    let operand = d.read_value_from_stream(r)?;
    Ok(d.build_op(
        opcode,
        SmallVec::from_slice(&[operand]),
        std::iter::once(result_ty),
        crate::cuda_tile_ir::ir::AttrMap::new(),
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

fn read_u32_var<'a>(r: &mut dyn ByteRead<'a>) -> Result<u32> {
    let v = r.read_var_u64()?;
    u32::try_from(v).map_err(|_| BytecodeError::ParseError("varint does not fit into u32".into()))
}
