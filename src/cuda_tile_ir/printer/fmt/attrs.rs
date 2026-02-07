use std::fmt::{self, Display};

use crate::cuda_tile_ir::OpAttrKey;
use crate::cuda_tile_ir::attrs::Attr;
use crate::cuda_tile_ir::enums::{
    AtomicRMWMode, ComparisonOrdering, ComparisonPredicate, MemoryOrdering, MemoryScope,
    RoundingMode, Signedness,
};
use crate::cuda_tile_ir::ir::Operation;

use super::super::Line;
use super::super::ctx::PrinterCtx;
use super::dense::fmt_float;

pub fn fmt_assume_predicate(attr: &Attr) -> Option<String> {
    match attr {
        Attr::DivBy {
            divisor,
            every,
            along,
            ..
        } => {
            let mut s = format!("div_by<{}", divisor);
            if let (Some(every), Some(along)) = (every, along) {
                s.push_str(&format!(", every {} along {}", every, along));
            }
            s.push('>');
            Some(s)
        }
        Attr::SameElements(v) => {
            let inner = v
                .iter()
                .map(|x| x.to_string())
                .collect::<Vec<_>>()
                .join(", ");
            Some(format!("same_elements<[{}]>", inner))
        }
        Attr::Bounded { lb, ub } => {
            let lb = lb.map(|x| x.to_string()).unwrap_or_else(|| "?".to_string());
            let ub = ub.map(|x| x.to_string()).unwrap_or_else(|| "?".to_string());
            Some(format!("bounded<{}, {}>", lb, ub))
        }
        Attr::NonNegative => Some("non_negative".to_string()),
        _ => None,
    }
}

pub fn fmt_optimization_hints_angle(ctx: &PrinterCtx<'_>, attr: &Attr) -> Option<String> {
    match attr {
        Attr::OptimizationHints(items) | Attr::Dict(items) => {
            let inner = items
                .iter()
                .map(|(k, v)| {
                    let a = ctx
                        .module
                        .arena
                        .attrs
                        .get(v.0 as usize)
                        .map(|x| fmt_value_for_opt_hints(ctx, x));
                    format!("{} = {}", k, a.unwrap_or_else(|| "?".to_string()))
                })
                .collect::<Vec<_>>()
                .join(", ");
            Some(format!("<{}>", inner))
        }
        _ => None,
    }
}

fn fmt_value_for_opt_hints(ctx: &PrinterCtx<'_>, a: &Attr) -> String {
    match a {
        Attr::Unit => "unit".to_string(),
        Attr::Bool(b) => b.to_string(),
        Attr::Int { value, .. } => value.to_string(),
        Attr::String(s) | Attr::FlatSymbolRef(s) => s.clone(),
        Attr::Dict(items) | Attr::OptimizationHints(items) => {
            let inner = items
                .iter()
                .map(|(k, vv)| {
                    let av = ctx
                        .module
                        .arena
                        .attrs
                        .get(vv.0 as usize)
                        .map(|x| fmt_value_for_opt_hints(ctx, x));
                    format!("{} = {}", k, av.unwrap_or_else(|| "?".to_string()))
                })
                .collect::<Vec<_>>()
                .join(", ");
            format!("{{{}}}", inner)
        }
        other => format!("{:?}", other),
    }
}

impl Display for RoundingMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            RoundingMode::NearestEven => "nearest_even",
            RoundingMode::Zero => "zero",
            RoundingMode::NegativeInf => "negative_inf",
            RoundingMode::PositiveInf => "positive_inf",
            RoundingMode::Approx => "approx",
            RoundingMode::Full => "full",
            RoundingMode::NearestIntToZero => "nearest_int_to_zero",
        })
    }
}

impl Display for Signedness {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Signedness::Signed => "signed",
            Signedness::Unsigned => "unsigned",
        })
    }
}

impl Display for MemoryOrdering {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            MemoryOrdering::Weak => "weak",
            MemoryOrdering::Relaxed => "relaxed",
            MemoryOrdering::Acquire => "acquire",
            MemoryOrdering::Release => "release",
            MemoryOrdering::AcqRel => "acq_rel",
        })
    }
}

impl Display for MemoryScope {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            MemoryScope::TlBlk => "tl_blk",
            MemoryScope::Device => "device",
            MemoryScope::Sys => "sys",
        })
    }
}

impl Display for ComparisonPredicate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            ComparisonPredicate::Equal => "equal",
            ComparisonPredicate::NotEqual => "not_equal",
            ComparisonPredicate::LessThan => "less_than",
            ComparisonPredicate::LessThanOrEqual => "less_than_or_equal",
            ComparisonPredicate::GreaterThan => "greater_than",
            ComparisonPredicate::GreaterThanOrEqual => "greater_than_or_equal",
        })
    }
}

impl Display for ComparisonOrdering {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            ComparisonOrdering::Ordered => "ordered",
            ComparisonOrdering::Unordered => "unordered",
        })
    }
}

impl Display for AtomicRMWMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            AtomicRMWMode::And => "and",
            AtomicRMWMode::Or => "or",
            AtomicRMWMode::Xor => "xor",
            AtomicRMWMode::Add => "add",
            AtomicRMWMode::AddF => "addf",
            AtomicRMWMode::Max => "max",
            AtomicRMWMode::Min => "min",
            AtomicRMWMode::UMax => "umax",
            AtomicRMWMode::UMin => "umin",
            AtomicRMWMode::Xchg => "xchg",
        })
    }
}

pub(in crate::cuda_tile_ir::printer) fn identity_literal(ctx: &PrinterCtx<'_>, attr: &Attr) -> String {
    match attr {
        Attr::Float { kind, bits } => {
            let value_bits = match kind {
                crate::cuda_tile_ir::enums::FloatKind::F16 | crate::cuda_tile_ir::enums::FloatKind::BF16 => {
                    (*bits) & 0xFFFF
                }
                crate::cuda_tile_ir::enums::FloatKind::F32 | crate::cuda_tile_ir::enums::FloatKind::TF32 => {
                    (*bits) & 0xFFFF_FFFF
                }
                crate::cuda_tile_ir::enums::FloatKind::F64 => *bits,
                crate::cuda_tile_ir::enums::FloatKind::F8E4M3FN | crate::cuda_tile_ir::enums::FloatKind::F8E5M2 => {
                    (*bits) & 0xFF
                }
            };
            let ty_str = match kind {
                crate::cuda_tile_ir::enums::FloatKind::F16 => "f16",
                crate::cuda_tile_ir::enums::FloatKind::BF16 => "bf16",
                crate::cuda_tile_ir::enums::FloatKind::F32 => "f32",
                crate::cuda_tile_ir::enums::FloatKind::TF32 => "tf32",
                crate::cuda_tile_ir::enums::FloatKind::F64 => "f64",
                crate::cuda_tile_ir::enums::FloatKind::F8E4M3FN => "f8E4M3FN",
                crate::cuda_tile_ir::enums::FloatKind::F8E5M2 => "f8E5M2",
            };
            let lit = match kind {
                crate::cuda_tile_ir::enums::FloatKind::F16 => {
                    let b = value_bits as u16;
                    let f = half::f16::from_bits(b).to_f32();
                    if !f.is_finite() {
                        format!("0x{:04X}", b)
                    } else {
                        fmt_float(f as f64)
                    }
                }
                crate::cuda_tile_ir::enums::FloatKind::BF16 => {
                    let b = value_bits as u16;
                    let f = half::bf16::from_bits(b).to_f32();
                    if !f.is_finite() {
                        format!("0x{:04X}", b)
                    } else {
                        fmt_float(f as f64)
                    }
                }
                crate::cuda_tile_ir::enums::FloatKind::F32 | crate::cuda_tile_ir::enums::FloatKind::TF32 => {
                    let b = value_bits as u32;
                    let f = f32::from_bits(b);
                    if !f.is_finite() {
                        format!("0x{:08X}", b)
                    } else {
                        fmt_float(f as f64)
                    }
                }
                crate::cuda_tile_ir::enums::FloatKind::F64 => {
                    let f = f64::from_bits(value_bits);
                    if !f.is_finite() {
                        format!("0x{:016X}", value_bits)
                    } else {
                        fmt_float(f)
                    }
                }
                crate::cuda_tile_ir::enums::FloatKind::F8E4M3FN | crate::cuda_tile_ir::enums::FloatKind::F8E5M2 => {
                    format!("0x{:02X}", value_bits as u8)
                }
            };
            format!("{} : {}", lit, ty_str)
        }
        Attr::Int { value, ty } => {
            let ty_str = match ctx.ty(*ty) {
                crate::cuda_tile_ir::types::Type::Int { width: 8 } => "i8",
                crate::cuda_tile_ir::types::Type::Int { width: 16 } => "i16",
                crate::cuda_tile_ir::types::Type::Int { width: 32 } => "i32",
                crate::cuda_tile_ir::types::Type::Int { width: 64 } => "i64",
                _ => "i32",
            };
            format!("{} : {}", value, ty_str)
        }
        _ => "0 : i32".to_string(),
    }
}

pub fn dense_i32_attr(ctx: &PrinterCtx<'_>, op: &Operation, key: OpAttrKey) -> Vec<i32> {
    match ctx.attr(op, key) {
        Some(Attr::DenseI32Array(v)) => v.clone(),
        _ => Vec::new(),
    }
}

pub fn emit_mem_semantics(line: &mut Line, ctx: &PrinterCtx<'_>, op: &Operation) {
    if let Some(Attr::MemoryOrdering(o)) = ctx.attr(op, OpAttrKey::MemoryOrderingSemantics) {
        line.push_str(&format!(" {}", o));
    }
    if let Some(Attr::MemoryScope(s)) = ctx.attr(op, OpAttrKey::MemoryScope) {
        line.push_str(&format!(" {}", s));
    }
}

pub fn emit_rounding_nonzero(line: &mut Line, ctx: &PrinterCtx<'_>, op: &Operation) {
    if let Some(Attr::RoundingMode(r)) = ctx.attr(op, OpAttrKey::RoundingMode) {
        if !matches!(r, RoundingMode::Zero) {
            line.push_str(&format!(" rounding<{}>", r));
        }
    }
}

pub fn emit_rounding_ieee(line: &mut Line, ctx: &PrinterCtx<'_>, op: &Operation) {
    if let Some(Attr::RoundingMode(r)) = ctx.attr(op, OpAttrKey::RoundingMode) {
        if !matches!(r, RoundingMode::NearestEven) {
            line.push_str(&format!(" rounding<{}>", r));
        } else {
            line.push_char(' ');
        }
    }
}

pub fn emit_rounding_non_ieee(line: &mut Line, ctx: &PrinterCtx<'_>, op: &Operation) {
    if let Some(Attr::RoundingMode(r)) = ctx.attr(op, OpAttrKey::RoundingMode) {
        if !matches!(r, RoundingMode::NearestEven) {
            line.push_str(&format!(" rounding<{}>", r));
        }
    }
}

pub fn emit_flush_to_zero(line: &mut Line, ctx: &PrinterCtx<'_>, op: &Operation) {
    if ctx.attr(op, OpAttrKey::FlushToZero).is_some() {
        line.push_str(" flush_to_zero");
    }
}

pub fn emit_signedness(line: &mut Line, ctx: &PrinterCtx<'_>, op: &Operation, key: OpAttrKey) {
    if let Some(Attr::Signedness(s)) = ctx.attr(op, key) {
        line.push_str(&format!(" {}", s));
    }
}

pub fn emit_comparison_predicate(
    line: &mut Line,
    ctx: &PrinterCtx<'_>,
    op: &Operation,
    key: OpAttrKey,
) {
    if let Some(Attr::ComparisonPredicate(p)) = ctx.attr(op, key) {
        line.push_str(&format!(" {}", p));
    }
}

pub fn emit_comparison_ordering(line: &mut Line, ctx: &PrinterCtx<'_>, op: &Operation) {
    if let Some(Attr::ComparisonOrdering(o)) = ctx.attr(op, OpAttrKey::ComparisonOrdering) {
        line.push_str(&format!(" {}", o));
    }
}

pub fn emit_atomic_rmw_mode(line: &mut Line, ctx: &PrinterCtx<'_>, op: &Operation) {
    if let Some(Attr::AtomicRMWMode(m)) = ctx.attr(op, OpAttrKey::Mode) {
        line.push_str(&format!(" {}", m));
    }
}

pub fn emit_permutation(line: &mut Line, ctx: &PrinterCtx<'_>, op: &Operation) {
    if let Some(Attr::DenseI32Array(arr)) = ctx.attr(op, OpAttrKey::Permutation) {
        let s = arr
            .iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        line.push_str(&format!(" [{}]", s));
    }
}
