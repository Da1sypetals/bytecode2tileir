use std::fmt;

use crate::cuda_tile_ir::OpAttrKey;
use crate::cuda_tile_ir::attrs::{Attr, DenseStorage};
use crate::cuda_tile_ir::ids::ValueId;
use crate::cuda_tile_ir::ir::Operation;
use crate::cuda_tile_ir::types::{Dim, Type};

use super::super::Line;
use super::super::Printer;
use super::super::escape::escape_mlir_string;
use super::super::fmt::{attrs, dense, types};
use super::super::indent::MlirPrinter;
use super::control_flow::print_region_with_block_args;

pub(super) fn print_constant<W: MlirPrinter + ?Sized>(
    p: &mut Printer<'_, '_, W>,
    op: &Operation,
) -> fmt::Result {
    let out = op.results.first().copied().unwrap_or(ValueId(0));
    let ty = p.ctx.value_ty(out);
    let ty_str = types::fmt_type(&p.ctx, ty);

    let (_elem_ty, elem_ty_str, shape) = match p.ctx.ty(ty) {
        Type::Tile { element, shape } => (
            *element,
            types::fmt_scalar(&p.ctx, *element),
            shape.0.clone(),
        ),
        _ => (ty, ty_str.clone(), Vec::new()),
    };
    let shape_i64: Vec<i64> = shape
        .iter()
        .map(|d| match d {
            Dim::Static(v) => *v,
            Dim::Dynamic => -1,
        })
        .collect();

    let (dense_ty, bytes) = match p.ctx.attr(op, OpAttrKey::Value) {
        Some(Attr::DenseElements { ty, storage }) => {
            let bytes = match storage {
                DenseStorage::Inline(v) => v.as_slice(),
                DenseStorage::Const(cid) => p.ctx.module.consts.get(*cid).unwrap_or(&[]),
                DenseStorage::Strings(_) => &[][..],
            };
            (*ty, bytes)
        }
        _ => panic!("cuda_tile.constant missing DenseElements value attribute"),
    };

    let elem = match p.ctx.ty(dense_ty) {
        Type::Tile { element, .. } => *element,
        _ => dense_ty,
    };
    let value_str = dense::fmt_dense_value(bytes, &p.ctx, elem, &shape_i64);

    p.write_indent()?;
    writeln!(
        p.w,
        "{} = constant <{}: {}> : {}",
        p.ctx.slot(out),
        elem_ty_str,
        value_str,
        ty_str
    )
}

pub(super) fn print_assert<W: MlirPrinter + ?Sized>(
    p: &mut Printer<'_, '_, W>,
    op: &Operation,
) -> fmt::Result {
    let cond = op
        .operands
        .first()
        .copied()
        .map(|v| p.ctx.slot(v).to_string())
        .unwrap_or_default();
    let cond_ty = op
        .operands
        .first()
        .copied()
        .map(|v| types::fmt_type(&p.ctx, p.ctx.value_ty(v)))
        .unwrap_or_default();
    let msg = p
        .ctx
        .attr(op, OpAttrKey::Message)
        .and_then(|a| match a {
            Attr::String(s) => Some(s.as_str()),
            _ => None,
        })
        .unwrap_or("");
    let escaped = escape_mlir_string(msg);
    p.write_indent()?;
    writeln!(p.w, "assert {}, \"{}\" : {}", cond, escaped, cond_ty)
}

pub(super) fn print_assume<W: MlirPrinter + ?Sized>(
    p: &mut Printer<'_, '_, W>,
    op: &Operation,
) -> fmt::Result {
    // Assume passes through the value unchanged; print as a cast-like op.
    let out = op.results.first().copied().unwrap_or(ValueId(0));
    let operand = op
        .operands
        .first()
        .copied()
        .map(|v| p.ctx.slot(v).to_string())
        .unwrap_or_default();
    let ty = types::fmt_type(&p.ctx, p.ctx.value_ty(out));
    let pred = p
        .ctx
        .attr(op, OpAttrKey::Predicate)
        .and_then(|a| attrs::fmt_assume_predicate(a));
    p.write_indent()?;
    if let Some(pred) = pred {
        writeln!(
            p.w,
            "{} = assume {}, {} : {}",
            p.ctx.slot(out),
            pred,
            operand,
            ty
        )
    } else {
        writeln!(p.w, "{} = assume {} : {}", p.ctx.slot(out), operand, ty)
    }
}

pub(super) fn print_get_global<W: MlirPrinter + ?Sized>(
    p: &mut Printer<'_, '_, W>,
    op: &Operation,
) -> fmt::Result {
    let out = op.results.first().copied().unwrap_or(ValueId(0));
    let ty = types::fmt_type(&p.ctx, p.ctx.value_ty(out));
    let sym = p
        .ctx
        .attr(op, OpAttrKey::GlobalName)
        .and_then(|a| match a {
            Attr::FlatSymbolRef(s) | Attr::String(s) => Some(s.as_str()),
            _ => None,
        })
        .unwrap_or("?");
    p.write_indent()?;
    writeln!(p.w, "{} = get_global @{} : {}", p.ctx.slot(out), sym, ty)
}

pub(super) fn print_print<W: MlirPrinter + ?Sized>(
    p: &mut Printer<'_, '_, W>,
    op: &Operation,
) -> fmt::Result {
    let fmt_str = p
        .ctx
        .attr(op, OpAttrKey::Format)
        .and_then(|a| match a {
            Attr::String(s) => Some(s.as_str()),
            _ => None,
        })
        .unwrap_or("");
    let escaped = escape_mlir_string(fmt_str);

    let operands: Vec<String> = op
        .operands
        .iter()
        .copied()
        .map(|v| p.ctx.slot(v).to_string())
        .collect();
    let types: Vec<String> = op
        .operands
        .iter()
        .copied()
        .map(|v| types::fmt_type(&p.ctx, p.ctx.value_ty(v)))
        .collect();
    p.write_indent()?;
    if operands.is_empty() {
        writeln!(p.w, "print \"{}\"", escaped)
    } else if types.len() == operands.len() {
        writeln!(
            p.w,
            "print \"{}\", {} : {}",
            escaped,
            operands.join(", "),
            types.join(", ")
        )
    } else {
        writeln!(p.w, "print \"{}\", {}", escaped, operands.join(", "))
    }
}

pub(super) fn print_make_token<W: MlirPrinter + ?Sized>(
    p: &mut Printer<'_, '_, W>,
    op: &Operation,
) -> fmt::Result {
    let out = op.results.first().copied().unwrap_or(ValueId(0));
    p.write_indent()?;
    writeln!(
        p.w,
        "{} = make_token : {}",
        p.ctx.slot(out),
        types::fmt_type(&p.ctx, p.ctx.value_ty(out))
    )
}

pub(super) fn print_mma<W: MlirPrinter + ?Sized>(
    p: &mut Printer<'_, '_, W>,
    op: &Operation,
) -> fmt::Result {
    let out = op.results.first().copied().unwrap_or(ValueId(0));
    let operand_types: Vec<String> = op
        .operands
        .iter()
        .copied()
        .map(|v| types::fmt_type(&p.ctx, p.ctx.value_ty(v)))
        .collect();

    let mut line = Line::new();
    let results = [out];
    line.results(&p.ctx, &results);
    line.op_name(op.opcode.name());
    line.operands(&p.ctx, &op.operands);
    if let Some(Attr::Signedness(s)) = p.ctx.attr(op, OpAttrKey::SignednessLhs) {
        line.push(&format!(" {}", s));
    }
    if let Some(Attr::Signedness(s)) = p.ctx.attr(op, OpAttrKey::SignednessRhs) {
        line.push(&format!(" {}", s));
    }
    if !operand_types.is_empty() {
        line.push(&format!(" : {}", operand_types.join(", ")));
    }
    p.write_indent()?;
    p.w.writeln(line.as_str())
}

#[derive(Debug, Clone, Copy)]
enum CmpKind {
    Int,
    Float,
}

fn print_cmp<W: MlirPrinter + ?Sized>(
    p: &mut Printer<'_, '_, W>,
    op: &Operation,
    kind: CmpKind,
) -> fmt::Result {
    let out = op.results.first().copied().unwrap_or(ValueId(0));
    let input_type = op
        .operands
        .first()
        .copied()
        .map(|v| types::fmt_type(&p.ctx, p.ctx.value_ty(v)))
        .unwrap_or_default();

    let mut line = Line::new();
    let results = [out];
    line.results(&p.ctx, &results);
    line.op_name(match kind {
        CmpKind::Int => "cmpi",
        CmpKind::Float => "cmpf",
    });

    if let Some(Attr::ComparisonPredicate(pred)) = p.ctx.attr(op, OpAttrKey::ComparisonPredicate) {
        line.push(&format!(" {}", pred));
    }
    if matches!(kind, CmpKind::Float) {
        if let Some(Attr::ComparisonOrdering(ord)) = p.ctx.attr(op, OpAttrKey::ComparisonOrdering) {
            line.push(&format!(" {}", ord));
        }
    }

    line.operands(&p.ctx, &op.operands);

    if matches!(kind, CmpKind::Int) {
        if let Some(Attr::Signedness(s)) = p.ctx.attr(op, OpAttrKey::Signedness) {
            line.push(&format!(", {}", s));
        }
    }

    line.push(&format!(
        " : {} -> {}",
        input_type,
        types::fmt_type(&p.ctx, p.ctx.value_ty(out))
    ));

    p.write_indent()?;
    p.w.writeln(line.as_str())
}

pub(super) fn print_cmpi<W: MlirPrinter + ?Sized>(
    p: &mut Printer<'_, '_, W>,
    op: &Operation,
) -> fmt::Result {
    print_cmp(p, op, CmpKind::Int)
}

pub(super) fn print_cmpf<W: MlirPrinter + ?Sized>(
    p: &mut Printer<'_, '_, W>,
    op: &Operation,
) -> fmt::Result {
    print_cmp(p, op, CmpKind::Float)
}

pub(super) fn print_select<W: MlirPrinter + ?Sized>(
    p: &mut Printer<'_, '_, W>,
    op: &Operation,
) -> fmt::Result {
    let out = op.results.first().copied().unwrap_or(ValueId(0));
    let cond_type = op
        .operands
        .first()
        .copied()
        .map(|v| types::fmt_type(&p.ctx, p.ctx.value_ty(v)))
        .unwrap_or_default();
    let mut line = Line::new();
    let results = [out];
    line.results(&p.ctx, &results);
    line.op_name("select");
    line.operands(&p.ctx, &op.operands);
    line.push(&format!(
        " : {}, {}",
        cond_type,
        types::fmt_type(&p.ctx, p.ctx.value_ty(out))
    ));
    p.write_indent()?;
    p.w.writeln(line.as_str())
}

fn print_reduce_like<W: MlirPrinter + ?Sized>(
    p: &mut Printer<'_, '_, W>,
    op: &Operation,
    kind: &str,
) -> fmt::Result {
    let mut line = Line::new();
    line.results(&p.ctx, &op.results);
    line.op_name(kind);
    line.operands(&p.ctx, &op.operands);
    if let Some(Attr::Int { value, .. }) = p.ctx.attr(op, OpAttrKey::Dim) {
        line.push(&format!(" dim={}", value));
    }
    if kind == "scan" {
        if let Some(Attr::Bool(b)) = p.ctx.attr(op, OpAttrKey::Reverse) {
            line.push(&format!(" reverse={}", if *b { "true" } else { "false" }));
        }
    }
    if let Some(Attr::Array(arr)) = p.ctx.attr(op, OpAttrKey::Identities) {
        if !arr.is_empty() {
            let ids: Vec<String> = arr
                .iter()
                .filter_map(|id| p.ctx.module.arena.attrs.get(id.0 as usize))
                .map(|a| attrs::identity_literal(&p.ctx, a))
                .collect();
            line.push(&format!(" identities=[{}]", ids.join(", ")));
        }
    }
    let input_types: Vec<String> = op
        .operands
        .iter()
        .copied()
        .map(|v| types::fmt_type(&p.ctx, p.ctx.value_ty(v)))
        .collect();
    let result_types: Vec<String> = op
        .results
        .iter()
        .copied()
        .map(|v| types::fmt_type(&p.ctx, p.ctx.value_ty(v)))
        .collect();
    if !input_types.is_empty() && !result_types.is_empty() {
        line.push(&format!(
            " : {} -> {}",
            input_types.join(", "),
            result_types.join(", ")
        ));
    }
    p.write_indent()?;
    p.w.writeln(line.as_str())?;

    if let Some(rid) = op.regions.first().copied() {
        let region = p.ctx.module.arena.region_(rid);
        print_region_with_block_args(p, region)?;
    }
    Ok(())
}

pub(super) fn print_reduce<W: MlirPrinter + ?Sized>(
    p: &mut Printer<'_, '_, W>,
    op: &Operation,
) -> fmt::Result {
    print_reduce_like(p, op, "reduce")
}

pub(super) fn print_scan<W: MlirPrinter + ?Sized>(
    p: &mut Printer<'_, '_, W>,
    op: &Operation,
) -> fmt::Result {
    print_reduce_like(p, op, "scan")
}
