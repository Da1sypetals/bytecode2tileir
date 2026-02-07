use std::fmt;

use crate::cuda_tile_ir::attrs::Attr;
use crate::cuda_tile_ir::ir::Operation;
use crate::cuda_tile_ir::{OpAttrKey, Opcode};

use super::super::Line;
use super::super::Printer;
use super::super::fmt::{attrs, types};
use super::super::indent::MlirPrinter;

#[derive(Debug, Clone, Copy)]
struct OperandSegments {
    num_indices: usize,
    has_mask: bool,
    has_other: bool,
    has_token: bool,
}

impl OperandSegments {
    fn for_load_view(ctx: &super::super::ctx::PrinterCtx<'_>, op: &Operation) -> Self {
        let segs = attrs::dense_i32_attr(ctx, op, OpAttrKey::OperandSegmentSizes);
        Self {
            num_indices: segs.get(1).copied().unwrap_or(0) as usize,
            has_token: segs.get(2).copied().unwrap_or(0) == 1,
            has_mask: false,
            has_other: false,
        }
    }

    fn for_load_ptr(ctx: &super::super::ctx::PrinterCtx<'_>, op: &Operation) -> Self {
        let segs = attrs::dense_i32_attr(ctx, op, OpAttrKey::OperandSegmentSizes);
        Self {
            num_indices: 0,
            has_mask: segs.get(1).copied().unwrap_or(0) == 1,
            has_other: segs.get(2).copied().unwrap_or(0) == 1,
            has_token: segs.get(3).copied().unwrap_or(0) == 1,
        }
    }

    fn for_store_ptr(ctx: &super::super::ctx::PrinterCtx<'_>, op: &Operation) -> Self {
        let segs = attrs::dense_i32_attr(ctx, op, OpAttrKey::OperandSegmentSizes);
        Self {
            num_indices: 0,
            has_mask: segs.get(2).copied().unwrap_or(0) == 1,
            has_other: false,
            has_token: segs.get(3).copied().unwrap_or(0) == 1,
        }
    }

    fn for_store_view(ctx: &super::super::ctx::PrinterCtx<'_>, op: &Operation) -> Self {
        let segs = attrs::dense_i32_attr(ctx, op, OpAttrKey::OperandSegmentSizes);
        Self {
            num_indices: segs.get(2).copied().unwrap_or(0) as usize,
            has_token: segs.get(3).copied().unwrap_or(0) == 1,
            has_mask: false,
            has_other: false,
        }
    }

    fn for_atomic_rmw(ctx: &super::super::ctx::PrinterCtx<'_>, op: &Operation) -> Self {
        let segs = attrs::dense_i32_attr(ctx, op, OpAttrKey::OperandSegmentSizes);
        Self {
            num_indices: 0,
            has_mask: segs.get(2).copied().unwrap_or(0) > 0,
            has_other: false,
            has_token: segs.get(3).copied().unwrap_or(0) > 0,
        }
    }

    fn for_atomic_cas(ctx: &super::super::ctx::PrinterCtx<'_>, op: &Operation) -> Self {
        let segs = attrs::dense_i32_attr(ctx, op, OpAttrKey::OperandSegmentSizes);
        Self {
            num_indices: 0,
            has_mask: segs.get(3).copied().unwrap_or(0) > 0,
            has_other: false,
            has_token: segs.get(4).copied().unwrap_or(0) > 0,
        }
    }
}

fn emit_opt_hints<W: MlirPrinter + ?Sized>(
    line: &mut Line,
    p: &Printer<'_, '_, W>,
    op: &Operation,
) {
    if let Some(h) = p
        .ctx
        .attr(op, OpAttrKey::OptimizationHints)
        .and_then(|a| attrs::fmt_optimization_hints_angle(&p.ctx, a))
    {
        line.push(&format!(" optimization_hints = {}", h));
    }
}

pub(super) fn print_load_tko<W: MlirPrinter + ?Sized>(
    p: &mut Printer<'_, '_, W>,
    op: &Operation,
) -> fmt::Result {
    let mut line = Line::new();
    line.results(&p.ctx, &op.results);
    line.op_name(op.opcode.name());

    if let Some(Attr::MemoryOrdering(o)) = p.ctx.attr(op, OpAttrKey::MemoryOrderingSemantics) {
        line.push(&format!(" {}", o));
    }
    if let Some(Attr::MemoryScope(s)) = p.ctx.attr(op, OpAttrKey::MemoryScope) {
        line.push(&format!(" {}", s));
    }

    if op.opcode == Opcode::LoadViewTko {
        let segs = OperandSegments::for_load_view(&p.ctx, op);
        if let Some(base) = op.operands.first().copied() {
            let indices_end = (1 + segs.num_indices).min(op.operands.len());
            let indices: Vec<String> = op
                .operands
                .iter()
                .copied()
                .skip(1)
                .take(indices_end.saturating_sub(1))
                .map(|v| p.ctx.slot(v).to_string())
                .collect();
            line.push(&format!(" {}[{}]", p.ctx.slot(base), indices.join(", ")));

            if segs.has_token && indices_end < op.operands.len() {
                line.push(&format!(
                    " token = {}",
                    p.ctx.slot(op.operands[indices_end])
                ));
            }
        }

        emit_opt_hints(&mut line, p, op);

        if !op.operands.is_empty() && !op.results.is_empty() {
            let ptr_ty = types::fmt_type(&p.ctx, p.ctx.value_ty(op.operands[0]));
            let idx_ty = op
                .operands
                .get(1)
                .copied()
                .map(|v| types::fmt_type(&p.ctx, p.ctx.value_ty(v)))
                .unwrap_or_else(|| "tile<i32>".to_string());
            let result_types: Vec<String> = op
                .results
                .iter()
                .copied()
                .map(|v| types::fmt_type(&p.ctx, p.ctx.value_ty(v)))
                .collect();
            line.push(&format!(
                " : {}, {} -> {}",
                ptr_ty,
                idx_ty,
                result_types.join(", ")
            ));
        }
    } else {
        let segs = OperandSegments::for_load_ptr(&p.ctx, op);

        if let Some(ptr) = op.operands.first().copied() {
            line.push(&format!(" {}", p.ctx.slot(ptr)));
            let mut idx = 1usize;
            if segs.has_mask {
                if let Some(mask) = op.operands.get(idx).copied() {
                    line.push(&format!(", {}", p.ctx.slot(mask)));
                    idx += 1;
                }
            }
            if segs.has_other {
                if let Some(other) = op.operands.get(idx).copied() {
                    line.push(&format!(", {}", p.ctx.slot(other)));
                    idx += 1;
                }
            }
            if segs.has_token {
                if let Some(tok) = op.operands.get(idx).copied() {
                    line.push(&format!(" token={}", p.ctx.slot(tok)));
                }
            }
        }

        emit_opt_hints(&mut line, p, op);

        let num_non_token_operands = 1 + (segs.has_mask as usize) + (segs.has_other as usize);
        let types_list: Vec<String> = op
            .operands
            .iter()
            .copied()
            .take(num_non_token_operands)
            .map(|v| types::fmt_type(&p.ctx, p.ctx.value_ty(v)))
            .collect();
        let result_types: Vec<String> = op
            .results
            .iter()
            .copied()
            .map(|v| types::fmt_type(&p.ctx, p.ctx.value_ty(v)))
            .collect();
        if !types_list.is_empty() && !result_types.is_empty() {
            line.push(&format!(
                " : {} -> {}",
                types_list.join(", "),
                result_types.join(", ")
            ));
        }
    }

    p.write_indent()?;
    p.w.writeln(line.as_str())
}

pub(super) fn print_store_tko<W: MlirPrinter + ?Sized>(
    p: &mut Printer<'_, '_, W>,
    op: &Operation,
) -> fmt::Result {
    let mut line = Line::new();
    line.results(&p.ctx, &op.results);
    line.op_name(op.opcode.name());

    if let Some(Attr::MemoryOrdering(o)) = p.ctx.attr(op, OpAttrKey::MemoryOrderingSemantics) {
        line.push(&format!(" {}", o));
    }
    if let Some(Attr::MemoryScope(s)) = p.ctx.attr(op, OpAttrKey::MemoryScope) {
        line.push(&format!(" {}", s));
    }

    if op.opcode == Opcode::StorePtrTko {
        let segs = OperandSegments::for_store_ptr(&p.ctx, op);

        if op.operands.len() >= 2 {
            line.push(&format!(
                " {}, {}",
                p.ctx.slot(op.operands[0]),
                p.ctx.slot(op.operands[1])
            ));
        }
        if segs.has_mask {
            if let Some(mask) = op.operands.get(2).copied() {
                line.push(&format!(", {}", p.ctx.slot(mask)));
            }
        }
        if segs.has_token {
            let tok_idx = if segs.has_mask { 3 } else { 2 };
            if let Some(tok) = op.operands.get(tok_idx).copied() {
                line.push(&format!(" token={}", p.ctx.slot(tok)));
            }
        }

        emit_opt_hints(&mut line, p, op);

        if op.operands.len() >= 2 {
            let mut types_list = vec![
                types::fmt_type(&p.ctx, p.ctx.value_ty(op.operands[0])),
                types::fmt_type(&p.ctx, p.ctx.value_ty(op.operands[1])),
            ];
            if segs.has_mask {
                if let Some(mask) = op.operands.get(2).copied() {
                    types_list.push(types::fmt_type(&p.ctx, p.ctx.value_ty(mask)));
                }
            }
            let result_types: Vec<String> = op
                .results
                .iter()
                .copied()
                .map(|v| types::fmt_type(&p.ctx, p.ctx.value_ty(v)))
                .collect();
            if !result_types.is_empty() {
                line.push(&format!(
                    " : {} -> {}",
                    types_list.join(", "),
                    result_types.join(", ")
                ));
            }
        }
    } else {
        let segs = OperandSegments::for_store_view(&p.ctx, op);
        if op.operands.len() >= 2 {
            let indices_end = (2 + segs.num_indices).min(op.operands.len());
            let indices: Vec<String> = op
                .operands
                .iter()
                .copied()
                .skip(2)
                .take(indices_end.saturating_sub(2))
                .map(|v| p.ctx.slot(v).to_string())
                .collect();
            line.push(&format!(
                " {}, {}[{}]",
                p.ctx.slot(op.operands[0]),
                p.ctx.slot(op.operands[1]),
                indices.join(", ")
            ));
            if segs.has_token && indices_end < op.operands.len() {
                line.push(&format!(
                    " token = {}",
                    p.ctx.slot(op.operands[indices_end])
                ));
            }
        }

        emit_opt_hints(&mut line, p, op);

        if op.operands.len() >= 2 {
            let ptr_ty = types::fmt_type(&p.ctx, p.ctx.value_ty(op.operands[0]));
            let val_ty = types::fmt_type(&p.ctx, p.ctx.value_ty(op.operands[1]));
            let idx_ty = op
                .operands
                .get(2)
                .copied()
                .map(|v| types::fmt_type(&p.ctx, p.ctx.value_ty(v)))
                .unwrap_or_else(|| "tile<i32>".to_string());
            let result_types: Vec<String> = op
                .results
                .iter()
                .copied()
                .map(|v| types::fmt_type(&p.ctx, p.ctx.value_ty(v)))
                .collect();
            if !result_types.is_empty() {
                line.push(&format!(
                    " : {}, {}, {} -> {}",
                    ptr_ty,
                    val_ty,
                    idx_ty,
                    result_types.join(", ")
                ));
            }
        }
    }

    p.write_indent()?;
    p.w.writeln(line.as_str())
}

pub(super) fn print_atomic_rmw_tko<W: MlirPrinter + ?Sized>(
    p: &mut Printer<'_, '_, W>,
    op: &Operation,
) -> fmt::Result {
    let segs = OperandSegments::for_atomic_rmw(&p.ctx, op);

    let mut line = Line::new();
    line.results(&p.ctx, &op.results);
    line.op_name("atomic_rmw_tko");

    if let Some(Attr::MemoryOrdering(ord)) = p.ctx.attr(op, OpAttrKey::MemoryOrderingSemantics) {
        line.push(&format!(" {}", ord));
    }
    if let Some(Attr::MemoryScope(scope)) = p.ctx.attr(op, OpAttrKey::MemoryScope) {
        line.push(&format!(" {}", scope));
    }
    if let Some(ptr) = op.operands.first().copied() {
        line.push(&format!(" {}", p.ctx.slot(ptr)));
    }
    if let Some(Attr::AtomicRMWMode(mode)) = p.ctx.attr(op, OpAttrKey::Mode) {
        line.push(&format!(", {}", mode));
    }
    if let Some(arg) = op.operands.get(1).copied() {
        line.push(&format!(", {}", p.ctx.slot(arg)));
    }
    if segs.has_mask {
        if let Some(mask) = op.operands.get(2).copied() {
            line.push(&format!(", {}", p.ctx.slot(mask)));
        }
    }
    if segs.has_token {
        let tok_idx = if segs.has_mask { 3 } else { 2 };
        if let Some(tok) = op.operands.get(tok_idx).copied() {
            line.push(&format!(" token={}", p.ctx.slot(tok)));
        }
    }

    let mut types_list = Vec::new();
    if let Some(v) = op.operands.first().copied() {
        types_list.push(types::fmt_type(&p.ctx, p.ctx.value_ty(v)));
    }
    if let Some(v) = op.operands.get(1).copied() {
        types_list.push(types::fmt_type(&p.ctx, p.ctx.value_ty(v)));
    }
    if segs.has_mask {
        if let Some(v) = op.operands.get(2).copied() {
            types_list.push(types::fmt_type(&p.ctx, p.ctx.value_ty(v)));
        }
    }
    let result_types: Vec<String> = op
        .results
        .iter()
        .copied()
        .map(|v| types::fmt_type(&p.ctx, p.ctx.value_ty(v)))
        .collect();
    line.push(&format!(
        " : {} -> {}",
        types_list.join(", "),
        result_types.join(", ")
    ));

    p.write_indent()?;
    p.w.writeln(line.as_str())
}

pub(super) fn print_atomic_cas_tko<W: MlirPrinter + ?Sized>(
    p: &mut Printer<'_, '_, W>,
    op: &Operation,
) -> fmt::Result {
    let segs = OperandSegments::for_atomic_cas(&p.ctx, op);

    let mut line = Line::new();
    line.results(&p.ctx, &op.results);
    line.op_name("atomic_cas_tko");

    if let Some(Attr::MemoryOrdering(ord)) = p.ctx.attr(op, OpAttrKey::MemoryOrderingSemantics) {
        line.push(&format!(" {}", ord));
    }
    if let Some(Attr::MemoryScope(scope)) = p.ctx.attr(op, OpAttrKey::MemoryScope) {
        line.push(&format!(" {}", scope));
    }
    if let Some(ptr) = op.operands.first().copied() {
        line.push(&format!(" {}", p.ctx.slot(ptr)));
    }
    if let Some(cmp) = op.operands.get(1).copied() {
        line.push(&format!(", {}", p.ctx.slot(cmp)));
    }
    if let Some(val) = op.operands.get(2).copied() {
        line.push(&format!(", {}", p.ctx.slot(val)));
    }
    if segs.has_mask {
        if let Some(mask) = op.operands.get(3).copied() {
            line.push(&format!(", {}", p.ctx.slot(mask)));
        }
    }
    if segs.has_token {
        let tok_idx = if segs.has_mask { 4 } else { 3 };
        if let Some(tok) = op.operands.get(tok_idx).copied() {
            line.push(&format!(" token={}", p.ctx.slot(tok)));
        }
    }

    let mut types_list = Vec::new();
    if let Some(v) = op.operands.first().copied() {
        types_list.push(types::fmt_type(&p.ctx, p.ctx.value_ty(v)));
    }
    if let Some(v) = op.operands.get(1).copied() {
        types_list.push(types::fmt_type(&p.ctx, p.ctx.value_ty(v)));
    }
    if segs.has_mask {
        if let Some(v) = op.operands.get(3).copied() {
            types_list.push(types::fmt_type(&p.ctx, p.ctx.value_ty(v)));
        }
    }
    let result_types: Vec<String> = op
        .results
        .iter()
        .copied()
        .map(|v| types::fmt_type(&p.ctx, p.ctx.value_ty(v)))
        .collect();
    line.push(&format!(
        " : {} -> {}",
        types_list.join(", "),
        result_types.join(", ")
    ));

    p.write_indent()?;
    p.w.writeln(line.as_str())
}
