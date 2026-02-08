use std::fmt;

use crate::cuda_tile_ir::ir::Operation;
use crate::cuda_tile_ir::{OpAttrKey, Opcode};

use super::super::ctx::PrinterCtx;
use super::super::fmt::{attrs, types};
use super::super::indent::MlirPrinter;
use super::super::Line;
use super::super::Printer;

#[derive(Debug, Clone, Copy)]
enum TypeSigPolicy {
    None,
    ResultOnly,
    CastLikeFirstOperand,
    AllOperandsArrow,
    OperandsOnly,
}

fn type_sig_policy(op: &Operation) -> TypeSigPolicy {
    match op.opcode {
        Opcode::Offset => TypeSigPolicy::AllOperandsArrow,
        Opcode::IToF
        | Opcode::FToI
        | Opcode::FToF
        | Opcode::Bitcast
        | Opcode::IntToPtr
        | Opcode::PtrToInt
        | Opcode::Extract
        | Opcode::Scan
        | Opcode::Cat
        | Opcode::ExtI
        | Opcode::TruncI
        | Opcode::GetIndexSpaceShape
        | Opcode::GetTensorShape
        | Opcode::Reshape
        | Opcode::Broadcast
        | Opcode::Permute => TypeSigPolicy::CastLikeFirstOperand,
        Opcode::Break | Opcode::Continue | Opcode::Yield => TypeSigPolicy::OperandsOnly,
        _ => {
            if op.results.is_empty() {
                TypeSigPolicy::None
            } else {
                TypeSigPolicy::ResultOnly
            }
        }
    }
}

fn uses_ieee_rounding(opcode: Opcode) -> bool {
    matches!(
        opcode,
        Opcode::AddF
            | Opcode::SubF
            | Opcode::MulF
            | Opcode::DivF
            | Opcode::Fma
            | Opcode::FToF
            | Opcode::IToF
            | Opcode::MmaF
    )
}

fn is_integer_div(opcode: Opcode) -> bool {
    matches!(opcode, Opcode::DivI | Opcode::RemI)
}

fn needs_signedness_before_rounding(opcode: Opcode) -> bool {
    matches!(opcode, Opcode::FToI | Opcode::IToF)
}

fn append_attrs(line: &mut Line, ctx: &PrinterCtx<'_>, op: &Operation) {
    attrs::emit_mem_semantics(line, ctx, op);

    let opcode = op.opcode;
    if is_integer_div(opcode) || needs_signedness_before_rounding(opcode) {
        attrs::emit_signedness(line, ctx, op, OpAttrKey::Signedness);
        attrs::emit_rounding_nonzero(line, ctx, op);
    } else {
        if uses_ieee_rounding(opcode) {
            attrs::emit_rounding_ieee(line, ctx, op);
        } else {
            attrs::emit_rounding_non_ieee(line, ctx, op);
        }
        attrs::emit_flush_to_zero(line, ctx, op);
        attrs::emit_signedness(line, ctx, op, OpAttrKey::Signedness);
    }

    attrs::emit_signedness(line, ctx, op, OpAttrKey::LhsSignedness);
    attrs::emit_signedness(line, ctx, op, OpAttrKey::RhsSignedness);
    attrs::emit_comparison_predicate(line, ctx, op, OpAttrKey::Predicate);
    attrs::emit_comparison_ordering(line, ctx, op);
    attrs::emit_atomic_rmw_mode(line, ctx, op);
    attrs::emit_permutation(line, ctx, op);
}

pub(super) fn print_simple<W: MlirPrinter + ?Sized>(
    p: &mut Printer<'_, '_, W>,
    op: &Operation,
) -> fmt::Result {
    let mut line = Line::new();
    line.results(&p.ctx, &op.results);
    line.op_name(op.opcode.name());
    line.operands(&p.ctx, &op.operands);

    append_attrs(&mut line, &p.ctx, op);

    match type_sig_policy(op) {
        TypeSigPolicy::AllOperandsArrow => {
            let operand_types: Vec<String> = op
                .operands
                .iter()
                .copied()
                .map(|v| types::fmt_type(&p.ctx, p.ctx.value_ty(v)))
                .collect();
            let result_ty = op
                .results
                .first()
                .copied()
                .map(|v| types::fmt_type(&p.ctx, p.ctx.value_ty(v)))
                .unwrap_or_default();
            line.push(&format!(" : {} -> {}", operand_types.join(", "), result_ty));
        }
        TypeSigPolicy::CastLikeFirstOperand => {
            let input_type = op
                .operands
                .first()
                .copied()
                .map(|v| types::fmt_type(&p.ctx, p.ctx.value_ty(v)))
                .unwrap_or_default();
            let result_ty = op
                .results
                .first()
                .copied()
                .map(|v| types::fmt_type(&p.ctx, p.ctx.value_ty(v)))
                .unwrap_or_default();
            line.push(&format!(" : {} -> {}", input_type, result_ty));
        }
        TypeSigPolicy::ResultOnly => {
            if let Some(v) = op.results.first().copied() {
                line.push(&format!(
                    " : {}",
                    types::fmt_type(&p.ctx, p.ctx.value_ty(v))
                ));
            }
        }
        TypeSigPolicy::OperandsOnly => {
            if !op.operands.is_empty() {
                let operand_types: Vec<String> = op
                    .operands
                    .iter()
                    .copied()
                    .map(|v| types::fmt_type(&p.ctx, p.ctx.value_ty(v)))
                    .collect();
                line.push(&format!(" : {}", operand_types.join(", ")));
            }
        }
        TypeSigPolicy::None => {}
    }

    p.write_indent()?;
    p.w.writeln(line.as_str())
}
