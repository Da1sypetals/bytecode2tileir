use std::fmt;

use crate::cuda_tile_ir::ids::ValueId;
use crate::cuda_tile_ir::ir::Operation;
use crate::cuda_tile_ir::OpAttrKey;

use super::super::fmt::{attrs, types};
use super::super::indent::MlirPrinter;
use super::super::Line;
use super::super::Printer;
use super::print_op;

pub(super) fn print_region_ops<W: MlirPrinter + ?Sized>(
    p: &mut Printer<'_, '_, W>,
    region: &crate::cuda_tile_ir::cfg::Region,
) -> fmt::Result {
    for &bid in &region.blocks {
        let block = p.ctx.module.arena.block_(bid);
        for &op_id in &block.ops {
            print_op(p, op_id)?;
        }
    }
    Ok(())
}

pub(super) fn print_region_with_block_args<W: MlirPrinter + ?Sized>(
    p: &mut Printer<'_, '_, W>,
    region: &crate::cuda_tile_ir::cfg::Region,
) -> fmt::Result {
    let Some(&bid) = region.blocks.first() else {
        return Ok(());
    };
    let block = p.ctx.module.arena.block_(bid);

    let args: Vec<String> = block
        .args
        .iter()
        .copied()
        .map(|arg| {
            format!(
                "{}: {}",
                p.ctx.slot(arg),
                types::fmt_type(&p.ctx, p.ctx.value_ty(arg))
            )
        })
        .collect();

    p.write_indent()?;
    p.w.writeln(&format!("({}) {{", args.join(", ")))?;

    p.indented(2, |p| print_region_ops(p, region))?;

    p.write_indent()?;
    p.w.writeln("}")
}

pub(super) fn print_if<W: MlirPrinter + ?Sized>(
    p: &mut Printer<'_, '_, W>,
    op: &Operation,
) -> fmt::Result {
    let cond = op.operands.first().copied().unwrap_or(ValueId(0));

    let mut line = Line::new();
    line.results(&p.ctx, &op.results);
    line.op_name("if");
    line.push(&format!(" {}", p.ctx.slot(cond)));
    if !op.results.is_empty() {
        let types: Vec<String> = op
            .results
            .iter()
            .copied()
            .map(|v| types::fmt_type(&p.ctx, p.ctx.value_ty(v)))
            .collect();
        line.push(&format!(" -> ({})", types.join(", ")));
    }
    line.push(" {");

    p.write_indent()?;
    p.w.writeln(line.as_str())?;

    if let Some(rid) = op.regions.first().copied() {
        let region = p.ctx.module.arena.region_(rid);
        p.indented(2, |p| print_region_ops(p, region))?;
    }

    if op.regions.len() > 1 {
        p.write_indent()?;
        p.w.writeln("} else {")?;
        let region = p.ctx.module.arena.region_(op.regions[1]);
        p.indented(2, |p| print_region_ops(p, region))?;
    }

    p.write_indent()?;
    p.w.writeln("}")
}

pub(super) fn print_for<W: MlirPrinter + ?Sized>(
    p: &mut Printer<'_, '_, W>,
    op: &Operation,
) -> fmt::Result {
    let operands: Vec<String> = op
        .operands
        .iter()
        .copied()
        .map(|v| p.ctx.slot(v).to_string())
        .collect();
    let segs = attrs::dense_i32_attr(&p.ctx, op, OpAttrKey::OperandSegmentSizes);

    // Get induction variable from region block args.
    let iv = op
        .regions
        .first()
        .copied()
        .and_then(|rid| p.ctx.module.arena.regions.get(rid.0 as usize))
        .and_then(|r| r.blocks.first().copied())
        .and_then(|bid| p.ctx.module.arena.blocks.get(bid.0 as usize))
        .and_then(|b| b.args.first().copied())
        .unwrap_or(ValueId(0));

    let iv_ty = types::fmt_type(&p.ctx, p.ctx.value_ty(iv));

    let mut line = Line::new();
    line.results(&p.ctx, &op.results);
    line.op_name("for");
    line.push(&format!(
        " {} in ({} to {}, step {}) : {}",
        p.ctx.slot(iv),
        operands.get(0).map(|s| s.as_str()).unwrap_or("%?"),
        operands.get(1).map(|s| s.as_str()).unwrap_or("%?"),
        operands.get(2).map(|s| s.as_str()).unwrap_or("%?"),
        iv_ty
    ));

    let num_init = operands.len().saturating_sub(3);
    let iter_arg_ids: Vec<ValueId> = op
        .regions
        .first()
        .copied()
        .and_then(|rid| p.ctx.module.arena.regions.get(rid.0 as usize))
        .and_then(|r| r.blocks.first().copied())
        .and_then(|bid| p.ctx.module.arena.blocks.get(bid.0 as usize))
        .map(|b| b.args.iter().copied().skip(1).take(num_init).collect())
        .unwrap_or_default();

    if num_init > 0 {
        let pairs: Vec<String> = (0..num_init)
            .map(|i| {
                let name = iter_arg_ids.get(i).copied().unwrap_or(ValueId(0));
                format!("{} = {}", p.ctx.slot(name), operands[3 + i])
            })
            .collect();
        line.push(&format!(" iter_values({})", pairs.join(", ")));
    }

    if !op.results.is_empty() {
        let types: Vec<String> = op
            .results
            .iter()
            .copied()
            .map(|v| types::fmt_type(&p.ctx, p.ctx.value_ty(v)))
            .collect();
        line.push(&format!(" -> ({})", types.join(", ")));
    }

    line.push(" {");
    p.write_indent()?;
    p.w.writeln(line.as_str())?;

    if let Some(rid) = op.regions.first().copied() {
        let region = p.ctx.module.arena.region_(rid);
        p.indented(2, |p| print_region_ops(p, region))?;
    }

    p.write_indent()?;
    p.w.writeln("}")?;

    // Note: the operand segment sizes are also used by lowering; keep printing consistent.
    let _ = segs;
    Ok(())
}

pub(super) fn print_loop<W: MlirPrinter + ?Sized>(
    p: &mut Printer<'_, '_, W>,
    op: &Operation,
) -> fmt::Result {
    let operands: Vec<String> = op
        .operands
        .iter()
        .copied()
        .map(|v| p.ctx.slot(v).to_string())
        .collect();

    let mut line = Line::new();
    line.results(&p.ctx, &op.results);
    line.op_name("loop");

    let num_init = operands.len();
    let iter_arg_ids: Vec<ValueId> = op
        .regions
        .first()
        .copied()
        .and_then(|rid| p.ctx.module.arena.regions.get(rid.0 as usize))
        .and_then(|r| r.blocks.first().copied())
        .and_then(|bid| p.ctx.module.arena.blocks.get(bid.0 as usize))
        .map(|b| b.args.iter().copied().take(num_init).collect())
        .unwrap_or_default();

    if !operands.is_empty() {
        let pairs: Vec<String> = operands
            .iter()
            .enumerate()
            .map(|(i, opnd)| {
                let name = iter_arg_ids.get(i).copied().unwrap_or(ValueId(0));
                format!("{} = {}", p.ctx.slot(name), opnd)
            })
            .collect();
        line.push(&format!(" iter_values({})", pairs.join(", ")));

        let iter_types: Vec<String> = op
            .operands
            .iter()
            .copied()
            .map(|v| types::fmt_type(&p.ctx, p.ctx.value_ty(v)))
            .collect();
        line.push(&format!(" : {}", iter_types.join(", ")));
    }

    if !op.results.is_empty() {
        let types: Vec<String> = op
            .results
            .iter()
            .copied()
            .map(|v| types::fmt_type(&p.ctx, p.ctx.value_ty(v)))
            .collect();
        if operands.is_empty() {
            line.push(&format!(" : {}", types.join(", ")));
        } else {
            line.push(&format!(" -> {}", types.join(", ")));
        }
    }

    line.push(" {");
    p.write_indent()?;
    p.w.writeln(line.as_str())?;

    if let Some(rid) = op.regions.first().copied() {
        let region = p.ctx.module.arena.region_(rid);
        p.indented(2, |p| print_region_ops(p, region))?;
    }

    p.write_indent()?;
    p.w.writeln("}")
}
