use std::fmt;

use crate::cuda_tile_ir::OpAttrKey;
use crate::cuda_tile_ir::attrs::Attr;
use crate::cuda_tile_ir::ids::ValueId;
use crate::cuda_tile_ir::ir::Operation;
use crate::cuda_tile_ir::types::{Dim, Type};

use super::super::Line;
use super::super::Printer;
use super::super::fmt::{attrs, types};
use super::super::indent::MlirPrinter;

fn print_unary_arrow<W: MlirPrinter + ?Sized>(
    p: &mut Printer<'_, '_, W>,
    op: &Operation,
    name: &str,
    extra: impl FnOnce(&mut Line),
) -> fmt::Result {
    let out = op.results.first().copied().unwrap_or(ValueId(0));
    let in_ref = op.operands.first().copied().unwrap_or(ValueId(0));
    let input_ty = types::fmt_type(&p.ctx, p.ctx.value_ty(in_ref));
    let out_ty = types::fmt_type(&p.ctx, p.ctx.value_ty(out));

    let mut line = Line::new();
    let results = [out];
    let operands = [in_ref];
    line.results(&p.ctx, &results);
    line.op_name(name);
    line.operands(&p.ctx, &operands);
    extra(&mut line);
    line.push(&format!(" : {} -> {}", input_ty, out_ty));

    p.write_indent()?;
    p.w.writeln(line.as_str())
}

pub(super) fn print_make_tensor_view<W: MlirPrinter + ?Sized>(
    p: &mut Printer<'_, '_, W>,
    op: &Operation,
) -> fmt::Result {
    let out = op.results.first().copied().unwrap_or(ValueId(0));
    let ty = p.ctx.value_ty(out);

    let mut seg_sizes = attrs::dense_i32_attr(&p.ctx, op, OpAttrKey::OperandSegmentSizes);
    if seg_sizes.len() < 3 {
        seg_sizes.resize(3, 0);
    }
    let num_shape = seg_sizes[1].max(0) as usize;
    let num_strides = seg_sizes[2].max(0) as usize;

    let base = p
        .ctx
        .slot(op.operands.first().copied().unwrap_or(ValueId(0)))
        .to_string();
    let shape_ops = &op.operands[1..(1 + num_shape).min(op.operands.len())];
    let stride_ops =
        &op.operands[(1 + num_shape)..(1 + num_shape + num_strides).min(op.operands.len())];

    let (type_shape, type_strides) = match p.ctx.ty(ty) {
        Type::TensorView { shape, strides, .. } => (shape.0.as_slice(), strides.as_slice()),
        _ => (&[][..], &[][..]),
    };

    let mut shape_it = shape_ops.iter();
    let shape_parts: Vec<String> = type_shape
        .iter()
        .map(|dim| match dim {
            Dim::Dynamic => shape_it
                .next()
                .copied()
                .map(|v| p.ctx.slot(v).to_string())
                .unwrap_or_else(|| "?".into()),
            Dim::Static(v) => v.to_string(),
        })
        .collect();

    let mut stride_it = stride_ops.iter();
    let stride_parts: Vec<String> = type_strides
        .iter()
        .copied()
        .map(|st| {
            if st < 0 {
                stride_it
                    .next()
                    .copied()
                    .map(|v| p.ctx.slot(v).to_string())
                    .unwrap_or_else(|| "?".into())
            } else {
                st.to_string()
            }
        })
        .collect();

    let dyn_present = !shape_ops.is_empty() || !stride_ops.is_empty();
    let dyn_ty = if dyn_present {
        shape_ops
            .first()
            .copied()
            .or_else(|| stride_ops.first().copied())
            .map(|v| types::fmt_type(&p.ctx, p.ctx.value_ty(v)))
            .unwrap_or_default()
    } else {
        String::new()
    };

    p.write_indent()?;
    if dyn_present {
        writeln!(
            p.w,
            "{} = make_tensor_view {}, shape = [{}], strides = [{}] : {} -> {}",
            p.ctx.slot(out),
            base,
            shape_parts.join(", "),
            stride_parts.join(", "),
            dyn_ty,
            types::fmt_type(&p.ctx, ty)
        )
    } else {
        writeln!(
            p.w,
            "{} = make_tensor_view {}, shape = [{}], strides = [{}] : {}",
            p.ctx.slot(out),
            base,
            shape_parts.join(", "),
            stride_parts.join(", "),
            types::fmt_type(&p.ctx, ty)
        )
    }
}

pub(super) fn print_make_partition_view<W: MlirPrinter + ?Sized>(
    p: &mut Printer<'_, '_, W>,
    op: &Operation,
) -> fmt::Result {
    let out = op.results.first().copied().unwrap_or(ValueId(0));
    let ty = types::fmt_type(&p.ctx, p.ctx.value_ty(out));
    let operand = op
        .operands
        .first()
        .copied()
        .map(|v| p.ctx.slot(v).to_string())
        .unwrap_or_default();
    p.write_indent()?;
    writeln!(
        p.w,
        "{} = make_partition_view {} : {}",
        p.ctx.slot(out),
        operand,
        ty
    )
}

pub(super) fn print_reshape<W: MlirPrinter + ?Sized>(
    p: &mut Printer<'_, '_, W>,
    op: &Operation,
) -> fmt::Result {
    print_unary_arrow(p, op, "reshape", |_| {})
}

pub(super) fn print_broadcast<W: MlirPrinter + ?Sized>(
    p: &mut Printer<'_, '_, W>,
    op: &Operation,
) -> fmt::Result {
    print_unary_arrow(p, op, "broadcast", |_| {})
}

pub(super) fn print_permute<W: MlirPrinter + ?Sized>(
    p: &mut Printer<'_, '_, W>,
    op: &Operation,
) -> fmt::Result {
    let perm = p
        .ctx
        .attr(op, OpAttrKey::Permutation)
        .and_then(|a| match a {
            Attr::DenseI32Array(v) => Some(v.clone()),
            _ => None,
        })
        .unwrap_or_default();
    let perm_str = perm
        .iter()
        .map(|x| x.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    print_unary_arrow(p, op, "permute", |line| {
        line.push(&format!(" [{}]", perm_str))
    })
}

pub(super) fn print_extract<W: MlirPrinter + ?Sized>(
    p: &mut Printer<'_, '_, W>,
    op: &Operation,
) -> fmt::Result {
    let mut line = Line::new();
    line.results(&p.ctx, &op.results);
    line.op_name("extract");
    if let Some(src) = op.operands.first().copied() {
        line.push(&format!(" {}", p.ctx.slot(src)));
        if op.operands.len() > 1 {
            let idx: Vec<String> = op
                .operands
                .iter()
                .copied()
                .skip(1)
                .map(|v| p.ctx.slot(v).to_string())
                .collect();
            line.push(&format!("[{}]", idx.join(", ")));
        }
    }
    let input_type = op
        .operands
        .first()
        .copied()
        .map(|v| types::fmt_type(&p.ctx, p.ctx.value_ty(v)))
        .unwrap_or_default();
    if let Some(out) = op.results.first().copied() {
        line.push(&format!(
            " : {} -> {}",
            input_type,
            types::fmt_type(&p.ctx, p.ctx.value_ty(out))
        ));
    }
    p.write_indent()?;
    p.w.writeln(line.as_str())
}

pub(super) fn print_cat<W: MlirPrinter + ?Sized>(
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
    line.op_name("cat");
    line.operands(&p.ctx, &op.operands);
    if let Some(Attr::Int { value, .. }) = p.ctx.attr(op, OpAttrKey::Dim) {
        line.push(&format!(" dim = {}", value));
    }
    if !operand_types.is_empty() {
        line.push(&format!(
            " : {} -> {}",
            operand_types.join(", "),
            types::fmt_type(&p.ctx, p.ctx.value_ty(out))
        ));
    }
    p.write_indent()?;
    p.w.writeln(line.as_str())
}
