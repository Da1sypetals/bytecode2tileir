//! Streaming MLIR printer for semantic IR (`crate::ir`).
//!
//! This printer targets the cuda-tile textual form used by tileiras/tilelang.

mod ctx;
mod escape;
mod fmt;
mod indent;
mod ops;

pub use ctx::{PrinterConfig, PrinterCtx};
pub use indent::MlirPrinter;

use std::fmt as std_fmt;
use std::fmt::Write as _;

use crate::bytecode::funcs::FunctionFlags;
use crate::cuda_tile_ir::cfg::{Function, Global, Module};
use crate::cuda_tile_ir::ids::{BlockId, TypeId, ValueId};
use crate::cuda_tile_ir::types::{Dim, Type};

#[derive(Default)]
struct Line(String);

impl Line {
    fn new() -> Self {
        Self::default()
    }

    fn push_str(&mut self, s: &str) {
        self.0.push_str(s);
    }

    fn push_char(&mut self, c: char) {
        self.0.push(c);
    }

    fn results(&mut self, ctx: &PrinterCtx<'_>, results: &[ValueId]) {
        if results.is_empty() {
            return;
        }
        for (idx, v) in results.iter().copied().enumerate() {
            if idx > 0 {
                self.0.push_str(", ");
            }
            let _ = write!(&mut self.0, "{}", ctx.slot(v));
        }
        self.0.push_str(" = ");
    }

    fn op_name(&mut self, name: &str) {
        self.0.push_str(name);
    }

    fn operands(&mut self, ctx: &PrinterCtx<'_>, operands: &[ValueId]) {
        if operands.is_empty() {
            return;
        }
        self.0.push(' ');
        for (idx, v) in operands.iter().copied().enumerate() {
            if idx > 0 {
                self.0.push_str(", ");
            }
            let _ = write!(&mut self.0, "{}", ctx.slot(v));
        }
    }

    fn push(&mut self, s: impl AsRef<str>) {
        self.0.push_str(s.as_ref());
    }

    fn as_str(&self) -> &str {
        self.0.as_str()
    }
}

pub fn module_to_mlir_text(module: &Module) -> String {
    let mut out = String::new();
    let _ = print_module(&mut out, module);
    out
}

pub fn print_module<W: MlirPrinter + ?Sized>(w: &mut W, module: &Module) -> std_fmt::Result {
    let mut p = Printer::new(w, module);
    p.print_module()
}

pub fn fmt_scalar(module: &Module, ty: TypeId) -> String {
    fmt::types::fmt_scalar(&PrinterCtx::new(module), ty)
}

pub fn fmt_type(module: &Module, ty: TypeId) -> String {
    fmt::types::fmt_type(&PrinterCtx::new(module), ty)
}

pub fn fmt_optimization_hints_angle(
    module: &Module,
    attr: &crate::cuda_tile_ir::attrs::Attr,
) -> Option<String> {
    fmt::attrs::fmt_optimization_hints_angle(&PrinterCtx::new(module), attr)
}

struct Printer<'m, 'w, W: MlirPrinter + ?Sized> {
    w: &'w mut W,
    ctx: PrinterCtx<'m>,
    indent: usize,
}

impl<'m, 'w, W: MlirPrinter + ?Sized> Printer<'m, 'w, W> {
    fn new(w: &'w mut W, module: &'m Module) -> Self {
        Self {
            w,
            ctx: PrinterCtx::new(module),
            indent: 0,
        }
    }

    fn write_indent(&mut self) -> std_fmt::Result {
        self.w.write_indent(self.indent)
    }

    fn indented<F>(&mut self, delta: usize, f: F) -> std_fmt::Result
    where
        F: FnOnce(&mut Self) -> std_fmt::Result,
    {
        self.indent += delta;
        let r = f(self);
        self.indent -= delta;
        r
    }

    fn print_module(&mut self) -> std_fmt::Result {
        writeln!(self.w, "cuda_tile.module @{} {{", self.ctx.module.name)?;
        self.indent = 2;

        for &gid in &self.ctx.module.globals {
            let g = self.ctx.module.arena.global_(gid);
            self.print_global(g)?;
        }

        if !self.ctx.module.globals.is_empty() && !self.ctx.module.functions.is_empty() {
            self.w.write_char('\n')?;
        }

        for (idx, &fid) in self.ctx.module.functions.iter().enumerate() {
            let func = self.ctx.module.arena.function_(fid);
            self.print_function(func)?;
            if idx + 1 < self.ctx.module.functions.len() {
                self.w.write_char('\n')?;
            }
        }

        self.indent = 0;
        self.w.writeln("}")
    }

    fn print_global(&mut self, global: &Global) -> std_fmt::Result {
        self.write_indent()?;

        let ty_str = fmt::types::fmt_type(&self.ctx, global.ty);
        let (elem_ty, elem_ty_str, shape) = match self.ctx.ty(global.ty) {
            Type::Tile { element, shape } => (
                *element,
                fmt::types::fmt_scalar(&self.ctx, *element),
                shape.0.clone(),
            ),
            _ => (global.ty, ty_str.clone(), Vec::new()),
        };
        let shape_i64: Vec<i64> = shape
            .iter()
            .map(|d| match d {
                Dim::Static(v) => *v,
                Dim::Dynamic => -1,
            })
            .collect();

        let bytes = self.ctx.module.consts.get(global.init).unwrap_or(&[]);
        let bytes: &[u8] = if bytes.is_empty() { &[0u8] } else { bytes };
        let value_str = fmt::dense::fmt_dense_value(bytes, &self.ctx, elem_ty, &shape_i64);

        let mut line = format!("global @{}", global.name);
        if global.alignment > 0 && global.alignment != 1 {
            line.push_str(&format!(" alignment = {}", global.alignment));
        }
        line.push_str(&format!(" <{}: {}> : {}", elem_ty_str, value_str, ty_str));
        writeln!(self.w, "{}", line)
    }

    fn print_function(&mut self, func: &Function) -> std_fmt::Result {
        let region = self.ctx.module.arena.region_(func.body);
        let entry_block = region.blocks.first().copied().unwrap_or(BlockId(0));
        let block = self.ctx.module.arena.block_(entry_block);

        let args: Vec<String> = block
            .args
            .iter()
            .copied()
            .map(|arg| {
                format!(
                    "{}: {}",
                    self.ctx.slot(arg),
                    fmt::types::fmt_type(&self.ctx, self.ctx.value_ty(arg))
                )
            })
            .collect();

        let keyword = if func.flags.contains(FunctionFlags::KERNEL_ENTRY) {
            "entry"
        } else {
            "func"
        };

        self.write_indent()?;
        let mut line = format!("{} @{}({})", keyword, func.name, args.join(", "));
        if func.flags.contains(FunctionFlags::KERNEL_ENTRY) {
            if let Some(hid) = func.opt_hints {
                if let Some(attr) = self.ctx.module.arena.attrs.get(hid.0 as usize) {
                    if let Some(h) = fmt::attrs::fmt_optimization_hints_angle(&self.ctx, attr) {
                        line.push_str(&format!(" optimization_hints={}", h));
                    }
                }
            }
        }

        writeln!(self.w, "{} {{", line)?;

        self.indented(2, |p| {
            for &op_id in &block.ops {
                ops::print_op(p, op_id)?;
            }
            Ok(())
        })?;

        self.write_indent()?;
        self.w.writeln("}")
    }
}
