use std::fmt::{self, Display};

use crate::cuda_tile_ir::enums::FloatKind;
use crate::cuda_tile_ir::ids::TypeId;
use crate::cuda_tile_ir::types::{Dim, Type};

use super::super::ctx::PrinterCtx;

pub struct ScalarDisplay<'a, 'm> {
    ctx: &'a PrinterCtx<'m>,
    ty: TypeId,
}

pub fn scalar<'a, 'm>(ctx: &'a PrinterCtx<'m>, ty: TypeId) -> ScalarDisplay<'a, 'm> {
    ScalarDisplay { ctx, ty }
}

impl Display for ScalarDisplay<'_, '_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.ctx.ty(self.ty) {
            Type::Int { width } => write!(f, "i{}", width),
            Type::Float(k) => f.write_str(match k {
                FloatKind::F16 => "f16",
                FloatKind::BF16 => "bf16",
                FloatKind::F32 => "f32",
                FloatKind::TF32 => "tf32",
                FloatKind::F64 => "f64",
                FloatKind::F8E4M3FN => "f8E4M3FN",
                FloatKind::F8E5M2 => "f8E5M2",
            }),
            _ => f.write_str("?"),
        }
    }
}

struct ElementDisplay<'a, 'm> {
    ctx: &'a PrinterCtx<'m>,
    ty: TypeId,
}

fn element<'a, 'm>(ctx: &'a PrinterCtx<'m>, ty: TypeId) -> ElementDisplay<'a, 'm> {
    ElementDisplay { ctx, ty }
}

impl Display for ElementDisplay<'_, '_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.ctx.ty(self.ty) {
            Type::Ptr { pointee } => write!(f, "ptr<{}>", scalar(self.ctx, *pointee)),
            _ => write!(f, "{}", scalar(self.ctx, self.ty)),
        }
    }
}

pub struct TypeDisplay<'a, 'm> {
    ctx: &'a PrinterCtx<'m>,
    ty: TypeId,
}

pub fn ty<'a, 'm>(ctx: &'a PrinterCtx<'m>, ty: TypeId) -> TypeDisplay<'a, 'm> {
    TypeDisplay { ctx, ty }
}

fn write_i64_dim(f: &mut fmt::Formatter<'_>, v: i64) -> fmt::Result {
    if v < 0 {
        f.write_str("?")
    } else {
        write!(f, "{}", v)
    }
}

impl Display for TypeDisplay<'_, '_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.ctx.ty(self.ty) {
            Type::Int { .. } | Type::Float(_) => write!(f, "{}", scalar(self.ctx, self.ty)),
            Type::Tile { element: el, shape } => {
                f.write_str("tile<")?;
                if shape.0.is_empty() {
                    write!(f, "{}>", element(self.ctx, *el))?;
                    return Ok(());
                }

                for (idx, d) in shape.0.iter().enumerate() {
                    if idx > 0 {
                        f.write_str("x")?;
                    }
                    match d {
                        Dim::Static(v) => write_i64_dim(f, *v)?,
                        Dim::Dynamic => f.write_str("?")?,
                    }
                }
                f.write_str("x")?;
                write!(f, "{}", element(self.ctx, *el))?;
                f.write_str(">")?;
                Ok(())
            }
            Type::Ptr { pointee } => write!(f, "ptr<{}>", scalar(self.ctx, *pointee)),
            Type::TensorView {
                element: el,
                shape,
                strides,
                ..
            } => {
                if shape.0.is_empty() && strides.is_empty() {
                    return write!(f, "tensor_view<{}>", scalar(self.ctx, *el));
                }

                f.write_str("tensor_view<")?;
                for (idx, d) in shape.0.iter().enumerate() {
                    if idx > 0 {
                        f.write_str("x")?;
                    }
                    match d {
                        Dim::Static(v) => write_i64_dim(f, *v)?,
                        Dim::Dynamic => f.write_str("?")?,
                    }
                }
                f.write_str("x")?;
                write!(f, "{}", scalar(self.ctx, *el))?;

                f.write_str(", strides=[")?;
                for (idx, st) in strides.iter().copied().enumerate() {
                    if idx > 0 {
                        f.write_str(",")?;
                    }
                    write_i64_dim(f, st)?;
                }
                f.write_str("]>")?;
                Ok(())
            }
            Type::PartitionView {
                tile_shape,
                view,
                dim_map,
                masked,
                padding_value: _,
            } => {
                f.write_str("partition_view<")?;
                if *masked {
                    f.write_str("masked ")?;
                }
                f.write_str("tile=(")?;
                for (idx, x) in tile_shape.iter().enumerate() {
                    if idx > 0 {
                        f.write_str("x")?;
                    }
                    write!(f, "{}", x)?;
                }
                f.write_str("), ")?;
                write!(f, "{}", ty(self.ctx, *view))?;
                if !dim_map.is_empty() {
                    f.write_str(", dim_map=[")?;
                    for (idx, x) in dim_map.iter().enumerate() {
                        if idx > 0 {
                            f.write_str(", ")?;
                        }
                        write!(f, "{}", x)?;
                    }
                    f.write_str("]")?;
                }
                f.write_str(">")?;
                Ok(())
            }
            Type::Token => f.write_str("token"),
            Type::Func { .. } => f.write_str("func"),
        }
    }
}

pub fn fmt_scalar(ctx: &PrinterCtx<'_>, ty: TypeId) -> String {
    scalar(ctx, ty).to_string()
}

pub fn fmt_type(ctx: &PrinterCtx<'_>, type_id: TypeId) -> String {
    ty(ctx, type_id).to_string()
}
