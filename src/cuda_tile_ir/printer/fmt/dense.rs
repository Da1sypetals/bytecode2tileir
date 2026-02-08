use half::{bf16, f16};

use crate::cuda_tile_ir::enums::FloatKind;
use crate::cuda_tile_ir::ids::TypeId;
use crate::cuda_tile_ir::types::Type;

use super::super::ctx::PrinterCtx;

pub fn fmt_dense_value(
    data: &[u8],
    ctx: &PrinterCtx<'_>,
    elem_ty: TypeId,
    shape: &[i64],
) -> String {
    let actual_data = if !data.is_empty() && data[0] as usize == data.len().saturating_sub(1) {
        &data[1..]
    } else {
        data
    };

    let num_elems: usize = if shape.is_empty() {
        1
    } else {
        shape
            .iter()
            .copied()
            .map(|x| if x < 0 { 1usize } else { x as usize })
            .product()
    };

    if matches!(ctx.ty(elem_ty), Type::Int { width: 1 }) {
        if num_elems == 0 {
            return "[]".into();
        }
        let required_bytes = (num_elems + 7) / 8;
        if required_bytes > 1 && actual_data.len() == 1 {
            return if (actual_data[0] & 1) != 0 {
                "true"
            } else {
                "false"
            }
            .into();
        }
        if actual_data.len() < required_bytes {
            panic!("DenseElements i1 payload too small");
        }
        if num_elems == 1 {
            return if (actual_data[0] & 1) != 0 {
                "true"
            } else {
                "false"
            }
            .into();
        }
        let values: Vec<&str> = (0..num_elems)
            .map(|i| {
                if (actual_data[i / 8] >> (i % 8)) & 1 != 0 {
                    "true"
                } else {
                    "false"
                }
            })
            .collect();
        return fmt_nested(&values, shape);
    }

    let elem_size = match ctx.ty(elem_ty) {
        Type::Int { width } => (*width as usize + 7) / 8,
        Type::Float(FloatKind::F8E4M3FN) | Type::Float(FloatKind::F8E5M2) => 1,
        Type::Float(FloatKind::F16) | Type::Float(FloatKind::BF16) => 2,
        Type::Float(FloatKind::TF32) => 3,
        Type::Float(FloatKind::F32) => 4,
        Type::Float(FloatKind::F64) => 8,
        other => panic!("unsupported DenseElements element type: {:?}", other),
    };

    if num_elems > 1 && actual_data.len() == elem_size {
        return fmt_single_value(actual_data, ctx, elem_ty);
    }
    if actual_data.len() < num_elems * elem_size {
        panic!("DenseElements payload too small");
    }
    if num_elems == 1 {
        return fmt_single_value(actual_data, ctx, elem_ty);
    }

    let values: Vec<String> = (0..num_elems)
        .map(|i| {
            fmt_single_value(
                &actual_data[i * elem_size..(i + 1) * elem_size],
                ctx,
                elem_ty,
            )
        })
        .collect();
    fmt_nested(&values, shape)
}

fn fmt_nested<S: AsRef<str>>(values: &[S], shape: &[i64]) -> String {
    if shape.len() <= 1 {
        return format!(
            "[{}]",
            values
                .iter()
                .map(|s| s.as_ref())
                .collect::<Vec<_>>()
                .join(", ")
        );
    }
    let inner_size: usize = shape[1..]
        .iter()
        .copied()
        .map(|x| if x < 0 { 1usize } else { x as usize })
        .product();
    let chunks: Vec<String> = values
        .chunks(inner_size)
        .map(|chunk| {
            fmt_nested(
                &chunk
                    .iter()
                    .map(|s| s.as_ref().to_string())
                    .collect::<Vec<_>>(),
                &shape[1..],
            )
        })
        .collect();
    format!("[{}]", chunks.join(", "))
}

fn fmt_single_value(data: &[u8], ctx: &PrinterCtx<'_>, elem_ty: TypeId) -> String {
    match ctx.ty(elem_ty) {
        Type::Int { width: 1 } => if data.first().map(|&b| b != 0).unwrap_or(false) {
            "true"
        } else {
            "false"
        }
        .into(),
        Type::Int { width: 8 } => (data.first().map(|&b| b as i8).unwrap_or(0)).to_string(),
        Type::Int { width: 16 } => {
            i16::from_le_bytes(data[..2].try_into().unwrap_or([0; 2])).to_string()
        }
        Type::Int { width: 32 } => {
            i32::from_le_bytes(data[..4].try_into().unwrap_or([0; 4])).to_string()
        }
        Type::Int { width: 64 } => {
            i64::from_le_bytes(data[..8].try_into().unwrap_or([0; 8])).to_string()
        }
        Type::Float(FloatKind::F32) => {
            let bits = u32::from_le_bytes(data[..4].try_into().unwrap_or([0; 4]));
            let f = f32::from_bits(bits);
            if !f.is_finite() {
                format!("0x{:08X}", bits)
            } else {
                fmt_float(f as f64)
            }
        }
        Type::Float(FloatKind::F64) => {
            let bits = u64::from_le_bytes(data[..8].try_into().unwrap_or([0; 8]));
            let f = f64::from_bits(bits);
            if !f.is_finite() {
                format!("0x{:016X}", bits)
            } else {
                fmt_float(f)
            }
        }
        Type::Float(FloatKind::F16) => {
            let bits = u16::from_le_bytes(data[..2].try_into().unwrap_or([0; 2]));
            let f = f16::from_bits(bits).to_f32();
            if !f.is_finite() {
                format!("0x{:04X}", bits)
            } else {
                fmt_float(f as f64)
            }
        }
        Type::Float(FloatKind::BF16) => {
            let bits = u16::from_le_bytes(data[..2].try_into().unwrap_or([0; 2]));
            let f = bf16::from_bits(bits).to_f32();
            if !f.is_finite() {
                format!("0x{:04X}", bits)
            } else {
                fmt_float(f as f64)
            }
        }
        Type::Float(FloatKind::TF32) => {
            let mut bytes = [0u8; 4];
            bytes[..data.len().min(3)].copy_from_slice(&data[..data.len().min(3)]);
            let bits = u32::from_le_bytes(bytes) << 13;
            let f32v = f32::from_bits(bits);
            if !f32v.is_finite() {
                return format!("0x{:08X}", bits);
            }
            let rounded = (f32v as f64 * 1e5).round() / 1e5;
            fmt_float(rounded)
        }
        Type::Float(FloatKind::F8E4M3FN) => fmt_float(f8e4m3fn_to_f64(data[0])),
        Type::Float(FloatKind::F8E5M2) => fmt_float(f8e5m2_to_f64(data[0])),
        other => panic!("unsupported DenseElements scalar type: {:?}", other),
    }
}

pub(super) fn fmt_float(v: f64) -> String {
    let s = format!("{:.6e}", v);
    if let Some(pos) = s.find('e') {
        let (mantissa, exp) = s.split_at(pos);
        format!("{}e{:+03}", mantissa, exp[1..].parse::<i32>().unwrap_or(0))
    } else {
        s
    }
}

fn f8e4m3fn_to_f64(bits: u8) -> f64 {
    let sign = ((bits >> 7) & 1) as i32;
    let exp = ((bits >> 3) & 0xf) as i32;
    let mant = (bits & 0x7) as u32;
    if exp == 0 {
        if mant == 0 {
            return if sign == 1 { -0.0 } else { 0.0 };
        }
        let v = (mant as f64) * 2.0f64.powi(-9);
        return if sign == 1 { -v } else { v };
    }
    if exp == 15 {
        return f64::NAN;
    }
    let v = (1.0 + (mant as f64) / 8.0) * 2.0f64.powi(exp - 7);
    if sign == 1 {
        -v
    } else {
        v
    }
}

fn f8e5m2_to_f64(bits: u8) -> f64 {
    let sign = ((bits >> 7) & 1) as i32;
    let exp = ((bits >> 2) & 0x1f) as i32;
    let mant = (bits & 0x3) as u32;
    if exp == 0 {
        if mant == 0 {
            return if sign == 1 { -0.0 } else { 0.0 };
        }
        let v = (mant as f64) * 2.0f64.powi(-16);
        return if sign == 1 { -v } else { v };
    }
    if exp == 31 {
        return if mant == 0 {
            if sign == 1 {
                f64::NEG_INFINITY
            } else {
                f64::INFINITY
            }
        } else {
            f64::NAN
        };
    }
    let v = (1.0 + (mant as f64) / 4.0) * 2.0f64.powi(exp - 15);
    if sign == 1 {
        -v
    } else {
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::cuda_tile_ir::arena::IrArena;
    use crate::cuda_tile_ir::cfg::Module;
    use crate::cuda_tile_ir::consts::ConstPool;
    use crate::cuda_tile_ir::types::Type;

    fn dummy_module_with_type(ty: Type) -> (Module, TypeId) {
        let mut arena = IrArena::new();
        let ty_id = arena.intern_type(ty);
        let module = Module {
            name: "test".into(),
            globals: Vec::new(),
            functions: Vec::new(),
            arena,
            consts: ConstPool::empty(),
        };
        (module, ty_id)
    }

    #[test]
    fn dense_formats_i1_bitpack() {
        let (module, ty_id) = dummy_module_with_type(Type::Int { width: 1 });
        let ctx = PrinterCtx::new(&module);
        let data = [0b0000_0011u8];
        assert_eq!(
            fmt_dense_value(&data, &ctx, ty_id, &[4]),
            "[true, true, false, false]"
        );
    }

    #[test]
    fn dense_formats_f32_nan_as_hex() {
        let (module, ty_id) = dummy_module_with_type(Type::Float(FloatKind::F32));
        let ctx = PrinterCtx::new(&module);
        let bits = 0x7FC0_0001u32;
        let data = bits.to_le_bytes();
        assert_eq!(fmt_dense_value(&data, &ctx, ty_id, &[]), "0x7FC00001");
    }

    #[test]
    fn dense_formats_broadcast_single_value_for_vector() {
        let (module, ty_id) = dummy_module_with_type(Type::Int { width: 8 });
        let ctx = PrinterCtx::new(&module);
        let data = [5u8];
        assert_eq!(fmt_dense_value(&data, &ctx, ty_id, &[3]), "5");
    }
}
