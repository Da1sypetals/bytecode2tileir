// Conversion operations for TileIR interpreter (Section 8.4)
//
// Implements: bitcast, exti, ftof, ftoi, int_to_ptr, itof, ptr_to_int, ptr_to_ptr, trunci

use crate::cuda_tile_ir::OpAttrKey;
use crate::cuda_tile_ir::attrs::Attr;
use crate::cuda_tile_ir::enums::{IntegerOverflow, RoundingMode, Signedness};
use crate::cuda_tile_ir::ir::Operation;
use crate::interpreter::data_structures::elem_type::ElemType;
use crate::interpreter::data_structures::interpreter::ExecutionContext;
use crate::interpreter::data_structures::tile::Tile;
use crate::interpreter::data_structures::value::Value;

impl ExecutionContext<'_> {
    // =========================================================================
    // Helper Functions (private)
    // =========================================================================

    fn extract_signedness(&self, op: &Operation) -> Signedness {
        let attr_id = op
            .attrs
            .iter()
            .find(|(key, _)| *key == OpAttrKey::Signedness)
            .map(|(_, id)| id)
            .expect("Operation missing Signedness attribute");
        match self.arena.attr_(*attr_id) {
            Attr::Signedness(s) => *s,
            _ => panic!("Signedness attribute must be Signedness enum"),
        }
    }

    fn extract_rounding_mode(&self, op: &Operation) -> RoundingMode {
        let attr_id = op
            .attrs
            .iter()
            .find(|(key, _)| *key == OpAttrKey::RoundingMode)
            .map(|(_, id)| id)
            .expect("Operation missing RoundingMode attribute");
        match self.arena.attr_(*attr_id) {
            Attr::RoundingMode(rm) => *rm,
            _ => panic!("RoundingMode attribute must be RoundingMode enum"),
        }
    }

    fn extract_overflow(&self, op: &Operation) -> IntegerOverflow {
        let attr_id = op
            .attrs
            .iter()
            .find(|(key, _)| *key == OpAttrKey::Overflow)
            .map(|(_, id)| id)
            .expect("Operation missing Overflow attribute");
        match self.arena.attr_(*attr_id) {
            Attr::IntegerOverflow(ovf) => *ovf,
            _ => panic!("Overflow attribute must be IntegerOverflow enum"),
        }
    }

    fn get_result_elem_type(&self, op: &Operation) -> ElemType {
        let result_value_id = op.results[0];
        let result_value_data = self.arena.value_(result_value_id);
        let result_ty = self.arena.type_(result_value_data.ty());
        crate::interpreter::type_conversion::type_to_elem_type(&result_ty, self.arena)
    }

    /// Round f32 to i32 according to the specified rounding mode.
    /// NaN → 0, Inf → panic (undefined behavior per spec)
    fn round_f32_to_i32(&self, v: f32, mode: RoundingMode) -> i32 {
        if v.is_nan() {
            return 0;
        }
        if v.is_infinite() {
            panic!("FToI: Infinity input is undefined behavior");
        }

        match mode {
            RoundingMode::NearestEven => v.round_ties_even() as i32,
            RoundingMode::Zero => v as i32,
            RoundingMode::NegativeInf => v.floor() as i32,
            RoundingMode::PositiveInf => v.ceil() as i32,
            RoundingMode::Approx | RoundingMode::Full | RoundingMode::NearestIntToZero => {
                // For these modes, use round to nearest (ties away from zero not standard in Rust)
                v.round() as i32
            }
        }
    }

    /// Round f64 to i64 according to the specified rounding mode.
    /// NaN → 0, Inf → panic (undefined behavior per spec)
    fn round_f64_to_i64(&self, v: f64, mode: RoundingMode) -> i64 {
        if v.is_nan() {
            return 0;
        }
        if v.is_infinite() {
            panic!("FToI: Infinity input is undefined behavior");
        }

        match mode {
            RoundingMode::NearestEven => v.round_ties_even() as i64,
            RoundingMode::Zero => v as i64,
            RoundingMode::NegativeInf => v.floor() as i64,
            RoundingMode::PositiveInf => v.ceil() as i64,
            RoundingMode::Approx | RoundingMode::Full | RoundingMode::NearestIntToZero => {
                v.round() as i64
            }
        }
    }

    fn validate_bitcast_compat(&self, src_elem: ElemType, dst_elem: ElemType) {
        assert_eq!(
            src_elem.bit_width(),
            dst_elem.bit_width(),
            "Bitcast requires same bit width: {} vs {}",
            src_elem.bit_width(),
            dst_elem.bit_width()
        );
        assert_ne!(src_elem, dst_elem, "Bitcast requires different types");
        // Disallow pointer types
        match src_elem {
            ElemType::Ptr => panic!("Bitcast does not support pointer types"),
            _ => {}
        }
        match dst_elem {
            ElemType::Ptr => panic!("Bitcast does not support pointer types"),
            _ => {}
        }
    }
}

impl ExecutionContext<'_> {
    // =========================================================================
    // Pointer Conversions (8.4.6-8.4.8)
    // =========================================================================

    /// 8.4.6. cuda_tile.int_to_ptr - Convert i64 tile to pointer tile
    pub fn execute_int_to_ptr(&mut self, op: &Operation) {
        let src_value = self.get_value(op.operands[0]);

        let result = match src_value {
            Value::Tile(Tile::I64(arr)) => {
                let ptrs: Vec<*mut u8> = arr.iter().map(|&v| v as *mut u8).collect();
                Tile::Ptr(
                    ndarray::Array::from_shape_vec(ndarray::IxDyn(arr.shape()), ptrs).unwrap(),
                )
            }
            _ => panic!("IntToPtr requires I64 tile operand"),
        };

        self.set_value(op.results[0], Value::Tile(result));
    }

    /// 8.4.7. cuda_tile.ptr_to_int - Convert pointer tile to i64 tile
    pub fn execute_ptr_to_int(&mut self, op: &Operation) {
        let src_value = self.get_value(op.operands[0]);

        let result = match src_value {
            Value::Tile(Tile::Ptr(arr)) => Tile::I64(arr.mapv(|p| p as i64)),
            _ => panic!("PtrToInt requires Ptr tile operand"),
        };

        self.set_value(op.results[0], Value::Tile(result));
    }

    /// 8.4.8. cuda_tile.ptr_to_ptr - Reinterpret pointer tile as another pointer type
    pub fn execute_ptr_to_ptr(&mut self, op: &Operation) {
        let src_value = self.get_value(op.operands[0]);

        // This is essentially a no-op for the interpreter - just copy the pointer values
        let result = match src_value {
            Value::Tile(Tile::Ptr(_)) => src_value.clone(),
            _ => panic!("PtrToPtr requires Ptr tile operand"),
        };

        self.set_value(op.results[0], result);
    }

    // =========================================================================
    // Bitcast (8.4.1)
    // =========================================================================

    /// 8.4.1. cuda_tile.bitcast - Reinterpret bits from one type to another
    pub fn execute_bitcast(&mut self, op: &Operation) {
        let src_value = self.get_value(op.operands[0]);
        let dst_elem_type = self.get_result_elem_type(op);

        let src_elem_type = match &src_value {
            Value::Tile(tile) => tile.elem_type(),
            _ => panic!("Bitcast requires Tile operand"),
        };

        self.validate_bitcast_compat(src_elem_type, dst_elem_type);

        let result = match (src_value, dst_elem_type) {
            // i32 <-> f32 (32 bits)
            (Value::Tile(Tile::I32(arr)), ElemType::F32) => {
                Tile::F32(arr.mapv(|v| f32::from_bits(v.cast_unsigned())))
            }
            (Value::Tile(Tile::F32(arr)), ElemType::I32) => {
                Tile::I32(arr.mapv(|v| v.to_bits().cast_signed()))
            }
            // i64 <-> f64 (64 bits)
            (Value::Tile(Tile::I64(arr)), ElemType::F64) => {
                Tile::F64(arr.mapv(|v| f64::from_bits(v.cast_unsigned())))
            }
            (Value::Tile(Tile::F64(arr)), ElemType::I64) => {
                Tile::I64(arr.mapv(|v| v.to_bits().cast_signed()))
            }
            // i16 <-> f16 (16 bits)
            (Value::Tile(Tile::I16(arr)), ElemType::F16) => {
                Tile::F16(arr.mapv(|v| f16::from_bits(v.cast_unsigned())))
            }
            (Value::Tile(Tile::F16(arr)), ElemType::I16) => {
                Tile::I16(arr.mapv(|v| v.to_bits().cast_signed()))
            }
            _ => panic!(
                "Unsupported bitcast: {:?} to {:?}",
                src_elem_type, dst_elem_type
            ),
        };

        self.set_value(op.results[0], Value::Tile(result));
    }

    // =========================================================================
    // ExtI - Integer Extension (8.4.2)
    // =========================================================================

    /// 8.4.2. cuda_tile.exti - Extend integer width (sign or zero extension)
    pub fn execute_exti(&mut self, op: &Operation) {
        let src_value = self.get_value(op.operands[0]);
        let signedness = self.extract_signedness(op);
        let dst_elem_type = self.get_result_elem_type(op);

        let result = match (src_value, signedness, dst_elem_type) {
            // Sign extension
            (Value::Tile(Tile::I1(arr)), Signedness::Signed, ElemType::I8) => {
                Tile::I8(arr.mapv(|b| b as i8))
            }
            (Value::Tile(Tile::I1(arr)), Signedness::Signed, ElemType::I16) => {
                Tile::I16(arr.mapv(|b| b as i16))
            }
            (Value::Tile(Tile::I1(arr)), Signedness::Signed, ElemType::I32) => {
                Tile::I32(arr.mapv(|b| b as i32))
            }
            (Value::Tile(Tile::I1(arr)), Signedness::Signed, ElemType::I64) => {
                Tile::I64(arr.mapv(|b| b as i64))
            }
            (Value::Tile(Tile::I8(arr)), Signedness::Signed, ElemType::I16) => {
                Tile::I16(arr.mapv(|v| v as i16))
            }
            (Value::Tile(Tile::I8(arr)), Signedness::Signed, ElemType::I32) => {
                Tile::I32(arr.mapv(|v| v as i32))
            }
            (Value::Tile(Tile::I8(arr)), Signedness::Signed, ElemType::I64) => {
                Tile::I64(arr.mapv(|v| v as i64))
            }
            (Value::Tile(Tile::I16(arr)), Signedness::Signed, ElemType::I32) => {
                Tile::I32(arr.mapv(|v| v as i32))
            }
            (Value::Tile(Tile::I16(arr)), Signedness::Signed, ElemType::I64) => {
                Tile::I64(arr.mapv(|v| v as i64))
            }
            (Value::Tile(Tile::I32(arr)), Signedness::Signed, ElemType::I64) => {
                Tile::I64(arr.mapv(|v| v as i64))
            }
            // Zero extension
            (Value::Tile(Tile::I1(arr)), Signedness::Unsigned, ElemType::I8) => {
                Tile::I8(arr.mapv(|b| b as u8 as i8))
            }
            (Value::Tile(Tile::I1(arr)), Signedness::Unsigned, ElemType::I16) => {
                Tile::I16(arr.mapv(|b| b as u16 as i16))
            }
            (Value::Tile(Tile::I1(arr)), Signedness::Unsigned, ElemType::I32) => {
                Tile::I32(arr.mapv(|b| b as u32 as i32))
            }
            (Value::Tile(Tile::I1(arr)), Signedness::Unsigned, ElemType::I64) => {
                Tile::I64(arr.mapv(|b| b as u64 as i64))
            }
            (Value::Tile(Tile::I8(arr)), Signedness::Unsigned, ElemType::I16) => {
                Tile::I16(arr.mapv(|v| v as u16 as i16))
            }
            (Value::Tile(Tile::I8(arr)), Signedness::Unsigned, ElemType::I32) => {
                Tile::I32(arr.mapv(|v| v as u32 as i32))
            }
            (Value::Tile(Tile::I8(arr)), Signedness::Unsigned, ElemType::I64) => {
                Tile::I64(arr.mapv(|v| v as u64 as i64))
            }
            (Value::Tile(Tile::I16(arr)), Signedness::Unsigned, ElemType::I32) => {
                Tile::I32(arr.mapv(|v| v as u32 as i32))
            }
            (Value::Tile(Tile::I16(arr)), Signedness::Unsigned, ElemType::I64) => {
                Tile::I64(arr.mapv(|v| v as u64 as i64))
            }
            (Value::Tile(Tile::I32(arr)), Signedness::Unsigned, ElemType::I64) => {
                Tile::I64(arr.mapv(|v| v as u64 as i64))
            }
            _ => panic!(
                "Unsupported exti: signedness={:?}, dst_type={:?}",
                signedness, dst_elem_type
            ),
        };

        self.set_value(op.results[0], Value::Tile(result));
    }

    // =========================================================================
    // TruncI - Integer Truncation (8.4.9)
    // =========================================================================

    /// 8.4.9. cuda_tile.trunci - Truncate integer width (discard high bits)
    pub fn execute_trunci(&mut self, op: &Operation) {
        let _overflow = self.extract_overflow(op); // Informational, not used
        let src_value = self.get_value(op.operands[0]);
        let dst_elem_type = self.get_result_elem_type(op);

        let result = match (src_value, dst_elem_type) {
            (Value::Tile(Tile::I64(arr)), ElemType::I32) => Tile::I32(arr.mapv(|v| v as i32)),
            (Value::Tile(Tile::I64(arr)), ElemType::I16) => Tile::I16(arr.mapv(|v| v as i16)),
            (Value::Tile(Tile::I64(arr)), ElemType::I8) => Tile::I8(arr.mapv(|v| v as i8)),
            (Value::Tile(Tile::I64(arr)), ElemType::Bool) => Tile::I1(arr.mapv(|v| v != 0)),
            (Value::Tile(Tile::I32(arr)), ElemType::I16) => Tile::I16(arr.mapv(|v| v as i16)),
            (Value::Tile(Tile::I32(arr)), ElemType::I8) => Tile::I8(arr.mapv(|v| v as i8)),
            (Value::Tile(Tile::I32(arr)), ElemType::Bool) => Tile::I1(arr.mapv(|v| v != 0)),
            (Value::Tile(Tile::I16(arr)), ElemType::I8) => Tile::I8(arr.mapv(|v| v as i8)),
            (Value::Tile(Tile::I16(arr)), ElemType::Bool) => Tile::I1(arr.mapv(|v| v != 0)),
            (Value::Tile(Tile::I8(arr)), ElemType::Bool) => Tile::I1(arr.mapv(|v| v != 0)),
            _ => panic!("Unsupported trunci to {:?}", dst_elem_type),
        };

        self.set_value(op.results[0], Value::Tile(result));
    }

    // =========================================================================
    // FToF - Float to Float (8.4.3)
    // =========================================================================

    /// 8.4.3. cuda_tile.ftof - Convert between floating-point types
    pub fn execute_ftof(&mut self, op: &Operation) {
        let _rounding_mode = self.extract_rounding_mode(op); // Informational for most cases
        let src_value = self.get_value(op.operands[0]);
        let dst_elem_type = self.get_result_elem_type(op);

        let result = match (src_value, dst_elem_type) {
            // Widening: use `as` for f16 and f32
            (Value::Tile(Tile::F16(arr)), ElemType::F32) => Tile::F32(arr.mapv(|v| v as f32)),
            (Value::Tile(Tile::F16(arr)), ElemType::F64) => Tile::F64(arr.mapv(|v| v as f64)),
            (Value::Tile(Tile::F32(arr)), ElemType::F64) => Tile::F64(arr.mapv(|v| v as f64)),
            // Narrowing: use `as f16`
            (Value::Tile(Tile::F64(arr)), ElemType::F32) => Tile::F32(arr.mapv(|v| v as f32)),
            (Value::Tile(Tile::F64(arr)), ElemType::F16) => Tile::F16(arr.mapv(|v| v as f16)),
            (Value::Tile(Tile::F32(arr)), ElemType::F16) => Tile::F16(arr.mapv(|v| v as f16)),
            _ => panic!("Unsupported ftof to {:?}", dst_elem_type),
        };

        self.set_value(op.results[0], Value::Tile(result));
    }

    // =========================================================================
    // IToF - Integer to Float (8.4.5)
    // =========================================================================

    /// 8.4.5. cuda_tile.itof - Convert integer to floating-point
    pub fn execute_itof(&mut self, op: &Operation) {
        let signedness = self.extract_signedness(op);
        let _rounding_mode = self.extract_rounding_mode(op); // Informational
        let src_value = self.get_value(op.operands[0]);
        let dst_elem_type = self.get_result_elem_type(op);

        let result = match (src_value, signedness, dst_elem_type) {
            // Signed conversions
            (Value::Tile(Tile::I1(arr)), Signedness::Signed, ElemType::F32) => {
                Tile::F32(arr.mapv(|b| b as i32 as f32))
            }
            (Value::Tile(Tile::I1(arr)), Signedness::Signed, ElemType::F64) => {
                Tile::F64(arr.mapv(|b| b as i32 as f64))
            }
            (Value::Tile(Tile::I1(arr)), Signedness::Signed, ElemType::F16) => {
                Tile::F16(arr.mapv(|b| (b as i32 as f32) as f16))
            }
            (Value::Tile(Tile::I8(arr)), Signedness::Signed, ElemType::F32) => {
                Tile::F32(arr.mapv(|v| v as f32))
            }
            (Value::Tile(Tile::I8(arr)), Signedness::Signed, ElemType::F64) => {
                Tile::F64(arr.mapv(|v| v as f64))
            }
            (Value::Tile(Tile::I8(arr)), Signedness::Signed, ElemType::F16) => {
                Tile::F16(arr.mapv(|v| (v as f32) as f16))
            }
            (Value::Tile(Tile::I16(arr)), Signedness::Signed, ElemType::F32) => {
                Tile::F32(arr.mapv(|v| v as f32))
            }
            (Value::Tile(Tile::I16(arr)), Signedness::Signed, ElemType::F64) => {
                Tile::F64(arr.mapv(|v| v as f64))
            }
            (Value::Tile(Tile::I16(arr)), Signedness::Signed, ElemType::F16) => {
                Tile::F16(arr.mapv(|v| (v as f32) as f16))
            }
            (Value::Tile(Tile::I32(arr)), Signedness::Signed, ElemType::F32) => {
                Tile::F32(arr.mapv(|v| v as f32))
            }
            (Value::Tile(Tile::I32(arr)), Signedness::Signed, ElemType::F64) => {
                Tile::F64(arr.mapv(|v| v as f64))
            }
            (Value::Tile(Tile::I32(arr)), Signedness::Signed, ElemType::F16) => {
                Tile::F16(arr.mapv(|v| (v as f32) as f16))
            }
            (Value::Tile(Tile::I64(arr)), Signedness::Signed, ElemType::F32) => {
                Tile::F32(arr.mapv(|v| v as f32))
            }
            (Value::Tile(Tile::I64(arr)), Signedness::Signed, ElemType::F64) => {
                Tile::F64(arr.mapv(|v| v as f64))
            }
            (Value::Tile(Tile::I64(arr)), Signedness::Signed, ElemType::F16) => {
                Tile::F16(arr.mapv(|v| (v as f32) as f16))
            }
            // Unsigned conversions - cast to unsigned first, then to float
            (Value::Tile(Tile::I1(arr)), Signedness::Unsigned, ElemType::F32) => {
                Tile::F32(arr.mapv(|b| (b as u8 as u32) as f32))
            }
            (Value::Tile(Tile::I1(arr)), Signedness::Unsigned, ElemType::F64) => {
                Tile::F64(arr.mapv(|b| (b as u8 as u64) as f64))
            }
            (Value::Tile(Tile::I1(arr)), Signedness::Unsigned, ElemType::F16) => {
                Tile::F16(arr.mapv(|b| ((b as u8 as u32) as f32) as f16))
            }
            (Value::Tile(Tile::I8(arr)), Signedness::Unsigned, ElemType::F32) => {
                Tile::F32(arr.mapv(|v| (v as u8) as f32))
            }
            (Value::Tile(Tile::I8(arr)), Signedness::Unsigned, ElemType::F64) => {
                Tile::F64(arr.mapv(|v| (v as u8) as f64))
            }
            (Value::Tile(Tile::I8(arr)), Signedness::Unsigned, ElemType::F16) => {
                Tile::F16(arr.mapv(|v| ((v as u8) as f32) as f16))
            }
            (Value::Tile(Tile::I16(arr)), Signedness::Unsigned, ElemType::F32) => {
                Tile::F32(arr.mapv(|v| (v as u16) as f32))
            }
            (Value::Tile(Tile::I16(arr)), Signedness::Unsigned, ElemType::F64) => {
                Tile::F64(arr.mapv(|v| (v as u16) as f64))
            }
            (Value::Tile(Tile::I16(arr)), Signedness::Unsigned, ElemType::F16) => {
                Tile::F16(arr.mapv(|v| ((v as u16) as f32) as f16))
            }
            (Value::Tile(Tile::I32(arr)), Signedness::Unsigned, ElemType::F32) => {
                Tile::F32(arr.mapv(|v| (v as u32) as f32))
            }
            (Value::Tile(Tile::I32(arr)), Signedness::Unsigned, ElemType::F64) => {
                Tile::F64(arr.mapv(|v| (v as u32) as f64))
            }
            (Value::Tile(Tile::I32(arr)), Signedness::Unsigned, ElemType::F16) => {
                Tile::F16(arr.mapv(|v| ((v as u32) as f32) as f16))
            }
            (Value::Tile(Tile::I64(arr)), Signedness::Unsigned, ElemType::F32) => {
                Tile::F32(arr.mapv(|v| (v as u64) as f32))
            }
            (Value::Tile(Tile::I64(arr)), Signedness::Unsigned, ElemType::F64) => {
                Tile::F64(arr.mapv(|v| (v as u64) as f64))
            }
            (Value::Tile(Tile::I64(arr)), Signedness::Unsigned, ElemType::F16) => {
                Tile::F16(arr.mapv(|v| ((v as u64) as f32) as f16))
            }
            _ => panic!(
                "Unsupported itof: signedness={:?}, dst_type={:?}",
                signedness, dst_elem_type
            ),
        };

        self.set_value(op.results[0], Value::Tile(result));
    }

    // =========================================================================
    // FToI - Float to Integer (8.4.4)
    // =========================================================================

    /// 8.4.4. cuda_tile.ftoi - Convert floating-point to integer
    pub fn execute_ftoi(&mut self, op: &Operation) {
        let signedness = self.extract_signedness(op);
        let rounding_mode = self.extract_rounding_mode(op);
        let src_value = self.get_value(op.operands[0]);
        let dst_elem_type = self.get_result_elem_type(op);

        // Helper closures for rounding
        let f32_to_i32 = |v: f32| self.round_f32_to_i32(v, rounding_mode);
        let f32_to_i64 = |v: f32| self.round_f32_to_i32(v, rounding_mode) as i64;
        let f64_to_i32 = |v: f64| self.round_f64_to_i64(v, rounding_mode) as i32;
        let f64_to_i64 = |v: f64| self.round_f64_to_i64(v, rounding_mode);
        let f16_to_i32 = |v: f16| self.round_f32_to_i32(v as f32, rounding_mode);
        let f16_to_i64 = |v: f16| self.round_f32_to_i32(v as f32, rounding_mode) as i64;

        let result = match (src_value, signedness, dst_elem_type) {
            // From f16
            (Value::Tile(Tile::F16(arr)), Signedness::Signed, ElemType::I8) => {
                Tile::I8(arr.mapv(|v| f16_to_i32(v) as i8))
            }
            (Value::Tile(Tile::F16(arr)), Signedness::Signed, ElemType::I16) => {
                Tile::I16(arr.mapv(|v| f16_to_i32(v) as i16))
            }
            (Value::Tile(Tile::F16(arr)), Signedness::Signed, ElemType::I32) => {
                Tile::I32(arr.mapv(f16_to_i32))
            }
            (Value::Tile(Tile::F16(arr)), Signedness::Signed, ElemType::I64) => {
                Tile::I64(arr.mapv(f16_to_i64))
            }
            (Value::Tile(Tile::F16(arr)), Signedness::Unsigned, ElemType::I8) => {
                Tile::I8(arr.mapv(|v| f16_to_i32(v) as u8 as i8))
            }
            (Value::Tile(Tile::F16(arr)), Signedness::Unsigned, ElemType::I16) => {
                Tile::I16(arr.mapv(|v| f16_to_i32(v) as u16 as i16))
            }
            (Value::Tile(Tile::F16(arr)), Signedness::Unsigned, ElemType::I32) => {
                Tile::I32(arr.mapv(|v| f16_to_i32(v) as u32 as i32))
            }
            (Value::Tile(Tile::F16(arr)), Signedness::Unsigned, ElemType::I64) => {
                Tile::I64(arr.mapv(|v| f16_to_i64(v) as u64 as i64))
            }
            // From f32
            (Value::Tile(Tile::F32(arr)), Signedness::Signed, ElemType::I8) => {
                Tile::I8(arr.mapv(|v| f32_to_i32(v) as i8))
            }
            (Value::Tile(Tile::F32(arr)), Signedness::Signed, ElemType::I16) => {
                Tile::I16(arr.mapv(|v| f32_to_i32(v) as i16))
            }
            (Value::Tile(Tile::F32(arr)), Signedness::Signed, ElemType::I32) => {
                Tile::I32(arr.mapv(f32_to_i32))
            }
            (Value::Tile(Tile::F32(arr)), Signedness::Signed, ElemType::I64) => {
                Tile::I64(arr.mapv(f32_to_i64))
            }
            (Value::Tile(Tile::F32(arr)), Signedness::Unsigned, ElemType::I8) => {
                Tile::I8(arr.mapv(|v| f32_to_i32(v) as u8 as i8))
            }
            (Value::Tile(Tile::F32(arr)), Signedness::Unsigned, ElemType::I16) => {
                Tile::I16(arr.mapv(|v| f32_to_i32(v) as u16 as i16))
            }
            (Value::Tile(Tile::F32(arr)), Signedness::Unsigned, ElemType::I32) => {
                Tile::I32(arr.mapv(|v| f32_to_i32(v) as u32 as i32))
            }
            (Value::Tile(Tile::F32(arr)), Signedness::Unsigned, ElemType::I64) => {
                Tile::I64(arr.mapv(|v| f32_to_i64(v) as u64 as i64))
            }
            // From f64
            (Value::Tile(Tile::F64(arr)), Signedness::Signed, ElemType::I8) => {
                Tile::I8(arr.mapv(|v| f64_to_i32(v) as i8))
            }
            (Value::Tile(Tile::F64(arr)), Signedness::Signed, ElemType::I16) => {
                Tile::I16(arr.mapv(|v| f64_to_i32(v) as i16))
            }
            (Value::Tile(Tile::F64(arr)), Signedness::Signed, ElemType::I32) => {
                Tile::I32(arr.mapv(f64_to_i32))
            }
            (Value::Tile(Tile::F64(arr)), Signedness::Signed, ElemType::I64) => {
                Tile::I64(arr.mapv(f64_to_i64))
            }
            (Value::Tile(Tile::F64(arr)), Signedness::Unsigned, ElemType::I8) => {
                Tile::I8(arr.mapv(|v| f64_to_i32(v) as u8 as i8))
            }
            (Value::Tile(Tile::F64(arr)), Signedness::Unsigned, ElemType::I16) => {
                Tile::I16(arr.mapv(|v| f64_to_i32(v) as u16 as i16))
            }
            (Value::Tile(Tile::F64(arr)), Signedness::Unsigned, ElemType::I32) => {
                Tile::I32(arr.mapv(|v| f64_to_i32(v) as u32 as i32))
            }
            (Value::Tile(Tile::F64(arr)), Signedness::Unsigned, ElemType::I64) => {
                Tile::I64(arr.mapv(|v| f64_to_i64(v) as u64 as i64))
            }
            _ => panic!(
                "Unsupported ftoi: signedness={:?}, dst_type={:?}",
                signedness, dst_elem_type
            ),
        };

        self.set_value(op.results[0], Value::Tile(result));
    }
}
