// Floating-point operations for TileIR interpreter (Section 8.7)
//
// Implements: absf, addf, ceil, cmpf, cos, cosh, divf, exp, exp2, floor,
//             fma, log, log2, maxf, minf, mmaf, mulf, negf, pow, remf,
//             rsqrt, sin, sinh, sqrt, subf, tan, tanh

use crate::cuda_tile_ir::OpAttrKey;
use crate::cuda_tile_ir::attrs::Attr;
use crate::cuda_tile_ir::enums::{ComparisonOrdering, ComparisonPredicate};
use crate::cuda_tile_ir::ir::Operation;
use crate::interpreter::data_structures::interpreter::ExecutionContext;
use crate::interpreter::data_structures::tile::Tile;
use crate::interpreter::data_structures::value::Value;
use ndarray::Zip;

impl ExecutionContext<'_> {
    // =========================================================================
    // Helper Functions (private)
    // =========================================================================

    fn extract_flush_to_zero(&self, op: &Operation) -> bool {
        op.attrs
            .iter()
            .find(|(key, _)| *key == OpAttrKey::FlushToZero)
            .map(|(_, id)| self.arena.attr_(*id))
            .map(|attr| matches!(attr, Attr::Bool(true)))
            .unwrap_or(false)
    }

    fn extract_propagate_nan(&self, op: &Operation) -> bool {
        op.attrs
            .iter()
            .find(|(key, _)| *key == OpAttrKey::PropagateNan)
            .map(|(_, id)| self.arena.attr_(*id))
            .map(|attr| matches!(attr, Attr::Bool(true)))
            .unwrap_or(false)
    }

    fn extract_comparison_predicate(&self, op: &Operation) -> ComparisonPredicate {
        let attr_id = op
            .attrs
            .iter()
            .find(|(key, _)| *key == OpAttrKey::ComparisonPredicate)
            .map(|(_, id)| id)
            .expect("CmpF operation missing ComparisonPredicate attribute");
        match self.arena.attr_(*attr_id) {
            Attr::ComparisonPredicate(p) => *p,
            _ => panic!("ComparisonPredicate attribute must be ComparisonPredicate enum"),
        }
    }

    fn extract_comparison_ordering(&self, op: &Operation) -> ComparisonOrdering {
        let attr_id = op
            .attrs
            .iter()
            .find(|(key, _)| *key == OpAttrKey::ComparisonOrdering)
            .map(|(_, id)| id)
            .expect("CmpF operation missing ComparisonOrdering attribute");
        match self.arena.attr_(*attr_id) {
            Attr::ComparisonOrdering(o) => *o,
            _ => panic!("ComparisonOrdering attribute must be ComparisonOrdering enum"),
        }
    }

    // =========================================================================
    // Unary Operations
    // =========================================================================

    /// 8.7.3. cuda_tile.absf - Element-wise floating-point absolute value
    pub fn execute_absf(&mut self, op: &Operation) {
        let src = self.get_value(op.operands[0]);
        let result = match src {
            Value::Tile(tile) => Value::Tile(tile.absf()),
            _ => panic!("Absf requires Tile operand"),
        };
        self.set_value(op.results[0], result);
    }

    /// 8.7.20. cuda_tile.negf - Element-wise floating-point negation
    pub fn execute_negf(&mut self, op: &Operation) {
        let src = self.get_value(op.operands[0]);
        let result = match src {
            Value::Tile(tile) => Value::Tile(tile.negf()),
            _ => panic!("Negf requires Tile operand"),
        };
        self.set_value(op.results[0], result);
    }

    /// 8.7.26. cuda_tile.sqrt - Element-wise square root
    pub fn execute_sqrt(&mut self, op: &Operation) {
        let src = self.get_value(op.operands[0]);
        let ftz = self.extract_flush_to_zero(op);
        let result = match src {
            Value::Tile(tile) => Value::Tile(tile.sqrt(ftz)),
            _ => panic!("Sqrt requires Tile operand"),
        };
        self.set_value(op.results[0], result);
    }

    /// 8.7.23. cuda_tile.rsqrt - Element-wise reciprocal square root
    pub fn execute_rsqrt(&mut self, op: &Operation) {
        let src = self.get_value(op.operands[0]);
        let ftz = self.extract_flush_to_zero(op);
        let result = match src {
            Value::Tile(tile) => Value::Tile(tile.rsqrt(ftz)),
            _ => panic!("Rsqrt requires Tile operand"),
        };
        self.set_value(op.results[0], result);
    }

    /// 8.7.25. cuda_tile.sin - Element-wise sine
    pub fn execute_sin(&mut self, op: &Operation) {
        let src = self.get_value(op.operands[0]);
        let result = match src {
            Value::Tile(tile) => Value::Tile(tile.sin()),
            _ => panic!("Sin requires Tile operand"),
        };
        self.set_value(op.results[0], result);
    }

    /// 8.7.8. cuda_tile.cos - Element-wise cosine
    pub fn execute_cos(&mut self, op: &Operation) {
        let src = self.get_value(op.operands[0]);
        let result = match src {
            Value::Tile(tile) => Value::Tile(tile.cos()),
            _ => panic!("Cos requires Tile operand"),
        };
        self.set_value(op.results[0], result);
    }

    /// 8.7.29. cuda_tile.tan - Element-wise tangent
    pub fn execute_tan(&mut self, op: &Operation) {
        let src = self.get_value(op.operands[0]);
        let result = match src {
            Value::Tile(tile) => Value::Tile(tile.tan()),
            _ => panic!("Tan requires Tile operand"),
        };
        self.set_value(op.results[0], result);
    }

    /// 8.7.24. cuda_tile.sinh - Element-wise hyperbolic sine
    pub fn execute_sinh(&mut self, op: &Operation) {
        let src = self.get_value(op.operands[0]);
        let result = match src {
            Value::Tile(tile) => Value::Tile(tile.sinh()),
            _ => panic!("Sinh requires Tile operand"),
        };
        self.set_value(op.results[0], result);
    }

    /// 8.7.7. cuda_tile.cosh - Element-wise hyperbolic cosine
    pub fn execute_cosh(&mut self, op: &Operation) {
        let src = self.get_value(op.operands[0]);
        let result = match src {
            Value::Tile(tile) => Value::Tile(tile.cosh()),
            _ => panic!("Cosh requires Tile operand"),
        };
        self.set_value(op.results[0], result);
    }

    /// 8.7.28. cuda_tile.tanh - Element-wise hyperbolic tangent
    pub fn execute_tanh(&mut self, op: &Operation) {
        let src = self.get_value(op.operands[0]);
        let result = match src {
            Value::Tile(tile) => Value::Tile(tile.tanh()),
            _ => panic!("Tanh requires Tile operand"),
        };
        self.set_value(op.results[0], result);
    }

    /// 8.7.11. cuda_tile.exp - Element-wise exponential
    pub fn execute_exp(&mut self, op: &Operation) {
        let src = self.get_value(op.operands[0]);
        let result = match src {
            Value::Tile(tile) => Value::Tile(tile.exp()),
            _ => panic!("Exp requires Tile operand"),
        };
        self.set_value(op.results[0], result);
    }

    /// 8.7.10. cuda_tile.exp2 - Element-wise power of two
    pub fn execute_exp2(&mut self, op: &Operation) {
        let src = self.get_value(op.operands[0]);
        let ftz = self.extract_flush_to_zero(op);
        let result = match src {
            Value::Tile(tile) => Value::Tile(tile.exp2(ftz)),
            _ => panic!("Exp2 requires Tile operand"),
        };
        self.set_value(op.results[0], result);
    }

    /// 8.7.15. cuda_tile.log - Element-wise natural logarithm
    pub fn execute_log(&mut self, op: &Operation) {
        let src = self.get_value(op.operands[0]);
        let result = match src {
            Value::Tile(tile) => Value::Tile(tile.log()),
            _ => panic!("Log requires Tile operand"),
        };
        self.set_value(op.results[0], result);
    }

    /// 8.7.14. cuda_tile.log2 - Element-wise base-2 logarithm
    pub fn execute_log2(&mut self, op: &Operation) {
        let src = self.get_value(op.operands[0]);
        let result = match src {
            Value::Tile(tile) => Value::Tile(tile.log2()),
            _ => panic!("Log2 requires Tile operand"),
        };
        self.set_value(op.results[0], result);
    }

    /// 8.7.12. cuda_tile.floor - Element-wise floor rounding
    pub fn execute_floor(&mut self, op: &Operation) {
        let src = self.get_value(op.operands[0]);
        let result = match src {
            Value::Tile(tile) => Value::Tile(tile.floor()),
            _ => panic!("Floor requires Tile operand"),
        };
        self.set_value(op.results[0], result);
    }

    /// 8.7.5. cuda_tile.ceil - Element-wise ceiling
    pub fn execute_ceil(&mut self, op: &Operation) {
        let src = self.get_value(op.operands[0]);
        let result = match src {
            Value::Tile(tile) => Value::Tile(tile.ceil()),
            _ => panic!("Ceil requires Tile operand"),
        };
        self.set_value(op.results[0], result);
    }

    // =========================================================================
    // Binary Operations
    // =========================================================================

    /// 8.7.4. cuda_tile.addf - Element-wise floating-point addition
    pub fn execute_addf(&mut self, op: &Operation) {
        let lhs = self.get_value(op.operands[0]);
        let rhs = self.get_value(op.operands[1]);
        let ftz = self.extract_flush_to_zero(op);
        // Note: rounding_mode attribute exists but ignored for initial implementation

        let result = match (lhs, rhs) {
            (Value::Tile(lhs_tile), Value::Tile(rhs_tile)) => {
                Value::Tile(lhs_tile.addf(&rhs_tile, ftz))
            }
            _ => panic!("Addf requires Tile operands"),
        };
        self.set_value(op.results[0], result);
    }

    /// 8.7.27. cuda_tile.subf - Element-wise floating-point subtraction
    pub fn execute_subf(&mut self, op: &Operation) {
        let lhs = self.get_value(op.operands[0]);
        let rhs = self.get_value(op.operands[1]);
        let ftz = self.extract_flush_to_zero(op);

        let result = match (lhs, rhs) {
            (Value::Tile(lhs_tile), Value::Tile(rhs_tile)) => {
                Value::Tile(lhs_tile.subf(&rhs_tile, ftz))
            }
            _ => panic!("Subf requires Tile operands"),
        };
        self.set_value(op.results[0], result);
    }

    /// 8.7.19. cuda_tile.mulf - Element-wise floating-point multiplication
    pub fn execute_mulf(&mut self, op: &Operation) {
        let lhs = self.get_value(op.operands[0]);
        let rhs = self.get_value(op.operands[1]);
        let ftz = self.extract_flush_to_zero(op);

        let result = match (lhs, rhs) {
            (Value::Tile(lhs_tile), Value::Tile(rhs_tile)) => {
                Value::Tile(lhs_tile.mulf(&rhs_tile, ftz))
            }
            _ => panic!("Mulf requires Tile operands"),
        };
        self.set_value(op.results[0], result);
    }

    /// 8.7.9. cuda_tile.divf - Element-wise floating-point division
    pub fn execute_divf(&mut self, op: &Operation) {
        let lhs = self.get_value(op.operands[0]);
        let rhs = self.get_value(op.operands[1]);
        let ftz = self.extract_flush_to_zero(op);

        let result = match (lhs, rhs) {
            (Value::Tile(lhs_tile), Value::Tile(rhs_tile)) => {
                Value::Tile(lhs_tile.divf(&rhs_tile, ftz))
            }
            _ => panic!("Divf requires Tile operands"),
        };
        self.set_value(op.results[0], result);
    }

    /// 8.7.22. cuda_tile.remf - Element-wise floating-point remainder
    pub fn execute_remf(&mut self, op: &Operation) {
        let lhs = self.get_value(op.operands[0]);
        let rhs = self.get_value(op.operands[1]);

        let result = match (lhs, rhs) {
            (Value::Tile(lhs_tile), Value::Tile(rhs_tile)) => Value::Tile(lhs_tile.remf(&rhs_tile)),
            _ => panic!("Remf requires Tile operands"),
        };
        self.set_value(op.results[0], result);
    }

    /// 8.7.17. cuda_tile.minf - Element-wise floating-point minimum
    pub fn execute_minf(&mut self, op: &Operation) {
        let lhs = self.get_value(op.operands[0]);
        let rhs = self.get_value(op.operands[1]);
        let propagate_nan = self.extract_propagate_nan(op);
        let ftz = self.extract_flush_to_zero(op);

        let result = match (lhs, rhs) {
            (Value::Tile(lhs_tile), Value::Tile(rhs_tile)) => {
                Value::Tile(lhs_tile.minf(&rhs_tile, propagate_nan, ftz))
            }
            _ => panic!("Minf requires Tile operands"),
        };
        self.set_value(op.results[0], result);
    }

    /// 8.7.16. cuda_tile.maxf - Element-wise floating-point maximum
    pub fn execute_maxf(&mut self, op: &Operation) {
        let lhs = self.get_value(op.operands[0]);
        let rhs = self.get_value(op.operands[1]);
        let propagate_nan = self.extract_propagate_nan(op);
        let ftz = self.extract_flush_to_zero(op);

        let result = match (lhs, rhs) {
            (Value::Tile(lhs_tile), Value::Tile(rhs_tile)) => {
                Value::Tile(lhs_tile.maxf(&rhs_tile, propagate_nan, ftz))
            }
            _ => panic!("Maxf requires Tile operands"),
        };
        self.set_value(op.results[0], result);
    }

    /// 8.7.21. cuda_tile.pow - Element-wise floating-point exponentiation
    pub fn execute_pow(&mut self, op: &Operation) {
        let base = self.get_value(op.operands[0]);
        let exp = self.get_value(op.operands[1]);

        let result = match (base, exp) {
            (Value::Tile(base_tile), Value::Tile(exp_tile)) => {
                Value::Tile(base_tile.pow(&exp_tile))
            }
            _ => panic!("Pow requires Tile operands"),
        };
        self.set_value(op.results[0], result);
    }

    // =========================================================================
    // Ternary Operations
    // =========================================================================

    /// 8.7.13. cuda_tile.fma - Floating point fused multiply-add
    pub fn execute_fma(&mut self, op: &Operation) {
        let lhs = self.get_value(op.operands[0]);
        let rhs = self.get_value(op.operands[1]);
        let acc = self.get_value(op.operands[2]);
        let ftz = self.extract_flush_to_zero(op);

        let result = match (lhs, rhs, acc) {
            (Value::Tile(lhs_tile), Value::Tile(rhs_tile), Value::Tile(acc_tile)) => {
                Value::Tile(lhs_tile.fma(&rhs_tile, &acc_tile, ftz))
            }
            _ => panic!("Fma requires Tile operands"),
        };
        self.set_value(op.results[0], result);
    }

    /// 8.7.18. cuda_tile.mmaf - Floating-point matrix-multiply-accumulate
    pub fn execute_mmaf(&mut self, op: &Operation) {
        let lhs = self.get_value(op.operands[0]);
        let rhs = self.get_value(op.operands[1]);
        let acc = self.get_value(op.operands[2]);

        let result = match (lhs, rhs, acc) {
            (Value::Tile(lhs_tile), Value::Tile(rhs_tile), Value::Tile(acc_tile)) => {
                Value::Tile(lhs_tile.mmaf(&rhs_tile, &acc_tile))
            }
            _ => panic!("Mmaf requires Tile operands"),
        };
        self.set_value(op.results[0], result);
    }

    // =========================================================================
    // Comparison Operations
    // =========================================================================

    /// 8.7.6. cuda_tile.cmpf - Element-wise floating-point comparison
    pub fn execute_cmpf(&mut self, op: &Operation) {
        let lhs = self.get_value(op.operands[2]); // operands[0] = predicate, [1] = ordering
        let rhs = self.get_value(op.operands[3]);
        let pred = self.extract_comparison_predicate(op);
        let ordering = self.extract_comparison_ordering(op);

        let result = match (lhs, rhs) {
            (Value::Tile(lhs_tile), Value::Tile(rhs_tile)) => {
                lhs_tile.cmpf(&rhs_tile, pred, ordering)
            }
            _ => panic!("Cmpf requires Tile operands"),
        };
        self.set_value(op.results[0], Value::Tile(result));
    }
}

// ============================================================================
// Tile Implementation
// ============================================================================

impl Tile {
    /// Helper: Check that two tiles have matching shapes (no implicit broadcast per general.md)
    fn check_shape_match(&self, other: &Tile) {
        let lhs_shape = self.shape();
        let rhs_shape = other.shape();
        assert_eq!(
            lhs_shape, rhs_shape,
            "Shape mismatch: lhs: {:?} vs rhs: {:?}",
            lhs_shape, rhs_shape
        );
    }

    // ========================================================================
    // Unary Operations
    // ========================================================================

    pub fn absf(&self) -> Self {
        match self {
            Tile::F16(arr) => Tile::F16(arr.mapv(|v| v.abs())),
            Tile::F32(arr) => Tile::F32(arr.mapv(|v| v.abs())),
            Tile::F64(arr) => Tile::F64(arr.mapv(|v| v.abs())),
            _ => panic!("Absf not supported for type {:?}", self.elem_type()),
        }
    }

    pub fn negf(&self) -> Self {
        match self {
            Tile::F16(arr) => Tile::F16(arr.mapv(|v| -v)),
            Tile::F32(arr) => Tile::F32(arr.mapv(|v| -v)),
            Tile::F64(arr) => Tile::F64(arr.mapv(|v| -v)),
            _ => panic!("Negf not supported for type {:?}", self.elem_type()),
        }
    }

    pub fn sqrt(&self, ftz: bool) -> Self {
        match self {
            Tile::F16(arr) => Tile::F16(arr.mapv(|v| Self::flush_subnormal_f16(v, ftz).sqrt())),
            Tile::F32(arr) => Tile::F32(arr.mapv(|v| Self::flush_subnormal_f32(v, ftz).sqrt())),
            Tile::F64(arr) => Tile::F64(arr.mapv(|v| Self::flush_subnormal_f64(v, ftz).sqrt())),
            _ => panic!("Sqrt not supported for type {:?}", self.elem_type()),
        }
    }

    pub fn rsqrt(&self, ftz: bool) -> Self {
        match self {
            Tile::F16(arr) => Tile::F16(arr.mapv(|v| {
                let v = Self::flush_subnormal_f16(v, ftz);
                if v == 0.0f16 {
                    f16::INFINITY
                } else {
                    v.sqrt().recip()
                }
            })),
            Tile::F32(arr) => Tile::F32(arr.mapv(|v| {
                let v = Self::flush_subnormal_f32(v, ftz);
                if v == 0.0 {
                    f32::INFINITY
                } else {
                    v.sqrt().recip()
                }
            })),
            Tile::F64(arr) => Tile::F64(arr.mapv(|v| {
                let v = Self::flush_subnormal_f64(v, ftz);
                if v == 0.0 {
                    f64::INFINITY
                } else {
                    v.sqrt().recip()
                }
            })),
            _ => panic!("Rsqrt not supported for type {:?}", self.elem_type()),
        }
    }

    pub fn sin(&self) -> Self {
        match self {
            Tile::F16(arr) => Tile::F16(arr.mapv(|v| v.sin())),
            Tile::F32(arr) => Tile::F32(arr.mapv(|v| v.sin())),
            Tile::F64(arr) => Tile::F64(arr.mapv(|v| v.sin())),
            _ => panic!("Sin not supported for type {:?}", self.elem_type()),
        }
    }

    pub fn cos(&self) -> Self {
        match self {
            Tile::F16(arr) => Tile::F16(arr.mapv(|v| v.cos())),
            Tile::F32(arr) => Tile::F32(arr.mapv(|v| v.cos())),
            Tile::F64(arr) => Tile::F64(arr.mapv(|v| v.cos())),
            _ => panic!("Cos not supported for type {:?}", self.elem_type()),
        }
    }

    pub fn tan(&self) -> Self {
        match self {
            Tile::F16(arr) => Tile::F16(arr.mapv(|v| v.tan())),
            Tile::F32(arr) => Tile::F32(arr.mapv(|v| v.tan())),
            Tile::F64(arr) => Tile::F64(arr.mapv(|v| v.tan())),
            _ => panic!("Tan not supported for type {:?}", self.elem_type()),
        }
    }

    pub fn sinh(&self) -> Self {
        match self {
            Tile::F16(arr) => Tile::F16(arr.mapv(|v| v.sinh())),
            Tile::F32(arr) => Tile::F32(arr.mapv(|v| v.sinh())),
            Tile::F64(arr) => Tile::F64(arr.mapv(|v| v.sinh())),
            _ => panic!("Sinh not supported for type {:?}", self.elem_type()),
        }
    }

    pub fn cosh(&self) -> Self {
        match self {
            Tile::F16(arr) => Tile::F16(arr.mapv(|v| v.cosh())),
            Tile::F32(arr) => Tile::F32(arr.mapv(|v| v.cosh())),
            Tile::F64(arr) => Tile::F64(arr.mapv(|v| v.cosh())),
            _ => panic!("Cosh not supported for type {:?}", self.elem_type()),
        }
    }

    pub fn tanh(&self) -> Self {
        match self {
            Tile::F16(arr) => Tile::F16(arr.mapv(|v| v.tanh())),
            Tile::F32(arr) => Tile::F32(arr.mapv(|v| v.tanh())),
            Tile::F64(arr) => Tile::F64(arr.mapv(|v| v.tanh())),
            _ => panic!("Tanh not supported for type {:?}", self.elem_type()),
        }
    }

    pub fn exp(&self) -> Self {
        match self {
            Tile::F16(arr) => Tile::F16(arr.mapv(|v| v.exp())),
            Tile::F32(arr) => Tile::F32(arr.mapv(|v| v.exp())),
            Tile::F64(arr) => Tile::F64(arr.mapv(|v| v.exp())),
            _ => panic!("Exp not supported for type {:?}", self.elem_type()),
        }
    }

    pub fn exp2(&self, ftz: bool) -> Self {
        match self {
            Tile::F16(arr) => Tile::F16(arr.mapv(|v| Self::flush_subnormal_f16(v, ftz).exp2())),
            Tile::F32(arr) => Tile::F32(arr.mapv(|v| Self::flush_subnormal_f32(v, ftz).exp2())),
            Tile::F64(arr) => Tile::F64(arr.mapv(|v| Self::flush_subnormal_f64(v, ftz).exp2())),
            _ => panic!("Exp2 not supported for type {:?}", self.elem_type()),
        }
    }

    pub fn log(&self) -> Self {
        match self {
            Tile::F16(arr) => Tile::F16(arr.mapv(|v| v.ln())),
            Tile::F32(arr) => Tile::F32(arr.mapv(|v| v.ln())),
            Tile::F64(arr) => Tile::F64(arr.mapv(|v| v.ln())),
            _ => panic!("Log not supported for type {:?}", self.elem_type()),
        }
    }

    pub fn log2(&self) -> Self {
        match self {
            Tile::F16(arr) => Tile::F16(arr.mapv(|v| v.log2())),
            Tile::F32(arr) => Tile::F32(arr.mapv(|v| v.log2())),
            Tile::F64(arr) => Tile::F64(arr.mapv(|v| v.log2())),
            _ => panic!("Log2 not supported for type {:?}", self.elem_type()),
        }
    }

    pub fn floor(&self) -> Self {
        match self {
            Tile::F16(arr) => Tile::F16(arr.mapv(|v| v.floor())),
            Tile::F32(arr) => Tile::F32(arr.mapv(|v| v.floor())),
            Tile::F64(arr) => Tile::F64(arr.mapv(|v| v.floor())),
            _ => panic!("Floor not supported for type {:?}", self.elem_type()),
        }
    }

    pub fn ceil(&self) -> Self {
        match self {
            Tile::F16(arr) => Tile::F16(arr.mapv(|v| v.ceil())),
            Tile::F32(arr) => Tile::F32(arr.mapv(|v| v.ceil())),
            Tile::F64(arr) => Tile::F64(arr.mapv(|v| v.ceil())),
            _ => panic!("Ceil not supported for type {:?}", self.elem_type()),
        }
    }

    // ========================================================================
    // Binary Operations
    // ========================================================================

    pub fn addf(&self, rhs: &Tile, ftz: bool) -> Self {
        self.check_shape_match(rhs);
        match (self, rhs) {
            (Tile::F16(a), Tile::F16(b)) => Tile::F16(
                Zip::from(a)
                    .and(b)
                    .map_collect(|&x, &y| Self::flush_subnormal_f16(x + y, ftz)),
            ),
            (Tile::F32(a), Tile::F32(b)) => Tile::F32(
                Zip::from(a)
                    .and(b)
                    .map_collect(|&x, &y| Self::flush_subnormal_f32(x + y, ftz)),
            ),
            (Tile::F64(a), Tile::F64(b)) => Tile::F64(
                Zip::from(a)
                    .and(b)
                    .map_collect(|&x, &y| Self::flush_subnormal_f64(x + y, ftz)),
            ),
            _ => panic!("Addf requires matching float types"),
        }
    }

    pub fn subf(&self, rhs: &Tile, ftz: bool) -> Self {
        self.check_shape_match(rhs);
        match (self, rhs) {
            (Tile::F16(a), Tile::F16(b)) => Tile::F16(
                Zip::from(a)
                    .and(b)
                    .map_collect(|&x, &y| Self::flush_subnormal_f16(x - y, ftz)),
            ),
            (Tile::F32(a), Tile::F32(b)) => Tile::F32(
                Zip::from(a)
                    .and(b)
                    .map_collect(|&x, &y| Self::flush_subnormal_f32(x - y, ftz)),
            ),
            (Tile::F64(a), Tile::F64(b)) => Tile::F64(
                Zip::from(a)
                    .and(b)
                    .map_collect(|&x, &y| Self::flush_subnormal_f64(x - y, ftz)),
            ),
            _ => panic!("Subf requires matching float types"),
        }
    }

    pub fn mulf(&self, rhs: &Tile, ftz: bool) -> Self {
        self.check_shape_match(rhs);
        match (self, rhs) {
            (Tile::F16(a), Tile::F16(b)) => Tile::F16(
                Zip::from(a)
                    .and(b)
                    .map_collect(|&x, &y| Self::flush_subnormal_f16(x * y, ftz)),
            ),
            (Tile::F32(a), Tile::F32(b)) => Tile::F32(
                Zip::from(a)
                    .and(b)
                    .map_collect(|&x, &y| Self::flush_subnormal_f32(x * y, ftz)),
            ),
            (Tile::F64(a), Tile::F64(b)) => Tile::F64(
                Zip::from(a)
                    .and(b)
                    .map_collect(|&x, &y| Self::flush_subnormal_f64(x * y, ftz)),
            ),
            _ => panic!("Mulf requires matching float types"),
        }
    }

    pub fn divf(&self, rhs: &Tile, ftz: bool) -> Self {
        self.check_shape_match(rhs);
        match (self, rhs) {
            (Tile::F16(a), Tile::F16(b)) => Tile::F16(
                Zip::from(a)
                    .and(b)
                    .map_collect(|&x, &y| Self::flush_subnormal_f16(x / y, ftz)),
            ),
            (Tile::F32(a), Tile::F32(b)) => Tile::F32(
                Zip::from(a)
                    .and(b)
                    .map_collect(|&x, &y| Self::flush_subnormal_f32(x / y, ftz)),
            ),
            (Tile::F64(a), Tile::F64(b)) => Tile::F64(
                Zip::from(a)
                    .and(b)
                    .map_collect(|&x, &y| Self::flush_subnormal_f64(x / y, ftz)),
            ),
            _ => panic!("Divf requires matching float types"),
        }
    }

    pub fn remf(&self, rhs: &Tile) -> Self {
        self.check_shape_match(rhs);
        match (self, rhs) {
            (Tile::F16(a), Tile::F16(b)) => {
                Tile::F16(Zip::from(a).and(b).map_collect(|&x, &y| x % y))
            }
            (Tile::F32(a), Tile::F32(b)) => {
                Tile::F32(Zip::from(a).and(b).map_collect(|&x, &y| x % y))
            }
            (Tile::F64(a), Tile::F64(b)) => {
                Tile::F64(Zip::from(a).and(b).map_collect(|&x, &y| x % y))
            }
            _ => panic!("Remf requires matching float types"),
        }
    }

    pub fn minf(&self, rhs: &Tile, propagate_nan: bool, ftz: bool) -> Self {
        self.check_shape_match(rhs);
        match (self, rhs) {
            (Tile::F16(a), Tile::F16(b)) => Tile::F16(Zip::from(a).and(b).map_collect(|&x, &y| {
                let x = Self::flush_subnormal_f16(x, ftz);
                let y = Self::flush_subnormal_f16(y, ftz);
                if propagate_nan {
                    if x.is_nan() || y.is_nan() {
                        f16::NAN
                    } else {
                        x.min(y)
                    }
                } else {
                    match (x.is_nan(), y.is_nan()) {
                        (true, true) => f16::NAN,
                        (true, false) => y,
                        (false, true) => x,
                        (false, false) => x.min(y),
                    }
                }
            })),
            (Tile::F32(a), Tile::F32(b)) => Tile::F32(Zip::from(a).and(b).map_collect(|&x, &y| {
                let x = Self::flush_subnormal_f32(x, ftz);
                let y = Self::flush_subnormal_f32(y, ftz);
                if propagate_nan {
                    if x.is_nan() || y.is_nan() {
                        f32::NAN
                    } else {
                        x.min(y)
                    }
                } else {
                    match (x.is_nan(), y.is_nan()) {
                        (true, true) => f32::NAN,
                        (true, false) => y,
                        (false, true) => x,
                        (false, false) => x.min(y),
                    }
                }
            })),
            (Tile::F64(a), Tile::F64(b)) => Tile::F64(Zip::from(a).and(b).map_collect(|&x, &y| {
                let x = Self::flush_subnormal_f64(x, ftz);
                let y = Self::flush_subnormal_f64(y, ftz);
                if propagate_nan {
                    if x.is_nan() || y.is_nan() {
                        f64::NAN
                    } else {
                        x.min(y)
                    }
                } else {
                    match (x.is_nan(), y.is_nan()) {
                        (true, true) => f64::NAN,
                        (true, false) => y,
                        (false, true) => x,
                        (false, false) => x.min(y),
                    }
                }
            })),
            _ => panic!("Minf requires matching float types"),
        }
    }

    pub fn maxf(&self, rhs: &Tile, propagate_nan: bool, ftz: bool) -> Self {
        self.check_shape_match(rhs);
        match (self, rhs) {
            (Tile::F16(a), Tile::F16(b)) => Tile::F16(Zip::from(a).and(b).map_collect(|&x, &y| {
                let x = Self::flush_subnormal_f16(x, ftz);
                let y = Self::flush_subnormal_f16(y, ftz);
                if propagate_nan {
                    if x.is_nan() || y.is_nan() {
                        f16::NAN
                    } else {
                        x.max(y)
                    }
                } else {
                    match (x.is_nan(), y.is_nan()) {
                        (true, true) => f16::NAN,
                        (true, false) => y,
                        (false, true) => x,
                        (false, false) => x.max(y),
                    }
                }
            })),
            (Tile::F32(a), Tile::F32(b)) => Tile::F32(Zip::from(a).and(b).map_collect(|&x, &y| {
                let x = Self::flush_subnormal_f32(x, ftz);
                let y = Self::flush_subnormal_f32(y, ftz);
                if propagate_nan {
                    if x.is_nan() || y.is_nan() {
                        f32::NAN
                    } else {
                        x.max(y)
                    }
                } else {
                    match (x.is_nan(), y.is_nan()) {
                        (true, true) => f32::NAN,
                        (true, false) => y,
                        (false, true) => x,
                        (false, false) => x.max(y),
                    }
                }
            })),
            (Tile::F64(a), Tile::F64(b)) => Tile::F64(Zip::from(a).and(b).map_collect(|&x, &y| {
                let x = Self::flush_subnormal_f64(x, ftz);
                let y = Self::flush_subnormal_f64(y, ftz);
                if propagate_nan {
                    if x.is_nan() || y.is_nan() {
                        f64::NAN
                    } else {
                        x.max(y)
                    }
                } else {
                    match (x.is_nan(), y.is_nan()) {
                        (true, true) => f64::NAN,
                        (true, false) => y,
                        (false, true) => x,
                        (false, false) => x.max(y),
                    }
                }
            })),
            _ => panic!("Maxf requires matching float types"),
        }
    }

    pub fn pow(&self, exp: &Tile) -> Self {
        self.check_shape_match(exp);
        match (self, exp) {
            (Tile::F16(a), Tile::F16(b)) => {
                Tile::F16(Zip::from(a).and(b).map_collect(|&x, &y| x.powf(y)))
            }
            (Tile::F32(a), Tile::F32(b)) => {
                Tile::F32(Zip::from(a).and(b).map_collect(|&x, &y| x.powf(y)))
            }
            (Tile::F64(a), Tile::F64(b)) => {
                Tile::F64(Zip::from(a).and(b).map_collect(|&x, &y| x.powf(y)))
            }
            _ => panic!("Pow requires matching float types"),
        }
    }

    // ========================================================================
    // Ternary Operations
    // ========================================================================

    pub fn fma(&self, rhs: &Tile, acc: &Tile, ftz: bool) -> Self {
        // Per precision.md: multiply in a/b type, cast to c type, add in c type
        self.check_shape_match(rhs);
        let (_, acc) = self.check_shape_match_return(rhs, acc);

        match (self, rhs, acc) {
            (Tile::F16(a), Tile::F16(b), Tile::F16(_c)) => {
                // Multiply in F16, add in F16
                let mul_tile = Tile::F16(
                    Zip::from(a)
                        .and(b)
                        .map_collect(|&x, &y| Self::flush_subnormal_f16(x * y, ftz)),
                );
                mul_tile.addf(acc, ftz)
            }
            (Tile::F32(a), Tile::F32(b), Tile::F32(_c)) => {
                let mul_tile = Tile::F32(
                    Zip::from(a)
                        .and(b)
                        .map_collect(|&x, &y| Self::flush_subnormal_f32(x * y, ftz)),
                );
                mul_tile.addf(acc, ftz)
            }
            (Tile::F64(a), Tile::F64(b), Tile::F64(_c)) => {
                let mul_tile = Tile::F64(
                    Zip::from(a)
                        .and(b)
                        .map_collect(|&x, &y| Self::flush_subnormal_f64(x * y, ftz)),
                );
                mul_tile.addf(acc, ftz)
            }
            _ => panic!("Fma requires matching float types for all operands"),
        }
    }

    pub fn mmaf(&self, rhs: &Tile, acc: &Tile) -> Self {
        // Matrix multiply: (M x K) * (K x N) + (M x N) = (M x N)
        // Or batched: (B x M x K) * (B x K x N) + (B x M x N) = (B x M x N)

        match (self, rhs, acc) {
            (Tile::F16(a), Tile::F16(b), Tile::F16(c)) => {
                let a_shape = a.shape();
                let b_shape = b.shape();
                let c_shape = c.shape();

                // Determine if batched or unbatched
                let is_batched = a_shape.len() == 3;

                if !is_batched {
                    // Unbatched: (M x K) * (K x N) + (M x N)
                    let m = a_shape[0];
                    let k = a_shape[1];
                    let n = b_shape[1];

                    assert_eq!(
                        b_shape[0], k,
                        "MMAF: lhs inner dim must match rhs outer dim"
                    );
                    assert_eq!(c_shape, &[m, n], "MMAF: accumulator shape mismatch");

                    self.matmul_unbatched_f16(a, b, c, m, k, n)
                } else {
                    // Batched: (B x M x K) * (B x K x N) + (B x M x N)
                    let bsz = a_shape[0];
                    let m = a_shape[1];
                    let k = a_shape[2];
                    let n = b_shape[2];

                    assert_eq!(b_shape[0], bsz, "MMAF: batch size mismatch");
                    assert_eq!(
                        b_shape[1], k,
                        "MMAF: lhs inner dim must match rhs outer dim"
                    );
                    assert_eq!(c_shape, &[bsz, m, n], "MMAF: accumulator shape mismatch");

                    self.matmul_batched_f16(a, b, c, bsz, m, k, n)
                }
            }
            (Tile::F32(a), Tile::F32(b), Tile::F32(c)) => {
                let a_shape = a.shape();
                let b_shape = b.shape();
                let c_shape = c.shape();

                let is_batched = a_shape.len() == 3;

                if !is_batched {
                    let m = a_shape[0];
                    let k = a_shape[1];
                    let n = b_shape[1];

                    assert_eq!(
                        b_shape[0], k,
                        "MMAF: lhs inner dim must match rhs outer dim"
                    );
                    assert_eq!(c_shape, &[m, n], "MMAF: accumulator shape mismatch");

                    self.matmul_unbatched_f32(a, b, c, m, k, n)
                } else {
                    let bsz = a_shape[0];
                    let m = a_shape[1];
                    let k = a_shape[2];
                    let n = b_shape[2];

                    assert_eq!(b_shape[0], bsz, "MMAF: batch size mismatch");
                    assert_eq!(
                        b_shape[1], k,
                        "MMAF: lhs inner dim must match rhs outer dim"
                    );
                    assert_eq!(c_shape, &[bsz, m, n], "MMAF: accumulator shape mismatch");

                    self.matmul_batched_f32(a, b, c, bsz, m, k, n)
                }
            }
            (Tile::F64(a), Tile::F64(b), Tile::F64(c)) => {
                let a_shape = a.shape();
                let b_shape = b.shape();
                let c_shape = c.shape();

                let is_batched = a_shape.len() == 3;

                if !is_batched {
                    let m = a_shape[0];
                    let k = a_shape[1];
                    let n = b_shape[1];

                    assert_eq!(
                        b_shape[0], k,
                        "MMAF: lhs inner dim must match rhs outer dim"
                    );
                    assert_eq!(c_shape, &[m, n], "MMAF: accumulator shape mismatch");

                    self.matmul_unbatched_f64(a, b, c, m, k, n)
                } else {
                    let bsz = a_shape[0];
                    let m = a_shape[1];
                    let k = a_shape[2];
                    let n = b_shape[2];

                    assert_eq!(b_shape[0], bsz, "MMAF: batch size mismatch");
                    assert_eq!(
                        b_shape[1], k,
                        "MMAF: lhs inner dim must match rhs outer dim"
                    );
                    assert_eq!(c_shape, &[bsz, m, n], "MMAF: accumulator shape mismatch");

                    self.matmul_batched_f64(a, b, c, bsz, m, k, n)
                }
            }
            // Mixed-precision: F16 * F16 + F32 = F32
            (Tile::F16(a), Tile::F16(b), Tile::F32(c)) => {
                let a_shape = a.shape();
                let b_shape = b.shape();
                let c_shape = c.shape();

                let is_batched = a_shape.len() == 3;

                if !is_batched {
                    let m = a_shape[0];
                    let k = a_shape[1];
                    let n = b_shape[1];

                    assert_eq!(
                        b_shape[0], k,
                        "MMAF: lhs inner dim must match rhs outer dim"
                    );
                    assert_eq!(c_shape, &[m, n], "MMAF: accumulator shape mismatch");

                    self.matmul_unbatched_f16_f32(a, b, c, m, k, n)
                } else {
                    let bsz = a_shape[0];
                    let m = a_shape[1];
                    let k = a_shape[2];
                    let n = b_shape[2];

                    assert_eq!(b_shape[0], bsz, "MMAF: batch size mismatch");
                    assert_eq!(
                        b_shape[1], k,
                        "MMAF: lhs inner dim must match rhs outer dim"
                    );
                    assert_eq!(c_shape, &[bsz, m, n], "MMAF: accumulator shape mismatch");

                    self.matmul_batched_f16_f32(a, b, c, bsz, m, k, n)
                }
            }
            // Mixed-precision: F16 * F16 + F64 = F64
            (Tile::F16(a), Tile::F16(b), Tile::F64(c)) => {
                let a_shape = a.shape();
                let b_shape = b.shape();
                let c_shape = c.shape();

                let is_batched = a_shape.len() == 3;

                if !is_batched {
                    let m = a_shape[0];
                    let k = a_shape[1];
                    let n = b_shape[1];

                    assert_eq!(
                        b_shape[0], k,
                        "MMAF: lhs inner dim must match rhs outer dim"
                    );
                    assert_eq!(c_shape, &[m, n], "MMAF: accumulator shape mismatch");

                    self.matmul_unbatched_f16_f64(a, b, c, m, k, n)
                } else {
                    let bsz = a_shape[0];
                    let m = a_shape[1];
                    let k = a_shape[2];
                    let n = b_shape[2];

                    assert_eq!(b_shape[0], bsz, "MMAF: batch size mismatch");
                    assert_eq!(
                        b_shape[1], k,
                        "MMAF: lhs inner dim must match rhs outer dim"
                    );
                    assert_eq!(c_shape, &[bsz, m, n], "MMAF: accumulator shape mismatch");

                    self.matmul_batched_f16_f64(a, b, c, bsz, m, k, n)
                }
            }
            // Mixed-precision: F32 * F32 + F64 = F64
            (Tile::F32(a), Tile::F32(b), Tile::F64(c)) => {
                let a_shape = a.shape();
                let b_shape = b.shape();
                let c_shape = c.shape();

                let is_batched = a_shape.len() == 3;

                if !is_batched {
                    let m = a_shape[0];
                    let k = a_shape[1];
                    let n = b_shape[1];

                    assert_eq!(
                        b_shape[0], k,
                        "MMAF: lhs inner dim must match rhs outer dim"
                    );
                    assert_eq!(c_shape, &[m, n], "MMAF: accumulator shape mismatch");

                    self.matmul_unbatched_f32_f64(a, b, c, m, k, n)
                } else {
                    let bsz = a_shape[0];
                    let m = a_shape[1];
                    let k = a_shape[2];
                    let n = b_shape[2];

                    assert_eq!(b_shape[0], bsz, "MMAF: batch size mismatch");
                    assert_eq!(
                        b_shape[1], k,
                        "MMAF: lhs inner dim must match rhs outer dim"
                    );
                    assert_eq!(c_shape, &[bsz, m, n], "MMAF: accumulator shape mismatch");

                    self.matmul_batched_f32_f64(a, b, c, bsz, m, k, n)
                }
            }
            _ => panic!("Mmaf requires lhs/rhs types to match and be <= accumulator precision"),
        }
    }

    fn matmul_unbatched_f16(
        &self,
        a: &ndarray::Array<f16, ndarray::IxDyn>,
        b: &ndarray::Array<f16, ndarray::IxDyn>,
        c: &ndarray::Array<f16, ndarray::IxDyn>,
        m: usize,
        k: usize,
        n: usize,
    ) -> Tile {
        let mut result: ndarray::Array<f16, _> =
            unsafe { ndarray::Array::uninit(ndarray::IxDyn(&[m, n])).assume_init() };
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f64;
                for kk in 0..k {
                    sum += a[[i, kk]] as f64 * b[[kk, j]] as f64;
                }
                result[[i, j]] = (sum + c[[i, j]] as f64) as f16;
            }
        }
        Tile::F16(result)
    }

    fn matmul_unbatched_f32(
        &self,
        a: &ndarray::Array<f32, ndarray::IxDyn>,
        b: &ndarray::Array<f32, ndarray::IxDyn>,
        c: &ndarray::Array<f32, ndarray::IxDyn>,
        m: usize,
        k: usize,
        n: usize,
    ) -> Tile {
        let mut result: ndarray::Array<f32, _> =
            unsafe { ndarray::Array::uninit(ndarray::IxDyn(&[m, n])).assume_init() };
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f64;
                for kk in 0..k {
                    sum += a[[i, kk]] as f64 * b[[kk, j]] as f64;
                }
                result[[i, j]] = (sum + c[[i, j]] as f64) as f32;
            }
        }
        Tile::F32(result)
    }

    fn matmul_unbatched_f64(
        &self,
        a: &ndarray::Array<f64, ndarray::IxDyn>,
        b: &ndarray::Array<f64, ndarray::IxDyn>,
        c: &ndarray::Array<f64, ndarray::IxDyn>,
        m: usize,
        k: usize,
        n: usize,
    ) -> Tile {
        let mut result: ndarray::Array<f64, _> =
            unsafe { ndarray::Array::uninit(ndarray::IxDyn(&[m, n])).assume_init() };
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f64;
                for kk in 0..k {
                    sum += a[[i, kk]] * b[[kk, j]];
                }
                result[[i, j]] = sum + c[[i, j]];
            }
        }
        Tile::F64(result)
    }

    fn matmul_batched_f16(
        &self,
        a: &ndarray::Array<f16, ndarray::IxDyn>,
        b: &ndarray::Array<f16, ndarray::IxDyn>,
        c: &ndarray::Array<f16, ndarray::IxDyn>,
        bsz: usize,
        m: usize,
        k: usize,
        n: usize,
    ) -> Tile {
        let mut result: ndarray::Array<f16, _> =
            unsafe { ndarray::Array::uninit(ndarray::IxDyn(&[bsz, m, n])).assume_init() };
        for batch in 0..bsz {
            for i in 0..m {
                for j in 0..n {
                    let mut sum = 0.0f64;
                    for kk in 0..k {
                        sum += a[[batch, i, kk]] as f64 * b[[batch, kk, j]] as f64;
                    }
                    result[[batch, i, j]] = (sum + c[[batch, i, j]] as f64) as f16;
                }
            }
        }
        Tile::F16(result)
    }

    fn matmul_batched_f32(
        &self,
        a: &ndarray::Array<f32, ndarray::IxDyn>,
        b: &ndarray::Array<f32, ndarray::IxDyn>,
        c: &ndarray::Array<f32, ndarray::IxDyn>,
        bsz: usize,
        m: usize,
        k: usize,
        n: usize,
    ) -> Tile {
        let mut result: ndarray::Array<f32, _> =
            unsafe { ndarray::Array::uninit(ndarray::IxDyn(&[bsz, m, n])).assume_init() };
        for batch in 0..bsz {
            for i in 0..m {
                for j in 0..n {
                    let mut sum = 0.0f64;
                    for kk in 0..k {
                        sum += a[[batch, i, kk]] as f64 * b[[batch, kk, j]] as f64;
                    }
                    result[[batch, i, j]] = (sum + c[[batch, i, j]] as f64) as f32;
                }
            }
        }
        Tile::F32(result)
    }

    fn matmul_batched_f64(
        &self,
        a: &ndarray::Array<f64, ndarray::IxDyn>,
        b: &ndarray::Array<f64, ndarray::IxDyn>,
        c: &ndarray::Array<f64, ndarray::IxDyn>,
        bsz: usize,
        m: usize,
        k: usize,
        n: usize,
    ) -> Tile {
        let mut result: ndarray::Array<f64, _> =
            unsafe { ndarray::Array::uninit(ndarray::IxDyn(&[bsz, m, n])).assume_init() };
        for batch in 0..bsz {
            for i in 0..m {
                for j in 0..n {
                    let mut sum = 0.0f64;
                    for kk in 0..k {
                        sum += a[[batch, i, kk]] * b[[batch, kk, j]];
                    }
                    result[[batch, i, j]] = sum + c[[batch, i, j]];
                }
            }
        }
        Tile::F64(result)
    }

    // Mixed-precision: F16 * F16 + F32
    fn matmul_unbatched_f16_f32(
        &self,
        a: &ndarray::Array<f16, ndarray::IxDyn>,
        b: &ndarray::Array<f16, ndarray::IxDyn>,
        c: &ndarray::Array<f32, ndarray::IxDyn>,
        m: usize,
        k: usize,
        n: usize,
    ) -> Tile {
        let mut result: ndarray::Array<f32, _> =
            unsafe { ndarray::Array::uninit(ndarray::IxDyn(&[m, n])).assume_init() };
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for kk in 0..k {
                    sum += a[[i, kk]] as f32 * b[[kk, j]] as f32;
                }
                result[[i, j]] = sum + c[[i, j]];
            }
        }
        Tile::F32(result)
    }

    fn matmul_batched_f16_f32(
        &self,
        a: &ndarray::Array<f16, ndarray::IxDyn>,
        b: &ndarray::Array<f16, ndarray::IxDyn>,
        c: &ndarray::Array<f32, ndarray::IxDyn>,
        bsz: usize,
        m: usize,
        k: usize,
        n: usize,
    ) -> Tile {
        let mut result: ndarray::Array<f32, _> =
            unsafe { ndarray::Array::uninit(ndarray::IxDyn(&[bsz, m, n])).assume_init() };
        for batch in 0..bsz {
            for i in 0..m {
                for j in 0..n {
                    let mut sum = 0.0f32;
                    for kk in 0..k {
                        sum += a[[batch, i, kk]] as f32 * b[[batch, kk, j]] as f32;
                    }
                    result[[batch, i, j]] = sum + c[[batch, i, j]];
                }
            }
        }
        Tile::F32(result)
    }

    // Mixed-precision: F16 * F16 + F64
    fn matmul_unbatched_f16_f64(
        &self,
        a: &ndarray::Array<f16, ndarray::IxDyn>,
        b: &ndarray::Array<f16, ndarray::IxDyn>,
        c: &ndarray::Array<f64, ndarray::IxDyn>,
        m: usize,
        k: usize,
        n: usize,
    ) -> Tile {
        let mut result: ndarray::Array<f64, _> =
            unsafe { ndarray::Array::uninit(ndarray::IxDyn(&[m, n])).assume_init() };
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f64;
                for kk in 0..k {
                    sum += a[[i, kk]] as f64 * b[[kk, j]] as f64;
                }
                result[[i, j]] = sum + c[[i, j]];
            }
        }
        Tile::F64(result)
    }

    fn matmul_batched_f16_f64(
        &self,
        a: &ndarray::Array<f16, ndarray::IxDyn>,
        b: &ndarray::Array<f16, ndarray::IxDyn>,
        c: &ndarray::Array<f64, ndarray::IxDyn>,
        bsz: usize,
        m: usize,
        k: usize,
        n: usize,
    ) -> Tile {
        let mut result: ndarray::Array<f64, _> =
            unsafe { ndarray::Array::uninit(ndarray::IxDyn(&[bsz, m, n])).assume_init() };
        for batch in 0..bsz {
            for i in 0..m {
                for j in 0..n {
                    let mut sum = 0.0f64;
                    for kk in 0..k {
                        sum += a[[batch, i, kk]] as f64 * b[[batch, kk, j]] as f64;
                    }
                    result[[batch, i, j]] = sum + c[[batch, i, j]];
                }
            }
        }
        Tile::F64(result)
    }

    // Mixed-precision: F32 * F32 + F64
    fn matmul_unbatched_f32_f64(
        &self,
        a: &ndarray::Array<f32, ndarray::IxDyn>,
        b: &ndarray::Array<f32, ndarray::IxDyn>,
        c: &ndarray::Array<f64, ndarray::IxDyn>,
        m: usize,
        k: usize,
        n: usize,
    ) -> Tile {
        let mut result: ndarray::Array<f64, _> =
            unsafe { ndarray::Array::uninit(ndarray::IxDyn(&[m, n])).assume_init() };
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f64;
                for kk in 0..k {
                    sum += a[[i, kk]] as f64 * b[[kk, j]] as f64;
                }
                result[[i, j]] = sum + c[[i, j]];
            }
        }
        Tile::F64(result)
    }

    fn matmul_batched_f32_f64(
        &self,
        a: &ndarray::Array<f32, ndarray::IxDyn>,
        b: &ndarray::Array<f32, ndarray::IxDyn>,
        c: &ndarray::Array<f64, ndarray::IxDyn>,
        bsz: usize,
        m: usize,
        k: usize,
        n: usize,
    ) -> Tile {
        let mut result: ndarray::Array<f64, _> =
            unsafe { ndarray::Array::uninit(ndarray::IxDyn(&[bsz, m, n])).assume_init() };
        for batch in 0..bsz {
            for i in 0..m {
                for j in 0..n {
                    let mut sum = 0.0f64;
                    for kk in 0..k {
                        sum += a[[batch, i, kk]] as f64 * b[[batch, kk, j]] as f64;
                    }
                    result[[batch, i, j]] = sum + c[[batch, i, j]];
                }
            }
        }
        Tile::F64(result)
    }

    // ========================================================================
    // Comparison Operations
    // ========================================================================

    pub fn cmpf(
        &self,
        rhs: &Tile,
        pred: ComparisonPredicate,
        _ordering: ComparisonOrdering,
    ) -> Tile {
        self.check_shape_match(rhs);

        // Note: ordering is ignored for initial implementation
        // Unordered vs ordered affects NaN handling, but we use standard Rust comparisons

        match (self, rhs, pred) {
            (Tile::F16(a), Tile::F16(b), ComparisonPredicate::Equal) => {
                Tile::I1(Zip::from(a).and(b).map_collect(|&x, &y| x == y))
            }
            (Tile::F16(a), Tile::F16(b), ComparisonPredicate::NotEqual) => {
                Tile::I1(Zip::from(a).and(b).map_collect(|&x, &y| x != y))
            }
            (Tile::F16(a), Tile::F16(b), ComparisonPredicate::LessThan) => {
                Tile::I1(Zip::from(a).and(b).map_collect(|&x, &y| x < y))
            }
            (Tile::F16(a), Tile::F16(b), ComparisonPredicate::LessThanOrEqual) => {
                Tile::I1(Zip::from(a).and(b).map_collect(|&x, &y| x <= y))
            }
            (Tile::F16(a), Tile::F16(b), ComparisonPredicate::GreaterThan) => {
                Tile::I1(Zip::from(a).and(b).map_collect(|&x, &y| x > y))
            }
            (Tile::F16(a), Tile::F16(b), ComparisonPredicate::GreaterThanOrEqual) => {
                Tile::I1(Zip::from(a).and(b).map_collect(|&x, &y| x >= y))
            }

            (Tile::F32(a), Tile::F32(b), ComparisonPredicate::Equal) => {
                Tile::I1(Zip::from(a).and(b).map_collect(|&x, &y| x == y))
            }
            (Tile::F32(a), Tile::F32(b), ComparisonPredicate::NotEqual) => {
                Tile::I1(Zip::from(a).and(b).map_collect(|&x, &y| x != y))
            }
            (Tile::F32(a), Tile::F32(b), ComparisonPredicate::LessThan) => {
                Tile::I1(Zip::from(a).and(b).map_collect(|&x, &y| x < y))
            }
            (Tile::F32(a), Tile::F32(b), ComparisonPredicate::LessThanOrEqual) => {
                Tile::I1(Zip::from(a).and(b).map_collect(|&x, &y| x <= y))
            }
            (Tile::F32(a), Tile::F32(b), ComparisonPredicate::GreaterThan) => {
                Tile::I1(Zip::from(a).and(b).map_collect(|&x, &y| x > y))
            }
            (Tile::F32(a), Tile::F32(b), ComparisonPredicate::GreaterThanOrEqual) => {
                Tile::I1(Zip::from(a).and(b).map_collect(|&x, &y| x >= y))
            }

            (Tile::F64(a), Tile::F64(b), ComparisonPredicate::Equal) => {
                Tile::I1(Zip::from(a).and(b).map_collect(|&x, &y| x == y))
            }
            (Tile::F64(a), Tile::F64(b), ComparisonPredicate::NotEqual) => {
                Tile::I1(Zip::from(a).and(b).map_collect(|&x, &y| x != y))
            }
            (Tile::F64(a), Tile::F64(b), ComparisonPredicate::LessThan) => {
                Tile::I1(Zip::from(a).and(b).map_collect(|&x, &y| x < y))
            }
            (Tile::F64(a), Tile::F64(b), ComparisonPredicate::LessThanOrEqual) => {
                Tile::I1(Zip::from(a).and(b).map_collect(|&x, &y| x <= y))
            }
            (Tile::F64(a), Tile::F64(b), ComparisonPredicate::GreaterThan) => {
                Tile::I1(Zip::from(a).and(b).map_collect(|&x, &y| x > y))
            }
            (Tile::F64(a), Tile::F64(b), ComparisonPredicate::GreaterThanOrEqual) => {
                Tile::I1(Zip::from(a).and(b).map_collect(|&x, &y| x >= y))
            }

            _ => panic!("Cmpf requires matching float types"),
        }
    }

    /// Helper: Check shapes match and return the tiles (for ternary ops)
    fn check_shape_match_return<'a>(
        &'a self,
        other: &'a Tile,
        third: &'a Tile,
    ) -> (&'a Tile, &'a Tile) {
        self.check_shape_match(other);
        self.check_shape_match(third);
        (other, third)
    }

    // ========================================================================
    // FTZ Helper Functions (private)
    // ========================================================================

    #[inline]
    fn flush_subnormal_f16(v: f16, ftz: bool) -> f16 {
        if ftz && v.is_subnormal() {
            if v.is_sign_negative() {
                -0.0f16
            } else {
                0.0f16
            }
        } else {
            v
        }
    }

    #[inline]
    fn flush_subnormal_f32(v: f32, ftz: bool) -> f32 {
        if ftz && v.is_subnormal() {
            if v.is_sign_negative() { -0.0 } else { 0.0 }
        } else {
            v
        }
    }

    #[inline]
    fn flush_subnormal_f64(v: f64, ftz: bool) -> f64 {
        if ftz && v.is_subnormal() {
            if v.is_sign_negative() { -0.0 } else { 0.0 }
        } else {
            v
        }
    }
}
