// Integer operations for TileIR interpreter (Section 8.8)
//
// Implements: absi, addi, cmpi, divi, maxi, mini, mmai, muli, mulhii,
//             negi, remi, shli, shri, subi

use crate::cuda_tile_ir::OpAttrKey;
use crate::cuda_tile_ir::attrs::Attr;
use crate::cuda_tile_ir::enums::{ComparisonPredicate, RoundingMode, Signedness};
use crate::cuda_tile_ir::ir::Operation;
use crate::interpreter::data_structures::interpreter::ExecutionContext;
use crate::interpreter::data_structures::tile::Tile;
use crate::interpreter::data_structures::value::Value;
use ndarray::Zip;
use std::ops::Neg;

impl ExecutionContext<'_> {
    // =========================================================================
    // Helper Functions (private)
    // =========================================================================

    fn extract_signedness_int(&self, op: &Operation) -> Signedness {
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

    fn extract_rounding_mode_int(&self, op: &Operation) -> RoundingMode {
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

    fn extract_comparison_predicate_int(&self, op: &Operation) -> ComparisonPredicate {
        let attr_id = op
            .attrs
            .iter()
            .find(|(key, _)| *key == OpAttrKey::ComparisonPredicate)
            .map(|(_, id)| id)
            .expect("CmpI operation missing ComparisonPredicate attribute");
        match self.arena.attr_(*attr_id) {
            Attr::ComparisonPredicate(p) => *p,
            _ => panic!("ComparisonPredicate attribute must be ComparisonPredicate enum"),
        }
    }

    // =========================================================================
    // Unary Operations
    // =========================================================================

    /// 8.8.2. cuda_tile.absi - Element-wise integer absolute value
    pub fn execute_absi(&mut self, op: &Operation) {
        let src = self.get_value(op.operands[0]);
        let result = match src {
            Value::Tile(tile) => Value::Tile(tile.absi()),
            _ => panic!("Absi requires Tile operand"),
        };
        self.set_value(op.results[0], result);
    }

    /// 8.8.11. cuda_tile.negi - Element-wise integer negation
    pub fn execute_negi(&mut self, op: &Operation) {
        let src = self.get_value(op.operands[0]);
        let result = match src {
            Value::Tile(tile) => Value::Tile(tile.negi()),
            _ => panic!("Negi requires Tile operand"),
        };
        self.set_value(op.results[0], result);
    }

    // =========================================================================
    // Binary Operations
    // =========================================================================

    /// 8.8.3. cuda_tile.addi - Element-wise integer addition
    pub fn execute_addi(&mut self, op: &Operation) {
        let lhs = self.get_value(op.operands[0]);
        let rhs = self.get_value(op.operands[1]);
        // Note: overflow attribute exists but is ignored for initial implementation

        let result = match (lhs, rhs) {
            (Value::Tile(lhs_tile), Value::Tile(rhs_tile)) => Value::Tile(lhs_tile.addi(&rhs_tile)),
            _ => panic!("Addi requires Tile operands"),
        };
        self.set_value(op.results[0], result);
    }

    /// 8.8.5. cuda_tile.divi - Element-wise integer division
    pub fn execute_divi(&mut self, op: &Operation) {
        let lhs = self.get_value(op.operands[0]);
        let rhs = self.get_value(op.operands[1]);
        let signedness = self.extract_signedness_int(op);
        let rounding = self.extract_rounding_mode_int(op);

        let result = match (lhs, rhs) {
            (Value::Tile(lhs_tile), Value::Tile(rhs_tile)) => {
                Value::Tile(lhs_tile.divi(&rhs_tile, signedness, rounding))
            }
            _ => panic!("Divi requires Tile operands"),
        };
        self.set_value(op.results[0], result);
    }

    /// 8.8.6. cuda_tile.maxi - Element-wise integer maximum
    pub fn execute_maxi(&mut self, op: &Operation) {
        let lhs = self.get_value(op.operands[0]);
        let rhs = self.get_value(op.operands[1]);
        let signedness = self.extract_signedness_int(op);

        let result = match (lhs, rhs) {
            (Value::Tile(lhs_tile), Value::Tile(rhs_tile)) => {
                Value::Tile(lhs_tile.maxi(&rhs_tile, signedness))
            }
            _ => panic!("Maxi requires Tile operands"),
        };
        self.set_value(op.results[0], result);
    }

    /// 8.8.7. cuda_tile.mini - Element-wise integer minimum
    pub fn execute_mini(&mut self, op: &Operation) {
        let lhs = self.get_value(op.operands[0]);
        let rhs = self.get_value(op.operands[1]);
        let signedness = self.extract_signedness_int(op);

        let result = match (lhs, rhs) {
            (Value::Tile(lhs_tile), Value::Tile(rhs_tile)) => {
                Value::Tile(lhs_tile.mini(&rhs_tile, signedness))
            }
            _ => panic!("Mini requires Tile operands"),
        };
        self.set_value(op.results[0], result);
    }

    /// 8.8.9. cuda_tile.muli - Element-wise integer multiplication
    pub fn execute_muli(&mut self, op: &Operation) {
        let lhs = self.get_value(op.operands[0]);
        let rhs = self.get_value(op.operands[1]);
        // Note: overflow attribute exists but is ignored for initial implementation

        let result = match (lhs, rhs) {
            (Value::Tile(lhs_tile), Value::Tile(rhs_tile)) => Value::Tile(lhs_tile.muli(&rhs_tile)),
            _ => panic!("Muli requires Tile operands"),
        };
        self.set_value(op.results[0], result);
    }

    /// 8.8.10. cuda_tile.mulhii - Element-wise high bits of integer multiplication
    pub fn execute_mulhii(&mut self, op: &Operation) {
        let lhs = self.get_value(op.operands[0]);
        let rhs = self.get_value(op.operands[1]);

        let result = match (lhs, rhs) {
            (Value::Tile(lhs_tile), Value::Tile(rhs_tile)) => {
                Value::Tile(lhs_tile.mulhii(&rhs_tile))
            }
            _ => panic!("Mulhii requires Tile operands"),
        };
        self.set_value(op.results[0], result);
    }

    /// 8.8.12. cuda_tile.remi - Element-wise integer remainder
    pub fn execute_remi(&mut self, op: &Operation) {
        let lhs = self.get_value(op.operands[0]);
        let rhs = self.get_value(op.operands[1]);
        let signedness = self.extract_signedness_int(op);

        let result = match (lhs, rhs) {
            (Value::Tile(lhs_tile), Value::Tile(rhs_tile)) => {
                Value::Tile(lhs_tile.remi(&rhs_tile, signedness))
            }
            _ => panic!("Remi requires Tile operands"),
        };
        self.set_value(op.results[0], result);
    }

    /// 8.8.13. cuda_tile.shli - Element-wise shift-left
    pub fn execute_shli(&mut self, op: &Operation) {
        let lhs = self.get_value(op.operands[0]);
        let rhs = self.get_value(op.operands[1]);
        // Note: overflow attribute exists but is ignored for initial implementation

        let result = match (lhs, rhs) {
            (Value::Tile(lhs_tile), Value::Tile(rhs_tile)) => Value::Tile(lhs_tile.shli(&rhs_tile)),
            _ => panic!("Shli requires Tile operands"),
        };
        self.set_value(op.results[0], result);
    }

    /// 8.8.14. cuda_tile.shri - Element-wise shift-right
    pub fn execute_shri(&mut self, op: &Operation) {
        let lhs = self.get_value(op.operands[0]);
        let rhs = self.get_value(op.operands[1]);
        let signedness = self.extract_signedness_int(op);

        let result = match (lhs, rhs) {
            (Value::Tile(lhs_tile), Value::Tile(rhs_tile)) => {
                Value::Tile(lhs_tile.shri(&rhs_tile, signedness))
            }
            _ => panic!("Shri requires Tile operands"),
        };
        self.set_value(op.results[0], result);
    }

    /// 8.8.15. cuda_tile.subi - Element-wise integer subtraction
    pub fn execute_subi(&mut self, op: &Operation) {
        let lhs = self.get_value(op.operands[0]);
        let rhs = self.get_value(op.operands[1]);
        // Note: overflow attribute exists but is ignored for initial implementation

        let result = match (lhs, rhs) {
            (Value::Tile(lhs_tile), Value::Tile(rhs_tile)) => Value::Tile(lhs_tile.subi(&rhs_tile)),
            _ => panic!("Subi requires Tile operands"),
        };
        self.set_value(op.results[0], result);
    }

    // =========================================================================
    // Ternary Operations
    // =========================================================================

    /// 8.8.8. cuda_tile.mmai - Integer matrix-multiply-accumulate
    pub fn execute_mmai(&mut self, op: &Operation) {
        use crate::cuda_tile_ir::OpAttrKey;
        use crate::cuda_tile_ir::attrs::Attr;

        let lhs = self.get_value(op.operands[0]);
        let rhs = self.get_value(op.operands[1]);
        let acc = self.get_value(op.operands[2]);

        let signedness_lhs = op
            .attrs
            .iter()
            .find(|(key, _)| *key == OpAttrKey::SignednessLhs)
            .map(|(_, id)| self.arena.attr_(*id))
            .map(|attr| match attr {
                Attr::Signedness(s) => *s,
                _ => panic!("SignednessLhs attribute must be Signedness enum"),
            })
            .unwrap_or(Signedness::Signed);

        let signedness_rhs = op
            .attrs
            .iter()
            .find(|(key, _)| *key == OpAttrKey::SignednessRhs)
            .map(|(_, id)| self.arena.attr_(*id))
            .map(|attr| match attr {
                Attr::Signedness(s) => *s,
                _ => panic!("SignednessRhs attribute must be Signedness enum"),
            })
            .unwrap_or(Signedness::Signed);

        let result = match (lhs, rhs, acc) {
            (Value::Tile(lhs_tile), Value::Tile(rhs_tile), Value::Tile(acc_tile)) => {
                Value::Tile(lhs_tile.mmai(&rhs_tile, &acc_tile, signedness_lhs, signedness_rhs))
            }
            _ => panic!("Mmai requires Tile operands"),
        };
        self.set_value(op.results[0], result);
    }

    // =========================================================================
    // Comparison Operations
    // =========================================================================

    /// 8.8.4. cuda_tile.cmpi - Element-wise integer comparison
    pub fn execute_cmpi(&mut self, op: &Operation) {
        let lhs = self.get_value(op.operands[1]); // operands[0] = predicate
        let rhs = self.get_value(op.operands[2]);
        let pred = self.extract_comparison_predicate_int(op);
        let signedness = self.extract_signedness_int(op);

        let result = match (lhs, rhs) {
            (Value::Tile(lhs_tile), Value::Tile(rhs_tile)) => {
                lhs_tile.cmpi(&rhs_tile, pred, signedness)
            }
            _ => panic!("Cmpi requires Tile operands"),
        };
        self.set_value(op.results[0], Value::Tile(result));
    }
}

// ============================================================================
// Tile Implementation
// ============================================================================

impl Tile {
    /// Helper: Check that two tiles have matching shapes (no implicit broadcast per general.md)
    fn check_shape_match_int(&self, other: &Tile) {
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

    pub fn absi(&self) -> Self {
        match self {
            Tile::I1(arr) => Tile::I1(arr.mapv(|v| v)),
            Tile::I8(arr) => Tile::I8(arr.mapv(|v| v.abs())),
            Tile::I16(arr) => Tile::I16(arr.mapv(|v| v.abs())),
            Tile::I32(arr) => Tile::I32(arr.mapv(|v| v.abs())),
            Tile::I64(arr) => Tile::I64(arr.mapv(|v| v.abs())),
            _ => panic!("Absi not supported for type {:?}", self.elem_type()),
        }
    }

    pub fn negi(&self) -> Self {
        match self {
            Tile::I1(arr) => Tile::I1(arr.mapv(|v| !v)),
            Tile::I8(arr) => Tile::I8(arr.mapv(|v| v.neg())),
            Tile::I16(arr) => Tile::I16(arr.mapv(|v| v.neg())),
            Tile::I32(arr) => Tile::I32(arr.mapv(|v| v.neg())),
            Tile::I64(arr) => Tile::I64(arr.mapv(|v| v.neg())),
            _ => panic!("Negi not supported for type {:?}", self.elem_type()),
        }
    }

    // ========================================================================
    // Binary Operations
    // ========================================================================

    pub fn addi(&self, rhs: &Tile) -> Self {
        self.check_shape_match_int(rhs);
        match (self, rhs) {
            (Tile::I1(a), Tile::I1(b)) => Tile::I1(Zip::from(a).and(b).map_collect(|&x, &y| x ^ y)),
            (Tile::I8(a), Tile::I8(b)) => {
                Tile::I8(Zip::from(a).and(b).map_collect(|&x, &y| x.wrapping_add(y)))
            }
            (Tile::I16(a), Tile::I16(b)) => {
                Tile::I16(Zip::from(a).and(b).map_collect(|&x, &y| x.wrapping_add(y)))
            }
            (Tile::I32(a), Tile::I32(b)) => {
                Tile::I32(Zip::from(a).and(b).map_collect(|&x, &y| x.wrapping_add(y)))
            }
            (Tile::I64(a), Tile::I64(b)) => {
                Tile::I64(Zip::from(a).and(b).map_collect(|&x, &y| x.wrapping_add(y)))
            }
            _ => panic!("Addi requires matching integer types"),
        }
    }

    pub fn subi(&self, rhs: &Tile) -> Self {
        self.check_shape_match_int(rhs);
        match (self, rhs) {
            (Tile::I1(a), Tile::I1(b)) => Tile::I1(Zip::from(a).and(b).map_collect(|&x, &y| x ^ y)),
            (Tile::I8(a), Tile::I8(b)) => {
                Tile::I8(Zip::from(a).and(b).map_collect(|&x, &y| x.wrapping_sub(y)))
            }
            (Tile::I16(a), Tile::I16(b)) => {
                Tile::I16(Zip::from(a).and(b).map_collect(|&x, &y| x.wrapping_sub(y)))
            }
            (Tile::I32(a), Tile::I32(b)) => {
                Tile::I32(Zip::from(a).and(b).map_collect(|&x, &y| x.wrapping_sub(y)))
            }
            (Tile::I64(a), Tile::I64(b)) => {
                Tile::I64(Zip::from(a).and(b).map_collect(|&x, &y| x.wrapping_sub(y)))
            }
            _ => panic!("Subi requires matching integer types"),
        }
    }

    pub fn muli(&self, rhs: &Tile) -> Self {
        self.check_shape_match_int(rhs);
        match (self, rhs) {
            (Tile::I1(a), Tile::I1(b)) => Tile::I1(Zip::from(a).and(b).map_collect(|&x, &y| x & y)),
            (Tile::I8(a), Tile::I8(b)) => {
                Tile::I8(Zip::from(a).and(b).map_collect(|&x, &y| x.wrapping_mul(y)))
            }
            (Tile::I16(a), Tile::I16(b)) => {
                Tile::I16(Zip::from(a).and(b).map_collect(|&x, &y| x.wrapping_mul(y)))
            }
            (Tile::I32(a), Tile::I32(b)) => {
                Tile::I32(Zip::from(a).and(b).map_collect(|&x, &y| x.wrapping_mul(y)))
            }
            (Tile::I64(a), Tile::I64(b)) => {
                Tile::I64(Zip::from(a).and(b).map_collect(|&x, &y| x.wrapping_mul(y)))
            }
            _ => panic!("Muli requires matching integer types"),
        }
    }

    pub fn mulhii(&self, rhs: &Tile) -> Self {
        self.check_shape_match_int(rhs);
        match (self, rhs) {
            (Tile::I1(a), Tile::I1(b)) => Tile::I1(Zip::from(a).and(b).map_collect(|&x, &y| {
                let x_u = x as u8;
                let y_u = y as u8;
                let full = (x_u as u64) * (y_u as u64);
                (full >> 1) as u8 != 0
            })),
            (Tile::I8(a), Tile::I8(b)) => Tile::I8(Zip::from(a).and(b).map_collect(|&x, &y| {
                let x_u = x as u8;
                let y_u = y as u8;
                let full = (x_u as u16) * (y_u as u16);
                (full >> 8) as i8
            })),
            (Tile::I16(a), Tile::I16(b)) => Tile::I16(Zip::from(a).and(b).map_collect(|&x, &y| {
                let x_u = x as u16;
                let y_u = y as u16;
                let full = (x_u as u32) * (y_u as u32);
                (full >> 16) as i16
            })),
            (Tile::I32(a), Tile::I32(b)) => Tile::I32(Zip::from(a).and(b).map_collect(|&x, &y| {
                let x_u = x as u32;
                let y_u = y as u32;
                let full = (x_u as u64) * (y_u as u64);
                (full >> 32) as i32
            })),
            (Tile::I64(a), Tile::I64(b)) => Tile::I64(Zip::from(a).and(b).map_collect(|&x, &y| {
                let x_u = x as u64;
                let y_u = y as u64;
                let full = (x_u as u128) * (y_u as u128);
                (full >> 64) as i64
            })),
            _ => panic!("Mulhii requires matching integer types"),
        }
    }

    pub fn divi(&self, rhs: &Tile, signedness: Signedness, rounding: RoundingMode) -> Self {
        self.check_shape_match_int(rhs);
        match (self, rhs, signedness, rounding) {
            // Unsigned division
            (Tile::I1(a), Tile::I1(b), Signedness::Unsigned, _) => Tile::I1(
                Zip::from(a)
                    .and(b)
                    .map_collect(|&_x, &y| if !y { true } else { false }),
            ),
            (Tile::I8(a), Tile::I8(b), Signedness::Unsigned, _) => Tile::I8(
                Zip::from(a)
                    .and(b)
                    .map_collect(|&x, &y| ((x as u8) / (y as u8)) as i8),
            ),
            (Tile::I16(a), Tile::I16(b), Signedness::Unsigned, _) => Tile::I16(
                Zip::from(a)
                    .and(b)
                    .map_collect(|&x, &y| ((x as u16) / (y as u16)) as i16),
            ),
            (Tile::I32(a), Tile::I32(b), Signedness::Unsigned, _) => Tile::I32(
                Zip::from(a)
                    .and(b)
                    .map_collect(|&x, &y| ((x as u32) / (y as u32)) as i32),
            ),
            (Tile::I64(a), Tile::I64(b), Signedness::Unsigned, _) => Tile::I64(
                Zip::from(a)
                    .and(b)
                    .map_collect(|&x, &y| ((x as u64) / (y as u64)) as i64),
            ),
            // Signed division with rounding towards zero (default)
            (Tile::I8(a), Tile::I8(b), Signedness::Signed, RoundingMode::Zero) => {
                Tile::I8(Zip::from(a).and(b).map_collect(|&x, &y| {
                    if y == 0 {
                        panic!("Division by zero");
                    }
                    x / y
                }))
            }
            (Tile::I16(a), Tile::I16(b), Signedness::Signed, RoundingMode::Zero) => {
                Tile::I16(Zip::from(a).and(b).map_collect(|&x, &y| {
                    if y == 0 {
                        panic!("Division by zero");
                    }
                    x / y
                }))
            }
            (Tile::I32(a), Tile::I32(b), Signedness::Signed, RoundingMode::Zero) => {
                Tile::I32(Zip::from(a).and(b).map_collect(|&x, &y| {
                    if y == 0 {
                        panic!("Division by zero");
                    }
                    x / y
                }))
            }
            (Tile::I64(a), Tile::I64(b), Signedness::Signed, RoundingMode::Zero) => {
                Tile::I64(Zip::from(a).and(b).map_collect(|&x, &y| {
                    if y == 0 {
                        panic!("Division by zero");
                    }
                    x / y
                }))
            }
            // Signed division with rounding towards negative infinity (floor)
            (Tile::I8(a), Tile::I8(b), Signedness::Signed, RoundingMode::NegativeInf) => {
                Tile::I8(Zip::from(a).and(b).map_collect(|&x, &y| {
                    if y == 0 {
                        panic!("Division by zero");
                    }
                    let q = x / y;
                    let r = x % y;
                    if r != 0 && ((x < 0) != (y < 0)) {
                        q - 1
                    } else {
                        q
                    }
                }))
            }
            (Tile::I16(a), Tile::I16(b), Signedness::Signed, RoundingMode::NegativeInf) => {
                Tile::I16(Zip::from(a).and(b).map_collect(|&x, &y| {
                    if y == 0 {
                        panic!("Division by zero");
                    }
                    let q = x / y;
                    let r = x % y;
                    if r != 0 && ((x < 0) != (y < 0)) {
                        q - 1
                    } else {
                        q
                    }
                }))
            }
            (Tile::I32(a), Tile::I32(b), Signedness::Signed, RoundingMode::NegativeInf) => {
                Tile::I32(Zip::from(a).and(b).map_collect(|&x, &y| {
                    if y == 0 {
                        panic!("Division by zero");
                    }
                    let q = x / y;
                    let r = x % y;
                    if r != 0 && ((x < 0) != (y < 0)) {
                        q - 1
                    } else {
                        q
                    }
                }))
            }
            (Tile::I64(a), Tile::I64(b), Signedness::Signed, RoundingMode::NegativeInf) => {
                Tile::I64(Zip::from(a).and(b).map_collect(|&x, &y| {
                    if y == 0 {
                        panic!("Division by zero");
                    }
                    let q = x / y;
                    let r = x % y;
                    if r != 0 && ((x < 0) != (y < 0)) {
                        q - 1
                    } else {
                        q
                    }
                }))
            }
            // Signed division with rounding towards positive infinity (ceil)
            (Tile::I8(a), Tile::I8(b), Signedness::Signed, RoundingMode::PositiveInf) => {
                Tile::I8(Zip::from(a).and(b).map_collect(|&x, &y| {
                    if y == 0 {
                        panic!("Division by zero");
                    }
                    let q = x / y;
                    let r = x % y;
                    if r != 0 && ((x < 0) == (y < 0)) {
                        q + 1
                    } else {
                        q
                    }
                }))
            }
            (Tile::I16(a), Tile::I16(b), Signedness::Signed, RoundingMode::PositiveInf) => {
                Tile::I16(Zip::from(a).and(b).map_collect(|&x, &y| {
                    if y == 0 {
                        panic!("Division by zero");
                    }
                    let q = x / y;
                    let r = x % y;
                    if r != 0 && ((x < 0) == (y < 0)) {
                        q + 1
                    } else {
                        q
                    }
                }))
            }
            (Tile::I32(a), Tile::I32(b), Signedness::Signed, RoundingMode::PositiveInf) => {
                Tile::I32(Zip::from(a).and(b).map_collect(|&x, &y| {
                    if y == 0 {
                        panic!("Division by zero");
                    }
                    let q = x / y;
                    let r = x % y;
                    if r != 0 && ((x < 0) == (y < 0)) {
                        q + 1
                    } else {
                        q
                    }
                }))
            }
            (Tile::I64(a), Tile::I64(b), Signedness::Signed, RoundingMode::PositiveInf) => {
                Tile::I64(Zip::from(a).and(b).map_collect(|&x, &y| {
                    if y == 0 {
                        panic!("Division by zero");
                    }
                    let q = x / y;
                    let r = x % y;
                    if r != 0 && ((x < 0) == (y < 0)) {
                        q + 1
                    } else {
                        q
                    }
                }))
            }
            _ => panic!("Divi: unsupported type/rounding combination"),
        }
    }

    pub fn remi(&self, rhs: &Tile, signedness: Signedness) -> Self {
        self.check_shape_match_int(rhs);
        match (self, rhs, signedness) {
            // Unsigned remainder
            (Tile::I1(a), Tile::I1(b), Signedness::Unsigned) => Tile::I1(
                Zip::from(a)
                    .and(b)
                    .map_collect(|&x, &y| if !y { x } else { false }),
            ),
            (Tile::I8(a), Tile::I8(b), Signedness::Unsigned) => Tile::I8(
                Zip::from(a)
                    .and(b)
                    .map_collect(|&x, &y| ((x as u8) % (y as u8)) as i8),
            ),
            (Tile::I16(a), Tile::I16(b), Signedness::Unsigned) => Tile::I16(
                Zip::from(a)
                    .and(b)
                    .map_collect(|&x, &y| ((x as u16) % (y as u16)) as i16),
            ),
            (Tile::I32(a), Tile::I32(b), Signedness::Unsigned) => Tile::I32(
                Zip::from(a)
                    .and(b)
                    .map_collect(|&x, &y| ((x as u32) % (y as u32)) as i32),
            ),
            (Tile::I64(a), Tile::I64(b), Signedness::Unsigned) => Tile::I64(
                Zip::from(a)
                    .and(b)
                    .map_collect(|&x, &y| ((x as u64) % (y as u64)) as i64),
            ),
            // Signed remainder (truncated division, sign follows dividend)
            (Tile::I8(a), Tile::I8(b), Signedness::Signed) => {
                Tile::I8(Zip::from(a).and(b).map_collect(|&x, &y| {
                    if y == 0 {
                        panic!("Division by zero");
                    }
                    x % y
                }))
            }
            (Tile::I16(a), Tile::I16(b), Signedness::Signed) => {
                Tile::I16(Zip::from(a).and(b).map_collect(|&x, &y| {
                    if y == 0 {
                        panic!("Division by zero");
                    }
                    x % y
                }))
            }
            (Tile::I32(a), Tile::I32(b), Signedness::Signed) => {
                Tile::I32(Zip::from(a).and(b).map_collect(|&x, &y| {
                    if y == 0 {
                        panic!("Division by zero");
                    }
                    x % y
                }))
            }
            (Tile::I64(a), Tile::I64(b), Signedness::Signed) => {
                Tile::I64(Zip::from(a).and(b).map_collect(|&x, &y| {
                    if y == 0 {
                        panic!("Division by zero");
                    }
                    x % y
                }))
            }
            _ => panic!("Remi requires matching integer types"),
        }
    }

    pub fn shli(&self, rhs: &Tile) -> Self {
        self.check_shape_match_int(rhs);
        match (self, rhs) {
            (Tile::I1(a), Tile::I1(b)) => Tile::I1(
                Zip::from(a)
                    .and(b)
                    .map_collect(|&x, &y| if x & !y { true } else { false }),
            ),
            (Tile::I8(a), Tile::I8(b)) => Tile::I8(
                Zip::from(a)
                    .and(b)
                    .map_collect(|&x, &y| x.wrapping_shl((y as u32) & 7)),
            ),
            (Tile::I16(a), Tile::I16(b)) => Tile::I16(
                Zip::from(a)
                    .and(b)
                    .map_collect(|&x, &y| x.wrapping_shl((y as u32) & 15)),
            ),
            (Tile::I32(a), Tile::I32(b)) => Tile::I32(
                Zip::from(a)
                    .and(b)
                    .map_collect(|&x, &y| x.wrapping_shl((y as u32) & 31)),
            ),
            (Tile::I64(a), Tile::I64(b)) => Tile::I64(
                Zip::from(a)
                    .and(b)
                    .map_collect(|&x, &y| x.wrapping_shl((y as u32) & 63)),
            ),
            _ => panic!("Shli requires matching integer types"),
        }
    }

    pub fn shri(&self, rhs: &Tile, signedness: Signedness) -> Self {
        self.check_shape_match_int(rhs);
        match (self, rhs, signedness) {
            // Unsigned (logical shift right, zero-fill)
            (Tile::I1(a), Tile::I1(b), Signedness::Unsigned) => Tile::I1(
                Zip::from(a)
                    .and(b)
                    .map_collect(|&x, &y| if x & !y { true } else { false }),
            ),
            (Tile::I8(a), Tile::I8(b), Signedness::Unsigned) => Tile::I8(
                Zip::from(a)
                    .and(b)
                    .map_collect(|&x, &y| ((x as u8) >> ((y as u8) & 7)) as i8),
            ),
            (Tile::I16(a), Tile::I16(b), Signedness::Unsigned) => Tile::I16(
                Zip::from(a)
                    .and(b)
                    .map_collect(|&x, &y| ((x as u16) >> ((y as u32) & 15)) as i16),
            ),
            (Tile::I32(a), Tile::I32(b), Signedness::Unsigned) => Tile::I32(
                Zip::from(a)
                    .and(b)
                    .map_collect(|&x, &y| ((x as u32) >> ((y as u32) & 31)) as i32),
            ),
            (Tile::I64(a), Tile::I64(b), Signedness::Unsigned) => Tile::I64(
                Zip::from(a)
                    .and(b)
                    .map_collect(|&x, &y| ((x as u64) >> ((y as u32) & 63)) as i64),
            ),
            // Signed (arithmetic shift right, sign-fill)
            (Tile::I8(a), Tile::I8(b), Signedness::Signed) => Tile::I8(
                Zip::from(a)
                    .and(b)
                    .map_collect(|&x, &y| x.wrapping_shr((y as u32) & 7)),
            ),
            (Tile::I16(a), Tile::I16(b), Signedness::Signed) => Tile::I16(
                Zip::from(a)
                    .and(b)
                    .map_collect(|&x, &y| x.wrapping_shr((y as u32) & 15)),
            ),
            (Tile::I32(a), Tile::I32(b), Signedness::Signed) => Tile::I32(
                Zip::from(a)
                    .and(b)
                    .map_collect(|&x, &y| x.wrapping_shr((y as u32) & 31)),
            ),
            (Tile::I64(a), Tile::I64(b), Signedness::Signed) => Tile::I64(
                Zip::from(a)
                    .and(b)
                    .map_collect(|&x, &y| x.wrapping_shr((y as u32) & 63)),
            ),
            _ => panic!("Shri requires matching integer types"),
        }
    }

    pub fn maxi(&self, rhs: &Tile, signedness: Signedness) -> Self {
        self.check_shape_match_int(rhs);
        match (self, rhs, signedness) {
            // For i1, interpret 0 as 0, 1 as -1 for signed, or 0/1 for unsigned
            (Tile::I1(a), Tile::I1(b), Signedness::Unsigned) => Tile::I1(
                Zip::from(a).and(b).map_collect(
                    |&x, &y| {
                        if (!x as u8) >= (!y as u8) { x } else { y }
                    },
                ),
            ),
            (Tile::I1(a), Tile::I1(b), Signedness::Signed) => {
                Tile::I1(Zip::from(a).and(b).map_collect(|&x, &y| {
                    let xi = if x { -1i8 } else { 0 };
                    let yi = if y { -1i8 } else { 0 };
                    if xi >= yi { x } else { y }
                }))
            }
            (Tile::I8(a), Tile::I8(b), Signedness::Unsigned) => Tile::I8(
                Zip::from(a).and(b).map_collect(
                    |&x, &y| {
                        if (x as u8) >= (y as u8) { x } else { y }
                    },
                ),
            ),
            (Tile::I8(a), Tile::I8(b), Signedness::Signed) => Tile::I8(
                Zip::from(a)
                    .and(b)
                    .map_collect(|&x, &y| if x >= y { x } else { y }),
            ),
            (Tile::I16(a), Tile::I16(b), Signedness::Unsigned) => Tile::I16(
                Zip::from(a).and(b).map_collect(
                    |&x, &y| {
                        if (x as u16) >= (y as u16) { x } else { y }
                    },
                ),
            ),
            (Tile::I16(a), Tile::I16(b), Signedness::Signed) => Tile::I16(
                Zip::from(a)
                    .and(b)
                    .map_collect(|&x, &y| if x >= y { x } else { y }),
            ),
            (Tile::I32(a), Tile::I32(b), Signedness::Unsigned) => Tile::I32(
                Zip::from(a).and(b).map_collect(
                    |&x, &y| {
                        if (x as u32) >= (y as u32) { x } else { y }
                    },
                ),
            ),
            (Tile::I32(a), Tile::I32(b), Signedness::Signed) => Tile::I32(
                Zip::from(a)
                    .and(b)
                    .map_collect(|&x, &y| if x >= y { x } else { y }),
            ),
            (Tile::I64(a), Tile::I64(b), Signedness::Unsigned) => Tile::I64(
                Zip::from(a).and(b).map_collect(
                    |&x, &y| {
                        if (x as u64) >= (y as u64) { x } else { y }
                    },
                ),
            ),
            (Tile::I64(a), Tile::I64(b), Signedness::Signed) => Tile::I64(
                Zip::from(a)
                    .and(b)
                    .map_collect(|&x, &y| if x >= y { x } else { y }),
            ),
            _ => panic!("Maxi requires matching integer types"),
        }
    }

    pub fn mini(&self, rhs: &Tile, signedness: Signedness) -> Self {
        self.check_shape_match_int(rhs);
        match (self, rhs, signedness) {
            // For i1, interpret 0 as 0, 1 as -1 for signed, or 0/1 for unsigned
            (Tile::I1(a), Tile::I1(b), Signedness::Unsigned) => Tile::I1(
                Zip::from(a).and(b).map_collect(
                    |&x, &y| {
                        if (!x as u8) <= (!y as u8) { x } else { y }
                    },
                ),
            ),
            (Tile::I1(a), Tile::I1(b), Signedness::Signed) => {
                Tile::I1(Zip::from(a).and(b).map_collect(|&x, &y| {
                    let xi = if x { -1i8 } else { 0 };
                    let yi = if y { -1i8 } else { 0 };
                    if xi <= yi { x } else { y }
                }))
            }
            (Tile::I8(a), Tile::I8(b), Signedness::Unsigned) => Tile::I8(
                Zip::from(a).and(b).map_collect(
                    |&x, &y| {
                        if (x as u8) <= (y as u8) { x } else { y }
                    },
                ),
            ),
            (Tile::I8(a), Tile::I8(b), Signedness::Signed) => Tile::I8(
                Zip::from(a)
                    .and(b)
                    .map_collect(|&x, &y| if x <= y { x } else { y }),
            ),
            (Tile::I16(a), Tile::I16(b), Signedness::Unsigned) => Tile::I16(
                Zip::from(a).and(b).map_collect(
                    |&x, &y| {
                        if (x as u16) <= (y as u16) { x } else { y }
                    },
                ),
            ),
            (Tile::I16(a), Tile::I16(b), Signedness::Signed) => Tile::I16(
                Zip::from(a)
                    .and(b)
                    .map_collect(|&x, &y| if x <= y { x } else { y }),
            ),
            (Tile::I32(a), Tile::I32(b), Signedness::Unsigned) => Tile::I32(
                Zip::from(a).and(b).map_collect(
                    |&x, &y| {
                        if (x as u32) <= (y as u32) { x } else { y }
                    },
                ),
            ),
            (Tile::I32(a), Tile::I32(b), Signedness::Signed) => Tile::I32(
                Zip::from(a)
                    .and(b)
                    .map_collect(|&x, &y| if x <= y { x } else { y }),
            ),
            (Tile::I64(a), Tile::I64(b), Signedness::Unsigned) => Tile::I64(
                Zip::from(a).and(b).map_collect(
                    |&x, &y| {
                        if (x as u64) <= (y as u64) { x } else { y }
                    },
                ),
            ),
            (Tile::I64(a), Tile::I64(b), Signedness::Signed) => Tile::I64(
                Zip::from(a)
                    .and(b)
                    .map_collect(|&x, &y| if x <= y { x } else { y }),
            ),
            _ => panic!("Mini requires matching integer types"),
        }
    }

    // ========================================================================
    // Ternary Operations
    // ========================================================================

    /// See further: https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html#cuda-tile-mmai
    pub fn mmai(
        &self,
        rhs: &Tile,
        acc: &Tile,
        signedness_lhs: Signedness,
        signedness_rhs: Signedness,
    ) -> Self {
        // Matrix multiply: (M x K) * (K x N) + (M x N) = (M x N)
        // Or batched: (B x M x K) * (B x K x N) + (B x M x N) = (B x M x N)
        // Input tiles lhs and rhs must be of integer type i8
        // The accumulator tile acc must be of type i32

        match (self, rhs, acc) {
            (Tile::I8(a), Tile::I8(b), Tile::I32(c)) => {
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
                        "MMAI: lhs inner dim must match rhs outer dim"
                    );
                    assert_eq!(c_shape, &[m, n], "MMAI: accumulator shape mismatch");

                    self.matmul_unbatched_i8(a, b, c, m, k, n, signedness_lhs, signedness_rhs)
                } else {
                    // Batched: (B x M x K) * (B x K x N) + (B x M x N)
                    let bsz = a_shape[0];
                    let m = a_shape[1];
                    let k = a_shape[2];
                    let n = b_shape[2];

                    assert_eq!(b_shape[0], bsz, "MMAI: batch size mismatch");
                    assert_eq!(
                        b_shape[1], k,
                        "MMAI: lhs inner dim must match rhs outer dim"
                    );
                    assert_eq!(c_shape, &[bsz, m, n], "MMAI: accumulator shape mismatch");

                    self.matmul_batched_i8(a, b, c, bsz, m, k, n, signedness_lhs, signedness_rhs)
                }
            }
            _ => panic!("Mmai requires lhs/rhs to be i8 and acc to be i32"),
        }
    }

    fn matmul_unbatched_i8(
        &self,
        a: &ndarray::Array<i8, ndarray::IxDyn>,
        b: &ndarray::Array<i8, ndarray::IxDyn>,
        c: &ndarray::Array<i32, ndarray::IxDyn>,
        m: usize,
        k: usize,
        n: usize,
        signedness_lhs: Signedness,
        signedness_rhs: Signedness,
    ) -> Tile {
        let mut result: ndarray::Array<i32, _> =
            unsafe { ndarray::Array::uninit(ndarray::IxDyn(&[m, n])).assume_init() };
        for i in 0..m {
            for j in 0..n {
                let mut sum: i32 = 0;
                for kk in 0..k {
                    let a_val = a[[i, kk]];
                    let b_val = b[[kk, j]];
                    let a_extended = if signedness_lhs == Signedness::Signed {
                        a_val as i32
                    } else {
                        (a_val as u8) as i32
                    };
                    let b_extended = if signedness_rhs == Signedness::Signed {
                        b_val as i32
                    } else {
                        (b_val as u8) as i32
                    };
                    sum += a_extended * b_extended;
                }
                result[[i, j]] = sum + c[[i, j]];
            }
        }
        Tile::I32(result)
    }

    fn matmul_batched_i8(
        &self,
        a: &ndarray::Array<i8, ndarray::IxDyn>,
        b: &ndarray::Array<i8, ndarray::IxDyn>,
        c: &ndarray::Array<i32, ndarray::IxDyn>,
        bsz: usize,
        m: usize,
        k: usize,
        n: usize,
        signedness_lhs: Signedness,
        signedness_rhs: Signedness,
    ) -> Tile {
        let mut result: ndarray::Array<i32, _> =
            unsafe { ndarray::Array::uninit(ndarray::IxDyn(&[bsz, m, n])).assume_init() };
        for batch in 0..bsz {
            for i in 0..m {
                for j in 0..n {
                    let mut sum: i32 = 0;
                    for kk in 0..k {
                        let a_val = a[[batch, i, kk]];
                        let b_val = b[[batch, kk, j]];
                        let a_extended = if signedness_lhs == Signedness::Signed {
                            a_val as i32
                        } else {
                            (a_val as u8) as i32
                        };
                        let b_extended = if signedness_rhs == Signedness::Signed {
                            b_val as i32
                        } else {
                            (b_val as u8) as i32
                        };
                        sum += a_extended * b_extended;
                    }
                    result[[batch, i, j]] = sum + c[[batch, i, j]];
                }
            }
        }
        Tile::I32(result)
    }

    // ========================================================================
    // Comparison Operations
    // ========================================================================

    pub fn cmpi(&self, rhs: &Tile, pred: ComparisonPredicate, signedness: Signedness) -> Tile {
        self.check_shape_match_int(rhs);

        match (self, rhs, pred, signedness) {
            // Equal comparison (same for signed and unsigned)
            (Tile::I1(a), Tile::I1(b), ComparisonPredicate::Equal, _) => {
                Tile::I1(Zip::from(a).and(b).map_collect(|&x, &y| x == y))
            }
            (Tile::I8(a), Tile::I8(b), ComparisonPredicate::Equal, _) => {
                Tile::I1(Zip::from(a).and(b).map_collect(|&x, &y| x == y))
            }
            (Tile::I16(a), Tile::I16(b), ComparisonPredicate::Equal, _) => {
                Tile::I1(Zip::from(a).and(b).map_collect(|&x, &y| x == y))
            }
            (Tile::I32(a), Tile::I32(b), ComparisonPredicate::Equal, _) => {
                Tile::I1(Zip::from(a).and(b).map_collect(|&x, &y| x == y))
            }
            (Tile::I64(a), Tile::I64(b), ComparisonPredicate::Equal, _) => {
                Tile::I1(Zip::from(a).and(b).map_collect(|&x, &y| x == y))
            }

            // Not equal comparison (same for signed and unsigned)
            (Tile::I1(a), Tile::I1(b), ComparisonPredicate::NotEqual, _) => {
                Tile::I1(Zip::from(a).and(b).map_collect(|&x, &y| x != y))
            }
            (Tile::I8(a), Tile::I8(b), ComparisonPredicate::NotEqual, _) => {
                Tile::I1(Zip::from(a).and(b).map_collect(|&x, &y| x != y))
            }
            (Tile::I16(a), Tile::I16(b), ComparisonPredicate::NotEqual, _) => {
                Tile::I1(Zip::from(a).and(b).map_collect(|&x, &y| x != y))
            }
            (Tile::I32(a), Tile::I32(b), ComparisonPredicate::NotEqual, _) => {
                Tile::I1(Zip::from(a).and(b).map_collect(|&x, &y| x != y))
            }
            (Tile::I64(a), Tile::I64(b), ComparisonPredicate::NotEqual, _) => {
                Tile::I1(Zip::from(a).and(b).map_collect(|&x, &y| x != y))
            }

            // Less than (signed)
            (Tile::I1(a), Tile::I1(b), ComparisonPredicate::LessThan, Signedness::Signed) => {
                Tile::I1(Zip::from(a).and(b).map_collect(|&x, &y| {
                    let xi = if x { -1i8 } else { 0 };
                    let yi = if y { -1i8 } else { 0 };
                    xi < yi
                }))
            }
            (Tile::I8(a), Tile::I8(b), ComparisonPredicate::LessThan, Signedness::Signed) => {
                Tile::I1(Zip::from(a).and(b).map_collect(|&x, &y| x < y))
            }
            (Tile::I16(a), Tile::I16(b), ComparisonPredicate::LessThan, Signedness::Signed) => {
                Tile::I1(Zip::from(a).and(b).map_collect(|&x, &y| x < y))
            }
            (Tile::I32(a), Tile::I32(b), ComparisonPredicate::LessThan, Signedness::Signed) => {
                Tile::I1(Zip::from(a).and(b).map_collect(|&x, &y| x < y))
            }
            (Tile::I64(a), Tile::I64(b), ComparisonPredicate::LessThan, Signedness::Signed) => {
                Tile::I1(Zip::from(a).and(b).map_collect(|&x, &y| x < y))
            }

            // Less than (unsigned)
            (Tile::I1(a), Tile::I1(b), ComparisonPredicate::LessThan, Signedness::Unsigned) => {
                Tile::I1(
                    Zip::from(a)
                        .and(b)
                        .map_collect(|&x, &y| (!x as u8) < (!y as u8)),
                )
            }
            (Tile::I8(a), Tile::I8(b), ComparisonPredicate::LessThan, Signedness::Unsigned) => {
                Tile::I1(
                    Zip::from(a)
                        .and(b)
                        .map_collect(|&x, &y| (x as u8) < (y as u8)),
                )
            }
            (Tile::I16(a), Tile::I16(b), ComparisonPredicate::LessThan, Signedness::Unsigned) => {
                Tile::I1(
                    Zip::from(a)
                        .and(b)
                        .map_collect(|&x, &y| (x as u16) < (y as u16)),
                )
            }
            (Tile::I32(a), Tile::I32(b), ComparisonPredicate::LessThan, Signedness::Unsigned) => {
                Tile::I1(
                    Zip::from(a)
                        .and(b)
                        .map_collect(|&x, &y| (x as u32) < (y as u32)),
                )
            }
            (Tile::I64(a), Tile::I64(b), ComparisonPredicate::LessThan, Signedness::Unsigned) => {
                Tile::I1(
                    Zip::from(a)
                        .and(b)
                        .map_collect(|&x, &y| (x as u64) < (y as u64)),
                )
            }

            // Less than or equal (signed)
            (
                Tile::I1(a),
                Tile::I1(b),
                ComparisonPredicate::LessThanOrEqual,
                Signedness::Signed,
            ) => Tile::I1(Zip::from(a).and(b).map_collect(|&x, &y| {
                let xi = if x { -1i8 } else { 0 };
                let yi = if y { -1i8 } else { 0 };
                xi <= yi
            })),
            (
                Tile::I8(a),
                Tile::I8(b),
                ComparisonPredicate::LessThanOrEqual,
                Signedness::Signed,
            ) => Tile::I1(Zip::from(a).and(b).map_collect(|&x, &y| x <= y)),
            (
                Tile::I16(a),
                Tile::I16(b),
                ComparisonPredicate::LessThanOrEqual,
                Signedness::Signed,
            ) => Tile::I1(Zip::from(a).and(b).map_collect(|&x, &y| x <= y)),
            (
                Tile::I32(a),
                Tile::I32(b),
                ComparisonPredicate::LessThanOrEqual,
                Signedness::Signed,
            ) => Tile::I1(Zip::from(a).and(b).map_collect(|&x, &y| x <= y)),
            (
                Tile::I64(a),
                Tile::I64(b),
                ComparisonPredicate::LessThanOrEqual,
                Signedness::Signed,
            ) => Tile::I1(Zip::from(a).and(b).map_collect(|&x, &y| x <= y)),

            // Less than or equal (unsigned)
            (
                Tile::I1(a),
                Tile::I1(b),
                ComparisonPredicate::LessThanOrEqual,
                Signedness::Unsigned,
            ) => Tile::I1(
                Zip::from(a)
                    .and(b)
                    .map_collect(|&x, &y| (!x as u8) <= (!y as u8)),
            ),
            (
                Tile::I8(a),
                Tile::I8(b),
                ComparisonPredicate::LessThanOrEqual,
                Signedness::Unsigned,
            ) => Tile::I1(
                Zip::from(a)
                    .and(b)
                    .map_collect(|&x, &y| (x as u8) <= (y as u8)),
            ),
            (
                Tile::I16(a),
                Tile::I16(b),
                ComparisonPredicate::LessThanOrEqual,
                Signedness::Unsigned,
            ) => Tile::I1(
                Zip::from(a)
                    .and(b)
                    .map_collect(|&x, &y| (x as u16) <= (y as u16)),
            ),
            (
                Tile::I32(a),
                Tile::I32(b),
                ComparisonPredicate::LessThanOrEqual,
                Signedness::Unsigned,
            ) => Tile::I1(
                Zip::from(a)
                    .and(b)
                    .map_collect(|&x, &y| (x as u32) <= (y as u32)),
            ),
            (
                Tile::I64(a),
                Tile::I64(b),
                ComparisonPredicate::LessThanOrEqual,
                Signedness::Unsigned,
            ) => Tile::I1(
                Zip::from(a)
                    .and(b)
                    .map_collect(|&x, &y| (x as u64) <= (y as u64)),
            ),

            // Greater than (signed)
            (Tile::I1(a), Tile::I1(b), ComparisonPredicate::GreaterThan, Signedness::Signed) => {
                Tile::I1(Zip::from(a).and(b).map_collect(|&x, &y| {
                    let xi = if x { -1i8 } else { 0 };
                    let yi = if y { -1i8 } else { 0 };
                    xi > yi
                }))
            }
            (Tile::I8(a), Tile::I8(b), ComparisonPredicate::GreaterThan, Signedness::Signed) => {
                Tile::I1(Zip::from(a).and(b).map_collect(|&x, &y| x > y))
            }
            (Tile::I16(a), Tile::I16(b), ComparisonPredicate::GreaterThan, Signedness::Signed) => {
                Tile::I1(Zip::from(a).and(b).map_collect(|&x, &y| x > y))
            }
            (Tile::I32(a), Tile::I32(b), ComparisonPredicate::GreaterThan, Signedness::Signed) => {
                Tile::I1(Zip::from(a).and(b).map_collect(|&x, &y| x > y))
            }
            (Tile::I64(a), Tile::I64(b), ComparisonPredicate::GreaterThan, Signedness::Signed) => {
                Tile::I1(Zip::from(a).and(b).map_collect(|&x, &y| x > y))
            }

            // Greater than (unsigned)
            (Tile::I1(a), Tile::I1(b), ComparisonPredicate::GreaterThan, Signedness::Unsigned) => {
                Tile::I1(
                    Zip::from(a)
                        .and(b)
                        .map_collect(|&x, &y| (!x as u8) > (!y as u8)),
                )
            }
            (Tile::I8(a), Tile::I8(b), ComparisonPredicate::GreaterThan, Signedness::Unsigned) => {
                Tile::I1(
                    Zip::from(a)
                        .and(b)
                        .map_collect(|&x, &y| (x as u8) > (y as u8)),
                )
            }
            (
                Tile::I16(a),
                Tile::I16(b),
                ComparisonPredicate::GreaterThan,
                Signedness::Unsigned,
            ) => Tile::I1(
                Zip::from(a)
                    .and(b)
                    .map_collect(|&x, &y| (x as u16) > (y as u16)),
            ),
            (
                Tile::I32(a),
                Tile::I32(b),
                ComparisonPredicate::GreaterThan,
                Signedness::Unsigned,
            ) => Tile::I1(
                Zip::from(a)
                    .and(b)
                    .map_collect(|&x, &y| (x as u32) > (y as u32)),
            ),
            (
                Tile::I64(a),
                Tile::I64(b),
                ComparisonPredicate::GreaterThan,
                Signedness::Unsigned,
            ) => Tile::I1(
                Zip::from(a)
                    .and(b)
                    .map_collect(|&x, &y| (x as u64) > (y as u64)),
            ),

            // Greater than or equal (signed)
            (
                Tile::I1(a),
                Tile::I1(b),
                ComparisonPredicate::GreaterThanOrEqual,
                Signedness::Signed,
            ) => Tile::I1(Zip::from(a).and(b).map_collect(|&x, &y| {
                let xi = if x { -1i8 } else { 0 };
                let yi = if y { -1i8 } else { 0 };
                xi >= yi
            })),
            (
                Tile::I8(a),
                Tile::I8(b),
                ComparisonPredicate::GreaterThanOrEqual,
                Signedness::Signed,
            ) => Tile::I1(Zip::from(a).and(b).map_collect(|&x, &y| x >= y)),
            (
                Tile::I16(a),
                Tile::I16(b),
                ComparisonPredicate::GreaterThanOrEqual,
                Signedness::Signed,
            ) => Tile::I1(Zip::from(a).and(b).map_collect(|&x, &y| x >= y)),
            (
                Tile::I32(a),
                Tile::I32(b),
                ComparisonPredicate::GreaterThanOrEqual,
                Signedness::Signed,
            ) => Tile::I1(Zip::from(a).and(b).map_collect(|&x, &y| x >= y)),
            (
                Tile::I64(a),
                Tile::I64(b),
                ComparisonPredicate::GreaterThanOrEqual,
                Signedness::Signed,
            ) => Tile::I1(Zip::from(a).and(b).map_collect(|&x, &y| x >= y)),

            // Greater than or equal (unsigned)
            (
                Tile::I1(a),
                Tile::I1(b),
                ComparisonPredicate::GreaterThanOrEqual,
                Signedness::Unsigned,
            ) => Tile::I1(
                Zip::from(a)
                    .and(b)
                    .map_collect(|&x, &y| (!x as u8) >= (!y as u8)),
            ),
            (
                Tile::I8(a),
                Tile::I8(b),
                ComparisonPredicate::GreaterThanOrEqual,
                Signedness::Unsigned,
            ) => Tile::I1(
                Zip::from(a)
                    .and(b)
                    .map_collect(|&x, &y| (x as u8) >= (y as u8)),
            ),
            (
                Tile::I16(a),
                Tile::I16(b),
                ComparisonPredicate::GreaterThanOrEqual,
                Signedness::Unsigned,
            ) => Tile::I1(
                Zip::from(a)
                    .and(b)
                    .map_collect(|&x, &y| (x as u16) >= (y as u16)),
            ),
            (
                Tile::I32(a),
                Tile::I32(b),
                ComparisonPredicate::GreaterThanOrEqual,
                Signedness::Unsigned,
            ) => Tile::I1(
                Zip::from(a)
                    .and(b)
                    .map_collect(|&x, &y| (x as u32) >= (y as u32)),
            ),
            (
                Tile::I64(a),
                Tile::I64(b),
                ComparisonPredicate::GreaterThanOrEqual,
                Signedness::Unsigned,
            ) => Tile::I1(
                Zip::from(a)
                    .and(b)
                    .map_collect(|&x, &y| (x as u64) >= (y as u64)),
            ),

            _ => panic!("Cmpi requires matching integer types"),
        }
    }
}
