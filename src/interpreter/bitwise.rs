// Bitwise operations for TileIR interpreter (Section 8.9)
//
// Implements: andi, ori, xori - Element-wise bitwise operations on integer tiles

use crate::cuda_tile_ir::ir::Operation;
use crate::interpreter::data_structures::interpreter::ExecutionContext;
use crate::interpreter::data_structures::tile::Tile;
use crate::interpreter::data_structures::value::Value;
use ndarray::Zip;

impl ExecutionContext<'_> {
    /// 8.9.1. cuda_tile.andi - Element-wise bitwise AND
    pub fn execute_andi(&mut self, op: &Operation) {
        let lhs = self.get_value(op.operands[0]);
        let rhs = self.get_value(op.operands[1]);

        let result = match (lhs, rhs) {
            (Value::Tile(lhs_tile), Value::Tile(rhs_tile)) => Value::Tile(lhs_tile.andi(&rhs_tile)),
            _ => panic!("AndI requires Tile operands"),
        };

        self.set_value(op.results[0], result);
    }

    /// 8.9.2. cuda_tile.ori - Element-wise bitwise OR
    pub fn execute_ori(&mut self, op: &Operation) {
        let lhs = self.get_value(op.operands[0]);
        let rhs = self.get_value(op.operands[1]);

        let result = match (lhs, rhs) {
            (Value::Tile(lhs_tile), Value::Tile(rhs_tile)) => Value::Tile(lhs_tile.ori(&rhs_tile)),
            _ => panic!("OrI requires Tile operands"),
        };

        self.set_value(op.results[0], result);
    }

    /// 8.9.3. cuda_tile.xori - Element-wise bitwise XOR
    pub fn execute_xori(&mut self, op: &Operation) {
        let lhs = self.get_value(op.operands[0]);
        let rhs = self.get_value(op.operands[1]);

        let result = match (lhs, rhs) {
            (Value::Tile(lhs_tile), Value::Tile(rhs_tile)) => Value::Tile(lhs_tile.xori(&rhs_tile)),
            _ => panic!("XOrI requires Tile operands"),
        };

        self.set_value(op.results[0], result);
    }
}

// ============================================================================
// Tile Implementation
// ============================================================================

impl Tile {
    // ========================================================================
    // Helper Functions (private)
    // ========================================================================

    /// Helper: Check that two tiles have matching shapes (no implicit broadcast per general.md)
    fn check_shape_match_bitwise(&self, other: &Tile) {
        let lhs_shape = self.shape();
        let rhs_shape = other.shape();
        assert_eq!(
            lhs_shape, rhs_shape,
            "Shape mismatch: lhs: {:?} vs rhs: {:?}",
            lhs_shape, rhs_shape
        );
    }

    // ========================================================================
    // Bitwise Binary Operations
    // ========================================================================

    /// Element-wise bitwise AND
    pub fn andi(&self, rhs: &Tile) -> Self {
        self.check_shape_match_bitwise(rhs);
        match (self, rhs) {
            (Tile::I1(a), Tile::I1(b)) => Tile::I1(Zip::from(a).and(b).map_collect(|&x, &y| x & y)),
            (Tile::I8(a), Tile::I8(b)) => Tile::I8(Zip::from(a).and(b).map_collect(|&x, &y| x & y)),
            (Tile::I16(a), Tile::I16(b)) => {
                Tile::I16(Zip::from(a).and(b).map_collect(|&x, &y| x & y))
            }
            (Tile::I32(a), Tile::I32(b)) => {
                Tile::I32(Zip::from(a).and(b).map_collect(|&x, &y| x & y))
            }
            (Tile::I64(a), Tile::I64(b)) => {
                Tile::I64(Zip::from(a).and(b).map_collect(|&x, &y| x & y))
            }
            _ => panic!("AndI requires matching integer types"),
        }
    }

    /// Element-wise bitwise OR
    pub fn ori(&self, rhs: &Tile) -> Self {
        self.check_shape_match_bitwise(rhs);
        match (self, rhs) {
            (Tile::I1(a), Tile::I1(b)) => Tile::I1(Zip::from(a).and(b).map_collect(|&x, &y| x | y)),
            (Tile::I8(a), Tile::I8(b)) => Tile::I8(Zip::from(a).and(b).map_collect(|&x, &y| x | y)),
            (Tile::I16(a), Tile::I16(b)) => {
                Tile::I16(Zip::from(a).and(b).map_collect(|&x, &y| x | y))
            }
            (Tile::I32(a), Tile::I32(b)) => {
                Tile::I32(Zip::from(a).and(b).map_collect(|&x, &y| x | y))
            }
            (Tile::I64(a), Tile::I64(b)) => {
                Tile::I64(Zip::from(a).and(b).map_collect(|&x, &y| x | y))
            }
            _ => panic!("OrI requires matching integer types"),
        }
    }

    /// Element-wise bitwise XOR
    pub fn xori(&self, rhs: &Tile) -> Self {
        self.check_shape_match_bitwise(rhs);
        match (self, rhs) {
            (Tile::I1(a), Tile::I1(b)) => Tile::I1(Zip::from(a).and(b).map_collect(|&x, &y| x ^ y)),
            (Tile::I8(a), Tile::I8(b)) => Tile::I8(Zip::from(a).and(b).map_collect(|&x, &y| x ^ y)),
            (Tile::I16(a), Tile::I16(b)) => {
                Tile::I16(Zip::from(a).and(b).map_collect(|&x, &y| x ^ y))
            }
            (Tile::I32(a), Tile::I32(b)) => {
                Tile::I32(Zip::from(a).and(b).map_collect(|&x, &y| x ^ y))
            }
            (Tile::I64(a), Tile::I64(b)) => {
                Tile::I64(Zip::from(a).and(b).map_collect(|&x, &y| x ^ y))
            }
            _ => panic!("XOrI requires matching integer types"),
        }
    }
}
