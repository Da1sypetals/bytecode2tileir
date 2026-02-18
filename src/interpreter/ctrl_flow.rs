// Control flow operations for TileIR interpreter.
// Implements cuda_tile.assert, cuda_tile.if, cuda_tile.for, cuda_tile.loop,
// cuda_tile.break, cuda_tile.continue, cuda_tile.yield, cuda_tile.return

use crate::cuda_tile_ir::OpAttrKey;
use crate::cuda_tile_ir::attrs::Attr;
use crate::cuda_tile_ir::ir::Operation;
use crate::interpreter::data_structures::elem_type::ElemType;
use crate::interpreter::data_structures::elem_type::Scalar;
use crate::interpreter::data_structures::interpreter::ExecutionContext;
use crate::interpreter::data_structures::tile::Tile;
use crate::interpreter::data_structures::value::Value;
use crate::interpreter::entry::ControlFlow;
use crate::interpreter::type_conversion::type_to_elem_type;
use log::error;

impl ExecutionContext<'_> {
    /// 8.5.1. cuda_tile.assert - Terminate kernel execution with an error message if condition is false-y
    pub fn execute_assert(&mut self, op: &Operation) {
        let condition = self.get_value(op.operands[0]);

        // Get the message attribute
        let message_attr_id = op
            .attrs
            .iter()
            .find(|(key, _)| *key == OpAttrKey::Message)
            .map(|(_, id)| id)
            .expect("Assert operation missing Message attribute");
        let message_attr = self.arena.attr_(*message_attr_id);

        let message = match message_attr {
            Attr::String(s) => s.clone(),
            _ => panic!("Assert Message attribute must be String"),
        };

        let condition_tile = match condition {
            Value::Tile(t) => t,
            _ => panic!("Assert condition must be a Tile"),
        };

        // Check if any element is false (0)
        let has_false = match condition_tile {
            Tile::I1(arr) => arr.iter().any(|&v| !v),
            _ => panic!("Assert condition must be tile<i1>"),
        };

        if has_false {
            error!("Assertion failed: {}", message);
            panic!("Assertion failed: {}", message);
        }
    }

    /// 8.5.5. cuda_tile.if - Conditional execution
    pub fn execute_if(&mut self, op: &Operation) {
        let condition = self.get_value(op.operands[0]);

        let condition_value = match condition {
            Value::Tile(Tile::I1(arr)) => {
                if arr.shape().is_empty() {
                    arr[ndarray::IxDyn(&[])]
                } else {
                    panic!("If condition must be a scalar (0-dimensional) tile<i1>");
                }
            }
            _ => panic!("If condition must be tile<i1>"),
        };

        // Check if the operation returns results
        let has_results = !op.results.is_empty();

        if has_results {
            // If has results, we need to execute the appropriate region and get yielded values
            let then_region_id = op.regions[0].0;
            let else_region_id = if op.regions.len() > 1 {
                Some(op.regions[1].0)
            } else {
                None
            };

            let result_values = if condition_value {
                let cf_result = self.execute_region(then_region_id, &[]);
                match cf_result {
                    Some(ControlFlow::Yield(values)) => values,
                    Some(_) => panic!("If then region must terminate with yield"),
                    None => panic!("If then region must terminate with yield"),
                }
            } else {
                if let Some(else_id) = else_region_id {
                    let cf_result = self.execute_region(else_id, &[]);
                    match cf_result {
                        Some(ControlFlow::Yield(values)) => values,
                        Some(_) => panic!("If else region must terminate with yield"),
                        None => panic!("If else region must terminate with yield"),
                    }
                } else {
                    panic!("If operation with results must have else region");
                }
            };

            // Set the result values
            for (&result_id, value) in op.results.iter().zip(result_values.into_iter()) {
                self.set_value(result_id, value);
            }
        } else {
            // If has no results, just execute the appropriate region
            if condition_value {
                self.execute_region(op.regions[0].0, &[]);
            } else if op.regions.len() > 1 {
                self.execute_region(op.regions[1].0, &[]);
            }
        }
    }

    /// 8.5.4. cuda_tile.for - For loop over integer range
    pub fn execute_for(&mut self, op: &Operation) {
        let lower_bound = self.get_value(op.operands[0]);
        let upper_bound = self.get_value(op.operands[1]);
        let step = self.get_value(op.operands[2]);

        // Get initial loop-carried values (operands after bounds)
        let init_values: Vec<Value> = op.operands[3..]
            .iter()
            .map(|&id| self.get_value(id).clone())
            .collect();

        // Check for unsigned attribute
        let unsigned = op
            .attrs
            .iter()
            .find(|(key, _)| *key == OpAttrKey::Signedness)
            .map(|(_, id)| {
                let attr = self.arena.attr_(*id);
                matches!(
                    attr,
                    Attr::Signedness(crate::cuda_tile_ir::enums::Signedness::Unsigned)
                )
            })
            .unwrap_or(false);

        // Extract bounds as i64 values
        let lower_i64 = extract_scalar_i64(lower_bound);
        let upper_i64 = extract_scalar_i64(upper_bound);
        let step_i64 = extract_scalar_i64(step);

        if step_i64 <= 0 {
            panic!("For loop step must be positive, got {}", step_i64);
        }

        let region_id = op.regions[0].0;

        // Get induction variable type
        let region = self
            .arena
            .region_(crate::cuda_tile_ir::ids::RegionId(region_id));
        let block_id = region.blocks[0];
        let block = self.arena.block_(block_id);
        let iv_id = block.args[0];

        // Get the element type for the induction variable
        let iv_value_data = self.arena.value_(iv_id);
        let iv_ty = self.arena.type_(iv_value_data.ty());
        let iv_elem_type = type_to_elem_type(iv_ty, self.arena);

        // Loop iteration
        let mut current_values = init_values;
        let mut iv_value = lower_i64;

        loop {
            // Check loop condition
            let in_range = if unsigned {
                (iv_value as u64) < (upper_i64 as u64)
            } else {
                iv_value < upper_i64
            };

            if !in_range {
                break;
            }

            // Create induction variable tile
            let iv_scalar = int_scalar(iv_value, iv_elem_type);
            let iv_tile = Tile::from_scalar(iv_scalar, iv_elem_type);

            // Prepare block arguments: IV + loop-carried values
            let mut block_args = vec![Value::Tile(iv_tile)];
            block_args.extend(current_values.iter().cloned());

            // Execute region: indirect recursion to execute_region
            let cf_result = self.execute_region(region_id, &block_args);

            match cf_result {
                Some(ControlFlow::Continue(values)) => {
                    current_values = values;
                }
                Some(ControlFlow::Break(values)) => {
                    current_values = values;
                    break;
                }
                Some(_) => panic!("For loop body must terminate with continue or break"),
                None => panic!("For loop body must terminate with continue"),
            }

            iv_value += step_i64;
        }

        // Set final loop-carried values as results
        for (&result_id, value) in op.results.iter().zip(current_values.into_iter()) {
            self.set_value(result_id, value);
        }
    }

    /// 8.5.6. cuda_tile.loop - Loop until a break operation
    pub fn execute_loop(&mut self, op: &Operation) {
        // Get initial loop-carried values
        let init_values: Vec<Value> = op
            .operands
            .iter()
            .map(|&id| self.get_value(id).clone())
            .collect();

        let region_id = op.regions[0].0;

        // Loop iteration
        let mut current_values = init_values;

        loop {
            // Execute region with current loop-carried values
            let cf_result = self.execute_region(region_id, &current_values);

            match cf_result {
                Some(ControlFlow::Continue(values)) => {
                    current_values = values;
                }
                Some(ControlFlow::Break(values)) => {
                    current_values = values;
                    break;
                }
                Some(_) => panic!("Loop body must terminate with continue or break"),
                None => panic!("Loop body must terminate with continue or break"),
            }
        }

        // Set final loop-carried values as results
        for (&result_id, value) in op.results.iter().zip(current_values.into_iter()) {
            self.set_value(result_id, value);
        }
    }
}

/// Extract a scalar i64 value from a Tile value
fn extract_scalar_i64(value: &Value) -> i64 {
    match value {
        Value::Tile(Tile::I8(arr)) => arr[ndarray::IxDyn(&[])] as i64,
        Value::Tile(Tile::I16(arr)) => arr[ndarray::IxDyn(&[])] as i64,
        Value::Tile(Tile::I32(arr)) => arr[ndarray::IxDyn(&[])] as i64,
        Value::Tile(Tile::I64(arr)) => arr[ndarray::IxDyn(&[])],
        _ => panic!("Loop bound must be an integer tile"),
    }
}

/// Convert an i64 value to the appropriate Scalar type
fn int_scalar(value: i64, elem_type: ElemType) -> Scalar {
    match elem_type {
        ElemType::Bool => Scalar::Bool(value != 0),
        ElemType::I8 => Scalar::I8(value as i8),
        ElemType::I16 => Scalar::I16(value as i16),
        ElemType::I32 => Scalar::I32(value as i32),
        ElemType::I64 => Scalar::I64(value),
        _ => panic!("Cannot convert i64 to {:?}", elem_type),
    }
}
