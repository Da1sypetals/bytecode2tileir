// Operation execution for TileIR interpreter.

use crate::cuda_tile_ir::ids::OpId;
use crate::cuda_tile_ir::types::Dim;
use crate::cuda_tile_ir::Opcode;

impl crate::interpreter::data_structures::interpreter::ExecutionContext<'_> {
    pub fn execute_op(&mut self, op_id: OpId) {
        let op = self.arena.op_(op_id);

        match op.opcode {
            // Core operations from llm_docs/tileir/core.md (8.3)
            Opcode::Broadcast => self.execute_broadcast(op),
            Opcode::Cat => self.execute_cat(op),
            Opcode::Constant => self.execute_constant(op),
            Opcode::Extract => self.execute_extract(op),
            Opcode::GetGlobal => self.execute_get_global(op),
            Opcode::GetNumTileBlocks => self.execute_get_num_tile_blocks(op),
            Opcode::GetTileBlockId => self.execute_get_tile_block_id(op),
            Opcode::Iota => self.execute_iota(op),
            Opcode::Offset => self.execute_offset(op),
            Opcode::Permute => self.execute_permute(op),
            Opcode::Reduce => self.execute_reduce(op),
            Opcode::Reshape => self.execute_reshape(op),
            Opcode::Scan => self.execute_scan(op),
            Opcode::Select => self.execute_select(op),

            // Conversion operations from llm_docs/tileir/conversions.md (8.4)
            Opcode::Bitcast => self.execute_bitcast(op),
            Opcode::ExtI => self.execute_exti(op),
            Opcode::FToF => self.execute_ftof(op),
            Opcode::FToI => self.execute_ftoi(op),
            Opcode::IntToPtr => self.execute_int_to_ptr(op),
            Opcode::IToF => self.execute_itof(op),
            Opcode::PtrToInt => self.execute_ptr_to_int(op),
            Opcode::PtrToPtr => self.execute_ptr_to_ptr(op),
            Opcode::TruncI => self.execute_trunci(op),

            // Memory operations from llm_docs/tileir/memory.md (8.6)
            Opcode::LoadPtrTko => self.execute_load_ptr_tko(op),
            Opcode::StorePtrTko => self.execute_store_ptr_tko(op),

            // Floating-point operations from llm_docs/tileir/float.md (8.7)
            Opcode::AbsF => self.execute_absf(op),
            Opcode::AddF => self.execute_addf(op),
            Opcode::Ceil => self.execute_ceil(op),
            Opcode::CmpF => self.execute_cmpf(op),
            Opcode::Cos => self.execute_cos(op),
            Opcode::CosH => self.execute_cosh(op),
            Opcode::DivF => self.execute_divf(op),
            Opcode::Exp => self.execute_exp(op),
            Opcode::Exp2 => self.execute_exp2(op),
            Opcode::Floor => self.execute_floor(op),
            Opcode::Fma => self.execute_fma(op),
            Opcode::Log => self.execute_log(op),
            Opcode::Log2 => self.execute_log2(op),
            Opcode::MaxF => self.execute_maxf(op),
            Opcode::MinF => self.execute_minf(op),
            Opcode::MmaF => self.execute_mmaf(op),
            Opcode::MulF => self.execute_mulf(op),
            Opcode::NegF => self.execute_negf(op),
            Opcode::Pow => self.execute_pow(op),
            Opcode::RemF => self.execute_remf(op),
            Opcode::Rsqrt => self.execute_rsqrt(op),
            Opcode::Sin => self.execute_sin(op),
            Opcode::SinH => self.execute_sinh(op),
            Opcode::Sqrt => self.execute_sqrt(op),
            Opcode::SubF => self.execute_subf(op),
            Opcode::Tan => self.execute_tan(op),
            Opcode::TanH => self.execute_tanh(op),

            _ => panic!("Opcode {:?} not implemented", op.opcode),
        }
    }
}
