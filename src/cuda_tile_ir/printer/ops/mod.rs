pub mod control_flow;
pub mod generic;
pub mod memory;
pub mod misc;
pub mod view;

use std::fmt;

use crate::cuda_tile_ir::ids::OpId;
use crate::cuda_tile_ir::Opcode;

use super::indent::MlirPrinter;
use super::Printer;

pub(super) fn print_op<W: MlirPrinter + ?Sized>(
    p: &mut Printer<'_, '_, W>,
    op_id: OpId,
) -> fmt::Result {
    let op = p.ctx.module.arena.op_(op_id);
    match op.opcode {
        Opcode::If => control_flow::print_if(p, op),
        Opcode::For => control_flow::print_for(p, op),
        Opcode::Loop => control_flow::print_loop(p, op),
        Opcode::Constant => misc::print_constant(p, op),
        Opcode::Assert => misc::print_assert(p, op),
        Opcode::Assume => misc::print_assume(p, op),
        Opcode::GetGlobal => misc::print_get_global(p, op),
        Opcode::LoadPtrTko | Opcode::LoadViewTko => memory::print_load_tko(p, op),
        Opcode::StorePtrTko | Opcode::StoreViewTko => memory::print_store_tko(p, op),
        Opcode::AtomicRMWTko => memory::print_atomic_rmw_tko(p, op),
        Opcode::AtomicCASTko => memory::print_atomic_cas_tko(p, op),
        Opcode::MakeToken => misc::print_make_token(p, op),
        Opcode::Print => misc::print_print(p, op),
        Opcode::MakeTensorView => view::print_make_tensor_view(p, op),
        Opcode::MakePartitionView => view::print_make_partition_view(p, op),
        Opcode::Reshape => view::print_reshape(p, op),
        Opcode::Broadcast => view::print_broadcast(p, op),
        Opcode::Extract => view::print_extract(p, op),
        Opcode::Cat => view::print_cat(p, op),
        Opcode::Permute => view::print_permute(p, op),
        Opcode::Reduce => misc::print_reduce(p, op),
        Opcode::Scan => misc::print_scan(p, op),
        Opcode::MmaI | Opcode::MmaF => misc::print_mma(p, op),
        Opcode::CmpI => misc::print_cmpi(p, op),
        Opcode::CmpF => misc::print_cmpf(p, op),
        Opcode::Select => misc::print_select(p, op),
        _ => generic::print_simple(p, op),
    }
}
