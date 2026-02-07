//! Function body decoder: instruction stream -> semantic IR.

use crate::bytecode::attrs::RawAttr;
use crate::bytecode::error::{BytecodeError, Result};
use crate::bytecode::format::{LocIndex, TypeId};
use crate::bytecode::module::BytecodeContext;
use crate::bytecode::reader::{ByteRead, Cursor};
use crate::cuda_tile_ir::arena::IrArena;
use crate::cuda_tile_ir::attrs::Attr;
use crate::cuda_tile_ir::debug::Location;
use crate::cuda_tile_ir::ids::{AttrId, BlockId, OpId, RegionId, ValueId};
use crate::cuda_tile_ir::ir::Operation;
use crate::cuda_tile_ir::value::{ValueData, ValueDef};

#[path = "sections/ops_arith.rs"]
mod ops_arith;
#[path = "sections/ops_control.rs"]
mod ops_control;
#[path = "sections/ops_mem.rs"]
mod ops_mem;
#[path = "sections/ops_misc.rs"]
mod ops_misc;
#[path = "sections/ops_tile.rs"]
mod ops_tile;

type OpDecodeFn =
    for<'a, 'm> fn(&mut BodyDecoder<'a, 'm>, &mut dyn ByteRead<'a>, Location) -> Result<OpId>;

pub struct BodyDecoder<'a, 'm> {
    pub ctx: &'m mut BytecodeContext<'a>,
    pub arena: &'m mut IrArena,
    pub value_stack: Vec<ValueId>,
    debug_func_loc: LocIndex,
    debug_op_index: u32,
}

impl<'a, 'm> BodyDecoder<'a, 'm> {
    pub fn new(ctx: &'m mut BytecodeContext<'a>, arena: &'m mut IrArena) -> Self {
        Self {
            ctx,
            arena,
            value_stack: Vec::new(),
            debug_func_loc: LocIndex(0),
            debug_op_index: 0,
        }
    }

    pub fn decode_function_body(
        &mut self,
        body: &'a [u8],
        entry_block: BlockId,
        func_loc: LocIndex,
    ) -> Result<()> {
        self.debug_func_loc = func_loc;
        self.debug_op_index = 0;

        let mut r = Cursor::new(body);
        while r.remaining() > 0 {
            let start = r.pos();
            let op = self.decode_op(&mut r).map_err(|e| {
                BytecodeError::with_context(format!("decode op stream at offset {start}"), e)
            })?;
            self.arena.blocks[entry_block.0 as usize].ops.push(op);
        }
        Ok(())
    }

    fn decode_region(&mut self, r: &mut dyn ByteRead<'a>) -> Result<RegionId> {
        let num_blocks = r.read_var_u64()? as usize;
        let mut blocks = Vec::with_capacity(num_blocks);
        for _ in 0..num_blocks {
            blocks.push(self.decode_block(r)?);
        }
        Ok(self.arena.new_region(crate::cuda_tile_ir::cfg::Region { blocks }))
    }

    fn decode_block(&mut self, r: &mut dyn ByteRead<'a>) -> Result<BlockId> {
        let num_args = r.read_var_u64()? as usize;

        let block = self.arena.new_block(crate::cuda_tile_ir::cfg::Block {
            args: Vec::new(),
            ops: Vec::new(),
        });

        let saved = self.value_stack.len();
        for i in 0..num_args {
            let ty = TypeId(read_u32_var(r)?);
            let v = self.arena.new_value(ValueData {
                ty,
                def: ValueDef::BlockArg {
                    block,
                    index: i as u32,
                },
            });
            self.arena.blocks[block.0 as usize].args.push(v);
            self.value_stack.push(v);
        }

        let num_ops = r.read_var_u64()? as usize;
        for _ in 0..num_ops {
            let op = self.decode_op(r)?;
            self.arena.blocks[block.0 as usize].ops.push(op);
        }

        self.value_stack.truncate(saved);
        Ok(block)
    }

    fn decode_op(&mut self, r: &mut dyn ByteRead<'a>) -> Result<OpId> {
        let start = r.pos();
        let opcode_var = r.read_var_u64()?;
        let opcode_byte: u8 = u8::try_from(opcode_var).map_err(|_| {
            BytecodeError::ParseError(format!("opcode varint {opcode_var} does not fit into u8"))
        })?;

        let loc = if let Some(debug) = self.ctx.debug.as_mut() {
            debug.resolve_op_location(
                self.debug_func_loc,
                self.debug_op_index,
                &self.ctx.strings,
                &mut self.ctx.types,
            )?
        } else {
            Location::default()
        };
        self.debug_op_index = self.debug_op_index.saturating_add(1);

        let decoder =
            opcode_decoder(opcode_byte).ok_or(BytecodeError::UnknownOpcode(opcode_byte))?;
        decoder(self, r, loc).map_err(|e| {
            let name = crate::cuda_tile_ir::Opcode::from_u64(opcode_byte as u64)
                .map(|op| op.name())
                .unwrap_or("unknown");
            BytecodeError::with_context(
                format!("opcode 0x{opcode_byte:02X} ({name}) at offset {start}"),
                e,
            )
        })
    }

    pub fn read_value(&self, idx: u32) -> Result<ValueId> {
        let idx_usize = idx as usize;
        self.value_stack
            .get(idx_usize)
            .copied()
            .ok_or_else(|| BytecodeError::IndexOutOfBounds {
                kind: "value",
                index: idx as u64,
                max: self.value_stack.len(),
            })
    }

    pub fn read_value_from_stream(&self, r: &mut dyn ByteRead<'a>) -> Result<ValueId> {
        let idx = read_u32_var(r)?;
        self.read_value(idx)
    }

    pub fn read_type_id_from_stream(&self, r: &mut dyn ByteRead<'a>) -> Result<TypeId> {
        Ok(TypeId(read_u32_var(r)?))
    }

    pub fn intern_raw_attr(&mut self, raw: RawAttr) -> AttrId {
        intern_raw_attr(self.arena, raw)
    }

    pub fn build_op(
        &mut self,
        opcode: crate::cuda_tile_ir::Opcode,
        operands: smallvec::SmallVec<[ValueId; 4]>,
        result_tys: impl IntoIterator<Item = TypeId>,
        attrs: crate::cuda_tile_ir::ir::AttrMap,
        regions: Vec<RegionId>,
        loc: Location,
    ) -> OpId {
        let op_id = self.arena.new_op(Operation {
            opcode,
            operands,
            results: smallvec::SmallVec::new(),
            attrs,
            regions,
            loc,
        });

        let mut results = smallvec::SmallVec::<[ValueId; 2]>::new();
        for (i, ty) in result_tys.into_iter().enumerate() {
            let v = self.arena.new_value(ValueData {
                ty,
                def: ValueDef::OpResult {
                    op: op_id,
                    index: i as u32,
                },
            });
            results.push(v);
            self.value_stack.push(v);
        }

        self.arena.ops[op_id.0 as usize].results = results;
        op_id
    }

    pub fn decode_region_from_stream(&mut self, r: &mut dyn ByteRead<'a>) -> Result<RegionId> {
        self.decode_region(r)
    }

    pub fn intern_builtin_int(&mut self, width: u8) -> TypeId {
        if let Some((idx, _)) = self
            .arena
            .types
            .iter()
            .enumerate()
            .find(|(_, t)| matches!(t, crate::cuda_tile_ir::types::Type::Int { width: w } if *w == width))
        {
            return TypeId(idx as u32);
        }
        self.arena
            .intern_type(crate::cuda_tile_ir::types::Type::Int { width })
    }
}

fn read_u32_var<'a>(r: &mut dyn ByteRead<'a>) -> Result<u32> {
    let v = r.read_var_u64()?;
    u32::try_from(v).map_err(|_| BytecodeError::ParseError("varint does not fit into u32".into()))
}

fn intern_raw_attr(arena: &mut IrArena, raw: RawAttr) -> AttrId {
    match raw {
        RawAttr::Unit => arena.intern_attr(Attr::Unit),
        RawAttr::Bool(v) => arena.intern_attr(Attr::Bool(v)),
        RawAttr::Int { ty, value } => arena.intern_attr(Attr::Int { ty, value }),
        RawAttr::Float { kind, bits } => arena.intern_attr(Attr::Float { kind, bits }),
        RawAttr::Type(t) => arena.intern_attr(Attr::Type(t)),
        RawAttr::String(s) => arena.intern_attr(Attr::String(s)),
        RawAttr::Array(v) => {
            let ids = v.into_iter().map(|a| intern_raw_attr(arena, a)).collect();
            arena.intern_attr(Attr::Array(ids))
        }
        RawAttr::DenseElements { ty, storage } => {
            arena.intern_attr(Attr::DenseElements { ty, storage })
        }
        RawAttr::Dict(entries) => {
            let entries = entries
                .into_iter()
                .map(|(k, v)| (k, intern_raw_attr(arena, v)))
                .collect();
            arena.intern_attr(Attr::Dict(entries))
        }
        RawAttr::OptimizationHints(entries) => {
            let entries = entries
                .into_iter()
                .map(|(k, v)| (k, intern_raw_attr(arena, v)))
                .collect();
            arena.intern_attr(Attr::OptimizationHints(entries))
        }
        RawAttr::DivBy {
            divisor,
            unsigned_int,
            every,
            along,
        } => arena.intern_attr(Attr::DivBy {
            divisor,
            unsigned_int,
            every,
            along,
        }),
        RawAttr::SameElements(v) => arena.intern_attr(Attr::SameElements(v)),
        RawAttr::Bounded { lb, ub } => arena.intern_attr(Attr::Bounded { lb, ub }),
        RawAttr::NonNegative => arena.intern_attr(Attr::NonNegative),
    }
}

fn opcode_decoder(byte: u8) -> Option<OpDecodeFn> {
    // NOTE: We avoid a giant match by dispatching through a dense table.
    // The table is built once and then indexed by the opcode byte.
    static TABLE: [Option<OpDecodeFn>; 256] = build_decoder_table();
    TABLE[byte as usize]
}

const fn build_decoder_table() -> [Option<OpDecodeFn>; 256] {
    use crate::cuda_tile_ir::Opcode;
    let mut t: [Option<OpDecodeFn>; 256] = [None; 256];

    // Arithmetic
    t[Opcode::AbsF as usize] = Some(ops_arith::decode_absf);
    t[Opcode::AbsI as usize] = Some(ops_arith::decode_absi);
    t[Opcode::AddF as usize] = Some(ops_arith::decode_addf);
    t[Opcode::AddI as usize] = Some(ops_arith::decode_addi);
    t[Opcode::AndI as usize] = Some(ops_arith::decode_andi);
    t[Opcode::Bitcast as usize] = Some(ops_arith::decode_bitcast);
    t[Opcode::Ceil as usize] = Some(ops_arith::decode_ceil);
    t[Opcode::CmpF as usize] = Some(ops_arith::decode_cmpf);
    t[Opcode::CmpI as usize] = Some(ops_arith::decode_cmpi);
    t[Opcode::Cos as usize] = Some(ops_arith::decode_cos);
    t[Opcode::CosH as usize] = Some(ops_arith::decode_cosh);
    t[Opcode::DivF as usize] = Some(ops_arith::decode_divf);
    t[Opcode::DivI as usize] = Some(ops_arith::decode_divi);
    t[Opcode::Exp as usize] = Some(ops_arith::decode_exp);
    t[Opcode::Exp2 as usize] = Some(ops_arith::decode_exp2);
    t[Opcode::ExtI as usize] = Some(ops_arith::decode_exti);
    t[Opcode::Floor as usize] = Some(ops_arith::decode_floor);
    t[Opcode::Fma as usize] = Some(ops_arith::decode_fma);
    t[Opcode::FToF as usize] = Some(ops_arith::decode_ftof);
    t[Opcode::FToI as usize] = Some(ops_arith::decode_ftoi);
    t[Opcode::IToF as usize] = Some(ops_arith::decode_itof);
    t[Opcode::Log as usize] = Some(ops_arith::decode_log);
    t[Opcode::Log2 as usize] = Some(ops_arith::decode_log2);
    t[Opcode::MaxF as usize] = Some(ops_arith::decode_maxf);
    t[Opcode::MaxI as usize] = Some(ops_arith::decode_maxi);
    t[Opcode::MinF as usize] = Some(ops_arith::decode_minf);
    t[Opcode::MinI as usize] = Some(ops_arith::decode_mini);
    t[Opcode::MulF as usize] = Some(ops_arith::decode_mulf);
    t[Opcode::MulhiI as usize] = Some(ops_arith::decode_mulhii);
    t[Opcode::MulI as usize] = Some(ops_arith::decode_muli);
    t[Opcode::NegF as usize] = Some(ops_arith::decode_negf);
    t[Opcode::NegI as usize] = Some(ops_arith::decode_negi);
    t[Opcode::OrI as usize] = Some(ops_arith::decode_ori);
    t[Opcode::Pow as usize] = Some(ops_arith::decode_pow);
    t[Opcode::PtrToInt as usize] = Some(ops_arith::decode_ptr_to_int);
    t[Opcode::PtrToPtr as usize] = Some(ops_arith::decode_ptr_to_ptr);
    t[Opcode::RemF as usize] = Some(ops_arith::decode_remf);
    t[Opcode::RemI as usize] = Some(ops_arith::decode_remi);
    t[Opcode::Rsqrt as usize] = Some(ops_arith::decode_rsqrt);
    t[Opcode::Select as usize] = Some(ops_arith::decode_select);
    t[Opcode::ShLI as usize] = Some(ops_arith::decode_shli);
    t[Opcode::ShRI as usize] = Some(ops_arith::decode_shri);
    t[Opcode::Sin as usize] = Some(ops_arith::decode_sin);
    t[Opcode::SinH as usize] = Some(ops_arith::decode_sinh);
    t[Opcode::Sqrt as usize] = Some(ops_arith::decode_sqrt);
    t[Opcode::SubF as usize] = Some(ops_arith::decode_subf);
    t[Opcode::SubI as usize] = Some(ops_arith::decode_subi);
    t[Opcode::Tan as usize] = Some(ops_arith::decode_tan);
    t[Opcode::TanH as usize] = Some(ops_arith::decode_tanh);
    t[Opcode::TruncI as usize] = Some(ops_arith::decode_trunci);
    t[Opcode::XOrI as usize] = Some(ops_arith::decode_xori);
    t[Opcode::IntToPtr as usize] = Some(ops_arith::decode_int_to_ptr);

    // Misc
    t[Opcode::Assert as usize] = Some(ops_misc::decode_assert);
    t[Opcode::Assume as usize] = Some(ops_misc::decode_assume);
    t[Opcode::Constant as usize] = Some(ops_misc::decode_constant);
    t[Opcode::Entry as usize] = Some(ops_misc::decode_entry);
    t[Opcode::GetGlobal as usize] = Some(ops_misc::decode_get_global);
    t[Opcode::GetIndexSpaceShape as usize] = Some(ops_misc::decode_get_index_space_shape);
    t[Opcode::GetNumTileBlocks as usize] = Some(ops_misc::decode_get_num_tile_blocks);
    t[Opcode::GetTensorShape as usize] = Some(ops_misc::decode_get_tensor_shape);
    t[Opcode::GetTileBlockId as usize] = Some(ops_misc::decode_get_tile_block_id);
    t[Opcode::Global as usize] = Some(ops_misc::decode_global);
    t[Opcode::Iota as usize] = Some(ops_misc::decode_iota);
    t[Opcode::JoinTokens as usize] = Some(ops_misc::decode_join_tokens);
    t[Opcode::MakePartitionView as usize] = Some(ops_misc::decode_make_partition_view);
    t[Opcode::MakeTensorView as usize] = Some(ops_misc::decode_make_tensor_view);
    t[Opcode::MakeToken as usize] = Some(ops_misc::decode_make_token);
    t[Opcode::Module as usize] = Some(ops_misc::decode_module);
    t[Opcode::Print as usize] = Some(ops_misc::decode_print);
    t[Opcode::Return as usize] = Some(ops_misc::decode_return);

    // Tile ops
    t[Opcode::Broadcast as usize] = Some(ops_tile::decode_broadcast);
    t[Opcode::Cat as usize] = Some(ops_tile::decode_cat);
    t[Opcode::Extract as usize] = Some(ops_tile::decode_extract);
    t[Opcode::Offset as usize] = Some(ops_tile::decode_offset);
    t[Opcode::Permute as usize] = Some(ops_tile::decode_permute);
    t[Opcode::Reduce as usize] = Some(ops_tile::decode_reduce);
    t[Opcode::Reshape as usize] = Some(ops_tile::decode_reshape);
    t[Opcode::Scan as usize] = Some(ops_tile::decode_scan);
    t[Opcode::MmaF as usize] = Some(ops_tile::decode_mmaf);
    t[Opcode::MmaI as usize] = Some(ops_tile::decode_mmai);

    // Control flow
    t[Opcode::For as usize] = Some(ops_control::decode_for);
    t[Opcode::Loop as usize] = Some(ops_control::decode_loop);
    t[Opcode::If as usize] = Some(ops_control::decode_if);
    t[Opcode::Break as usize] = Some(ops_control::decode_break);
    t[Opcode::Continue as usize] = Some(ops_control::decode_continue);
    t[Opcode::Yield as usize] = Some(ops_control::decode_yield);

    // Memory
    t[Opcode::LoadPtrTko as usize] = Some(ops_mem::decode_load_ptr_tko);
    t[Opcode::LoadViewTko as usize] = Some(ops_mem::decode_load_view_tko);
    t[Opcode::StorePtrTko as usize] = Some(ops_mem::decode_store_ptr_tko);
    t[Opcode::StoreViewTko as usize] = Some(ops_mem::decode_store_view_tko);
    t[Opcode::AtomicCASTko as usize] = Some(ops_mem::decode_atomic_cas_tko);
    t[Opcode::AtomicRMWTko as usize] = Some(ops_mem::decode_atomic_rmw_tko);

    t
}
