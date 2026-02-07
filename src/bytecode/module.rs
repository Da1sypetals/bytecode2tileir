//! BytecodeModule: parsed section tables + function/global declarations.

use crate::bytecode::consts::ConstPool;
use crate::bytecode::debug::DebugTable;
use crate::bytecode::decode_body::BodyDecoder;
use crate::bytecode::error::{BytecodeError, Result};
use crate::bytecode::file::BytecodeFile;
use crate::bytecode::format::{
    FuncId, SECTION_CONSTS, SECTION_DEBUG, SECTION_FUNCS, SECTION_GLOBALS, SECTION_STRINGS,
    SECTION_TYPES, Version,
};
use crate::bytecode::funcs::{FunctionDecl, parse_function_table};
use crate::bytecode::globals::{GlobalDecl, parse_global_table};
use crate::bytecode::strings::StringTable;
use crate::bytecode::types::TypeTable;
use crate::cuda_tile_ir::builder::IrBuilder;
use crate::cuda_tile_ir::cfg::Function as IrFunction;
use crate::cuda_tile_ir::cfg::Global as IrGlobal;
use crate::cuda_tile_ir::debug::Location;
use crate::cuda_tile_ir::ids::{BlockId, FunctionId, RegionId, ValueId};
use crate::cuda_tile_ir::types::Type;
use crate::cuda_tile_ir::value::{ValueData, ValueDef};

pub struct BytecodeContext<'a> {
    pub version: Version,
    pub strings: StringTable<'a>,
    pub types: TypeTable<'a>,
    pub consts: ConstPool<'a>,
    pub debug: Option<DebugTable<'a>>,
}

pub struct BytecodeModule<'a> {
    pub ctx: BytecodeContext<'a>,
    pub globals: Vec<GlobalDecl>,
    pub funcs: Vec<FunctionDecl<'a>>,
}

impl<'a> BytecodeModule<'a> {
    pub fn parse(file: BytecodeFile<'a>) -> Result<Self> {
        let strings_bytes = file
            .section_bytes(SECTION_STRINGS)
            .ok_or_else(|| BytecodeError::ParseError("missing required string section".into()))?;
        let strings = StringTable::parse(strings_bytes)
            .map_err(|e| BytecodeError::with_context("parse strings section", e))?;

        let types_bytes = file
            .section_bytes(SECTION_TYPES)
            .ok_or_else(|| BytecodeError::ParseError("missing required type section".into()))?;
        let mut types = TypeTable::parse(types_bytes)
            .map_err(|e| BytecodeError::with_context("parse types section", e))?;

        let consts = if let Some(bytes) = file.section_bytes(SECTION_CONSTS) {
            ConstPool::parse(bytes)
                .map_err(|e| BytecodeError::with_context("parse consts section", e))?
        } else {
            ConstPool::empty()
        };

        let debug = if let Some(bytes) = file.section_bytes(SECTION_DEBUG) {
            Some(
                DebugTable::parse(bytes)
                    .map_err(|e| BytecodeError::with_context("parse debug section", e))?,
            )
        } else {
            None
        };

        let globals = if let Some(bytes) = file.section_bytes(SECTION_GLOBALS) {
            parse_global_table(bytes)
                .map_err(|e| BytecodeError::with_context("parse globals section", e))?
        } else {
            Vec::new()
        };

        let funcs_bytes = file
            .section_bytes(SECTION_FUNCS)
            .ok_or_else(|| BytecodeError::ParseError("missing required function section".into()))?;
        let funcs = parse_function_table(funcs_bytes, &strings, &mut types, &consts)
            .map_err(|e| BytecodeError::with_context("parse function table", e))?;

        Ok(Self {
            ctx: BytecodeContext {
                version: file.version,
                strings,
                types,
                consts,
                debug,
            },
            globals,
            funcs,
        })
    }

    pub fn func(&self, id: FuncId) -> Option<&FunctionDecl<'a>> {
        self.funcs.get(id.0 as usize)
    }

    pub fn intern_all_types(&mut self, arena: &mut crate::cuda_tile_ir::arena::IrArena) -> Result<()> {
        if !arena.types.is_empty() {
            return Ok(());
        }
        for i in 0..self.ctx.types.len() {
            let ty = self.ctx.types.get(crate::cuda_tile_ir::ids::TypeId(i as u32))?;
            arena.intern_type(ty);
        }
        Ok(())
    }

    pub fn decode_function_to_ir(
        &mut self,
        id: FuncId,
        builder: &mut IrBuilder,
        decode_body: bool,
    ) -> Result<FunctionId> {
        let func = self
            .func(id)
            .ok_or_else(|| BytecodeError::IndexOutOfBounds {
                kind: "function",
                index: id.0 as u64,
                max: self.funcs.len(),
            })?
            .clone();

        self.intern_all_types(&mut builder.arena)?;

        // Resolve function signature parameters.
        let sig_ty = builder
            .arena
            .types
            .get(func.signature.0 as usize)
            .ok_or_else(|| BytecodeError::IndexOutOfBounds {
                kind: "type",
                index: func.signature.0 as u64,
                max: builder.arena.types.len(),
            })?;
        let params = match sig_ty {
            Type::Func { params, .. } => params.clone(),
            _ => Vec::new(),
        };

        let name = self.ctx.strings.get(func.name)?.to_string();
        let loc = if let Some(debug) = self.ctx.debug.as_mut() {
            debug.resolve_location(func.loc, &self.ctx.strings, &mut self.ctx.types)?
        } else {
            Location::default()
        };

        let opt_hints = func.opt_hints.map(|raw| {
            // Reuse BodyDecoder's interner logic.
            let mut tmp = BodyDecoder::new(&mut self.ctx, &mut builder.arena);
            tmp.intern_raw_attr(raw)
        });

        let (body_region, entry_block, entry_args) = create_entry_body(&mut builder.arena, &params);

        if decode_body {
            let mut decoder = BodyDecoder::new(&mut self.ctx, &mut builder.arena);
            decoder.value_stack = entry_args;
            decoder
                .decode_function_body(func.body, entry_block, func.loc)
                .map_err(|e| {
                    BytecodeError::with_context(
                        format!("decode body of function {} (id {})", name.as_str(), id.0),
                        e,
                    )
                })?;
        }

        let ir_func = IrFunction {
            name,
            signature: func.signature,
            flags: func.flags,
            loc,
            opt_hints,
            body: body_region,
        };
        Ok(builder.add_function(ir_func))
    }

    pub fn decode_global_to_ir(
        &mut self,
        g: &GlobalDecl,
        builder: &mut IrBuilder,
    ) -> Result<crate::cuda_tile_ir::ids::GlobalId> {
        self.intern_all_types(&mut builder.arena)?;
        let name = self.ctx.strings.get(g.name)?.to_string();
        let ir_g = IrGlobal {
            name,
            ty: g.ty,
            init: g.init,
            alignment: g.alignment,
        };
        Ok(builder.add_global(ir_g))
    }
}

fn create_entry_body(
    arena: &mut crate::cuda_tile_ir::arena::IrArena,
    params: &[crate::cuda_tile_ir::ids::TypeId],
) -> (RegionId, BlockId, Vec<ValueId>) {
    let block = arena.new_block(crate::cuda_tile_ir::cfg::Block {
        args: Vec::new(),
        ops: Vec::new(),
    });

    let mut args = Vec::with_capacity(params.len());
    for (i, &ty) in params.iter().enumerate() {
        let v = arena.new_value(ValueData {
            ty,
            def: ValueDef::BlockArg {
                block,
                index: i as u32,
            },
        });
        arena.blocks[block.0 as usize].args.push(v);
        args.push(v);
    }

    let region = arena.new_region(crate::cuda_tile_ir::cfg::Region {
        blocks: vec![block],
    });
    (region, block, args)
}
