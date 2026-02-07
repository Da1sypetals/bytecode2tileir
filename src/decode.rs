//! Public API for bytecode parsing + IR decoding.

use crate::bytecode::error::Result;

#[derive(Debug, Clone)]
pub struct DecodeOptions {
    pub lazy_functions: bool,
    pub attach_debug: bool,
    pub keep_const_refs: bool,
}

impl Default for DecodeOptions {
    fn default() -> Self {
        Self {
            lazy_functions: true,
            attach_debug: true,
            keep_const_refs: true,
        }
    }
}

pub fn parse_bytecode<'a>(data: &'a [u8]) -> Result<crate::bytecode::module::BytecodeModule<'a>> {
    let file = crate::bytecode::BytecodeFile::parse(data)?;
    crate::bytecode::module::BytecodeModule::parse(file)
}

pub fn decode_module<'a>(data: &'a [u8], opts: &DecodeOptions) -> Result<crate::cuda_tile_ir::cfg::Module> {
    let mut bc = parse_bytecode(data)?;

    if !opts.attach_debug {
        bc.ctx.debug = None;
    }

    let mut builder = crate::cuda_tile_ir::builder::IrBuilder::new("kernels");

    // Copy constant pool into owned module storage.
    builder.consts = crate::cuda_tile_ir::consts::ConstPool::new(
        bc.ctx.consts.offsets().to_vec(),
        bc.ctx.consts.blob().to_vec(),
    );

    // Intern all bytecode types into the arena (TypeId matches bytecode indices).
    bc.intern_all_types(&mut builder.arena)?;

    // Globals
    let globals = bc.globals.clone();
    for g in &globals {
        bc.decode_global_to_ir(g, &mut builder)?;
    }

    // Functions
    let decode_bodies = !opts.lazy_functions;
    for i in 0..bc.funcs.len() {
        bc.decode_function_to_ir(
            crate::bytecode::format::FuncId(i as u32),
            &mut builder,
            decode_bodies,
        )?;
    }

    Ok(builder.build())
}
