pub mod attrs;
pub mod consts;
pub mod debug;
pub mod decode_body;
pub mod error;
pub mod file;
pub mod format;
pub mod funcs;
pub mod globals;
pub mod module;
pub mod reader;
pub mod strings;
pub mod table;
pub mod tags;
pub mod types;

#[cfg(test)]
mod tests;

pub use error::{BytecodeError, Result};
pub use file::BytecodeFile;
pub use tags::{AttributeTag, TypeTag};

pub fn is_tilir_bytecode(data: &[u8]) -> bool {
    data.len() >= format::MAGIC.len() && data[..format::MAGIC.len()] == format::MAGIC
}
