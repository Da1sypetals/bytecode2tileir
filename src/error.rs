//! Error types for bytecode2mlir

use thiserror::Error;

#[derive(Error, Debug)]
pub enum Error {
    #[error("bytecode error: {0}")]
    Bytecode(#[from] crate::bytecode::error::BytecodeError),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("config error: {0}")]
    Config(String),
}

impl Error {
    pub fn config(msg: impl Into<String>) -> Self {
        Self::Config(msg.into())
    }
}

pub type Result<T> = std::result::Result<T, Error>;
