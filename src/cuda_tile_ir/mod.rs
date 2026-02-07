//! Semantic IR for FlashTile.
//!
//! This module contains the CuTile IR layer used by the frontend and lowering pipeline.

pub mod arena;
pub mod attr_keys;
pub mod attrs;
pub mod builder;
pub mod cfg;
pub mod consts;
pub mod debug;
pub mod enums;
pub mod ids;
pub mod ir;
pub mod opcodes;
pub mod printer;
pub mod types;
pub mod value;

pub use attr_keys::OpAttrKey;
pub use opcodes::Opcode;
