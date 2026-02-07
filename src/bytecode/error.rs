use std::fmt;

#[derive(Debug)]
pub enum BytecodeError {
    InvalidMagic,
    UnsupportedVersion {
        major: u8,
        minor: u8,
        tag: u16,
    },
    UnexpectedEof {
        at: usize,
        needed: usize,
        remaining: usize,
    },
    InvalidAlignment(u64),
    DuplicateSection(u8),
    InvalidPadding {
        at: usize,
        expected: u8,
        found: u8,
    },
    InvalidTypeTag(u8),
    InvalidAttrTag(u8),
    IndexOutOfBounds {
        kind: &'static str,
        index: u64,
        max: usize,
    },
    CorruptTable {
        table: &'static str,
        idx: usize,
        offset: u64,
        blob_len: usize,
    },
    InvalidUtf8,
    InvalidEnum {
        name: &'static str,
        value: u64,
    },
    UnknownOpcode(u8),
    ParseError(String),
    Context {
        ctx: String,
        source: Box<BytecodeError>,
    },
}

impl fmt::Display for BytecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidMagic => write!(f, "invalid magic number"),
            Self::UnsupportedVersion { major, minor, tag } => {
                write!(f, "unsupported version {major}.{minor}.{tag}")
            }
            Self::UnexpectedEof {
                at,
                needed,
                remaining,
            } => write!(
                f,
                "unexpected EOF at offset {at} (needed {needed} bytes, have {remaining})"
            ),
            Self::InvalidAlignment(a) => write!(f, "invalid alignment: {a}"),
            Self::DuplicateSection(id) => write!(f, "duplicate section id: 0x{id:02X}"),
            Self::InvalidPadding {
                at,
                expected,
                found,
            } => write!(
                f,
                "invalid padding byte at offset {at}: expected 0x{expected:02X}, found 0x{found:02X}"
            ),
            Self::InvalidTypeTag(tag) => write!(f, "invalid type tag: {tag}"),
            Self::InvalidAttrTag(tag) => write!(f, "invalid attribute tag: {tag}"),
            Self::IndexOutOfBounds { kind, index, max } => {
                write!(f, "{kind} index {index} out of bounds (max {max})")
            }
            Self::CorruptTable {
                table,
                idx,
                offset,
                blob_len,
            } => write!(
                f,
                "corrupt {table} offset table: idx={idx} offset={offset} blob_len={blob_len}"
            ),
            Self::InvalidUtf8 => write!(f, "invalid utf8 string"),
            Self::InvalidEnum { name, value } => write!(f, "invalid {name} enum value: {value}"),
            Self::UnknownOpcode(op) => write!(f, "unknown opcode: 0x{op:02X}"),
            Self::ParseError(msg) => write!(f, "{msg}"),
            Self::Context { ctx, source } => write!(f, "{ctx}: {source}"),
        }
    }
}

impl std::error::Error for BytecodeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Context { source, .. } => Some(source),
            _ => None,
        }
    }
}

impl BytecodeError {
    pub fn with_context(ctx: impl Into<String>, err: BytecodeError) -> BytecodeError {
        BytecodeError::Context {
            ctx: ctx.into(),
            source: Box::new(err),
        }
    }
}

pub type Result<T> = std::result::Result<T, BytecodeError>;
