//! Owned constant pool for semantic IR.

use crate::bytecode::format::ConstId;

#[derive(Debug, Default, Clone)]
pub struct ConstPool {
    offsets: Vec<u64>,
    blob: Vec<u8>,
}

impl ConstPool {
    pub fn new(offsets: Vec<u64>, blob: Vec<u8>) -> Self {
        Self { offsets, blob }
    }

    pub fn empty() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.offsets.len()
    }

    pub fn is_empty(&self) -> bool {
        self.offsets.is_empty()
    }

    pub fn get(&self, id: ConstId) -> Option<&[u8]> {
        let idx = id.0 as usize;
        if idx >= self.offsets.len() {
            return None;
        }
        let start = self.offsets[idx] as usize;
        let end = if idx + 1 < self.offsets.len() {
            self.offsets[idx + 1] as usize
        } else {
            self.blob.len()
        };
        self.blob.get(start..end)
    }
}
