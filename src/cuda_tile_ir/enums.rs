//! Shared enums for the semantic IR.

/// Float kind used by Tile IR types and attributes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FloatKind {
    F16,
    BF16,
    F32,
    TF32,
    F64,
    F8E4M3FN,
    F8E5M2,
}

impl FloatKind {
    pub fn bit_width(self) -> u32 {
        match self {
            Self::F8E4M3FN | Self::F8E5M2 => 8,
            Self::F16 | Self::BF16 => 16,
            Self::TF32 => 19,
            Self::F32 => 32,
            Self::F64 => 64,
        }
    }
}

/// Macro to define enums with TryFrom<u32> implementation.
macro_rules! define_enum {
    ($(#[$meta:meta])* $vis:vis enum $name:ident { $($variant:ident = $val:expr),* $(,)? }) => {
        $(#[$meta])*
        $vis enum $name { $($variant = $val),* }

        impl TryFrom<u32> for $name {
            type Error = ();
            fn try_from(v: u32) -> Result<Self, ()> {
                match v { $($val => Ok(Self::$variant),)* _ => Err(()) }
            }
        }
    };
}

define_enum! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub enum PaddingValue {
        Zero = 0,
        NegZero = 1,
        Nan = 2,
        PosInf = 3,
        NegInf = 4,
    }
}

define_enum! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub enum RoundingMode {
        NearestEven = 0,
        Zero = 1,
        NegativeInf = 2,
        PositiveInf = 3,
        Approx = 4,
        Full = 5,
        NearestIntToZero = 6,
    }
}

define_enum! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub enum Signedness {
        Unsigned = 0,
        Signed = 1,
    }
}

define_enum! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub enum ComparisonPredicate {
        Equal = 0,
        NotEqual = 1,
        LessThan = 2,
        LessThanOrEqual = 3,
        GreaterThan = 4,
        GreaterThanOrEqual = 5,
    }
}

define_enum! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub enum ComparisonOrdering {
        Unordered = 0,
        Ordered = 1,
    }
}

define_enum! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub enum AtomicRMWMode {
        And = 0,
        Or = 1,
        Xor = 2,
        Add = 3,
        AddF = 4,
        Max = 5,
        Min = 6,
        UMax = 7,
        UMin = 8,
        Xchg = 9,
    }
}

define_enum! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub enum MemoryScope {
        TlBlk = 0,
        Device = 1,
        Sys = 2,
    }
}

define_enum! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub enum MemoryOrdering {
        Weak = 0,
        Relaxed = 1,
        Acquire = 2,
        Release = 3,
        AcqRel = 4,
    }
}

define_enum! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub enum IntegerOverflow {
        None = 0,
        Nsw = 1,
        Nuw = 2,
        Nw = 3,
    }
}
