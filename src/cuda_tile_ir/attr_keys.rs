//! Operation attribute keys.

/// Operation attribute key.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OpAttrKey {
    // Common
    Value,
    OperandSegmentSizes,
    OptimizationHints,

    // Arithmetic
    RoundingMode,
    FlushToZero,
    PropagateNan,
    Overflow,
    Signedness,
    LhsSignedness,
    RhsSignedness,
    SignednessLhs,
    SignednessRhs,

    // Comparison
    ComparisonPredicate,
    ComparisonOrdering,

    // Memory
    MemoryOrderingSemantics,
    MemoryScope,
    Mode,

    // Tile
    Dim,
    Permutation,
    Identities,
    Reverse,

    // Misc
    GlobalName,
    Name,
    Format,
    Message,
    Predicate,
}

impl OpAttrKey {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Value => "value",
            Self::OperandSegmentSizes => "operand_segment_sizes",
            Self::OptimizationHints => "optimization_hints",
            Self::RoundingMode => "rounding_mode",
            Self::FlushToZero => "flush_to_zero",
            Self::PropagateNan => "propagate_nan",
            Self::Overflow => "overflow",
            Self::Signedness => "signedness",
            Self::LhsSignedness => "lhs_signedness",
            Self::RhsSignedness => "rhs_signedness",
            Self::SignednessLhs => "signedness_lhs",
            Self::SignednessRhs => "signedness_rhs",
            Self::ComparisonPredicate => "comparison_predicate",
            Self::ComparisonOrdering => "comparison_ordering",
            Self::MemoryOrderingSemantics => "memory_ordering_semantics",
            Self::MemoryScope => "memory_scope",
            Self::Mode => "mode",
            Self::Dim => "dim",
            Self::Permutation => "permutation",
            Self::Identities => "identities",
            Self::Reverse => "reverse",
            Self::GlobalName => "global_name",
            Self::Name => "name",
            Self::Format => "format",
            Self::Message => "message",
            Self::Predicate => "predicate",
        }
    }

    pub fn from_str(s: &str) -> Option<Self> {
        Some(match s {
            "value" => Self::Value,
            "operand_segment_sizes" => Self::OperandSegmentSizes,
            "optimization_hints" => Self::OptimizationHints,
            "rounding_mode" => Self::RoundingMode,
            "flush_to_zero" => Self::FlushToZero,
            "propagate_nan" => Self::PropagateNan,
            "overflow" => Self::Overflow,
            "signedness" => Self::Signedness,
            "lhs_signedness" => Self::LhsSignedness,
            "rhs_signedness" => Self::RhsSignedness,
            "signedness_lhs" => Self::SignednessLhs,
            "signedness_rhs" => Self::SignednessRhs,
            "comparison_predicate" => Self::ComparisonPredicate,
            "comparison_ordering" => Self::ComparisonOrdering,
            "memory_ordering_semantics" => Self::MemoryOrderingSemantics,
            "memory_scope" => Self::MemoryScope,
            "mode" => Self::Mode,
            "dim" => Self::Dim,
            "permutation" => Self::Permutation,
            "identities" => Self::Identities,
            "reverse" => Self::Reverse,
            "global_name" => Self::GlobalName,
            "name" => Self::Name,
            "format" => Self::Format,
            "message" => Self::Message,
            "predicate" => Self::Predicate,
            _ => return None,
        })
    }
}

impl std::fmt::Display for OpAttrKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}
