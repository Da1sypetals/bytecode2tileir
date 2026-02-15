/// depender depends on dependee
struct Dependency {
    /// SSA id
    pub(crate) depender: u32,

    /// SSA id
    pub(crate) dependee: u32,
}
