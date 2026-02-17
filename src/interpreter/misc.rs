use crate::cuda_tile_ir::ir::Operation;
use crate::interpreter::data_structures::interpreter::ExecutionContext;

impl ExecutionContext<'_> {
    /// Execute the `assume` operation.
    ///
    /// `assume` passes through the value unchanged. The predicate is ignored
    /// for now since the interpreter does not perform optimization.
    pub(crate) fn execute_assume(&mut self, op: &Operation) {
        // FIXME: Ignore predicate for now - interpreter does not use optimization hints
        let src_value = self.get_value(op.operands[0]).clone();
        self.set_value(op.results[0], src_value);
    }
}
