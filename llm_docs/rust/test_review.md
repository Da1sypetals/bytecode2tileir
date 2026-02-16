# Test Review Instructions

## Objective

*   Act as a Test Validity Auditor to determine if tests provide genuine verification of system behavior rather than merely simulating success to satisfy a pipeline.

## Non-goal

*   Ignore code style, code quality, naming conventions, formatting, DRY principles, comment quality, file organization, or general maintainability concerns.
*   Ignore coverage metrics, branch percentages, or any quantitative measure of how much code is touched during execution.
*   Ignore architectural patterns, design decisions, modularity, dependency injection styles, structural elegance, or performance characteristics.

## Goal

*   Verify assertions check for logic correctness against specific expected values rather than just checking return metadata, existence, type, or overly broad constraints.
*   Confirm the test invokes the real implemented API or logic rather than calling mocks, stubs, copy-paste implemented code, or wrappers that bypass the actual computation.
*   Detect cheating approaches such as suppressing errors, catching exceptions without failing, manipulating setup to let all tests pass regardless of implementation quality, asserting tautologies.
*   Confirm tests do not use very trivial test data, like 1-d or small 2-d tiles with only a dozen of elements, each dimension of the tile of size 2~4, which is too small and trivial to be representative of real-world data.

## Your output

For EACH test function, if they have not cheated, report; if they HAVE cheated, point out where and how to fix.

DO NOT add quotes for cheating tests. They are cheating, CHEATING, not "cheating".