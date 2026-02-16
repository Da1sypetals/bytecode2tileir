## Obligatory Readings

llm_docs/libs/iterate.md
llm_docs/rust/float16.md
llm_docs/rust/quality.md
llm_docs/tileir/general.md
llm_docs/tileir/types.md


## Rules

UNLESS I EXPLICITLY REQUEST IT, **PROHIBITED** to create or modify any document. This includes creation via command line or through a tool call; both are **PROHIBITED**.

**STRICTLY PROHIBITED** to embed excessively long code (Python, Rust, etc.) within bash commands. Specifically, if the Python code will exceed 8 lines, you must first write it to a temporary file (created via a tool call in the current directory, not in the terminal) and then execute that file.

If you wish to write into a file, use **tool call**. IT IS STRICTLY PROHIBITED to write into files with heredocs (e.g. `cat ... > file.py`).

When provided a web link or documentation, you must first understand the **COMPLETE** content of the linked page before beginning the task. If you discover incorrect library usage, you should **RE-VIEW** the **COMPLETE** content of the provided web link, **RATHER THAN** consulting the library's source code. If you wish to view the library's source code, you must seek approval.

When provided a web link, DO NOT think "document may exist in local" and try to find it. Your ONLY option is to fetch and read the **COMPLETE** content of the provided web link.

Python specific: Unless explicitly requested, the use of try-except is **PROHIBITED**. To reiterate: unless explicitly requested, the use of try-except is **PROHIBITED**.

Code should be written to seek fast-fail; it should crash in place at the point of error, **RATHER THAN** catching errors.

If you are unable to fulfill my request  (because of theoretical limitations, NOT practicality like "the task is too long", where you should just execute it), you should actively terminate the task and tell me "TASK CANNOT BE COMPLETED," **RATHER THAN** attempting to skip, omit, or deceive me.
    - A bug is not this case. You should solve the bug by YOURSELF, not asking me to solve bugs, nor stopping.

Use the Python from the currently active conda environment, not the system Python.

If you want to write exceedingly long content into 1 file, you should use file writing tools multiple tiles to append to the same file. DO NOT create huge bash command to concatenate files.

You are STRICTLY PROHIBITED from creating markdown files IN ANY FORM (tool call, heredoc, etc.) unless explicitly specified, either it is summary, specification, notes, etc. 

Remember to also read EVERYTHING under llm/docs_rust.

You are STRICTLY PROHIBITED to make ANY kind of simplification or suggest me to simplify something, or use mocks in any kind.

DO NOT DARE TO TRY TO REMOVE ANY TEST TO "PASS" THEM YOU IDIOT. YOU WILL BE FUCKED IF YOU DO THIS IDIOTIC ACT.

When implementing interpreter, you MUST NOT modify any file under src/cuda_tile_ir.

When reading docs, DO NOT filter or search (sed, grep, etc.), READ COMPLETE FILE.

Your EVERY response should conform to llm_docs/rust/float16.md

We EXPLICITLY do not mind (or even encourage if unsafe does the job better) using unsafe, and you should NOT prioritize using safe code just because it is safe.

The typical style is implement tile associated method in float.rs, and call those methods in execute_xxx methods. Do not put heavy logic or computation in execute_xxx methods. You should refer to existing implementations for this.

It is PROHIBITED to first create a `Vec` then construct `ndarray` Array with it. Instead you should create `ndarray` Array in the first place and update arrays with indexing.