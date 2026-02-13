UNLESS I EXPLICITLY REQUEST IT, **PROHIBITED** to create or modify any document. This includes creation via command line or through a tool call; both are **PROHIBITED**.

**STRICTLY PROHIBITED** to embed excessively long Python code within bash commands. Specifically, if the Python code will exceed 8 lines, you must first write it to a temporary file (created via a tool call in the current directory, not in the terminal) and then execute that file.

When providing a web link, you must first understand the **COMPLETE** content of the linked page before beginning the task. If you discover incorrect library usage, you should **RE-VIEW** the **COMPLETE** content of the provided web link, **RATHER THAN** consulting the library's source code. If you wish to view the library's source code, you must seek approval.

Unless explicitly requested, the use of try-except is **PROHIBITED**. To reiterate: unless explicitly requested, the use of try-except is **PROHIBITED**.

Code should be written to seek fast-fail; it should crash in place at the point of error, **RATHER THAN** catching errors.

If you are unable to fulfill my request, you should actively terminate the task and tell me "TASK CANNOT BE COMPLETED," **RATHER THAN** attempting to skip, omit, or deceive me.

Use the Python from the currently active conda environment, not the system Python. If conda is not found, run `/opt/conda/bin/conda init bash`.
