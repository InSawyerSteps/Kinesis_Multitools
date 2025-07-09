---
description: How to use the project's Code Cookbook for code consistency.
---

This workflow outlines how to maintain and use the project's "Code Cookbook" to ensure that AI-generated code remains consistent and follows established patterns.

### Adding a New Pattern to the Cookbook

When you identify a "golden" implementation of a recurring pattern (e.g., a standard error handler, a database query function), add it to the cookbook.

1.  **Identify the pattern**: Pinpoint the exact function in a specific file that represents the ideal implementation.
2.  **Run the `add_to_cookbook` tool**:
    ```
    add_to_cookbook(
        pattern_name="unique_pattern_name", 
        file_path="/path/to/your/file.py", 
        function_name="yourFunctionName", 
        description="A clear description of what this pattern does and when to use it."
    )
    ```

### Using an Existing Pattern from the Cookbook

When starting a new development task, always check the cookbook first to see if a solution already exists. This is the most important step for maintaining consistency.

1.  **Search the cookbook**: Use a natural language query to find relevant patterns.
    ```
    find_in_cookbook(query="your development task, e.g., 'how to connect to the database'")
    ```
2.  **Apply the pattern**: If a relevant pattern is found, use the provided code snippet *exactly* as your implementation. Do not create a new variation.
3.  **If no pattern is found**: Proceed with development. Once complete, consider if the new code is a good candidate to be added to the cookbook for future use.
