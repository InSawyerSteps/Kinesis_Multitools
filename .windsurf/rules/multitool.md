---
trigger: always_on
---

---
trigger: always_on
---

<rules>
    <instructions>
        You are an expert-level software engineering assistant. You have access to a powerful local multitool for searching and a tool for indexing the project.

        IMPORTANT: Before you can use search types 'semantic', 'similarity', or 'task_verification', the project MUST be indexed. If the user asks for a semantic search and you suspect the index is old or non-existent, you should first use the @index_project tool.

        All search subtools now honor 'includes' and 'max_results' parameters for focused, efficient queries.
        AST, references, similarity, and task_verification search tools have robust error handling and skip binary/internal files.
        The embedding/indexing tool uses sentence-transformers with GPU acceleration if available.
        .venv and dependency folders are always skipped during indexing/search.
        requirements.txt is updated with all needed dependencies and correct versions.
        Usage instructions for each search mode and parameter are accurate and up-to-date.

        When asked to find code, analyze the user's request carefully to select the best `search_type` from the @search tool's description. For complex tasks like finding all references, you may need to chain multiple search calls (e.g., first use 'ast' to find the definition, then 'references' to find its usages).
    </instructions>

    <!-- Tool 1: The Indexer (Prerequisite) -->
    <tool>
        <name>@index_project</name>
        <url>http://localhost:8000/index_project_files</url>
        <method>POST</method>
        <description>
            Scans the current project, chunks all relevant text files, and creates a vector index for semantic search. 
            This is a necessary first step before using the 'semantic', 'similarity', or 'task_verification' search types.
            The tool returns a JSON summary of the indexing operation.
            Payload: {"project_name": "MCP-Server"}
        </description>
    </tool>

    <!-- Tool 2: The Search Multitool -->
    <tool>
        <name>@search</name>
        <url>http://localhost:8000/search</url>
        <method>POST</method>
        <description>
            Performs a comprehensive search of the codebase using various methods. 
            You MUST specify the search method in the `search_type` field of the JSON request body.

            Available search_type options:

            - 'keyword': For finding exact, literal strings or variable names. Very fast.
              Use this for finding specific error messages or exact function names.
              Payload: {"project_name": "MCP-Server", "search_type": "keyword", "query": "the exact string", "params": {"includes": [...], "max_results": 10}}

            - 'semantic': For finding code based on natural language concepts.
              Use this when you don't know the exact names, like "find user login logic".
              Payload: {"project_name": "MCP-Server", "search_type": "semantic", "query": "natural language description", "params": {"includes": [...], "max_results": 10}}

            - 'ast': For structurally finding DEFINITIONS of functions or classes.
              Use this to find exactly where a specific function or class is defined.
              Payload: {"project_name": "MCP-Server", "search_type": "ast", "query": "ClassNameOrFunctionName", "params": {"includes": [...], "max_results": 10}}

            - 'references': For finding all USAGES of a symbol (function, class, variable).
              Use this to see where a function is called from before you change it.
              Requires the file path, line, and column of the symbol's definition in the `params` object.
              Payload: {"project_name": "MCP-Server", "search_type": "references", "query": "functionName", "params": {"file_path": "/path/to/file.py", "line": 42, "column": 15, "max_results": 10}}

            - 'similarity': For finding duplicate or functionally similar code blocks.
              The `query` MUST be the actual block of source code to check against.
              Payload: {"project_name": "MCP-Server", "search_type": "similarity", "query": "def my_func(a, b):\n  return a + b", "params": {"includes": [...], "max_results": 10}}

            - 'task_verification': Checks if a task description is implemented in the code.
              Use this to check progress on a task. The `query` is the task description.
              Payload: {"project_name": "MCP-Server", "search_type": "task_verification", "query": "Implement user auth caching", "params": {"includes": [...], "max_results": 10}}
        </description>
    </tool>
</rules>