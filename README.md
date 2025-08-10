# Kinesis Multitools

Kinesis Multitools is a robust, extensible MCP server for IDE-integrated code intelligence, semantic code search, and canonical code pattern management. Built for reliability and security, it enables both developers and AI agents to analyze, search, and interact with codebases efficiently—while remaining open to new tool ideas from the community.

## Key Features

- **Modern FastMCP Backend:**
  - Uses `fastmcp==2.9.2` for maximum performance and compatibility (see `requirements.txt`).
- **Supported MCP Tools:**
  - `index_project_files`: Incremental semantic indexing of project files for search.
  - `search`: Multi-modal codebase search (keyword, regex, semantic, ast, references, similarity, task_verification.
  - `cookbook_multitool`: Unified tool for capturing and searching canonical code patterns.
  - `get_snippet`: Extract a precise code snippet from a file by function, class, or line range. Supports extracting the full source of a named function or class (using AST), or any arbitrary line range. Returns a structured status, snippet content, and message. See below for usage details.
  - `introspect`: Multi-modal code/project introspection multitool for fast, read-only analysis of code and config files. Supports modes for config file reading, structural outline, file statistics, and deep inspection of a function or class. Returns structured results for each mode. See below for usage details.
  - File read/list utilities: Safe listing and reading of project files.
  - `anchor_multitool`: Project root management (modes: drop, list, remove, rename). Persist external project roots across restarts.
- **Reliable by Design:** All tools run in isolated processes with hard timeouts, preventing hangs and ensuring the server remains responsive.
- **Incremental Indexing:** Only changed files are re-embedded, making semantic search fast and efficient.
- **Secure & Sandboxed:** All file operations are validated to ensure they remain within the configured project root.
- **Extensible:** A development guide (`tooldevguide.md`) provides a blueprint for adding new capabilities.

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/[your-username]/Kinesis-Multitools.git
cd Kinesis-Multitools
pip install -r requirements.txt
```

**Dependency Note:**
All required libraries and versions are listed in `requirements.txt`. Key packages include:
- fastmcp==2.9.2
- mcp==1.9.4
- pydantic[email]==2.11.7
- pydantic-settings==2.5.2
- ollama==0.5.1
- fastapi==0.115.9
- uvicorn[standard]==0.30.6
- faiss-cpu==1.11.0
- sentence-transformers==4.1.0
- jedi==0.19.2
- chromadb==1.0.5
- onnxruntime==1.21.1
- numpy==2.3.1
- tokenizers==0.21.1
- pylint==3.2.2
- mypy==1.16.1
- bandit==1.8.5
- vulture==2.11
- radon==6.0.1
- libcst==1.2.0
- rich==14.0.0
- typer==0.15.2
- python-dotenv==1.1.0
- requests==2.32.3

All dependencies are up-to-date as of July 2025. For GPU support, see the torch install instructions in `requirements.txt`.

## Windsurf IDE Configuration

Configure the MCP server in your `mcp_config.json`:

```json
"kinesis_multitools": {
  "command": "C:\\Projects\\MCP Server\\.venv\\Scripts\\fastmcp.exe",
  "args": [
    "run",
    "C:\\Projects\\MCP Server\\src\\toolz.py",
    "--transport", "stdio"
  ],
  "env": {},
  "disabled": false
}
```

**Note:** Ensure the `command` path points to the correct `fastmcp.exe` in your `.venv`.

## Usage

Run the MCP server via FastMCP (stdio transport):

```bash
fastmcp run src/toolz.py --transport stdio
```

Notes:

- The server is designed to be launched by an MCP client (e.g., Windsurf) over stdio.
- Direct `python src/toolz.py` is not the supported entrypoint for serving tools.
- HTTP mode is not provided by default. If you need HTTP, wrap the FastMCP server yourself.

## Supported Tools

- `index_project_files`: Incremental semantic indexing for search.
- `search`: Multi-modal codebase search (keyword, regex, semantic, ast, references, similarity, task_verification).
- `cookbook_multitool`: Unified tool for capturing and searching canonical code patterns.
- `get_snippet`: Extract by function/class/lines.
- `introspect`: Config/outline/stats/inspect.
- `anchor_multitool`: Manage project roots (drop/list/remove/rename) and persist across restarts.
- File read/list utilities.

**Removed tools:**
- The legacy `analyze` and `edit` tools have been removed for stability. Only the tools above are supported in this release.

For details on tool usage and extension, see `tooldevguide.md`.

---

## Canonical Patterns in the Cookbook

The following key patterns from `src/toolz.py` are available in the project Cookbook for rapid reuse and code consistency. Use the `cookbook_multitool` to search for or add these patterns to new projects:

| Pattern Name                         | Function Name                     | Description                                                                                                     |
|--------------------------------------|-----------------------------------|-----------------------------------------------------------------------------------------------------------------|
| incremental_vector_indexing          | index_project_files               | Efficiently indexes project files for semantic search by reusing unchanged vectors and only embedding changes.   |
| multimodal_search_dispatch           | unified_search                    | Unified entrypoint for multi-modal codebase search with robust error handling and subtool dispatch.              |
| process_timeout_and_error_decorator  | tool_process_timeout_and_errors   | Decorator for running a function in a separate process with a hard timeout and robust error handling.            |
| thread_timeout_and_error_decorator   | tool_timeout_and_errors           | Decorator for running a function in a thread with a hard timeout, logging all exceptions and returning errors.   |
| project_path_safety_check            | _is_safe_path                     | Ensures a given path is inside one of the allowed project roots. Prevents unsafe file access or traversal.       |
| precise_code_snippet_extractor       | get_snippet                       | Extracts a precise code snippet from a file by function, class, or line range using AST or line ranges.          |
| multi_modal_code_introspection       | introspect                        | Multi-modal code/project introspection multitool for fast, read-only analysis of code and config files.           |

If you implement a new utility or multitool that solves a general problem (e.g., logging, config loading, error formatting), consider adding it to the Cookbook for future reuse.


### Indexing the Project

Embedding‑based search modes (`semantic`, `similarity`, `task_verification`) require a FAISS vector index. The index is stored in a hidden folder under the project root. If the index is missing, these modes will return an error asking you to run `index_project_files` first.

Indexing is incremental: unchanged files reuse existing vectors; only added/modified files are re‑embedded.

Invoke the tool from your MCP client:

```json
{
  "tool": "index_project_files",
  "args": { "project_name": "MCP-Server" }
}
```

Example response:

```json
{
  "status": "success",
  "message": "Project 'MCP-Server' indexed incrementally.",
  "files_scanned_and_included": 150,
  "unchanged_files": 140,
  "updated_files": 10,
  "deleted_files": 0,
  "total_chunks_indexed": 1250,
  "indexing_duration_seconds": 15.75
}
```

### Searching the Codebase

Call the `search` tool with the desired `search_type` and parameters.

Semantic search example:

```json
{
  "tool": "search",
  "args": {
    "search_type": "semantic",
    "query": "how does the server manage timeouts on Windows?",
    "project_name": "MCP-Server",
    "params": { "max_results": 5 }
  }
}
```

AST search for a function definition:

```json
{
  "tool": "search",
  "args": {
    "search_type": "ast",
    "query": "unified_search",
    "project_name": "MCP-Server",
    "params": { "target_node_type": "function" }
  }
}
```

Other supported `search_type` values: `keyword`, `regex`, `references`, `similarity`, `task_verification`.

Tip: narrow scope with `params.includes` (glob patterns) and set `params.max_results` to limit output.

---

## Windows Reliability Notes

- Tools are decorated for strict timeouts. On Windows, thread‑based timeouts are used where process isolation is unsafe for certain workloads; on POSIX, process‑based timeouts are used.
- Decorator order matters for Windows: `@mcp.tool(...)` must be the outermost decorator. This repository follows that rule across all tools.
- Enable verbose search diagnostics by setting `MCP_DEBUG_SEARCH=1` in the environment before launching the server.

## Roadmap

**Roadmap & Community Involvement:**

We welcome suggestions and contributions for new tools! If you have an idea for a code intelligence, search, or automation tool that would benefit the community, please open an issue or submit a pull request.

- `index_project_files`: Incremental semantic indexing of project files for search.
- `search`: Multi-modal codebase search (keyword, regex, semantic, ast, references, similarity, task_verification.
- `cookbook_multitool`: Unified tool for capturing and searching canonical code patterns (see below).
- `get_snippet`: Extract a function, class, or line range from any project file. See below for details.
- `introspect`: Multi-modal code/project introspection for config, outline, stats, and inspect. See below for details.
- File read/list utilities.

---

## Code Cookbook Multitool

The `cookbook_multitool` is a unified endpoint for capturing and searching canonical code patterns ("golden patterns") across multiple languages. It replaces the older `add_to_cookbook` and `find_in_cookbook` tools.

### Purpose
- **Add** a pattern's source and metadata to the project cookbook (supports Python, JavaScript/TypeScript, HTML, JSON, and plain text).
- **Find** patterns by name, description, or function name, optionally filtering by language.

### Multi-language add options
Use `/cookbook_multitool` with `mode: "add"`. You can add patterns via one of the following strategies:

- File + function/class name (language auto-detected by extension, with override):
```json
{
  "mode": "add",
  "pattern_name": "secure_path_validator",
  "file_path": "C:/Projects/MCP Server/src/toolz.py",
  "function_name": "_is_safe_path",
  "description": "Ensures a given path stays within allowed project roots.",
  "language": "python"
}
```

- JavaScript/TypeScript function extraction (regex + brace balancing):
```json
{
  "mode": "add",
  "pattern_name": "fetch_with_retry",
  "file_path": "C:/Projects/app/src/utils.ts",
  "function_name": "fetchWithRetry",
  "description": "Wrapper around fetch with exponential backoff.",
  "language": "typescript"
}
```

- HTML: extract from <script> blocks or between markers:
```json
{
  "mode": "add",
  "pattern_name": "form_validate_handler",
  "file_path": "C:/Projects/site/index.html",
  "start_marker": "<!-- VALIDATE_START -->",
  "end_marker": "<!-- VALIDATE_END -->",
  "description": "Client-side validation handler embedded in script block.",
  "language": "html"
}
```

- JSON/Text: store a snippet directly or via line markers:
```json
{
  "mode": "add",
  "pattern_name": "pytest_config",
  "code_snippet": "{\n  \"pytest\": { \"addopts\": \"-q\" }\n}",
  "description": "Minimal pytest JSON config snippet.",
  "language": "json"
}
```

### Find with optional language filter
```json
{
  "mode": "find",
  "query": "secure path",
  "language": "python"
}
```

### Response
- On success, returns a status and message (for add), or a list of matching patterns (for find).
- Stored metadata includes `language`, `extraction_strategy`, and a `locator` describing how the snippet was extracted (e.g., function name, markers, or direct snippet).
- Cookbook patterns are stored as JSON files in `.project_cookbook/` in the project root.

See `tooldevguide.md` for full schema, edge cases, and additional examples.

---

## get_snippet Tool

The `get_snippet` tool extracts a precise code snippet from a file by function, class, or line range.

**Capabilities:**
- Extract the full source of a named function (including decorators and signature)
- Extract the full source of a named class (including decorators and signature)
- Extract an arbitrary range of lines from any text file

**Arguments:**
- `file_path` (str): Absolute path to the file to extract from. Must be within the project root.
- `mode` (str): Extraction mode. One of:
    * `function`: Extract a named function by its name.
    * `class`: Extract a named class by its name.
    * `lines`: Extract a specific line range.
- `name` (str, optional): Required for `function` and `class` modes. The name of the function or class to extract.
- `start_line` (int, optional): Required for `lines` mode. 1-indexed starting line.
- `end_line` (int, optional): Required for `lines` mode. 1-indexed ending line (inclusive).

**Returns:**
- `status` (str): 'success', 'error', or 'not_found'
- `snippet` (str, optional): The extracted code snippet (if found)
- `message` (str): Human-readable status or error message

**Usage Examples:**
```json
{
  "file_path": "/project/src/foo.py",
  "mode": "function",
  "name": "my_func"
}
```
```json
{
  "file_path": "/project/src/foo.py",
  "mode": "lines",
  "start_line": 10,
  "end_line": 20
}
```

---

## introspect Tool

The `introspect` tool is a multi-modal code/project introspection multitool for fast, read-only analysis of code and config files.

**Modes:**
- `config`: Reads project configuration files (pyproject.toml or requirements.txt).
    * Args: config_type ('pyproject' or 'requirements').
    * Returns: For 'pyproject', the full TOML text. For 'requirements', a list of package strings.
- `outline`: Returns a high-level structural map of a Python file: all top-level functions and classes (with their methods).
    * Args: file_path (str)
    * Returns: functions (list), classes (list of {name, methods})
- `stats`: Calculates basic file statistics: total lines, code lines, comment lines, file size in bytes.
    * Args: file_path (str)
    * Returns: total_lines (int), code_lines (int), comment_lines (int), file_size_bytes (int)
- `inspect`: Provides details about a single function or class in a file: name, arguments/methods, docstring.
    * Args: file_path (str), function_name (str, optional), class_name (str, optional)
    * Returns: type ('function' or 'class'), name, args/methods, docstring

**Arguments:**
- `mode` (str): One of 'config', 'outline', 'stats', 'inspect'.
- `file_path` (str, optional): Path to the file for inspection (required for all except config).
- `config_type` (str, optional): 'pyproject' or 'requirements' for config mode.
- `function_name` (str, optional): Name of function to inspect (for 'inspect' mode).
- `class_name` (str, optional): Name of class to inspect (for 'inspect' mode).

**Returns:**
- `status` (str): 'success', 'error', or 'not_found'
- `message` (str): Human-readable status or error message
- Mode-specific fields:
    - config: config_type, content (str) or packages (list)
    - outline: functions (list), classes (list of {name, methods})
    - stats: total_lines (int), code_lines (int), comment_lines (int), file_size_bytes (int)
    - inspect: type, name, args/methods, docstring

**Usage Examples:**
```json
{
  "mode": "outline",
  "file_path": "/project/src/foo.py"
}
```
```json
{
  "mode": "inspect",
  "file_path": "/project/src/foo.py",
  "function_name": "my_func"
}
```
```json
{
  "mode": "config",
  "config_type": "requirements"
}
```

---

## License

This project is licensed under the MIT License. See the [LICENSE.md](LICENSE.md) file for details.
