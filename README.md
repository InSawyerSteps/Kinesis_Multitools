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

To run the server:

```bash
python src/toolz.py
```

The server will start on `http://localhost:8000`.

## Supported Tools

- `index_project_files`: Incremental semantic indexing for search.
- `search`: Multi-modal codebase search (keyword, regex, semantic, ast, references, similarity, task_verification).
- `cookbook_multitool`: Unified tool for capturing and searching canonical code patterns.
- File read/list utilities.

**Removed tools:**
- The legacy `analyze` and `edit` tools have been removed for stability. Only the tools above are supported in this release.

For details on tool usage and extension, see `tooldevguide.md`.

---

## Canonical Patterns in the Cookbook

The following key patterns from `src/toolz.py` are now available in the project Cookbook for rapid reuse and code consistency. Use the `cookbook_multitool` to search for or add these patterns to new projects:

| Pattern Name                       | Function Name                | Description                                                                                       |
|------------------------------------|------------------------------|---------------------------------------------------------------------------------------------------|
| secure_project_file_listing        | list_project_files           | Canonical function for recursively listing files in a project, with robust filtering and validation.|
| safe_file_read_with_validation     | read_project_file            | Safely reads a file, enforcing project root constraints and handling text/binary content.          |
| incremental_vector_indexing        | index_project_files          | Efficiently indexes project files for semantic search, only embedding changed/added files.         |
| multimodal_search_dispatch         | unified_search               | Unified entrypoint for multi-modal codebase search with robust error handling.                     |
| process_timeout_and_error_decorator| tool_process_timeout_and_errors | Decorator for process-based timeout and error handling for MCP tools.                              |
| thread_timeout_and_error_decorator | tool_timeout_and_errors      | Decorator for thread-based timeout and error handling for MCP tools.                               |
| secure_path_validation             | _is_safe_path                | Validates that a path is within the configured project root(s), preventing unauthorized access.    |
| batch_embedding_with_lazy_model    | _embed_batch                 | Batch embedding logic with lazy model loading and device selection.                                |


### Indexing the Project

Several of the `/search` subtools rely on a FAISS vector index generated from
your source files.  This index is stored in a hidden folder inside the project root.  If the index does not exist,
embedding‑based modes such as `semantic`, `similarity` and `task_verification`
return an error:

```
Index not found for project. Please run 'index_project_files' first.
```

Run the indexing tool whenever you update your code so that these subtools have
up‑to‑date embeddings.  Indexing is incremental and will only reprocess files
that changed.

Start by indexing your project:

```http
POST http://localhost:8000/index_project_files
Content-Type: application/json

{
  "project_name": "MCP-Server"
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

Send a request to `/search` with the desired `search_type` and query.

Example – semantic search:

```http
POST http://localhost:8000/search
Content-Type: application/json

{
  "search_type": "semantic",
  "query": "how does the server handle user authentication?",
  "project_name": "MCP-Server",
  "params": {
    "max_results": 5
  }
}
```

Example – AST search for a function definition:

```http
POST http://localhost:8000/search
Content-Type: application/json

{
  "search_type": "ast",
  "query": "unified_search",
  "project_name": "MCP-Server",
  "params": {
    "target_node_type": "function"
  }
}
```

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

The `cookbook_multitool` is a unified endpoint for capturing and searching canonical code patterns ("golden patterns") in your project. It replaces the older `add_to_cookbook` and `find_in_cookbook` tools.

### Purpose
- **Add** a function's source and metadata to the project cookbook for future reuse and consistency.
- **Find** patterns by name, description, or function name for rapid code reuse and enforcement of best practices.

### Usage

Send a request to `/cookbook_multitool` with the desired `mode`:

#### Add a Pattern
```json
{
  "mode": "add",
  "pattern_name": "My Golden Pattern",
  "file_path": "C:/Projects/MCP Server/src/toolz.py",
  "function_name": "_is_safe_path",
  "description": "A canonical function for secure path validation."
}
```

#### Find a Pattern
```json
{
  "mode": "find",
  "query": "secure path"
}
```

### Response
- On success, returns a status and message (for add), or a list of matching patterns (for find).
- All cookbook patterns are stored as JSON files in `.project_cookbook/` in the project root.

See the developer guide for advanced usage and schema details.

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
