# MCP Server Next-Gen Tool Development Guide

> **Supported MCP Tools (as of July 2025):**
>
> - `index_project_files`: Incremental semantic indexing of project files for search (see `.windsurf/rules/indexing.md`).
> - `search`: Multi-modal codebase search (keyword, regex, semantic, ast, references, similarity, task_verification; see `.windsurf/rules/search.md`).
> - `cookbook_multitool`: Unified tool for capturing and searching canonical code patterns (see below).
> - `get_snippet`: Extract a precise code snippet from a file by function, class, or line range (see below).
> - `introspect`: Multi-modal code/project introspection multitool for fast, read-only analysis of code and config files (see below).
> - File read/list utilities: Safe recursive listing and reading of project files.
> - `anchor_drop`: Persistently register external project roots for safe tool access across restarts.

---

## FastMCP Version and Dependency Management

- This project uses `fastmcp==2.9.2` (see `requirements.txt`).
- All dependencies and versions are listed in `requirements.txt` and must be kept in sync with actual usage in `src/toolz.py` and the server runtime.
- For GPU support, see the torch install instructions in `requirements.txt`.

---

## Code Cookbook Multitool (`cookbook_multitool`)

The `cookbook_multitool` is a unified endpoint for capturing and searching canonical code patterns ("golden patterns") in your project. It supersedes the old `add_to_cookbook` and `find_in_cookbook` tools.

### Purpose
- **Add** a function's source and metadata to the project cookbook for future reuse and code consistency.
- **Find** patterns by name, description, or function name for rapid code reuse and enforcement of best practices.

### Request Schema
- `mode` (str): `"add"` or `"find"`. Required.
- For `add` mode (multi-language):
    - `pattern_name` (str): Unique name for the pattern. Required.
    - `description` (str): Human-readable description. Required.
    - One of the following extraction strategies:
        * `file_path` (str) + `function_name` (str): Extract a function/class by name. Language auto-detected by extension; override with `language`.
        * `file_path` (str) + `start_marker` (str) + `end_marker` (str): Extract code between markers (HTML/JSON/text supported).
        * `code_snippet` (str): Store a raw snippet directly (JSON/text supported).
    - `language` (str, optional): One of `python`, `javascript`, `typescript`, `html`, `json`, `text`. If omitted, detected from `file_path`.
- For `find` mode:
    - `query` (str): Search query for names/descriptions/function names. Required.
    - `language` (str, optional): Filter results by language.

### Example Usage

#### Add a Pattern
```json
{
  "mode": "add",
  "pattern_name": "secure_path_validator",
  "file_path": "C:/Projects/MCP Server/src/toolz.py",
  "function_name": "_is_safe_path",
  "description": "Ensures a path stays within registered project roots.",
  "language": "python"
}
```

#### Add from HTML markers
```json
{
  "mode": "add",
  "pattern_name": "form_validate_handler",
  "file_path": "C:/Projects/site/index.html",
  "start_marker": "<!-- VALIDATE_START -->",
  "end_marker": "<!-- VALIDATE_END -->",
  "description": "Client-side validation handler in script block.",
  "language": "html"
}
```

#### Add a raw JSON snippet
```json
{
  "mode": "add",
  "pattern_name": "pytest_config",
  "code_snippet": "{\n  \"pytest\": { \"addopts\": \"-q\" }\n}",
  "description": "Minimal pytest JSON config.",
  "language": "json"
}
```

#### Find a Pattern
```json
{
  "mode": "find",
  "query": "secure path",
  "language": "python"
}
```

### Response
- On success, returns a status and message (for add), or a list of matching patterns (for find).
- Stored metadata includes `language`, `extraction_strategy`, and a `locator` for how it was extracted (function name, markers, or direct snippet).
- Patterns are stored as JSON files under `.project_cookbook/` in the project root.

### Best Practices
- Use descriptive, unique pattern names to avoid collisions.
- Store only canonical, well-tested functions as cookbook patterns.
- Use the find mode to enforce code consistency and accelerate onboarding.

## Project Roots Persistence & Path Safety

All file operations are sandboxed to registered project roots to ensure security and predictable behavior.

### Persistence
- Registered roots are saved to `.project_roots.json` in the server root.
- The MCP-Server root is always included by default.
- Roots are loaded on startup and validated (existence, normalization).

### Registering roots with `anchor_drop`
Use the `anchor_drop` tool to add an external project root (optionally with an alias) and persist it across restarts.

Example request:
```json
{
  "path": "C:/Projects/AnotherRepo",
  "project_name": "AnotherRepo"
}
```

Example response includes `status`, `message`, and the updated `project_roots` mapping.

### Path validation
- The canonical safety check `'_is_safe_path'` ensures any file path used by tools resolves inside one of the registered roots (prevents traversal or escape).
- Tools must reject operations on paths outside these roots and return a clear error message.

### Best practices
- Prefer passing logical project aliases where supported to simplify requests.
- Avoid duplicate registrations; the tool handles de-duplication and returns an informative message.

### Manual Testing Checklist (Quick)
- __anchor_drop__: Register an external root and confirm persistence in `.project_roots.json`.
  - Expect `status: success`, updated `project_roots` mapping.
- __index_project_files__: Run once, then modify 1-2 files and re-run.
  - Expect `updated_files` > 0 only on the second run; `unchanged_files` dominates.
- __search__:
  - `keyword`: Query a known symbol; expect file hits.
  - `regex`: Provide a simple regex; expect matching lines.
  - `semantic`: If index exists, expect relevant chunks; otherwise a clear error to index first.
  - `ast`: Query a known function/class name; expect structured results.
  - `references` (if configured): Provide file/line/column for a symbol; expect usages.
  - `similarity`: Provide a code snippet; expect similar blocks.
  - `task_verification`: Provide a task sentence; expect presence/absence assessment.
- __cookbook_multitool__:
  - `add` (Python): `file_path` + `function_name` for a known function.
  - `add` (HTML markers): `start_marker`/`end_marker` in an HTML file.
  - `add` (raw JSON): `code_snippet` with `language: json`.
  - `find`: Query by keywords; optionally filter by `language`.
- __get_snippet__: Test `mode: function`, `mode: class`, and `mode: lines`.
  - Expect `status`, `snippet`, and helpful `message` when not found.
- __introspect__:
  - `config` with `requirements`.
  - `outline` on a Python file.
  - `stats` and `inspect` for a known symbol.
- __file read/list utilities__: List project files and read a small text file; verify path safety rejects out-of-root paths.

---

## Overview
This guide is your blueprint for extending `toolz.py` with advanced, multi-command "mega-tools" for the MCP Server. It covers:
- Authoring style and architecture
- Implementation patterns for multitools (tools-within-tools)
- Subtool design and parameterization
- Logging, error handling, and security
- Best practices for static analysis, code transformation (LibCST), code generation, visualization, and project management
- Search multitool: design, subtool taxonomy, and extensibility

**Read this carefully before contributing any new tool.**

---

## 1. MCP Tool Authoring Principles

- **All tools are exposed via `@mcp.tool()` and registered with the FastMCP server.**
- **Each tool can be a multitool:** it exposes a main endpoint (e.g., `/analyze`, `/refactor`, `/search`) with a `mode` or `subcommand` parameter that selects the specific operation.
- **Every tool and subtool must:**
    - Use precise Python type hints and Pydantic models for arguments/return types
    - Provide an exhaustive, usage-focused docstring (see below)
    - Log all actions and errors
    - Validate all parameters and file paths
    - Return structured, machine-readable output (dicts, lists, etc.)
    - Handle all exceptions gracefully (never crash the server)
    - Operate only within validated project roots

---

## 2. Tool Docstring and Usage Pattern

Every tool must have a docstring with:
- **Summary**: One-line description
- **Args**: Each argument, its type, allowed values, default, and usage
- **Returns**: Structure and meaning of return value(s)
- **Usage**: Example calls, edge cases, safety notes

**Example:**
```python
@mcp.tool()
def analyze(request: AnalysisRequest) -> dict:
    """
    Run static analyses on files or directories (quality, types, security, etc.).

    Args:
        request (AnalysisRequest):
            - path (str): File or directory to analyze.
            - analyses (List[str]): List of analyses to run (see supported).

    Returns:
        dict: { 'status': ..., 'results': {analysis_type: result, ...}, ... }

    Usage:
        - Run multiple analyses in one call.
        - Output is always structured and machine-readable.
    """
```

---

## 3. Multitool Design Pattern

- **One endpoint, many subtools:**
    - Each multitool exposes a main function (e.g., `analyze`, `refactor`, `search`)
    - The main function takes a Pydantic model with a `mode` (or `subcommand`, `search_type`, etc.) and a `params` dict
    - The function dispatches to the correct subtool implementation based on the mode
    - All subtools share parameter validation, logging, and result formatting

**Example Pydantic Model:**
```python
class RefactorRequest(BaseModel):
    file_path: str
    operation: Literal["rename_symbol", "add_docstring", "extract_method"]
    params: Dict[str, Any] # e.g., {"old_name": ..., "new_name": ...}
```

---

## 4. Subtool Implementation and Parameterization

---

## Incremental Project Indexing (`index_project_files`)

The `index_project_files` tool now supports **incremental indexing** for semantic and hybrid search. This dramatically improves speed and efficiency for large projects.

### How it Works
- On each run, the tool:
  - Loads previous indexing metadata (`metadata.json`) containing chunk embeddings, file modification times (mtimes), and sizes.
  - Scans all relevant project files, ignoring dependency and binary folders.
  - Compares each file’s mtime and size to the metadata.
    - **Unchanged:** Reuses existing vectors (no re-embedding).
    - **Updated/New:** Re-chunks and re-embeds only changed files.
    - **Deleted:** Removes vectors for files no longer present.
  - Rebuilds the FAISS index from the union of reused and new vectors.
  - Saves updated metadata for all indexed chunks.

### Metadata Format
Each chunk entry in `metadata.json`:
```json
{
  "path": "relative/path/to/file.py",
  "content": "<chunk text>",
  "vector": [ ...float32 list... ],
  "file_mtime": 1719690000.123,
  "file_size": 2048
}
```

### Return Value
The tool returns a summary dict:
```json
{
  "status": "success",
  "message": "...",
  "files_scanned_and_included": <int>,
  "unchanged_files": <int>,
  "updated_files": <int>,
  "deleted_files": <int>,
  "total_chunks_indexed": <int>,
  "indexing_duration_seconds": <float>
}
```

### Usage Notes
- **First run:** All files are indexed as new.
- **Subsequent runs:** Only changed, added, or deleted files are reprocessed.
- **Performance:** For large codebases with few changes, incremental indexing is much faster than full reindexing.
- **Robustness:** Handles missing or corrupt metadata gracefully by falling back to a full reindex.
- **Logs:** Detailed logging of all file actions (unchanged, updated, deleted) is provided.
- **Edge Cases:** File renames are treated as delete+add.

### Best Practices
- Always run `index_project_files` after adding, editing, or deleting files to keep the semantic index up to date.
- For subfolder indexing, use the `subfolder` argument.
- Ensure that project root and subfolder paths are validated for safety.


- **Each subtool is a function or method called by the main multitool.**
- **Subtools must:**
    - Validate all required and optional params
    - Log their invocation and results
    - Return a structured dict (with status, results, and error messages)
    - Handle all exceptions
    - For destructive actions, support a `preview_only` or `dry_run` mode

---

## 5. Reliability and Performance Patterns

### Known Issues
- The `analyze` tool may hang, time out, or fail to return results. This is a known issue and is under investigation. All other tools are stable and reliable.

To ensure the MCP server is robust and responsive, especially when dealing with long-running tasks or heavy libraries, follow these critical patterns.

### 5.1. Process-Based Timeouts for Reliability

Standard threading timeouts in Python cannot interrupt blocking I/O or long-running native code (e.g., in libraries like `torch` or `pylint`). This can cause tools to hang indefinitely, freezing the server.

**Solution:** Use the `@tool_process_timeout_and_errors` decorator.

This decorator runs the entire tool function in a separate `multiprocessing.Process`.
- It enforces a hard timeout, forcibly terminating the process if it exceeds the limit.
- It catches any exceptions within the process, including `TimeoutExpired`.
- It ensures a structured error message is always returned to the client, preventing hangs.

**Usage:**
```python
from tool_utils import tool_process_timeout_and_errors

@tool_process_timeout_and_errors(timeout=60)
@mcp.tool()
def my_long_running_tool(request: MyRequest) -> dict:
    # ... logic that might hang ...
    return {"status": "success"}
```

**Rule:** All tools that perform heavy computation, run external subprocesses, or use libraries with native C/C++ extensions **must** use this decorator.

### 5.2. Lazy Loading for Performance

Some libraries, particularly AI models like `sentence-transformers`, have a very high startup cost. Loading them at the top of `toolz.py` means this cost is paid every time a new process is spawned by our timeout decorator.

**Solution:** Implement lazy loading. Load expensive resources only when they are first needed within a process.

**Pattern:**
1.  Define a global variable for the resource, initialized to `None`.
2.  Create a getter function that checks if the resource is `None`. If it is, load it and store it in the global variable.
3.  All parts of the code must access the resource via the getter function.

**Example (`_get_st_model`):**
```python
# Global variable for the model
_st_model = None

def _get_st_model():
    """Lazily loads and returns the SentenceTransformer model."""
    global _st_model
    if _st_model is None:
        logger.info("Lazy loading SentenceTransformer model...")
        _st_model = SentenceTransformer(MODEL_NAME)
    return _st_model

# Usage in a tool
def _embed_batch(batch_texts: list[str]) -> list[list[float]]:
    model = _get_st_model() # Access via getter
    embeddings = model.encode(batch_texts, show_progress_bar=False)
    return [e.tolist() for e in embeddings]
```

**Rule:** Any dependency with a significant import/initialization time (>100ms) should be lazy-loaded.

---

## 6. Logging, Error Handling, and Security

- Use the shared `logger` instance (`logger.info`, `logger.error`, etc.)
- Log all actions, inputs, outputs, and errors
- Wrap risky code in try/except and return errors in the output dict
- Always validate file paths using `_is_safe_path` or equivalent
- Never operate outside project roots
- Never overwrite files unless explicitly requested

---

## 7. Static Analysis Multitool: `/analyze`

**Purpose:** Run multiple static analyses (quality, types, security, dead_code, complexity, todos, dependencies, licenses) on files or directories.

**Model:**
```python
class AnalysisRequest(BaseModel):
    path: str
    analyses: List[Literal[
        "quality", "types", "security", "dead_code",
        "complexity", "todos", "dependencies", "licenses"
    ]]
```

**Subtool Logic:**
- `quality`: Run Pylint with `--output-format=json`, parse and return issues
- `types`: Run MyPy, parse output lines into error dicts
- `security`: Run Bandit with `-f json`, parse and return vulnerabilities
- `dead_code`: Run Vulture, parse output for unused code
- `complexity`: Use Radon (preferably as a library), return cyclomatic complexity metrics
- `todos`: Scan files for TODO/FIXME/NOTE comments using regex
- `dependencies`, `licenses`: Parse requirements/pyproject and check with pip-audit or license libraries

**Return:** Always a dict with subtool results keyed by analysis type.

---

## 8. Code Transformation Multitool: `/refactor` (LibCST Required)

**Purpose:** Perform safe, format-preserving code changes (rename, docstring insertion, extraction, etc.).

**Model:**
```python
class RefactorRequest(BaseModel):
    file_path: str
    operation: Literal["rename_symbol", "add_docstring", "extract_method"]
    params: Dict[str, Any]
```

**Subtool Logic:**
- `rename_symbol`: Use LibCST to rename all instances of a symbol in a file
- `add_docstring`: Use LLM to generate docstring, then LibCST to insert it
- `extract_method`: Use LibCST to extract code into a new function/method

**Best Practices:**
- Always use LibCST (never ast) for code modifications.
- Validate all params (e.g., `old_name`, `new_name` for rename).
- Support `preview_only` mode for all destructive actions.
- Return both the new code and a diff (if preview).
- **Stability Note:** As of the latest update, advanced LibCST features in the `edit` tool (like `rename_symbol`) are disabled to prevent hangs caused by implementation issues. The tool returns a clear error message for these operations. This is a valid temporary pattern to ensure server stability.

---

## 9. Automated Code Generation Multitool: `/generate`

**Purpose:** Generate tests, documentation, or other code artifacts using LLMs and/or static analysis.

**Model:**
```python
class GenerateRequest(BaseModel):
    task: Literal["tests", "docs_build"]
    params: Dict[str, Any]
```

**Subtool Logic:**
- `tests`: Use LLM to generate pytest suites for a function or class
- `docs_build`: Run `mkdocs build` or `sphinx-build` as a subprocess, return output

---

## 10. Visualization Multitool: `/visualize`

**Purpose:** Generate diagrams of code structure and dependencies.

**Model:**
```python
class VisualizeRequest(BaseModel):
    diagram_type: Literal["call_graph", "dependency_graph"]
    path: str
```

**Subtool Logic:**
- `call_graph`: Use pyan to generate DOT, render with graphviz to SVG/PNG
- `dependency_graph`: Similar, but focus on imports and package dependencies

---

## 11. Project Management Multitool: `/project`

**Purpose:** Manage project metadata, dependencies, releases, etc.

**Model:**
```python
class ProjectRequest(BaseModel):
    action: Literal["audit_dependencies", "check_licenses", "bump_version", "gen_changelog"]
    params: Dict[str, Any]
```

**Subtool Logic:**
- `audit_dependencies`: Run pip-audit, parse and return JSON output
- `check_licenses`: Check all dependencies for license compliance
- `bump_version`: Run bump2version for patch/minor/major
- `gen_changelog`: Run towncrier to generate CHANGELOG.md

---

## 12. Search Multitool: `/search` (Unified Interface)

**Purpose:** Provide lexical, semantic, structural, and referential search across the codebase.

**Model:**
```python
class SearchRequest(BaseModel):
    search_type: Literal[
        "keyword", "regex", "semantic", "ast", "references", "similarity", "task_verification"
    ]
    query: str
    params: Optional[Dict[str, Any]] = None
```

**Subtool Logic:**
- `keyword`: Literal search across files
- `regex`: Advanced regex search across files
- `semantic`: Embedding-based search (FAISS)
- `ast`: AST-based structural search
- `references`: Find symbol usages (Jedi or LibCST)
- `similarity`: Embedding-based code similarity
- `task_verification`: Meta-search for task implementation status

**Best Practices:**
- All search subtools must honor `includes`, `max_results`, and other filter params
- Always skip binary/dependency files
- Return structured, deduplicated results

---

## 13. General Best Practices and Checklist

- [ ] Use Pydantic models for all multitool requests
- [ ] Document every argument, return value, and subtool in the docstring
- [ ] Log every action and error
- [ ] Validate all file paths and parameters
- [ ] Use LibCST for all code modifications
- [ ] Support preview/dry-run modes for destructive actions
- [ ] Return structured, machine-readable output
- [ ] Update requirements.txt for new dependencies
- [ ] Update Windsurf rules and documentation after every change

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

**By following this guide, you will ensure all new MCP tools are robust, extensible, and agent-ready.**
