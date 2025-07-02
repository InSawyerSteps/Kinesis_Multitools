# MCP Server Next-Gen Tool Development Guide

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
        "keyword", "semantic", "ast", "references", "similarity", "task_verification"
    ]
    query: str
    params: Optional[Dict[str, Any]] = None
```

**Subtool Logic:**
- `keyword`: Regex/literal search across files
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

## 14. Example Multitool Skeleton

```python
@mcp.tool()
def refactor(request: RefactorRequest) -> dict:
    """
    Perform safe, format-preserving code refactors (rename, docstring, extract) using LibCST.

    Args:
        request (RefactorRequest):
            - file_path (str): File to modify
            - operation (str): Refactor operation (see supported)
            - params (dict): Operation parameters

    Returns:
        dict: { 'status': ..., 'result': ..., 'message': ... }
    """
    logger.info(f"[refactor] op={request.operation} file={request.file_path}")
    if not _is_safe_path(request.file_path):
        return {"status": "error", "message": "Unsafe path"}
    try:
        if request.operation == "rename_symbol":
            return _rename_symbol(request.file_path, **request.params)
        elif request.operation == "add_docstring":
            return _add_docstring(request.file_path, **request.params)
        ...
        else:
            return {"status": "error", "message": "Unknown operation"}
    except Exception as e:
        logger.error(f"[refactor] error: {e}")
        return {"status": "error", "message": str(e)}
```

---

**By following this guide, you will ensure all new MCP tools are robust, extensible, and agent-ready.**
