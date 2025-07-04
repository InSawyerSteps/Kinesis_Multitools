"""
MCP Server with a consolidated, multi-modal search tool.

This module provides two primary tools for an AI agent:
  - index_project_files: Scans and creates a searchable vector index of the project. This must be run before using semantic search capabilities.
  - search: A powerful "multitool" that provides multiple modes of searching:
    - 'keyword': Literal text search.
    - 'semantic': Natural language concept search using the vector index.
    - 'ast': Structural search for definitions (functions, classes).
    - 'references': Finds all usages of a symbol.
    - 'similarity': Finds code blocks semantically similar to a given snippet.
    - 'task_verification': A meta-search to check the implementation status of a task.
"""
import ast
import json
import logging
import os
import pathlib
import re
import time
from typing import Any, Dict, List, Literal, Optional

from fastmcp import FastMCP
from pydantic import BaseModel, Field

# --- Add this block with the other imports ---
import subprocess
import difflib

# Attempt to import new dependencies for /analyze and /edit tools
try:
    from pylint.lint import Run as pylint_run
except ImportError:
    pylint_run = None

try:
    from mypy import api as mypy_api
except ImportError:
    mypy_api = None

try:
    from bandit.core import manager as bandit_manager
    from bandit.core import config as bandit_config
except ImportError:
    bandit_manager = None
    bandit_config = None

try:
    from vulture import Vulture
except ImportError:
    Vulture = None

try:
    from radon.cli import cc_visit
    from radon.cli.tools import CCHarvestor
except ImportError:
    cc_visit = None
    CCHarvestor = None

try:
    import libcst as cst
except ImportError:
    cst = None

from typing import Any, Dict, List, Literal

class AnalysisRequest(BaseModel):
    path: str
    analyses: List[Literal["quality", "types", "security", "dead_code", "complexity", "todos"]]

class EditOperation(BaseModel):
    operation: Literal[
        "replace_block", "insert_at", "delete_block",  # Text-based
        "rename_symbol", "add_docstring"              # LibCST-based
    ]
    params: Dict[str, Any]

class BatchEditRequest(BaseModel):
    file_path: str
    edits: List[EditOperation]
    preview_only: bool = False

# --- End of new imports and models ---

# --- Dependency Imports ---
# These are required for the tools to function.
# Ensure you have them installed:
# pip install faiss-cpu sentence-transformers torch numpy jedi

try:
    import faiss
    import numpy as np
    from sentence_transformers import SentenceTransformer
    import torch
    
    # --- Global Model and Device Configuration ---
    # This setup is done once when the module is loaded for efficiency.
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    _ST_MODEL = SentenceTransformer('all-MiniLM-L6-v2', device=DEVICE)
    
    # Suppress the noisy INFO log from SentenceTransformer
    st_logger = logging.getLogger("sentence_transformers.SentenceTransformer")
    st_logger.setLevel(logging.WARNING)

    LIBS_AVAILABLE = True
except ImportError:
    LIBS_AVAILABLE = False

try:
    import jedi
except ImportError:
    jedi = None

# ---------------------------------------------------------------------------
# FastMCP initialisation & logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("mcp_search_tools")

kinesis_mcp = FastMCP(
    name="Kinesis_Multitools",
)

# ---------------------------------------------------------------------------
# Project root configuration
# ---------------------------------------------------------------------------
# Assumes this file is in 'src/' and the project root is its parent's parent.
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
PROJECT_ROOTS: Dict[str, pathlib.Path] = {
    "Kinesis_Multitools": PROJECT_ROOT,
}
INDEX_DIR_NAME = ".windsurf_search_index"

# ---------------------------------------------------------------------------
# Core Helper Functions
# ---------------------------------------------------------------------------

def _get_project_path(project_name: str) -> Optional[pathlib.Path]:
    """Gets the root path for a given project name."""
    return PROJECT_ROOTS.get(project_name)

def _iter_files(root: pathlib.Path, extensions: Optional[List[str]] = None):
    """Yields all files under root, skipping common dependency/VCS and binary files."""
    exclude_dirs = {".git", ".venv", "venv", "__pycache__", "node_modules", ".vscode", ".idea", "dist", "build"}
    binary_extensions = {
        ".zip", ".gz", ".tar", ".rar", ".7z", ".exe", ".dll", ".so", ".a",
        ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".pdf", ".doc",
        ".docx", ".ppt", ".pptx", ".xls", ".xlsx", ".pyc", ".pyo", ".db",
        ".sqlite", ".sqlite3", ".iso", ".img", ".mp3", ".mp4", ".avi",
        ".mkv", ".mov"
    }
    norm_exts = {f".{e.lower().lstrip('.')}" for e in extensions} if extensions else None

    for p in root.rglob('*'):
        # Check if any part of the path is in the exclude list
        if any(part in exclude_dirs for part in p.parts):
            continue

        if not p.is_file():
            continue
        
        # Skip binary files
        if p.suffix.lower() in binary_extensions:
            continue

        # Exclude .windsurf_search_index and any .json files (case-insensitive)
        p_str = str(p).lower()
        if ".windsurf_search_index" in p_str or p_str.endswith(".json"):
            continue
        if extensions and p.suffix.lower() not in norm_exts:
            continue
        yield p

def _embed_batch(texts: list[str]) -> list[list[float]]:
    """Encodes a batch of texts into vector embeddings using the loaded model."""
    if not LIBS_AVAILABLE or _ST_MODEL is None:
        raise RuntimeError("Embedding libraries (torch, sentence-transformers) are not available.")
    logger.info(f"Embedding a batch of {len(texts)} texts on {DEVICE}...")
    with torch.no_grad():
        return _ST_MODEL.encode(texts, batch_size=32, show_progress_bar=False, device=DEVICE).tolist()

def _is_safe_path(path: pathlib.Path) -> bool:
    """Ensure *path* is inside one of the PROJECT_ROOTS roots."""
    try:
        resolved_path = path.resolve()
        for root in PROJECT_ROOTS.values():
            if resolved_path.is_relative_to(root.resolve()):
                return True
    except (OSError, ValueError):  # Catches resolution errors or invalid paths
        return False
    return False

# ---------------------------------------------------------------------------
# General-purpose Project Tools (migrated from toolz.py)
# ---------------------------------------------------------------------------

@kinesis_mcp.tool()
def list_project_files(project_name: str, extensions: Optional[List[str]] = None, max_items: int = 1000) -> List[str]:
    """
    Recursively list files for a given project.

    Args:
        project_name (str): Name of the project as defined in PROJECT_ROOTS (e.g., "MCP-Server").
        extensions (List[str], optional): List of file extensions to include (e.g., ["py", "md"]). If omitted, all files are included.
        max_items (int, optional): Maximum number of files to return (default: 1000).

    Returns:
        List[str]: Absolute file paths as strings.

    Usage:
        Use this tool to get a list of all source files in a project, optionally filtered by extension. Useful for building file pickers, search indexes, or for pre-filtering files for other tools.
    """
    logger.info("[list_files] project=%s extensions=%s", project_name, extensions)
    root = PROJECT_ROOTS.get(project_name)
    if not root:
        logger.error("Invalid project name: %s", project_name)
        return []
    results = []
    try:
        for fp in _iter_files(root, extensions):
            if len(results) >= max_items:
                logger.warning("Hit max_items limit of %d. Returning partial results.", max_items)
                break
            results.append(str(fp.resolve()))
        logger.info("[list_files] Found %d paths.", len(results))
    except Exception as e:
        logger.error("Error listing files for project '%s': %s", project_name, e, exc_info=True)
    return results

@kinesis_mcp.tool()
def read_project_file(absolute_file_path: str, max_bytes: int = 2_000_000) -> Dict[str, Any]:
    """
    Read a file from disk with path safety checks.

    Args:
        absolute_file_path (str): Full absolute path to the file (must be within the project root).
        max_bytes (int, optional): Maximum number of bytes to read (default: 2,000,000).

    Returns:
        dict: {
            'status': 'success'|'error',
            'file_path': str,
            'content': str or None,  # UTF-8 text or hex preview for binary
            'message': str
        }

    Usage:
        Use this tool to safely read the contents of a file for display, editing, or analysis. Binary files return a hex preview. Files outside project roots are blocked.
    """
    path = pathlib.Path(absolute_file_path)
    if not _is_safe_path(path):
        return {"status": "error", "file_path": absolute_file_path, "content": None, "message": "Access denied: Path is outside configured project roots."}
    if not path.is_file():
        return {"status": "error", "file_path": absolute_file_path, "content": None, "message": "Not a file."}
    try:
        data = path.read_bytes()[:max_bytes]
        try:
            content = data.decode("utf-8")
            message = f"Successfully read {len(data)} bytes as text."
        except UnicodeDecodeError:
            content = data.hex()[:1000]  # Return a hex preview for binary files
            message = f"Read {len(data)} bytes of binary data (showing hex preview)."
        return {"status": "success", "file_path": absolute_file_path, "content": content, "message": message}
    except Exception as e:
        logger.error("Failed to read file '%s': %s", absolute_file_path, e, exc_info=True)
        return {"status": "error", "file_path": absolute_file_path, "content": None, "message": str(e)}

# ---------------------------------------------------------------------------
# Tool 1: Indexing (Prerequisite for semantic searches)
# ---------------------------------------------------------------------------

@kinesis_mcp.tool()
def index_project_files(project_name: str, subfolder: Optional[str] = None, max_file_size_mb: int = 5) -> Dict:
    """
    Index the project for semantic search **incrementally**.

    This version detects file additions, modifications, and deletions so that we **avoid
    recomputing embeddings for every chunk on every run**.  It works by:

    1. Loading any existing FAISS index and accompanying ``metadata.json``.
    2. Collecting current project files (respecting ignore-lists / size / extension filters).
    3. Comparing each file's *mtime* and *size* against what was stored during the last
       indexing pass.
       * Unchanged  →  re-use the already-stored vectors (no embedding cost).
       * Added / modified → re-chunk + embed just those files.
       * Deleted → their vectors are dropped.
    4. Re-building a **fresh** FAISS index from the union of reused + new vectors.  Using a
       fresh index keeps the logic simple and avoids tricky in-place deletions while still
       saving the heavy embedding cost.

    NOTE: Each chunk entry in ``metadata.json`` now contains::

        {
          "path": "relative/path/to/file.py",
          "content": "<chunk text>",
          "vector": [..float32 list..],
          "file_mtime": 1719690000.123,
          "file_size": 2048
        }

    Args:
        project_name (str): Name of the project as defined in ``PROJECT_ROOTS``.
        subfolder (str, optional): If set, only index this subfolder within the project.
        max_file_size_mb (int, optional): Maximum file size (in MB) to include (default ``5``).

    Returns:
        dict: {
            'status': 'success' | 'error',
            'message': str,
            'files_scanned_and_included': int,   # Total files considered this run
            'unchanged_files': int,              # Re-used without re-embedding
            'updated_files': int,                # Added or modified & re-embedded
            'deleted_files': int,                # Removed from index
            'total_chunks_indexed': int,
            'indexing_duration_seconds': float
        }
    """
    if not LIBS_AVAILABLE:
        return {"status": "error", "message": "Indexing failed: Required libraries (faiss, numpy, sentence-transformers) are not installed."}

    # ------------------------------------------------------------
    # Config & Setup
    # ------------------------------------------------------------
    directories_to_ignore = {
        'node_modules', '.git', '__pycache__', 'venv', '.venv', 'target',
        'build', 'dist', '.cache', '.idea', '.vscode', 'eggs', '.eggs'
    }
    text_extensions_to_include = {
        '.py', '.js', '.ts', '.jsx', '.tsx', '.md', '.json', '.yaml', '.yml',
        '.html', '.css', '.scss', '.txt', '.sh', '.bat', '.ps1', '.xml', '.rb',
        '.java', '.c', '.h', '.cpp', '.go', '.rs', '.php'
    }
    max_file_size_bytes = max_file_size_mb * 1024 * 1024

    CHUNK_SIZE = 512
    CHUNK_OVERLAP = 64
    BATCH_SIZE = 128

    start_time = time.monotonic()

    project_path = _get_project_path(project_name)
    if not project_path:
        return {"status": "error", "message": f"Project '{project_name}' not found."}

    scan_path = project_path / subfolder if subfolder else project_path
    if subfolder and not scan_path.is_dir():
        return {"status": "error", "message": f"Subfolder '{subfolder}' not found in project."}

    index_path = scan_path / INDEX_DIR_NAME
    index_path.mkdir(exist_ok=True)

    # ------------------------------------------------------------
    # Step 1.  Gather current relevant files
    # ------------------------------------------------------------
    relevant_files: list[pathlib.Path] = [
        p for p in scan_path.rglob('*') if p.is_file() and
        not any(ignored in p.parts for ignored in directories_to_ignore) and
        p.suffix.lower() in text_extensions_to_include and
        p.stat().st_size <= max_file_size_bytes and
        ".windsurf_search_index" not in str(p) and not str(p).endswith(".json")
    ]
    logger.info("[index] Found %d relevant files to consider.", len(relevant_files))

    # ------------------------------------------------------------
    # Step 2.  Load previous metadata (if any)
    # ------------------------------------------------------------
    old_metadata_file = index_path / "metadata.json"
    old_metadata: list[dict] = []
    if old_metadata_file.exists():
        try:
            with open(old_metadata_file, "r", encoding="utf-8") as f:
                old_metadata = json.load(f)
            logger.info("[index] Loaded previous metadata with %d chunks.", len(old_metadata))
        except Exception as e:
            logger.warning("[index] Failed to load existing metadata.json: %s. A full re-index will be performed.", e)
            old_metadata = []

    # Helper: map file path ➜ first chunk entry (to inspect stored stats)
    old_stats_by_path: dict[str, dict] = {}
    for entry in old_metadata:
        path_key = entry.get("path")
        if path_key and path_key not in old_stats_by_path:
            old_stats_by_path[path_key] = entry

    # Current file stats lookup
    current_stats: dict[str, tuple] = {}
    for fp in relevant_files:
        stat = fp.stat()
        current_stats[str(fp.relative_to(project_path))] = (stat.st_mtime, stat.st_size, fp)

    # ------------------------------------------------------------
    # Step 3.  Categorise files
    # ------------------------------------------------------------
    unchanged_paths: set[str] = set()
    updated_paths: set[str] = set()
    for rel_path, (mtime, size, _fp) in current_stats.items():
        old = old_stats_by_path.get(rel_path)
        if old and old.get("file_mtime") == mtime and old.get("file_size") == size and "vector" in old:
            unchanged_paths.add(rel_path)
        else:
            updated_paths.add(rel_path)
    deleted_paths: set[str] = set(old_stats_by_path.keys()) - set(current_stats.keys())

    logger.info(
        "[index] unchanged=%d updated=%d deleted=%d",
        len(unchanged_paths), len(updated_paths), len(deleted_paths)
    )

    # ------------------------------------------------------------
    # Step 4.  Reuse vectors for unchanged chunks
    # ------------------------------------------------------------
    new_metadata: list[dict] = []
    all_vectors: list[list[float]] = []
    for entry in old_metadata:
        if entry.get("path") in unchanged_paths and "vector" in entry:
            new_metadata.append(entry)
            all_vectors.append(entry["vector"])

    # ------------------------------------------------------------
    # Step 5.  Process updated (new/modified) files
    # ------------------------------------------------------------
    batch_texts: list[str] = []
    batch_meta: list[tuple[str, int, float, int]] = []  # (rel_path, offset, mtime, size)

    for rel_path in updated_paths:
        fp = current_stats[rel_path][2]
        try:
            text = fp.read_text("utf-8", errors="ignore")
            if not text.strip():
                continue
            mtime, size, _ = current_stats[rel_path]
            for i in range(0, len(text), CHUNK_SIZE - CHUNK_OVERLAP):
                chunk_content = text[i : i + CHUNK_SIZE]
                batch_texts.append(chunk_content)
                batch_meta.append((rel_path, i, mtime, size))
        except Exception as e:
            logger.warning("[index] Could not read or chunk file %s: %s", fp, e)

    # Embed in batches
    for i in range(0, len(batch_texts), BATCH_SIZE):
        chunk_batch = batch_texts[i : i + BATCH_SIZE]
        vectors = _embed_batch(chunk_batch)
        for vec, (rel_path, _offset, mtime, size) in zip(vectors, batch_meta[i : i + BATCH_SIZE]):
            new_metadata.append({
                "path": rel_path,
                "content": chunk_batch.pop(0),  # pop to keep memory low
                "vector": vec,
                "file_mtime": mtime,
                "file_size": size,
            })
            all_vectors.append(vec)
        logger.info("[index] Embedded batch of %d chunks. Total chunks so far: %d", len(vectors), len(all_vectors))

    if not all_vectors:
        return {"status": "error", "message": "No chunks available to build index (all files empty or unreadable)."}

    # ------------------------------------------------------------
    # Step 6.  Build & save FAISS index + metadata
    # ------------------------------------------------------------
    try:
        dim = len(all_vectors[0])
        index = faiss.IndexFlatL2(dim)
        index.add(np.array(all_vectors, dtype=np.float32))

        faiss.write_index(index, str(index_path / "index.faiss"))
        with open(old_metadata_file, "w", encoding="utf-8") as f:
            json.dump(new_metadata, f)
    except Exception as e:
        logger.exception("[index] Failed to build or save index: %s", e)
        return {"status": "error", "message": f"Failed to build or save index: {e}"}

    duration = time.monotonic() - start_time
    return {
        "status": "success",
        "message": f"Project '{project_name}' indexed incrementally.",
        "files_scanned_and_included": len(relevant_files),
        "unchanged_files": len(unchanged_paths),
        "updated_files": len(updated_paths),
        "deleted_files": len(deleted_paths),
        "total_chunks_indexed": len(all_vectors),
        "indexing_duration_seconds": round(duration, 2),
    }


# ---------------------------------------------------------------------------
# Tool 2: The Search Multitool
# ---------------------------------------------------------------------------

# --- Pydantic Model for the Unified Search Request ---
class SearchRequest(BaseModel):
    search_type: Literal[
        "keyword", "semantic", "ast", "references", "similarity", "task_verification"
    ]
    query: str
    project_name: str
    params: Optional[Dict[str, Any]] = Field(default_factory=dict)

# --- Internal Logic for Each Search Type ---

def _search_by_keyword(query: str, project_path: pathlib.Path, params: Dict) -> Dict:
    """Performs a literal substring search across project files, honoring includes/extensions/max_results params."""
    import logging
    logger = logging.getLogger("mcp_search_tools")
    results = []
    files_scanned = 0

    includes = params.get("includes")
    extensions = params.get("extensions")
    max_results = params.get("max_results", 1000)

    # Build file list
    if includes:
        files = [project_path / inc if not os.path.isabs(inc) else pathlib.Path(inc) for inc in includes]
    else:
        files = list(_iter_files(project_path, extensions=extensions))

    for fp in files:
        logger.debug(f"[keyword] Scanning file: {fp}")
        # Filter out internal index/metadata files
        if ".windsurf_search_index" in str(fp):
            continue
        try:
            with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                for i, line_content in enumerate(f, 1):
                    if query in line_content:
                        results.append({
                            "file_path": str(fp),
                            "line_number": i,
                            "line_content": line_content.strip()[:200] + ("..." if len(line_content.strip()) > 200 else "")
                        })
                        if len(results) >= max_results:
                            return {
                                "status": "success",
                                "results": results,
                                "files_scanned": files_scanned + 1
                            }
            files_scanned += 1
        except Exception:
            continue
    return {
        "status": "success",
        "results": results,
        "files_scanned": files_scanned
    }

def _search_by_semantic(query: str, project_path: pathlib.Path, params: Dict) -> Dict:
    """Performs semantic search using the FAISS index.

    Supports params:
        includes: optional list of file paths to restrict the returned results.
        max_results: number of results to return (default 10).
    """
    if not LIBS_AVAILABLE:
        return {"status": "error", "message": "Semantic search failed: Required libraries not installed."}

    includes = params.get("includes")
    max_results = params.get("max_results", 10)
    subfolder = params.get("subfolder")

    search_root = project_path
    if subfolder:
        search_root = project_path / subfolder

    index_path = search_root / INDEX_DIR_NAME
    faiss_index_file = index_path / "index.faiss"
    metadata_file = index_path / "metadata.json"

    if not (faiss_index_file.exists() and metadata_file.exists()):
        return {"status": "error", "message": f"Index not found for project. Please run 'index_project_files' first."}

    try:
        index = faiss.read_index(str(faiss_index_file))
        with open(metadata_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        
        query_vec = _embed_batch([query])
        distances, indices = index.search(np.array(query_vec, dtype=np.float32), max_results)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx == -1:
                continue
            meta = metadata[idx]
            path_match = True
            if includes:
                # Check if the meta path is in the includes list.
                # The path in metadata is relative to the project root, as is the 'includes' path.
                # Normalize both for a reliable comparison.
                meta_path_obj = pathlib.Path(meta["path"])
                path_match = any(meta_path_obj == pathlib.Path(inc) for inc in includes)

            if path_match:
                results.append({
                    "score": float(distances[0][i]),
                    "path": meta["path"],
                    "content": meta["content"]
                })
            
            if len(results) >= max_results:
                break
        return {"status": "success", "results": results}
    except Exception as e:
        return {"status": "error", "message": f"An error occurred during semantic search: {e}"}

def _search_by_ast(query: str, project_path: pathlib.Path, params: Dict) -> Dict:
    """Finds definitions of functions or classes using AST.

    Supports params:
        includes: optional list of file paths to restrict the search to.
        target_node_type: 'function', 'class', or 'any'
        max_results: stop after this many matches (default 50).
    """
    target_node_type = params.get("target_node_type", "any")
    includes = params.get("includes")
    max_results = params.get("max_results", 50)

    class DefinitionFinder(ast.NodeVisitor):
        def __init__(self, query):
            self.query = query
            self.findings = []
            self.logger = logging.getLogger("mcp_search_tools")

        def visit_FunctionDef(self, node):
            self.logger.debug(f"[AST] Visiting function: {node.name}")
            is_match = self.query.lower() in node.name.lower()
            if is_match and (target_node_type in ["function", "any"]):
                self.logger.info(f"[AST] Found potential function match: {node.name}")
                try:
                    content = ast.get_source_segment(self.source_code, node)
                    if content:
                        self.findings.append({
                            "type": "function_definition",
                            "name": node.name,
                            "line_number": node.lineno,
                            "content": content[:200] + ("..." if len(content) > 200 else "")
                        })
                except Exception as e:
                    self.logger.warning(f"[AST] Could not get source for function {node.name}: {e}")
            self.generic_visit(node)

        def visit_ClassDef(self, node):
            self.logger.debug(f"[AST] Visiting class: {node.name}")
            is_match = self.query.lower() in node.name.lower()
            if is_match and (target_node_type in ["class", "any"]):
                self.logger.info(f"[AST] Found potential class match: {node.name}")
                try:
                    content = ast.get_source_segment(self.source_code, node)
                    if content:
                        self.findings.append({
                            "type": "class_definition",
                            "name": node.name,
                            "line_number": node.lineno,
                            "content": content[:200] + ("..." if len(content) > 200 else "")
                        })
                except Exception as e:
                    self.logger.warning(f"[AST] Could not get source for class {node.name}: {e}")
            self.generic_visit(node)

    results = []
    if includes:
        files = [project_path / inc if not os.path.isabs(inc) else pathlib.Path(inc) for inc in includes]
    else:
        files = _iter_files(project_path, extensions=[".py"])

    for fp in files:
        if len(results) >= max_results:
            break
        if ".windsurf_search_index" in str(fp) or str(fp).endswith(".json"):
            continue
        try:
            source_code = fp.read_text("utf-8")
            tree = ast.parse(source_code)
            visitor = DefinitionFinder(query)
            visitor.source_code = source_code
            visitor.visit(tree)
            for finding in visitor.findings:
                finding['file_path'] = str(fp)
                results.append(finding)
                if len(results) >= max_results:
                    break
        except Exception:
            continue

    if not results:
        return {"status": "not_found", "message": f"No AST matches found for '{query}'."}
    return {"status": "success", "results": results}

def _search_for_references(query: str, project_path: pathlib.Path, params: Dict) -> Dict:
    """Finds all usages of a symbol using Jedi, or falls back to grep if minimal context provided."""
    file_path = params.get("file_path")
    line = params.get("line")
    column = params.get("column")
    includes = params.get("includes")
    max_results = params.get("max_results", 100)

    # If file_path/line/column are provided, use Jedi's precise reference search
    if file_path and line is not None and column is not None:
        if not jedi:
            return {"status": "error", "message": "Jedi is not installed, which is required for precise reference search."}
        try:
            abs_file = project_path / file_path
            if not abs_file.exists():
                return {"status": "error", "message": f"File '{file_path}' not found in project."}
            source = abs_file.read_text("utf-8")
            script = jedi.Script(source, path=str(abs_file))
            refs = script.get_references(line=int(line), column=int(column), include_builtins=False)
            results = []
            for ref in refs:
                if len(results) >= max_results:
                    break
                results.append({
                    "file_path": str(ref.module_path),
                    "line": ref.line,
                    "column": ref.column,
                    "code": ref.get_line_code().strip(),
                    "is_definition": ref.is_definition()
                })
            return {"status": "success", "results": results}
        except Exception as e:
            return {"status": "error", "message": f"Jedi reference search failed: {e}"}

    # If not enough context for Jedi, fall back to grep-like search
    elif query:
        grep_results = []
        if includes:
            files = [project_path / inc if not os.path.isabs(inc) else pathlib.Path(inc) for inc in includes]
        else:
            files = _iter_files(project_path, extensions=[".py"])

        for fp in files:
            if len(grep_results) >= max_results:
                break
            if ".windsurf_search_index" in str(fp) or str(fp).endswith(".json"):
                continue
            try:
                with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                    for i, line_content in enumerate(f, 1):
                        if query in line_content:
                            grep_results.append({
                                "file_path": str(fp),
                                "line_number": i,
                                "line_content": line_content.strip()[:200] + ("..." if len(line_content.strip()) > 200 else "")
                            })
                            if len(grep_results) >= max_results:
                                break
            except Exception:
                continue
        
        if not grep_results:
            return {"status": "not_found", "message": "No references found."}
        return {"status": "success", "results": grep_results}

    return {"status": "error", "message": "Not enough parameters for reference search."}

def _search_for_similarity(query: str, project_path: pathlib.Path, params: Dict) -> Dict:
    """Finds code chunks semantically similar to the query block of code."""
    # This is essentially a semantic search where the query is a block of code.
    # We can reuse the semantic search logic directly.
    logger.info("Performing similarity search by reusing semantic search logic.")
    return _search_by_semantic(query, project_path, params)

def _verify_task_implementation(query: str, project_path: pathlib.Path, params: Dict) -> Dict:
    """A meta-search to find code related to a task description for agent assessment."""
    # Step 1: Semantic search for top relevant chunks
    sem_results = _search_by_semantic(query, project_path, {"max_results": 10})
    if sem_results.get("status") != "success":
        return sem_results
    # Step 2: Try to find a function/class definition matching task intent
    try:
        ast_results = _search_by_ast(query.split()[0], project_path, {"target_node_type": "any"})
    except Exception as e:
        ast_results = {"status": "error", "results": [], "message": f"AST search failed: {e}"}
    # Step 3: Combine and score
    found = False
    code_matches = []
    if ast_results and ast_results.get("status") == "success":
        for node in ast_results.get("results", []):
            content = node.get("content")
            if content and query.lower() in content.lower():
                found = True
                code_matches.append(node)
    score = 1.0 if found else 0.5 if sem_results["results"] else 0.0
    message = (
        "Task implementation likely present." if found else
        "No direct implementation found, but relevant code exists." if sem_results["results"] else
        "No relevant code found."
    )
    if ast_results and ast_results.get("status") != "success":
        message += f" [AST warning: {ast_results.get('message','AST context unavailable')}]"
    return {
        "status": "success" if found or sem_results["results"] else "not_found",
        "score": score,
        "semantic_results": sem_results["results"],
        "ast_matches": code_matches,
        "message": message
    }

# --- The Single MCP Tool Endpoint for Searching ---
@kinesis_mcp.tool(name="search")
def unified_search(request: SearchRequest) -> Dict[str, Any]:
    """
    Multi-modal codebase search tool. Supports keyword, semantic, AST, references, similarity, and task_verification modes.

    Args:
        request (SearchRequest):
            - search_type (str): One of ['keyword', 'semantic', 'ast', 'references', 'similarity', 'task_verification'].
            - query (str): The search string or code snippet.
            - project_name (str): Name of the project as defined in PROJECT_ROOTS.
            - params (dict, optional):
                - includes (List[str], optional): Restrict search to these files or folders.
                - max_results (int, optional): Maximum number of results to return.
                - extensions (List[str], optional): Filter files by extension (for 'keyword').
                - target_node_type (str, optional): 'function', 'class', or 'any' (for 'ast').
                - file_path, line, column (for 'references').

    Returns:
        dict: {'status': 'success'|'error'|'not_found', 'results': list, ...}

    Usage:
        - 'keyword': Fast literal search. Supports includes, extensions, max_results.
        - 'semantic': Natural language/code search. Requires prior indexing. Supports includes, max_results.
        - 'ast': Find definitions by structure. Supports includes, target_node_type, max_results.
        - 'references': Find usages of a symbol. Supports includes, file_path, line, column, max_results.
        - 'similarity': Find similar code blocks. Query must be a code snippet. Supports includes, max_results.
        - 'task_verification': Check if a task is implemented. Query is a task description. Supports includes, max_results.
    """
    search_type = request.search_type
    project_name = request.project_name
    logger.info(f"[search] type='{search_type}' project='{project_name}' q='{request.query[:50]}...'")

    project_path = _get_project_path(project_name)
    if not project_path:
        return {"status": "error", "message": f"Project '{project_name}' not found."}

    # --- Router logic to call the correct internal search function ---
    search_functions = {
        "keyword": _search_by_keyword,
        "semantic": _search_by_semantic,
        "ast": _search_by_ast,
        "references": _search_for_references,
        "similarity": _search_for_similarity,
        "task_verification": _verify_task_implementation,
    }
    
    search_func = search_functions.get(search_type)

    if search_func:
        return search_func(request.query, project_path, request.params)
    else:
        return {"status": "error", "message": "Invalid search type specified."}

# ---------------------------------------------------------------------------
# Main execution block to run the server
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if not LIBS_AVAILABLE:
        logger.error("Critical libraries (torch, sentence-transformers, faiss-cpu) are not installed.")

# ---------------------------------------------------------------------------
# Tool 3: The /analyze Multitool
# ---------------------------------------------------------------------------
@kinesis_mcp.tool()
def analyze(request: AnalysisRequest) -> dict:
    """
    Run static analyses on files or directories (quality, types, security, dead code, complexity, TODOs).

    Args:
        request (AnalysisRequest):
            - path (str): File or directory to analyze
            - analyses (List[str]): Which analyses to run. Allowed: 'quality', 'types', 'security', 'dead_code', 'complexity', 'todos'.

    Returns:
        dict: {
            'status': 'success'|'error',
            'results': Dict[str, Any],
            'message': str
        }
    Usage:
        analyze({"path": "foo.py", "analyses": ["quality", "types"]})
        analyze({"path": "src/", "analyses": ["dead_code", "complexity"]})
        analyze({"path": "src/", "analyses": ["audit_package_versions"]})
        analyze({"path": "src/", "analyses": ["check_licenses"]})
    """
    logger.info(f"[analyze] path={request.path} analyses={request.analyses}")
    results = {}
    path = pathlib.Path(request.path)
    if not _is_safe_path(path):
        return {"status": "error", "results": {}, "message": "Unsafe path."}
    if not path.exists():
        return {"status": "error", "results": {}, "message": "Path does not exist."}

    try:
        for analysis in request.analyses:
            if analysis == "quality":
                if pylint_run is None:
                    results["quality"] = "pylint not installed"
                else:
                    logger.info(f"[analyze] Running pylint on {path}")
                    try:
                        import sys
                        venv_scripts = os.path.dirname(sys.executable)
                        if sys.platform == 'win32':
                            pylint_exec = 'pylint.exe'
                        else:
                            pylint_exec = 'pylint'
                        pylint_path = os.path.join(venv_scripts, pylint_exec)
                        logger.info(f"[analyze] Using pylint at: {pylint_path}")
                        pylint_output = subprocess.run([
                            pylint_path, str(path)
                        ], capture_output=True, text=True, timeout=60)
                        results["quality"] = pylint_output.stdout
                    except Exception as e:
                        results["quality"] = f"pylint error: {e}"
            elif analysis == "types":
                if mypy_api is None:
                    results["types"] = "mypy not installed"
                else:
                    logger.info(f"[analyze] Running mypy on {path}")
                    try:
                        mypy_result = mypy_api.run([str(path)])
                        results["types"] = mypy_result[0]
                    except Exception as e:
                        results["types"] = f"mypy error: {e}"
            elif analysis == "security":
                if bandit_manager is None or bandit_config is None:
                    results["security"] = "bandit not installed"
                else:
                    logger.info(f"[analyze] Running bandit on {path}")
                    try:
                        conf = bandit_config.BanditConfig()
                        mgr = bandit_manager.BanditManager(conf, "file")
                        mgr.discover_files([str(path)])
                        mgr.run_tests()
                        # Convert Bandit issues to dicts, compatibly for all Bandit versions
                        try:
                            issues = mgr.get_issue_list(sev_filter=None, conf_filter=None)
                        except TypeError as te:
                            logger.warning(f"[analyze] Bandit get_issue_list() does not accept sev_filter/conf_filter: {te}. Retrying without arguments.")
                            issues = mgr.get_issue_list()
                        results["security"] = [issue.as_dict() if hasattr(issue, 'as_dict') else str(issue) for issue in issues]
                    except Exception as e:
                        results["security"] = f"bandit error: {e}"
            elif analysis == "dead_code":
                if Vulture is None:
                    results["dead_code"] = "vulture not installed"
                else:
                    logger.info(f"[analyze] Running vulture on {path}")
                    try:
                        v = Vulture()
                        v.scavenge([str(path)])
                        # Vulture returns a list of Item objects; convert them to dicts
                        unused = v.get_unused_code()
                        results["dead_code"] = [item.__dict__ if hasattr(item, '__dict__') else str(item) for item in unused]
                    except Exception as e:
                        results["dead_code"] = f"vulture error: {e}"
            elif analysis == "complexity":
                if cc_visit is None:
                    results["complexity"] = "radon not installed"
                else:
                    logger.info(f"[analyze] Running radon on {path}")
                    try:
                        if sys.platform == 'win32':
                            radon_exec = 'radon.exe'
                        else:
                            radon_exec = 'radon'
                        radon_path = os.path.join(os.path.dirname(sys.executable), radon_exec)
                        logger.info(f"[analyze] Using radon at: {radon_path}")
                        cc_results = subprocess.run([
                            radon_path, 'cc', str(path)
                        ], capture_output=True, text=True, timeout=60)
                        results["complexity"] = cc_results.stdout
                    except Exception as e:
                        results["complexity"] = f"radon error: {e}"
            elif analysis == "todos":
                logger.info(f"[analyze] Scanning for TODOs in {path}")
                try:
                    todos = []
                    if path.is_file():
                        with path.open("r", encoding="utf-8", errors="ignore") as f:
                            for i, line in enumerate(f, 1):
                                if "TODO" in line:
                                    todos.append({"line": i, "text": line.strip()})
                    else:
                        for file in path.rglob("*.py"):
                            with file.open("r", encoding="utf-8", errors="ignore") as f:
                                for i, line in enumerate(f, 1):
                                    if "TODO" in line:
                                        todos.append({"file": str(file), "line": i, "text": line.strip()})
                    results["todos"] = todos
                except Exception as e:
                    results["todos"] = f"TODO scan error: {e}"
            elif analysis == "check_licenses":
                try:
                    license_req = CheckLicensesRequest()
                    results["check_licenses"] = check_licenses(license_req)
                except Exception as e:
                    results["check_licenses"] = f"check_licenses error: {e}"
            else:
                results[analysis] = f"Unknown analysis: {analysis}"
        return {"status": "success", "results": results, "message": "Analysis complete."}
    except Exception as e:
        logger.error(f"[analyze] error: {e}", exc_info=True)
        return {"status": "error", "results": results, "message": str(e)}

# ---------------------------------------------------------------------------
# Tool 4: The /edit Multitool
# ---------------------------------------------------------------------------
@kinesis_mcp.tool()
def edit(request: BatchEditRequest) -> dict:
    """
    Perform batch code edits (replace, insert, delete, rename symbol, add docstring) with preview/diff support.

    Args:
        request (BatchEditRequest):
            - file_path (str): File to edit
            - edits (List[EditOperation]): List of edit operations to apply
            - preview_only (bool): If True, only preview/diff changes, do not write

    Returns:
        dict: {
            'status': 'success'|'error',
            'preview': bool,
            'diff': str (if preview),
            'modified_content': str (if preview),
            'message': str
        }
    Usage:
        edit({"file_path": "foo.py", "edits": [{"operation": "replace_block", ...}], "preview_only": true})
    """
    logger.info(f"[edit] file_path={request.file_path} preview_only={request.preview_only}")
    target_path = pathlib.Path(request.file_path)
    if not _is_safe_path(target_path):
        return {"status": "error", "message": "Unsafe path."}
    if not target_path.is_file():
        return {"status": "error", "message": "File not found."}
    try:
        original_content = target_path.read_text("utf-8")
        modified_content = original_content

        # Text-based operations
        for edit_op in request.edits:
            if edit_op.operation == "replace_block":
                logger.info(f"[edit] replace_block params={edit_op.params}")
                old = edit_op.params.get("old")
                new = edit_op.params.get("new")
                if old is None or new is None:
                    raise ValueError("replace_block requires 'old' and 'new' params")
                modified_content = modified_content.replace(old, new)
            elif edit_op.operation == "insert_at":
                logger.info(f"[edit] insert_at params={edit_op.params}")
                idx = edit_op.params.get("index")
                text = edit_op.params.get("text")
                if idx is None or text is None:
                    raise ValueError("insert_at requires 'index' and 'text' params")
                lines = modified_content.splitlines(keepends=True)
                lines.insert(idx, text)
                modified_content = "".join(lines)
            elif edit_op.operation == "delete_block":
                logger.info(f"[edit] delete_block params={edit_op.params}")
                block = edit_op.params.get("block")
                if block is None:
                    raise ValueError("delete_block requires 'block' param")
                modified_content = modified_content.replace(block, "")

        # LibCST-based operations (rename_symbol, add_docstring)
        if any(op.operation in ("rename_symbol", "add_docstring") for op in request.edits):
            if not cst:
                return {"status": "error", "message": "LibCST is not installed, cannot perform edits."}
            # TODO: Replace with actual LibCST logic for code transformations
            class LibCSTEditor:
                def __init__(self, code):
                    self.code = code
                def rename(self, old, new):
                    raise NotImplementedError("LibCST-based rename not implemented yet.")
                def add_docstring(self, node, doc):
                    raise NotImplementedError("LibCST-based docstring insertion not implemented yet.")
            editor = LibCSTEditor(modified_content)
            for edit_op in request.edits:
                if edit_op.operation == "rename_symbol":
                    old = edit_op.params.get("old")
                    new = edit_op.params.get("new")
                    if old is None or new is None:
                        raise ValueError("rename_symbol requires 'old' and 'new' params")
                    editor.rename(old, new)
                elif edit_op.operation == "add_docstring":
                    node = edit_op.params.get("node")
                    doc = edit_op.params.get("doc")
                    if node is None or doc is None:
                        raise ValueError("add_docstring requires 'node' and 'doc' params")
                    editor.add_docstring(node, doc)
            modified_content = editor.code

        if request.preview_only:
            diff = "".join(difflib.unified_diff(
                original_content.splitlines(keepends=True),
                modified_content.splitlines(keepends=True),
                fromfile=f"a/{request.file_path}",
                tofile=f"b/{request.file_path}",
            ))
            return {
                "status": "success",
                "preview": True,
                "diff": diff,
                "modified_content": modified_content
            }
        else:
            target_path.write_text(modified_content, "utf-8")
            return {
                "status": "success",
                "preview": False,
                "message": f"Successfully applied {len(request.edits)} edits to '{request.file_path}'."
            }
    except Exception as e:
        logger.error(f"[edit] Transaction failed for file '{request.file_path}': {e}", exc_info=True)
        return {"status": "error", "message": f"Edit failed: {e}"}

# ---------------------------------------------------------------------------
# Main execution block to run the server
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if not LIBS_AVAILABLE:
        logger.error("Critical libraries (torch, sentence-transformers, faiss-cpu) are not installed.")
        logger.error("Please run: pip install faiss-cpu sentence-transformers torch numpy jedi")
    else:
        logger.info("Starting Project Search & Index MCP Server...")
        kinesis_mcp.run()