"""
MCP Server with a consolidated, multi-modal search tool."""

# --- DEBUG: Enhanced Python Environment Info ---
import sys
import os
import importlib
import pkgutil
import site

print("\n=== DEBUG: Python Environment ===")
print(f"Python Executable: {sys.executable}")
print(f"Python Version: {sys.version}")
print(f"Virtual Env: {os.environ.get('VIRTUAL_ENV', 'Not in virtual environment')}")
print("\n=== sys.path ===")
for p in sys.path:
    print(f" - {p}")

print("\n=== Installed Packages ===")
for m in sorted(sys.modules.keys()):
    if m.startswith('radon') or m.startswith('_radon') or m in ('mcp', 'fastmcp'):
        try:
            mod = sys.modules[m]
            print(f"{m}: {getattr(mod, '__file__', 'no __file__')}")
        except Exception as e:
            print(f"{m}: Error accessing - {str(e)}")

# --- SUPPRESS ALL WARNINGS AND NOISY STARTUP MESSAGES (for Windsurf handshake) ---
import warnings
warnings.filterwarnings("ignore")  # Ignore all warnings (Deprecation, User, etc.)
import os
os.environ["PYTHONWARNINGS"] = "ignore"
# Suppress numpy warnings and info
try:
    import numpy as _np
    _np.warnings.filterwarnings("ignore")
    _np.seterr(all="ignore")
except Exception:
    pass
# Silence bandit warnings if possible
try:
    import logging as _logging
    _logging.getLogger("bandit").setLevel(_logging.ERROR)
except Exception:
    pass

import concurrent.futures
import functools
import traceback
import multiprocessing
import time
from datetime import datetime, timedelta

# This module provides two primary tools for an AI agent:
#   - index_project_files: Scans and creates a searchable vector index of the project. This must be run before using semantic search capabilities.
#   - search: A powerful "multitool" that provides multiple modes of searching:
#     - 'keyword': Literal text search.
#     - 'semantic': Natural language concept search using the vector index.
#     - 'ast': Structural search for definitions (functions, classes).
#     - 'references': Finds all usages of a symbol.
#     - 'similarity': Finds code blocks semantically similar to a given snippet.
#     - 'task_verification': A meta-search to check the implementation status of a task.

import ast
import json
import logging
import os
import pathlib
import re
import time
from typing import Any, Dict, List, Literal, Optional, Set
import uvicorn

from fastmcp import FastMCP
from pydantic import BaseModel, Field

# --- Add this block with the other imports ---
import subprocess
import difflib

try:
    print("\n=== DEBUG: Attempting to import radon ===")
    print("Searching for radon in:")
    for finder, name, _ in pkgutil.iter_modules():
        if 'radon' in name:
            print(f" - Found: {name} at {getattr(finder, 'path', 'unknown path')}")
    # Try importing with debug info
    try:
        import radon
        print(f"Radon imported successfully from: {radon.__file__}")
        print(f"Radon version: {getattr(radon, '__version__', 'unknown')}")
        # Try importing specific modules
        from radon.cli.harvest import CCHarvester
        # Radon 6.x defines Config inside radon.cli.__init__, not radon.cli.config
        from radon.cli import Config
        print("Radon CLI modules imported successfully")
        RADON_AVAILABLE = True
    except Exception as e:
        print(f"Radon import failed: {type(e).__name__}: {str(e)}")
        print("Available modules in radon package:")
        if 'radon' in sys.modules:
            radon_path = os.path.dirname(sys.modules['radon'].__file__)
            for item in os.listdir(radon_path):
                if item.endswith('.py') and not item.startswith('_'):
                    print(f" - {item}")
        RADON_AVAILABLE = False
        CCHarvester = None
        Config = None
        # Do not raise here, just mark as unavailable
except Exception as e:
    print(f"[DEBUG] Error in radon debug import block: {e}")
    RADON_AVAILABLE = False
    CCHarvester = None
    Config = None
    # Do not raise here, just mark as unavailable
    print("\n=== RADON IMPORT FAILED ===")
    print("Error: ")
    print("This suggests either:")
    print("1. Radon is not installed in the current environment")
    print("2. The installation is corrupted")
    print("3. There's a version mismatch")
    print("\nTo fix this, try:")
    print("1. pip uninstall -y radon")
    print("2. pip install --no-cache-dir radon==6.0.1")
    print("3. Verify with: python -c 'import radon; print(radon.__file__)'")
    print("\nCurrent Python environment:")
    print(f"  - Executable: {sys.executable}")
    print(f"  - Path: {sys.path}")
    
    CCHarvester = None
    Config = None
    RADON_AVAILABLE = False

try:
    import libcst as cst
except ImportError:
    cst = None

from typing import Any, Dict, List, Literal

class AnalysisRequest(BaseModel):
    path: str
    analyses: List[Literal["quality", "types", "security", "dead_code", "complexity", "todos"]]

class SuggestTestsRequest(BaseModel):
    file_path: str = Field(..., description="The absolute path to the file containing the function.")
    function_name: str = Field(..., description="The name of the function to generate tests for.")
    model: Optional[str] = Field(None, description="(Optional) Ollama model to use for test generation. Overrides the default if set.")

class AddToCookbookRequest(BaseModel):
    pattern_name: str = Field(..., description="A unique, file-safe name for the pattern (e.g., 'standard_error_handler').")
    file_path: str = Field(..., description="The absolute path to the file containing the reference function.")
    function_name: str = Field(..., description="The name of the reference function to save as a pattern.")
    description: str = Field(..., description="A clear, one-sentence description of what this pattern does and when to use it.")

class FindInCookbookRequest(BaseModel):
    query: str = Field(..., description="A natural language query to search for a relevant code pattern.")

# --- End of new imports and models ---

# --- New Tool Request Models ---
from typing import Literal, Optional
from pydantic import BaseModel, Field

class IntrospectRequest(BaseModel):
    mode: Literal[
        "config", "outline", "stats", "inspect"
    ] = Field(..., description="The introspection mode to use.")
    file_path: Optional[str] = Field(None, description="The absolute path to the file for inspection.")
    # For 'inspect' mode
    function_name: Optional[str] = Field(None, description="The name of the function to inspect.")
    class_name: Optional[str] = Field(None, description="The name of the class to inspect.")
    # For 'config' mode
    config_type: Optional[Literal["pyproject", "requirements"]] = Field(None, description="The type of config file to read.")

class SnippetRequest(BaseModel):
    file_path: str = Field(..., description="The absolute path to the file.")
    mode: Literal["function", "class", "lines"] = Field(..., description="The extraction mode.")
    # For function/class mode
    name: Optional[str] = Field(None, description="The name of the function or class to extract.")

    # For lines mode
    start_line: Optional[int] = Field(None, description="The starting line number (1-indexed).")
    end_line: Optional[int] = Field(None, description="The ending line number (inclusive).")


# --- Dependency Imports ---
# These are required for the tools to function.
# Ensure you have them installed:
# pip install faiss-cpu sentence-transformers torch numpy jedi

try:
    import faiss
    import numpy as np
    from sentence_transformers import SentenceTransformer
    import torch
    
    # --- Global Model and Device Configuration (with Lazy Loading) ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    _ST_MODEL_INSTANCE = None

    def _get_st_model():
        """Lazily loads the SentenceTransformer model on first use."""
        global _ST_MODEL_INSTANCE
        if _ST_MODEL_INSTANCE is None:
            logger.info(f"[Lazy Load] Loading sentence-transformer model 'all-MiniLM-L6-v2' onto device '{DEVICE}'...")
            _ST_MODEL_INSTANCE = SentenceTransformer('all-MiniLM-L6-v2', device=DEVICE)
            # Suppress the noisy INFO log from SentenceTransformer after loading
            st_logger = logging.getLogger("sentence_transformers.SentenceTransformer")
            st_logger.setLevel(logging.WARNING)
            logger.info("[Lazy Load] Model loaded successfully.")
        return _ST_MODEL_INSTANCE

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
logger = logging.getLogger("mcp_search_tools")

logger.info("[MCP SERVER IMPORT] Importing src/toolz.py and initializing FastMCP instance...")
mcp = FastMCP(
    name="MCP-Server",
)


# ---------------------------------------------------------------------------
# Project root configuration
# ---------------------------------------------------------------------------
# Assumes this file is in 'src/' and the project root is its parent's parent.
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
PROJECT_ROOTS: Dict[str, pathlib.Path] = {
    "MCP-Server": PROJECT_ROOT,
}
INDEX_DIR_NAME = ".windsurf_search_index"

# ---------------------------------------------------------------------------
# Core Helper Functions
# ---------------------------------------------------------------------------


import concurrent.futures
import functools
import traceback
import multiprocessing
import time

def tool_process_timeout_and_errors(timeout=60):
    """
    Decorator to run a function in a separate process with a hard timeout and robust error handling.
    Returns MCP-protocol-compliant error dicts on timeout or exception.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            def target_fn(q, *a, **k):
                try:
                    result = func(*a, **k)
                    q.put(("success", result))
                except Exception as e:
                    tb = traceback.format_exc()
                    q.put(("error", {"status": "error", "message": str(e), "traceback": tb}))
            q = multiprocessing.Queue()
            p = multiprocessing.Process(target=target_fn, args=(q, *args), kwargs=kwargs)
            p.start()
            try:
                status, result = q.get(timeout=timeout)
                p.join(1)
                if status == "success":
                    return result
                else:
                    logging.error(f"[tool_process_timeout_and_errors] Tool error: {result.get('message')}")
                    return result
            except Exception as e:
                p.terminate()
                logging.error(f"[tool_process_timeout_and_errors] Tool timed out or crashed: {e}")
                return {"status": "error", "message": f"Tool timed out after {timeout} seconds or crashed.", "exception": str(e)}
        return wrapper
    return decorator

def tool_timeout_and_errors(timeout=60):
    """
    Decorator to enforce timeout and robust error handling for MCP tool functions.
    Logs all exceptions and timeouts. Returns a dict with status and error message on failure.
    Args:
        timeout (int): Timeout in seconds for the tool execution.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger = logging.getLogger("mcp_search_tools")
            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(func, *args, **kwargs)
                    try:
                        return future.result(timeout=timeout)
                    except concurrent.futures.TimeoutError:
                        logger.error(f"[TIMEOUT] Tool '{func.__name__}' timed out after {timeout} seconds.")
                        return {"status": "error", "message": f"Tool '{func.__name__}' timed out after {timeout} seconds."}
            except Exception as e:
                tb = traceback.format_exc()
                logger.error(f"[EXCEPTION] Tool '{func.__name__}' failed: {e}\n{tb}")
                return {"status": "error", "message": f"Tool '{func.__name__}' failed: {e}", "traceback": tb}
        return wrapper
    return decorator



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
    """Encodes a batch of texts into vector embeddings using the lazily-loaded model."""
    if not LIBS_AVAILABLE:
        raise RuntimeError("Embedding libraries (torch, sentence-transformers) are not available.")
    model = _get_st_model()
    logger.info(f"Embedding a batch of {len(texts)} texts on {DEVICE}...")
    with torch.no_grad():
        return model.encode(texts, batch_size=32, show_progress_bar=False, device=DEVICE).tolist()

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

@mcp.tool()
def list_files_anywhere(
    directory_path: str,
    extensions: Optional[List[str]] = None,
    max_items: int = 1000
) -> dict:
    """
    List files under any absolute directory path, optionally filtering by extension.

    Args:
        directory_path (str): Absolute path to the directory to scan.
        extensions (List[str], optional): List of file extensions to include (e.g., ["py", "md"]). If omitted, all files are included.
        max_items (int, optional): Maximum number of files to return (default: 1000).

    Returns:
        dict: {
            'status': 'success'|'error',
            'files': List[str],  # Only present if status == 'success'
            'message': str       # Error message if status == 'error'
        }
    """
    import pathlib, os
    from typing import List, Optional, Dict, Any
    results = []
    try:
        root = pathlib.Path(directory_path)
        if not root.is_dir():
            return {"status": "error", "message": f"Not a directory: {directory_path}"}
        if extensions:
            norm_exts = {f".{str(e).lower().lstrip('.')}" for e in extensions if isinstance(e, str)}
        else:
            norm_exts = None
        for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
            for fname in filenames:
                fpath = pathlib.Path(dirpath) / fname
                if fpath.is_symlink():
                    continue
                if norm_exts and fpath.suffix.lower() not in norm_exts:
                    continue
                results.append(str(fpath.resolve()))
                if len(results) >= max_items:
                    return {"status": "success", "files": results}
        return {"status": "success", "files": results}
    except Exception as e:
        return {"status": "error", "message": str(e)}
@tool_timeout_and_errors(timeout=60)
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

@tool_process_timeout_and_errors(timeout=300)
@mcp.tool(name="index_project_files")
def index_project_files(project_name: str, subfolder: Optional[str] = None, max_file_size_mb: int = 5) -> dict:
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
# ---------------------------------------------------------------------------

# --- Pydantic Model for the Unified Search Request ---
class SearchRequest(BaseModel):
    search_type: Literal[
        "keyword", "regex", "semantic", "ast", "references", "similarity", "task_verification"
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

def _search_by_regex(query: str, project_path: pathlib.Path, params: Dict) -> Dict:
    """
    Search project files using a regular expression.

    Params dictionary may contain:
        - includes (List[str]): optional paths relative to project root
        - extensions (List[str]): file extensions when includes is omitted
        - ignore_case (bool)
        - multiline (bool)
        - dotall (bool)
        - max_results (int, default 1000)
    """
    import logging
    import re
    logger = logging.getLogger("mcp_search_tools")
    flags = 0
    if params.get("ignore_case"):
        flags |= re.IGNORECASE
    if params.get("multiline"):
        flags |= re.MULTILINE
    if params.get("dotall"):
        flags |= re.DOTALL

    try:
        pattern = re.compile(query, flags)
    except re.error as exc:
        return {"status": "error", "message": f"Invalid regex: {exc}"}

    includes = params.get("includes")
    extensions = params.get("extensions")
    max_results = params.get("max_results", 1000)

    # Build file list
    if includes:
        files = [project_path / inc if not os.path.isabs(inc) else pathlib.Path(inc) for inc in includes]
    else:
        files = list(_iter_files(project_path, extensions=extensions))

    results = []
    files_scanned = 0

    for fp in files:
        logger.debug(f"[regex] Scanning file: {fp}")
        if ".windsurf_search_index" in str(fp):
            continue
        try:
            with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                for lineno, line in enumerate(f, 1):
                    for match in pattern.finditer(line):
                        snippet = match.group(0)
                        if len(snippet) > 200:
                            snippet = snippet[:200] + "..."
                        results.append({
                            "file_path": str(fp),
                            "line_number": lineno,
                            "match": snippet,
                        })
                        if len(results) >= max_results:
                            return {
                                "status": "success",
                                "results": results,
                                "files_scanned": files_scanned + 1,
                            }
            files_scanned += 1
        except Exception:
            continue

    return {"status": "success", "results": results, "files_scanned": files_scanned}

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
@tool_process_timeout_and_errors(timeout=120)  # Increased timeout for potentially long searches
@mcp.tool(name="search")
def unified_search(request: SearchRequest) -> Dict[str, Any]:
    """
    Multi-modal codebase search tool. Supports keyword, regex, semantic, AST, references, similarity, and task_verification modes.
    This tool enforces a hard timeout and robust error handling using a separate process.
    If the tool fails or times out, it returns a structured MCP error response.

    Args:
        request (SearchRequest):
            - search_type (str): One of ['keyword', 'regex', 'semantic', 'ast', 'references', 'similarity', 'task_verification'].
            - query (str): The search string or code snippet.
            - project_name (str): Name of the project as defined in PROJECT_ROOTS.
            - params (dict, optional):
                - includes (List[str], optional): Restrict search to these files or folders.
                - max_results (int, optional): Maximum number of results to return.
                - extensions (List[str], optional): Filter files by extension (for 'keyword' and 'regex').
                - ignore_case (bool): Case-insensitive regex search (for 'regex').
                - multiline (bool): Multiline regex flag (for 'regex').
                - dotall (bool): Dot matches newline (for 'regex').
                - target_node_type (str, optional): 'function', 'class', or 'any' (for 'ast').
                - file_path, line, column (for 'references').

    Returns:
        dict: {
            'status': 'success'|'error'|'not_found', 'results': list, ...
        }

    Usage:
        - 'keyword': Fast literal search. Supports includes, extensions, max_results.
        - 'regex': Advanced regex search. Supports includes, extensions, ignore_case, multiline, dotall, max_results.
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
        "regex": _search_by_regex,
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

# Tool 4: Test Generation
# ---------------------------------------------------------------------------
# --- Ollama Model Selection for Test Generation ---
import os

# Main and fallback models
_OLLAMA_MAIN_MODEL = "goekdenizguelmez/JOSIEFIED-Qwen3:8b-deepseek-r1-0528"
_OLLAMA_FALLBACK_MODEL = "qwen3:4b"

# Allow override via environment variable
OLLAMA_MODEL_FOR_TESTS = os.environ.get("OLLAMA_MODEL_FOR_TESTS", _OLLAMA_MAIN_MODEL)

# === Code Cookbook Tools ===
COOKBOOK_DIR_NAME = ".project_cookbook"

class CookbookMultitoolRequest(BaseModel):
    mode: str = Field(..., description="Operation mode: 'add' or 'find'.")
    # For 'add' mode
    pattern_name: Optional[str] = Field(None, description="Unique name for the pattern (required for 'add').")
    file_path: Optional[str] = Field(None, description="Absolute path to the file containing the function (required for 'add').")
    function_name: Optional[str] = Field(None, description="Name of the function to save as a pattern (required for 'add').")
    description: Optional[str] = Field(None, description="Short description of the pattern (required for 'add').")
    # For 'find' mode
    query: Optional[str] = Field(None, description="Search query for finding patterns (required for 'find').")

def _add_to_cookbook_impl(request: AddToCookbookRequest) -> dict:
    logger.info(f"[add_to_cookbook] Adding pattern '{request.pattern_name}' from {request.file_path}::{request.function_name}")
    project_path = _get_project_path("MCP-Server")
    if not project_path:
        return {"status": "error", "message": "Project 'MCP-Server' not found."}
    cookbook_dir = project_path / COOKBOOK_DIR_NAME
    cookbook_dir.mkdir(exist_ok=True)
    safe_filename = re.sub(r'[^\w\-_\. ]', '_', request.pattern_name) + ".json"
    output_path = cookbook_dir / safe_filename
    if output_path.exists():
        return {"status": "error", "message": f"Pattern '{request.pattern_name}' already exists at {output_path}"}
    source_file_path = pathlib.Path(request.file_path)
    if not _is_safe_path(source_file_path):
        return {"status": "error", "message": "Access denied: Source file path is outside configured project roots."}
    if not source_file_path.is_file():
        return {"status": "error", "message": f"Source file not found at: {request.file_path}"}
    try:
        source_code_text = source_file_path.read_text("utf-8")
        tree = ast.parse(source_code_text)
        class FunctionFinder(ast.NodeVisitor):
            def __init__(self, target_name):
                self.target_name = target_name
                self.found_node = None
            def visit_FunctionDef(self, node):
                if node.name == self.target_name:
                    self.found_node = node
                self.generic_visit(node)
        finder = FunctionFinder(request.function_name)
        finder.visit(tree)
        if not finder.found_node:
            return {"status": "error", "message": f"Could not find function '{request.function_name}' in '{request.file_path}'."}
        function_source = ast.get_source_segment(source_code_text, finder.found_node)
        if not function_source:
             return {"status": "error", "message": f"Could not extract source for function '{request.function_name}'."}
    except Exception as e:
        logger.error(f"[add_to_cookbook] AST parsing failed for {request.file_path}: {e}")
        return {"status": "error", "message": f"Failed to parse or read source file: {e}"}
    pattern_data = {
        "pattern_name": request.pattern_name,
        "description": request.description,
        "source_file": request.file_path,
        "function_name": request.function_name,
        "source_code": function_source,
        "added_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(pattern_data, f, indent=4)
        logger.info(f"[add_to_cookbook] Successfully saved pattern to {output_path}")
        return {"status": "success", "message": f"Pattern '{request.pattern_name}' was successfully added to the cookbook."}
    except Exception as e:
        logger.error(f"[add_to_cookbook] Failed to write pattern file {output_path}: {e}")
        return {"status": "error", "message": f"Failed to write pattern file: {e}"}

def _find_in_cookbook_impl(request: FindInCookbookRequest) -> dict:
    logger.info(f"[find_in_cookbook] Searching for pattern with query: '{request.query}'")
    project_path = _get_project_path("MCP-Server")
    if not project_path:
        return {"status": "error", "message": "Project 'MCP-Server' not found."}
    cookbook_dir = project_path / COOKBOOK_DIR_NAME
    if not cookbook_dir.is_dir():
        return {"status": "not_found", "message": "Cookbook directory does not exist. Add a pattern first."}
    matches = []
    query_lower = request.query.lower()
    for pattern_file in cookbook_dir.glob("*.json"):
        try:
            with open(pattern_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            searchable_text = f"{data.get('pattern_name', '')} {data.get('description', '')} {data.get('function_name', '')}".lower()
            if query_lower in searchable_text:
                matches.append(data)
        except Exception as e:
            logger.warning(f"[find_in_cookbook] Could not read or parse pattern file {pattern_file}: {e}")
            continue
    if not matches:
        return {"status": "not_found", "message": f"No patterns found matching the query: '{request.query}'"}
    return {"status": "success", "results": matches}

@mcp.tool()
@tool_timeout_and_errors(timeout=30)
def cookbook_multitool(request: CookbookMultitoolRequest) -> dict:
    """
    Unified Code Cookbook multitool for adding and searching code patterns.

    Args:
        request (CookbookMultitoolRequest):
            - mode (str): 'add' to save a new pattern, 'find' to search for patterns.
            - pattern_name (str, required for 'add'): Unique name for the pattern.
            - file_path (str, required for 'add'): Absolute path to the file containing the function.
            - function_name (str, required for 'add'): Name of the function to save.
            - description (str, required for 'add'): Description of the pattern.
            - query (str, required for 'find'): Search query for finding patterns.

    Returns:
        dict: Status and results or error message.

    Usage:
        - To add: mode='add', pattern_name, file_path, function_name, description
        - To find: mode='find', query
    """
    logger.info(f"[cookbook_multitool] mode={request.mode}")
    if request.mode == "add":
        missing = [f for f in ["pattern_name", "file_path", "function_name", "description"] if getattr(request, f) is None]
        if missing:
            return {"status": "error", "message": f"Missing required fields for 'add': {', '.join(missing)}"}
        add_req = AddToCookbookRequest(
            pattern_name=request.pattern_name,
            file_path=request.file_path,
            function_name=request.function_name,
            description=request.description
        )
        return _add_to_cookbook_impl(add_req)
    elif request.mode == "find":
        if not request.query:
            return {"status": "error", "message": "Missing 'query' for 'find' mode."}
        find_req = FindInCookbookRequest(query=request.query)
        return _find_in_cookbook_impl(find_req)
    else:
        return {"status": "error", "message": f"Invalid mode: {request.mode}. Use 'add' or 'find'."}


# --- Tool: get_snippet ---
import ast

@mcp.tool()
@tool_timeout_and_errors(timeout=10)
def get_snippet(request: SnippetRequest) -> dict:
    """
    Extract a precise code snippet from a file by function, class, or line range.

    This tool allows you to programmatically extract:
      - The full source of a named function (including decorators and signature)
      - The full source of a named class (including decorators and signature)
      - An arbitrary range of lines from any text file

    Args:
        request (SnippetRequest):
            - file_path (str): Absolute path to the file to extract from. Must be within the project root.
            - mode (str): Extraction mode. One of:
                * 'function': Extract a named function by its name.
                * 'class': Extract a named class by its name.
                * 'lines': Extract a specific line range.
            - name (str, optional): Required for 'function' and 'class' modes. The name of the function or class to extract.
            - start_line (int, optional): Required for 'lines' mode. 1-indexed starting line.
            - end_line (int, optional): Required for 'lines' mode. 1-indexed ending line (inclusive).

    Returns:
        dict: Structured JSON response with status and result:
            - status (str): 'success', 'error', or 'not_found'
            - snippet (str, optional): The extracted code snippet (if found)
            - message (str): Human-readable status or error message

    Usage Examples:
        # Extract a function
        {
            "file_path": "/project/src/foo.py",
            "mode": "function",
            "name": "my_func"
        }
        # Extract lines 10-20
        {
            "file_path": "/project/src/foo.py",
            "mode": "lines",
            "start_line": 10,
            "end_line": 20
        }

    Notes:
        - Returns an error if the file is outside the project root or does not exist.
        - For 'function' and 'class', uses Python AST for precise extraction.
        - For 'lines', both start and end are inclusive and 1-indexed.
    """
    import pathlib
    logger = globals().get('logger', None)
    try:
        path = pathlib.Path(request.file_path)
        if not _is_safe_path(path):
            msg = f"Unsafe or out-of-project file path: {request.file_path}"
            if logger: logger.error(f"[get_snippet] {msg}")
            return {"status": "error", "message": msg}
        if not path.exists():
            msg = f"File does not exist: {request.file_path}"
            if logger: logger.error(f"[get_snippet] {msg}")
            return {"status": "error", "message": msg}
        source = path.read_text(encoding="utf-8")
        if request.mode in ("function", "class"):
            if not request.name:
                return {"status": "error", "message": f"'name' must be provided for mode '{request.mode}'"}
            try:
                tree = ast.parse(source, filename=str(path))
                for node in ast.walk(tree):
                    if request.mode == "function" and isinstance(node, ast.FunctionDef) and node.name == request.name:
                        snippet = ast.get_source_segment(source, node)
                        if snippet:
                            return {"status": "success", "snippet": snippet, "message": "Function extracted."}
                        else:
                            return {"status": "not_found", "message": "Function found but could not extract source."}
                    if request.mode == "class" and isinstance(node, ast.ClassDef) and node.name == request.name:
                        snippet = ast.get_source_segment(source, node)
                        if snippet:
                            return {"status": "success", "snippet": snippet, "message": "Class extracted."}
                        else:
                            return {"status": "not_found", "message": "Class found but could not extract source."}
                return {"status": "not_found", "message": f"No {request.mode} named '{request.name}' found."}
            except Exception as e:
                if logger: logger.error(f"[get_snippet] AST parse error: {e}")
                return {"status": "error", "message": f"AST parse error: {e}"}
        elif request.mode == "lines":
            if request.start_line is None or request.end_line is None:
                return {"status": "error", "message": "start_line and end_line must be provided for 'lines' mode."}
            lines = source.splitlines()
            # 1-indexed, inclusive
            if request.start_line < 1 or request.end_line > len(lines) or request.start_line > request.end_line:
                return {"status": "error", "message": "Invalid line range."}
            snippet = "\n".join(lines[request.start_line - 1:request.end_line])
            return {"status": "success", "snippet": snippet, "message": "Lines extracted."}
        else:
            return {"status": "error", "message": f"Invalid mode: {request.mode}."}
    except Exception as e:
        if logger: logger.error(f"[get_snippet] Unexpected error: {e}")
        return {"status": "error", "message": f"Unexpected error: {e}"}

def introspect(request: IntrospectRequest) -> dict:
    """
    Multi-modal code/project introspection multitool for fast, read-only analysis of code and config files.

    This tool provides several sub-tools (modes) for lightweight, high-speed codebase introspection:

    Modes:
        - 'config':
            * Reads project configuration files (pyproject.toml or requirements.txt).
            * Args: config_type ('pyproject' or 'requirements').
            * Returns: For 'pyproject', the full TOML text. For 'requirements', a list of package strings.
        - 'outline':
            * Returns a high-level structural map of a Python file: all top-level functions and classes (with their methods).
            * Args: file_path (str)
            * Returns: functions (list), classes (list of {name, methods})
        - 'stats':
            * Calculates basic file statistics: total lines, code lines, comment lines, file size in bytes.
            * Args: file_path (str)
            * Returns: total_lines (int), code_lines (int), comment_lines (int), file_size_bytes (int)
        - 'inspect':
            * Provides details about a single function or class in a file: name, arguments/methods, docstring.
            * Args: file_path (str), function_name (str, optional), class_name (str, optional)
            * Returns: type ('function' or 'class'), name, args/methods, docstring

    Args:
        request (IntrospectRequest):
            - mode (str): One of 'config', 'outline', 'stats', 'inspect'.
            - file_path (str, optional): Path to the file for inspection (required for all except config).
            - config_type (str, optional): 'pyproject' or 'requirements' for config mode.
            - function_name (str, optional): Name of function to inspect (for 'inspect' mode).
            - class_name (str, optional): Name of class to inspect (for 'inspect' mode).

    Returns:
        dict: Structured JSON response. Always includes:
            - status (str): 'success', 'error', or 'not_found'
            - message (str): Human-readable status or error message
        Mode-specific fields:
            - config: config_type, content (str) or packages (list)
            - outline: functions (list), classes (list of {name, methods})
            - stats: total_lines, code_lines, comment_lines, file_size_bytes
            - inspect: type, name, args/methods, docstring

    Usage Examples:
        # Get outline of a file
        {
            "mode": "outline",
            "file_path": "/project/src/foo.py"
        }
        # Inspect a function
        {
            "mode": "inspect",
            "file_path": "/project/src/foo.py",
            "function_name": "my_func"
        }
        # Get requirements
        {
            "mode": "config",
            "config_type": "requirements"
        }

    Notes:
        - All file paths are validated for project safety.
        - All operations are fast and read-only (no mutation, no heavy analysis).
        - Returns 'not_found' if the target file, function, or class does not exist.
    """
    import pathlib, ast, os, json
    logger = globals().get('logger', None)
    try:
        # --- config mode ---
        if request.mode == "config":
            if request.config_type not in ("pyproject", "requirements"):
                return {"status": "error", "message": "config_type must be 'pyproject' or 'requirements' for config mode."}
            root = next(iter(PROJECT_ROOTS.values()), None)
            if not root:
                return {"status": "error", "message": "No project root found."}
            if request.config_type == "pyproject":
                cfg_path = pathlib.Path(root) / "pyproject.toml"
                if not cfg_path.exists():
                    return {"status": "not_found", "message": "pyproject.toml not found."}
                content = cfg_path.read_text(encoding="utf-8")
                return {"status": "success", "config_type": "pyproject", "content": content}
            elif request.config_type == "requirements":
                req_path = pathlib.Path(root) / "requirements.txt"
                if not req_path.exists():
                    return {"status": "not_found", "message": "requirements.txt not found."}
                lines = req_path.read_text(encoding="utf-8").splitlines()
                pkgs = [l.strip() for l in lines if l.strip() and not l.strip().startswith("#")]
                return {"status": "success", "config_type": "requirements", "packages": pkgs}

        # --- outline, stats, inspect modes require file_path ---
        if not request.file_path:
            return {"status": "error", "message": "file_path must be provided for this mode."}
        path = pathlib.Path(request.file_path)
        if not _is_safe_path(path):
            msg = f"Unsafe or out-of-project file path: {request.file_path}"
            if logger: logger.error(f"[introspect] {msg}")
            return {"status": "error", "message": msg}
        if not path.exists():
            return {"status": "not_found", "message": f"File does not exist: {request.file_path}"}
        source = path.read_text(encoding="utf-8")

        # --- outline mode ---
        if request.mode == "outline":
            try:
                tree = ast.parse(source, filename=str(path))
                functions = []
                classes = []
                for node in tree.body:
                    if isinstance(node, ast.FunctionDef):
                        functions.append(node.name)
                    elif isinstance(node, ast.ClassDef):
                        cls = {"name": node.name, "methods": [n.name for n in node.body if isinstance(n, ast.FunctionDef)]}
                        classes.append(cls)
                return {"status": "success", "functions": functions, "classes": classes}
            except Exception as e:
                if logger: logger.error(f"[introspect] AST parse error: {e}")
                return {"status": "error", "message": f"AST parse error: {e}"}

        # --- stats mode ---
        elif request.mode == "stats":
            lines = source.splitlines()
            total_lines = len(lines)
            comment_lines = sum(1 for l in lines if l.strip().startswith("#"))
            code_lines = sum(1 for l in lines if l.strip() and not l.strip().startswith("#"))
            file_size_bytes = os.path.getsize(str(path))
            return {
                "status": "success",
                "total_lines": total_lines,
                "code_lines": code_lines,
                "comment_lines": comment_lines,
                "file_size_bytes": file_size_bytes
            }

        # --- inspect mode ---
        elif request.mode == "inspect":
            if not request.function_name and not request.class_name:
                return {"status": "error", "message": "Must provide function_name or class_name for inspect mode."}
            try:
                tree = ast.parse(source, filename=str(path))
                # Inspect function
                if request.function_name:
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef) and node.name == request.function_name:
                            args = [a.arg for a in node.args.args]
                            doc = ast.get_docstring(node)
                            return {
                                "status": "success",
                                "type": "function",
                                "name": node.name,
                                "args": args,
                                "docstring": doc
                            }
                # Inspect class
                if request.class_name:
                    for node in ast.walk(tree):
                        if isinstance(node, ast.ClassDef) and node.name == request.class_name:
                            doc = ast.get_docstring(node)
                            methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
                            return {
                                "status": "success",
                                "type": "class",
                                "name": node.name,
                                "methods": methods,
                                "docstring": doc
                            }
                return {"status": "not_found", "message": "Requested function/class not found."}
            except Exception as e:
                if logger: logger.error(f"[introspect] AST parse error: {e}")
                return {"status": "error", "message": f"AST parse error: {e}"}

        else:
            return {"status": "error", "message": f"Invalid mode: {request.mode}."}
    except Exception as e:
        if logger: logger.error(f"[introspect] Unexpected error: {e}")
        return {"status": "error", "message": f"Unexpected error: {e}"}

    import pathlib
    logger = globals().get('logger', None)
    try:
        path = pathlib.Path(request.file_path)
        if not _is_safe_path(path):
            msg = f"Unsafe or out-of-project file path: {request.file_path}"
            if logger: logger.error(f"[get_snippet] {msg}")
            return {"status": "error", "message": msg}
        if not path.exists():
            msg = f"File does not exist: {request.file_path}"
            if logger: logger.error(f"[get_snippet] {msg}")
            return {"status": "error", "message": msg}
        source = path.read_text(encoding="utf-8")
        if request.mode in ("function", "class"):
            if not request.name:
                return {"status": "error", "message": f"'name' must be provided for mode '{request.mode}'"}
            try:
                tree = ast.parse(source, filename=str(path))
                for node in ast.walk(tree):
                    if request.mode == "function" and isinstance(node, ast.FunctionDef) and node.name == request.name:
                        snippet = ast.get_source_segment(source, node)
                        if snippet:
                            return {"status": "success", "snippet": snippet, "message": "Function extracted."}
                        else:
                            return {"status": "not_found", "message": "Function found but could not extract source."}
                    if request.mode == "class" and isinstance(node, ast.ClassDef) and node.name == request.name:
                        snippet = ast.get_source_segment(source, node)
                        if snippet:
                            return {"status": "success", "snippet": snippet, "message": "Class extracted."}
                        else:
                            return {"status": "not_found", "message": "Class found but could not extract source."}
                return {"status": "not_found", "message": f"No {request.mode} named '{request.name}' found."}
            except Exception as e:
                if logger: logger.error(f"[get_snippet] AST parse error: {e}")
                return {"status": "error", "message": f"AST parse error: {e}"}
        elif request.mode == "lines":
            if request.start_line is None or request.end_line is None:
                return {"status": "error", "message": "start_line and end_line must be provided for 'lines' mode."}
            lines = source.splitlines()
            # 1-indexed, inclusive
            if request.start_line < 1 or request.end_line > len(lines) or request.start_line > request.end_line:
                return {"status": "error", "message": "Invalid line range."}
            snippet = "\n".join(lines[request.start_line - 1:request.end_line])
            return {"status": "success", "snippet": snippet, "message": "Lines extracted."}
        else:
            return {"status": "error", "message": f"Invalid mode: {request.mode}."}
    except Exception as e:
        if logger: logger.error(f"[get_snippet] Unexpected error: {e}")
        return {"status": "error", "message": f"Unexpected error: {e}"}