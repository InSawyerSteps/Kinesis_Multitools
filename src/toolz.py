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
from typing import Any, Dict, List, Literal, Optional
import uvicorn

from fastmcp import FastMCP
from pydantic import BaseModel, Field

# --- Add this block with the other imports ---
import subprocess
import difflib
import base64
import platform

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

class AnchorMultitoolRequest(BaseModel):
    mode: str = Field(..., description="Operation mode: 'drop', 'list', 'remove', or 'rename'.")
    # For 'drop' mode
    path: Optional[str] = Field(None, description="Absolute path to the folder to register as a project root (required for 'drop').")
    project_name: Optional[str] = Field(None, description="Alias for the project (optional for 'drop', defaults to folder name).")
    # For 'remove' mode
    # project_name is reused
    # For 'rename' mode
    old_name: Optional[str] = Field(None, description="Existing alias to rename (required for 'rename').")
    new_name: Optional[str] = Field(None, description="New alias name (required for 'rename').")


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

# Debug toggle for search instrumentation
DEBUG_SEARCH = str(os.environ.get("MCP_DEBUG_SEARCH", "0")).lower() in ("1", "true", "yes", "y")
IS_WINDOWS = platform.system() == "Windows"

def _dlog(msg: str) -> None:
    """Emit debug logs only when MCP_DEBUG_SEARCH is enabled."""
    if DEBUG_SEARCH:
        logger.info(msg)


# ---------------------------------------------------------------------------
# Project root configuration
# ---------------------------------------------------------------------------
# Assumes this file is in 'src/' and the project root is its parent's parent.
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
PROJECT_ROOTS_CONFIG_FILE = PROJECT_ROOT / ".project_roots.json"
INDEX_DIR_NAME = ".windsurf_search_index"

def _load_project_roots() -> Dict[str, pathlib.Path]:
    """Load project roots from persistent storage, with fallback to defaults."""
    default_roots = {
        "MCP-Server": PROJECT_ROOT,
    }
    
    if not PROJECT_ROOTS_CONFIG_FILE.exists():
        # Don't log on first startup to avoid noise
        pass
        return default_roots
    
    try:
        with open(PROJECT_ROOTS_CONFIG_FILE, "r", encoding="utf-8") as f:
            config = json.load(f)
        
        # Convert string paths back to pathlib.Path objects and validate
        loaded_roots = {}
        for name, path_str in config.items():
            try:
                path_obj = pathlib.Path(path_str).resolve()
                if path_obj.exists() and path_obj.is_dir():
                    loaded_roots[name] = path_obj
                    logger.info(f"[project_roots] Loaded: {name} -> {path_obj}")
                else:
                    logger.warning(f"[project_roots] Skipping invalid path for '{name}': {path_str}")
            except Exception as e:
                logger.warning(f"[project_roots] Error loading path for '{name}': {e}")
        
        # Always include default MCP-Server root
        loaded_roots["MCP-Server"] = PROJECT_ROOT
        return loaded_roots
        
    except Exception as e:
        logger.error(f"[project_roots] Failed to load config file, using defaults: {e}")
        return default_roots

def _save_project_roots(roots: Dict[str, pathlib.Path]) -> bool:
    """Save project roots to persistent storage."""
    try:
        # Convert pathlib.Path objects to strings for JSON serialization
        config = {name: str(path) for name, path in roots.items()}
        
        # Create parent directory if it doesn't exist
        PROJECT_ROOTS_CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
        
        # Write with pretty formatting
        with open(PROJECT_ROOTS_CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, sort_keys=True)
        
        logger.info(f"[project_roots] Saved {len(config)} project roots to {PROJECT_ROOTS_CONFIG_FILE}")
        return True
        
    except Exception as e:
        logger.error(f"[project_roots] Failed to save config file: {e}")
        return False

# Load project roots at module initialization
PROJECT_ROOTS: Dict[str, pathlib.Path] = _load_project_roots()

# ---------------------------------------------------------------------------
# Core Helper Functions
# ---------------------------------------------------------------------------

def _normalize_search_includes(includes: List[str], project_path: pathlib.Path, extensions: Optional[List[str]] = None) -> List[pathlib.Path]:
    """
    Normalize search includes to handle directories, files, and glob patterns properly.
    
    Args:
        includes: List of include patterns/paths
        project_path: Base project path
        extensions: Optional file extensions to filter by
    
    Returns:
        List of file paths to search
    """
    start_ts = time.perf_counter()
    _dlog(f"[_normalize_search_includes] start includes={includes} extensions={extensions}")
    files = []
    # Normalize extensions to plain (no leading dot) lowercase for consistent comparison
    norm_exts = {e.lower().lstrip('.') for e in extensions} if extensions else None
    for inc in includes:
        # Handle absolute vs relative paths
        inc_path = project_path / inc if not os.path.isabs(inc) else pathlib.Path(inc)
        _dlog(f"[_normalize_search_includes] processing include='{inc}' as path='{inc_path}'")
        
        if inc_path.is_dir():
            # Directory: get all files within it
            _dlog(f"[_normalize_search_includes] dir -> scanning {inc_path}")
            files.extend(_iter_files(inc_path, extensions=extensions))
        elif inc_path.is_file():
            # File: add directly if it matches extension filter
            if not norm_exts or inc_path.suffix.lstrip('.').lower() in norm_exts:
                _dlog(f"[_normalize_search_includes] file accepted: {inc_path}")
                files.append(inc_path)
        else:
            # Try as glob pattern
            try:
                _dlog(f"[_normalize_search_includes] glob expand: base={project_path} pattern='{inc}'")
                glob_matches = list(project_path.glob(inc))
                for match in glob_matches:
                    if match.is_file():
                        if not norm_exts or match.suffix.lstrip('.').lower() in norm_exts:
                            files.append(match)
                    elif match.is_dir():
                        files.extend(_iter_files(match, extensions=extensions))
            except (OSError, ValueError) as e:
                # Invalid glob pattern, skip
                logger.warning(f"[_normalize_search_includes] glob error for pattern='{inc}': {e}")
                continue
    
    elapsed_ms = int((time.perf_counter() - start_ts) * 1000)
    _dlog(f"[_normalize_search_includes] end total_files={len(files)} elapsed_ms={elapsed_ms}")
    return files

import concurrent.futures
import functools
import traceback
import multiprocessing
import time

def tool_process_timeout_and_errors(timeout=60):
    """
    Decorator to run a function with a hard timeout and robust error handling.
    Uses multiprocessing on POSIX. On Windows, falls back to a thread-based runner
    to avoid spawn/pickling issues that can hang MCP tools.
    Returns MCP-protocol-compliant error dicts on timeout or exception.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            import platform
            if platform.system() == "Windows":
                # Fallback to thread-based timeout (same behavior as tool_timeout_and_errors)
                logger = logging.getLogger("mcp_search_tools")
                try:
                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                        future = executor.submit(func, *args, **kwargs)
                        try:
                            return future.result(timeout=timeout)
                        except concurrent.futures.TimeoutError:
                            logger.error(f"[TIMEOUT] Tool '{func.__name__}' timed out after {timeout} seconds (thread fallback).")
                            return {"status": "error", "message": f"Tool '{func.__name__}' timed out after {timeout} seconds."}
                except Exception as e:
                    tb = traceback.format_exc()
                    logger.error(f"[EXCEPTION] Tool '{func.__name__}' failed (thread fallback): {e}\n{tb}")
                    return {"status": "error", "message": f"Tool '{func.__name__}' failed: {e}", "traceback": tb}
            else:
                # Use separate process on POSIX
                def target_fn(q, *a, **k):
                    try:
                        result = func(*a, **k)
                        q.put(("success", result))
                    except Exception:
                        tb = traceback.format_exc()
                        q.put(("error", {"status": "error", "message": "Subprocess error", "traceback": tb}))
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
                    try:
                        p.terminate()
                    except Exception:
                        pass
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

def _iter_files(root: pathlib.Path, extensions: Optional[List[str]] = None, max_return: int = 1000):
    """Yields up to max_return files under root, skipping common dependency/VCS and binary files."""
    exclude_dirs = {".git", ".venv", "venv", "__pycache__", "node_modules", ".vscode", ".idea", "dist", "build"}
    binary_extensions = {
        ".zip", ".gz", ".tar", ".rar", ".7z", ".exe", ".dll", ".so", ".a",
        ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".pdf", ".doc",
        ".docx", ".ppt", ".pptx", ".xls", ".xlsx", ".pyc", ".pyo", ".db",
        ".sqlite", ".sqlite3", ".iso", ".img", ".mp3", ".mp4", ".avi",
        ".mkv", ".mov"
    }
    norm_exts = {f".{e.lower().lstrip('.')}" for e in extensions} if extensions else None

    from collections import deque
    dirs_to_scan = deque([root])
    files_processed = 0
    files_yielded = 0
    max_files = 20000  # Hard safety cap

    _dlog(f"[_iter_files] start root='{root}' extensions={extensions} max_return={max_return}")
    while dirs_to_scan and files_processed < max_files and files_yielded < max_return:
        current_dir = dirs_to_scan.popleft()
        try:
            if current_dir.name in exclude_dirs:
                continue
            for item in current_dir.iterdir():
                files_processed += 1
                if files_processed >= max_files:
                    logger.warning(f"[_iter_files] Processed over {max_files} items. Stopping traversal as a safety measure.")
                    _dlog(f"[_iter_files] safety-stop files_processed={files_processed} files_yielded={files_yielded}")
                    return
                if files_yielded >= max_return:
                    _dlog(f"[_iter_files] reached max_return files_yielded={files_yielded}")
                    return
                if item.is_dir():
                    if item.name not in exclude_dirs:
                        dirs_to_scan.append(item)
                elif item.is_file():
                    if item.suffix.lower() in binary_extensions:
                        continue
                    item_str = str(item).lower()
                    if ".windsurf_search_index" in item_str or item_str.endswith(".json"):
                        continue
                    if extensions and item.suffix.lower() not in norm_exts:
                        continue
                    yield item
                    files_yielded += 1
        except (PermissionError, OSError) as e:
            logger.warning(f"Skipping directory {current_dir} due to access error: {e}")
            continue
    _dlog(f"[_iter_files] end files_processed={files_processed} files_yielded={files_yielded}")

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
# Project Root Registration Tool
# ---------------------------------------------------------------------------

@mcp.tool()
@tool_timeout_and_errors(timeout=10)
def anchor_multitool(request: AnchorMultitoolRequest) -> dict:
    """
    Unified project root management multitool for dynamic project registration and management.
    
    Args:
        request (AnchorMultitoolRequest): Contains mode and mode-specific parameters.
    
    Returns:
        dict: Status, message, and results based on the selected mode.
    
    Modes:
        - 'drop': Register a new project root (path required, project_name optional)
        - 'list': List all registered project roots
        - 'remove': Remove a project root (project_name required)  
        - 'rename': Rename a project root alias (old_name and new_name required)
    """
    logger.info(f"[anchor_multitool] Mode: {request.mode}")
    
    try:
        if request.mode == 'drop':
            return _anchor_drop_impl(request)
        elif request.mode == 'list':
            return _anchor_list_impl()
        elif request.mode == 'remove':
            return _anchor_remove_impl(request)
        elif request.mode == 'rename':
            return _anchor_rename_impl(request)
        else:
            return {"status": "error", "message": f"Invalid mode: {request.mode}. Use 'drop', 'list', 'remove', or 'rename'."}
    except Exception as e:
        logger.error(f"[anchor_multitool] Failed: {e}", exc_info=True)
        return {"status": "error", "message": f"Operation failed: {e}"}

def _anchor_drop_impl(request: AnchorMultitoolRequest) -> dict:
    """Implementation for drop mode."""
    if not request.path:
        return {"status": "error", "message": "Missing 'path' for 'drop' mode"}
    
    logger.info(f"[anchor_drop] Registering project root: path={request.path}, project_name={request.project_name}")
    base_path = pathlib.Path(request.path).resolve()
    if not base_path.exists() or not base_path.is_dir():
        return {"status": "error", "message": f"Path does not exist or is not a directory: {request.path}"}
    
    alias = request.project_name if request.project_name else base_path.name
    
    # Check if already registered
    if alias in PROJECT_ROOTS and PROJECT_ROOTS[alias] == base_path:
        return {
            "status": "success", 
            "message": f"Project root '{alias}' is already registered at '{base_path}'.",
            "project_roots": {k: str(v) for k, v in PROJECT_ROOTS.items()}
        }
    
    # Add to in-memory registry
    PROJECT_ROOTS[alias] = base_path
    
    # Persist to disk
    if _save_project_roots(PROJECT_ROOTS):
        persistence_msg = " (persisted to disk)"
    else:
        persistence_msg = " (WARNING: failed to persist to disk)"
    
    logger.info(f"[anchor_drop] Registered project root: {alias} -> {base_path}{persistence_msg}")
    return {
        "status": "success",
        "message": f"Registered project root '{alias}' at '{base_path}'{persistence_msg}.",
        "project_roots": {k: str(v) for k, v in PROJECT_ROOTS.items()}
    }

def _anchor_list_impl() -> dict:
    """Implementation for list mode."""
    roots = {k: str(v) for k, v in PROJECT_ROOTS.items()}
    logger.info("[anchor_list] Returning %d project roots.", len(roots))
    return {"status": "success", "project_roots": roots, "count": len(roots)}

def _anchor_remove_impl(request: AnchorMultitoolRequest) -> dict:
    """Implementation for remove mode."""
    if not request.project_name:
        return {"status": "error", "message": "Missing 'project_name' for 'remove' mode"}
    
    logger.info("[anchor_remove] Removing project root alias='%s'", request.project_name)
    if request.project_name not in PROJECT_ROOTS:
        return {"status": "error", "message": f"Project alias '{request.project_name}' not found."}
    if request.project_name == "MCP-Server":
        return {"status": "error", "message": "Cannot remove the default 'MCP-Server' root."}

    removed_path = PROJECT_ROOTS.pop(request.project_name)
    persisted = _save_project_roots(PROJECT_ROOTS)
    logger.info("[anchor_remove] Removed '%s' -> %s (persisted=%s)", request.project_name, removed_path, persisted)
    return {
        "status": "success",
        "message": f"Removed project root '{request.project_name}'.",
        "project_roots": {k: str(v) for k, v in PROJECT_ROOTS.items()},
    }

def _anchor_rename_impl(request: AnchorMultitoolRequest) -> dict:
    """Implementation for rename mode."""
    if not request.old_name or not request.new_name:
        return {"status": "error", "message": "Both 'old_name' and 'new_name' must be provided for 'rename' mode."}
    
    logger.info("[anchor_rename] Renaming alias '%s' -> '%s'", request.old_name, request.new_name)
    if request.old_name not in PROJECT_ROOTS:
        return {"status": "error", "message": f"Project alias '{request.old_name}' not found."}
    if request.new_name in PROJECT_ROOTS:
        return {"status": "error", "message": f"Project alias '{request.new_name}' already exists."}
    if request.old_name == "MCP-Server":
        return {"status": "error", "message": "Cannot rename the default 'MCP-Server' root."}

    path = PROJECT_ROOTS.pop(request.old_name)
    PROJECT_ROOTS[request.new_name] = path
    persisted = _save_project_roots(PROJECT_ROOTS)
    logger.info("[anchor_rename] Renamed '%s' -> '%s' at %s (persisted=%s)", request.old_name, request.new_name, path, persisted)
    return {
        "status": "success",
        "message": f"Renamed project root '{request.old_name}' to '{request.new_name}'.",
        "project_roots": {k: str(v) for k, v in PROJECT_ROOTS.items()},
    }



# ---------------------------------------------------------------------------
# General-purpose Project Tools (migrated from toolz.py)

@mcp.tool()
@tool_timeout_and_errors(timeout=60)
def list_project_files(project_name: str, extensions: Optional[List[str]] = None, max_items: int = 1000) -> Dict[str, Any]:
    """
    List files under a registered project root (supports dynamic roots via anchor_drop),
    optionally filtering by extension.

    Args:
        project_name (str): Name of the registered project (see PROJECT_ROOTS / anchor_drop).
        extensions (List[str], optional): List of file extensions to include (e.g., ["py", "md"]). If omitted, all files are included.
        max_items (int, optional): Maximum number of files to return (default: 1000).

    Returns:
        dict: {
            'status': 'success'|'error',
            'files': List[str],        # Only present if status == 'success'
            'count': int,              # Number of files returned
            'project_root': str,       # Absolute path to root used
            'message': str             # Error message if status == 'error'
        }
    """
    results: List[str] = []
    try:
        project_path = _get_project_path(project_name)
        if project_path is None:
            available = ", ".join(sorted(PROJECT_ROOTS.keys()))
            msg = f"Unknown project '{project_name}'. Available projects: [{available}]"
            logger.error("[list_project_files] %s", msg)
            return {"status": "error", "message": msg}

        for fp in _iter_files(project_path, extensions, max_return=max_items):
            results.append(str(fp.resolve()))

        logger.info("[list_project_files] Found %d paths for project '%s' at root %s.", len(results), project_name, project_path)
        return {
            "status": "success",
            "files": results,
            "count": len(results),
            "project_root": str(project_path),
        }
    except Exception as e:
        logger.error("[list_project_files] Error listing files for project '%s': %s", project_name, e, exc_info=True)
        return {"status": "error", "message": str(e)}

@mcp.tool()
@tool_timeout_and_errors(timeout=60)
def read_project_file(
    absolute_file_path: str,
    max_bytes: int = 2_000_000,
    offset: int = 0,
    length: Optional[int] = None,
    binary_mode: Literal["auto", "text", "hex", "base64"] = "auto",
    encoding: str = "utf-8",
) -> Dict[str, Any]:
    """
    Read a file from disk with path safety checks.

    Args:
        absolute_file_path (str): Full absolute path to the file (must be within the project root).
        max_bytes (int, optional): Safety cap for maximum number of bytes to read in a single call (default: 2,000,000).
        offset (int, optional): Byte offset to start reading from (default: 0). Must be >= 0.
        length (int, optional): Maximum number of bytes to read starting at offset. When omitted, reads up to max_bytes.
        binary_mode (str, optional): How to return content bytes. One of:
            - 'auto' (default): Try text decode using 'encoding'; on failure, return short hex preview.
            - 'text': Force text decode using 'encoding'. On failure, returns error.
            - 'hex': Return hex string of the bytes (may be preview-limited).
            - 'base64': Return base64-encoded string of the bytes.
        encoding (str, optional): Text encoding to use when decoding in 'auto' or 'text' modes. Default 'utf-8'.

    Returns:
        dict: {
            'status': 'success'|'error',
            'file_path': str,
            'content': str or None,  # UTF-8 text, hex string, or base64 depending on mode
            'message': str,
            'total_size': int,
            'offset': int,
            'bytes_read': int,
            'eof': bool,
            'binary_mode_used': str,
            'encoding': str
        }

    Usage:
        Use this tool to safely read the contents of a file for display, editing, or analysis. Binary files can be returned
        as hex or base64, and large files can be paginated using 'offset' and 'length'. Files outside project roots are blocked.
    """
    path = pathlib.Path(absolute_file_path)
    if not _is_safe_path(path):
        return {
            "status": "error",
            "file_path": absolute_file_path,
            "content": None,
            "message": "Access denied: Path is outside configured project roots.",
        }
    if not path.is_file():
        return {
            "status": "error",
            "file_path": absolute_file_path,
            "content": None,
            "message": "Not a file.",
        }
    if offset < 0:
        return {
            "status": "error",
            "file_path": absolute_file_path,
            "content": None,
            "message": "Invalid offset: must be >= 0.",
        }
    if length is not None and length < 0:
        return {
            "status": "error",
            "file_path": absolute_file_path,
            "content": None,
            "message": "Invalid length: must be >= 0 when provided.",
        }
    if binary_mode not in ("auto", "text", "hex", "base64"):
        return {
            "status": "error",
            "file_path": absolute_file_path,
            "content": None,
            "message": "Invalid binary_mode. Use one of: auto, text, hex, base64.",
        }
    try:
        total_size = path.stat().st_size
        if offset >= total_size:
            # Nothing to read; we're at or beyond EOF
            return {
                "status": "success",
                "file_path": absolute_file_path,
                "content": "",
                "message": "Offset beyond EOF; returning empty content.",
                "total_size": total_size,
                "offset": offset,
                "bytes_read": 0,
                "eof": True,
                "binary_mode_used": binary_mode,
                "encoding": encoding,
            }

        # Determine how many bytes to read this call (respect both 'length' and 'max_bytes')
        max_window = max_bytes if length is None else min(length, max_bytes)
        end = min(total_size, offset + max_window)
        read_len = max(0, end - offset)

        with open(path, "rb") as f:
            f.seek(offset)
            data = f.read(read_len)

        eof = (end >= total_size)

        # Encode content depending on requested binary_mode
        content: Optional[str] = None
        message: str
        used_mode = binary_mode

        if binary_mode == "text":
            # Strict text decode; error on failure
            content = data.decode(encoding)
            message = f"Read {len(data)} bytes as text from offset {offset}."
        elif binary_mode == "hex":
            # Keep preview to avoid huge payloads in hex mode
            content = data.hex()
            if len(content) > 2000:
                content = content[:2000]
                message = f"Read {len(data)} bytes (hex preview truncated)."
            else:
                message = f"Read {len(data)} bytes (hex)."
        elif binary_mode == "base64":
            content = base64.b64encode(data).decode("ascii")
            message = f"Read {len(data)} bytes (base64)."
        else:  # auto
            try:
                content = data.decode(encoding)
                message = f"Read {len(data)} bytes as text from offset {offset}."
            except UnicodeDecodeError:
                used_mode = "hex"
                hex_preview = data.hex()
                # Keep same preview size behavior as legacy: 1000 chars
                if len(hex_preview) > 1000:
                    hex_preview = hex_preview[:1000]
                    message = f"Read {len(data)} bytes of binary data (showing hex preview)."
                else:
                    message = f"Read {len(data)} bytes of binary data (hex)."
                content = hex_preview

        return {
            "status": "success",
            "file_path": absolute_file_path,
            "content": content,
            "message": message,
            "total_size": total_size,
            "offset": offset,
            "bytes_read": len(data),
            "eof": eof,
            "binary_mode_used": used_mode,
            "encoding": encoding,
        }
    except Exception as e:
        logger.error("Failed to read file '%s': %s", absolute_file_path, e, exc_info=True)
        return {
            "status": "error",
            "file_path": absolute_file_path,
            "content": None,
            "message": str(e),
        }

# ---------------------------------------------------------------------------
# Tool 1: Indexing (Prerequisite for semantic searches)
# ---------------------------------------------------------------------------

@mcp.tool(name="index_project_files")
@tool_process_timeout_and_errors(timeout=300)
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
    t0 = time.perf_counter()
    results = []
    files_scanned = 0

    includes = params.get("includes")
    extensions = params.get("extensions")
    max_results = params.get("max_results", 1000)

    _dlog(f"[keyword] start q_len={len(query) if query else 0} includes={includes} extensions={extensions} max_results={max_results}")

    # Build file list
    if includes:
        files = _normalize_search_includes(includes, project_path, extensions=extensions)
    else:
        files = list(_iter_files(project_path, extensions=extensions))

    files = list(files)
    _dlog(f"[keyword] files_to_scan={len(files)} sample={[str(files[0]) if files else None]}")

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
                            elapsed_ms = int((time.perf_counter() - t0) * 1000)
                            _dlog(f"[keyword] early-return results={len(results)} files_scanned={files_scanned+1} took_ms={elapsed_ms}")
                            return {
                                "status": "success",
                                "results": results,
                                "files_scanned": files_scanned + 1
                            }
            files_scanned += 1
        except Exception as e:
            logger.debug(f"[keyword] skipping {fp} due to error: {e}")
            continue
    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    _dlog(f"[keyword] end results={len(results)} files_scanned={files_scanned} took_ms={elapsed_ms}")
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
        files = _normalize_search_includes(includes, project_path, extensions=extensions)
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
        files = _normalize_search_includes(includes, project_path, extensions=["py"])
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
            files = _normalize_search_includes(includes, project_path, extensions=["py"])
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

def _search_by_semantic(query: str, project_path: pathlib.Path, params: Dict) -> Dict:
    """Semantic search over the prebuilt FAISS index.

    Args:
        query: Natural language or code text to search for semantically.
        project_path: Absolute path to the project root.
        params: Optional dict with keys:
            - max_results (int): number of results to return (default 10)
            - includes (List[str]): optional include filters; if provided, only results
              whose absolute file path is in the normalized include set are returned.

    Returns:
        Dict with keys: status, results (list of {file_path, score, snippet}).
    """
    if not LIBS_AVAILABLE:
        return {"status": "error", "message": "Semantic search requires faiss, numpy, and sentence-transformers. Please install dependencies or disable this mode."}

    max_results = int(params.get("max_results", 10) or 10)
    includes = params.get("includes")

    index_dir = project_path / INDEX_DIR_NAME
    index_file = index_dir / "index.faiss"
    meta_file = index_dir / "metadata.json"
    if not index_file.exists() or not meta_file.exists():
        return {"status": "error", "message": "Semantic index not found. Run 'index_project_files' first."}

    try:
        with open(meta_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    except Exception as e:
        return {"status": "error", "message": f"Failed to load metadata.json: {e}"}

    if not metadata:
        return {"status": "error", "message": "metadata.json is empty; run indexing again."}

    try:
        index = faiss.read_index(str(index_file))
    except Exception as e:
        return {"status": "error", "message": f"Failed to read FAISS index: {e}"}

    # Build include set if provided (absolute paths)
    include_set: Optional[set[str]] = None
    if includes:
        try:
            inc_paths = _normalize_search_includes(includes, project_path)
            include_set = {str(p.resolve()) for p in inc_paths}
        except Exception:
            include_set = None

    # Embed the query and search
    try:
        vec = _embed_batch([query])[0]
        q = np.array([vec], dtype=np.float32)
        search_k = max_results * 3
        search_k = min(search_k if search_k > 0 else 10, len(metadata))
        D, I = index.search(q, search_k if search_k > 0 else 10)
    except Exception as e:
        return {"status": "error", "message": f"Semantic search failed during embedding or index search: {e}"}

    results = []
    for idx, dist in zip(I[0], D[0]):
        if idx < 0 or idx >= len(metadata):
            continue
        m = metadata[idx]
        abs_path = str((project_path / m.get("path", "")).resolve())
        if include_set is not None and abs_path not in include_set:
            continue
        content = (m.get("content") or "").strip()
        snippet = content[:200] + ("..." if len(content) > 200 else "")
        results.append({
            "file_path": abs_path,
            "score": float(dist),  # L2 distance (lower is better)
            "snippet": snippet,
        })
        if len(results) >= max_results:
            break

    if not results:
        return {"status": "not_found", "message": "No semantic matches found."}
    return {"status": "success", "results": results}

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
# Windows: enforce thread-based timeout with shorter limit to surface hangs sooner.
# POSIX: keep process-based isolation with a longer timeout.
_SEARCH_DECORATOR = tool_timeout_and_errors(timeout=60) if IS_WINDOWS else tool_process_timeout_and_errors(timeout=120)
@mcp.tool(name="search")
@_SEARCH_DECORATOR
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
    t0 = time.perf_counter()
    search_type = request.search_type
    project_name = request.project_name
    q_preview = (request.query or "")[:80].replace("\n", " ")
    logger.info(f"[search] type='{search_type}' project='{project_name}' q='{q_preview}...'")
    _dlog(f"[search] params={request.params}")

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
        _dlog(f"[search] route -> {search_func.__name__}")
        
        def _run():
            return search_func(request.query, project_path, request.params)

        # Extra inner watchdog on Windows to guarantee return even if outer decorator fails
        if IS_WINDOWS:
            try:
                import concurrent.futures as _fut
                with _fut.ThreadPoolExecutor(max_workers=1) as ex:
                    fut = ex.submit(_run)
                    result = fut.result(timeout=55)
            except _fut.TimeoutError:
                elapsed_ms = int((time.perf_counter() - t0) * 1000)
                logger.error(f"[search] inner-timeout type='{search_type}' after {elapsed_ms} ms")
                return {"status": "error", "message": f"Search '{search_type}' timed out after 55s (inner watchdog)."}
        else:
            result = _run()

        try:
            res_count = len(result.get("results", [])) if isinstance(result, dict) else None
        except Exception:
            res_count = None
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        _dlog(f"[search] done type='{search_type}' took_ms={elapsed_ms} results={res_count}")
        return result
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

class AnchorMultitoolRequest(BaseModel):
    mode: str = Field(..., description="Operation mode: 'drop', 'list', 'remove', or 'rename'.")
    # For 'drop' mode
    path: Optional[str] = Field(None, description="Absolute path to the folder to register as a project root (required for 'drop').")
    project_name: Optional[str] = Field(None, description="Alias for the project (optional for 'drop', defaults to folder name).")
    # For 'remove' mode
    # project_name is reused
    # For 'rename' mode
    old_name: Optional[str] = Field(None, description="Existing alias to rename (required for 'rename').")
    new_name: Optional[str] = Field(None, description="New alias name (required for 'rename').")

class CookbookMultitoolRequest(BaseModel):
    mode: str = Field(..., description="Operation mode: 'add', 'find', 'remove', or 'update'.")
    # For 'add' and 'update' modes
    pattern_name: Optional[str] = Field(None, description="Unique name for the pattern (required for 'add', 'remove', 'update').")
    file_path: Optional[str] = Field(None, description="Absolute path to the file containing the function (required for 'add', optional for 'update').")
    function_name: Optional[str] = Field(None, description="Name of the function to save as a pattern (required for 'add', optional for 'update').")
    description: Optional[str] = Field(None, description="Short description of the pattern (required for 'add', optional for 'update').")
    # For 'find' mode
    query: Optional[str] = Field(None, description="Search query for finding patterns (required for 'find').")
    # Multi-language support (new fields)
    language: Optional[str] = Field(None, description="Language hint: 'python', 'javascript', 'typescript', 'html', 'json', 'text', or 'auto' (default: 'auto').")
    code_snippet: Optional[str] = Field(None, description="Raw code or text snippet to store directly (alternative to file extraction).")
    start_marker: Optional[str] = Field(None, description="Start marker for text extraction between markers (works with end_marker).")
    end_marker: Optional[str] = Field(None, description="End marker for text extraction between markers (works with start_marker).")

class AddToCookbookRequest(BaseModel):
    pattern_name: str = Field(..., description="Unique name for the pattern.")
    file_path: str = Field(..., description="Absolute path to the file containing the function.")
    function_name: str = Field(..., description="Name of the function to save as a pattern.")
    description: str = Field(..., description="Short description of the pattern.")

class FindInCookbookRequest(BaseModel):
    query: str = Field(..., description="Search query for finding patterns.")

# Multi-language extraction helpers
def _detect_language_by_ext(file_path: str, explicit_language: Optional[str]) -> str:
    """Detect language by file extension or explicit hint."""
    if explicit_language and explicit_language.lower() != "auto":
        return explicit_language.lower()
    
    ext = pathlib.Path(file_path).suffix.lower() if file_path else ""
    language_map = {
        ".py": "python",
        ".js": "javascript",
        ".mjs": "javascript",
        ".cjs": "javascript",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".jsx": "javascript",
        ".html": "html",
        ".htm": "html",
        ".json": "json",
        ".md": "text",
        ".txt": "text",
    }
    return language_map.get(ext, "text")

def _extract_js_function(text: str, function_name: str) -> Optional[str]:
    """Extract JavaScript/TypeScript function using regex and brace balancing."""
    patterns = [
        re.compile(rf"(?:export\s+)?function\s+{re.escape(function_name)}\s*\(", re.MULTILINE),
        re.compile(rf"const\s+{re.escape(function_name)}\s*=\s*function\s*\(", re.MULTILINE),
        re.compile(rf"const\s+{re.escape(function_name)}\s*=\s*\([^)]*\)\s*=>\s*\{{", re.MULTILINE),
        re.compile(rf"function\s+{re.escape(function_name)}\s*\(", re.MULTILINE),
        re.compile(rf"let\s+{re.escape(function_name)}\s*=\s*function\s*\(", re.MULTILINE),
        re.compile(rf"var\s+{re.escape(function_name)}\s*=\s*function\s*\(", re.MULTILINE),
    ]
    
    for pattern in patterns:
        match = pattern.search(text)
        if not match:
            continue
            
        # Find the opening brace and extract balanced content
        start_pos = match.start()
        brace_pos = text.find("{", match.end() - 1)
        if brace_pos == -1:
            continue
            
        # Simple brace counting (not perfect but functional)
        depth = 1
        pos = brace_pos + 1
        in_string = False
        string_char = None
        escaped = False
        
        while pos < len(text) and depth > 0:
            char = text[pos]
            
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif in_string:
                if char == string_char:
                    in_string = False
            elif char in ('"', "'", "`"):
                in_string = True
                string_char = char
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                
            pos += 1
        
        if depth == 0:
            return text[start_pos:pos]
    
    return None

def _extract_from_html(text: str, function_name: Optional[str] = None, 
                      start_marker: Optional[str] = None, end_marker: Optional[str] = None) -> Optional[str]:
    """Extract content from HTML file using function name or markers."""
    if start_marker and end_marker:
        start_idx = text.find(start_marker)
        if start_idx == -1:
            return None
        end_idx = text.find(end_marker, start_idx + len(start_marker))
        if end_idx == -1:
            return None
        return text[start_idx:end_idx + len(end_marker)]
    
    if function_name:
        # Look for JavaScript functions within <script> blocks
        script_pattern = re.compile(r'<script[^>]*>(.*?)</script>', re.DOTALL | re.IGNORECASE)
        for match in script_pattern.finditer(text):
            script_content = match.group(1)
            extracted = _extract_js_function(script_content, function_name)
            if extracted:
                return extracted
    
    return None

def _extract_by_markers(text: str, start_marker: str, end_marker: str) -> Optional[str]:
    """Extract text content between start and end markers."""
    start_idx = text.find(start_marker)
    if start_idx == -1:
        return None
    end_idx = text.find(end_marker, start_idx + len(start_marker))
    if end_idx == -1:
        return None
    return text[start_idx:end_idx + len(end_marker)]

def _add_to_cookbook_impl(request) -> dict:
    """Multi-language cookbook implementation supporting both legacy AddToCookbookRequest and new CookbookMultitoolRequest."""
    # Handle both legacy and new request types
    if hasattr(request, 'mode'):  # CookbookMultitoolRequest
        pattern_name = request.pattern_name
        file_path = request.file_path
        function_name = request.function_name
        description = request.description
        language = getattr(request, 'language', None)
        code_snippet = getattr(request, 'code_snippet', None)
        start_marker = getattr(request, 'start_marker', None)
        end_marker = getattr(request, 'end_marker', None)
    else:  # Legacy AddToCookbookRequest
        pattern_name = request.pattern_name
        file_path = request.file_path
        function_name = request.function_name
        description = request.description
        language = None
        code_snippet = None
        start_marker = None
        end_marker = None
    
    logger.info(f"[add_to_cookbook] Adding pattern '{pattern_name}' (language: {language or 'auto'})")
    
    base_dir = pathlib.Path(r"C:\Projects\MCP Server")
    cookbook_dir = base_dir / COOKBOOK_DIR_NAME
    cookbook_dir.mkdir(exist_ok=True)
    safe_filename = re.sub(r'[^\w\-_\. ]', '_', pattern_name) + ".json"
    output_path = cookbook_dir / safe_filename
    
    if output_path.exists():
        return {"status": "error", "message": f"Pattern '{pattern_name}' already exists at {output_path}"}
    
    # Handle direct code snippet
    if code_snippet:
        detected_language = _detect_language_by_ext(file_path or "", language)
        pattern_data = {
            "pattern_name": pattern_name,
            "description": description,
            "language": detected_language,
            "locator": {"type": "snippet"},
            "extraction_strategy": "snippet",
            "source_code": code_snippet,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        if file_path:
            pattern_data["file_path"] = file_path
        
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(pattern_data, f, indent=4)
            logger.info(f"[add_to_cookbook] Successfully saved snippet pattern to {output_path}")
            return {"status": "success", "message": f"Pattern '{pattern_name}' was successfully added to the cookbook."}
        except Exception as e:
            logger.error(f"[add_to_cookbook] Failed to write pattern file {output_path}: {e}")
            return {"status": "error", "message": f"Failed to write pattern file: {e}"}
    
    # File-based extraction
    if not file_path:
        return {"status": "error", "message": "Either 'file_path' or 'code_snippet' must be provided."}
    
    source_file_path = pathlib.Path(file_path)
    if not _is_safe_path(source_file_path):
        return {"status": "error", "message": "Access denied: Source file path is outside configured project roots."}
    if not source_file_path.is_file():
        return {"status": "error", "message": f"Source file not found at: {file_path}"}
    
    try:
        source_code_text = source_file_path.read_text("utf-8")
    except Exception as e:
        return {"status": "error", "message": f"Failed to read source file: {e}"}
    
    detected_language = _detect_language_by_ext(file_path, language)
    extracted_code = None
    extraction_strategy = None
    locator = {}
    
    # Try marker-based extraction first
    if start_marker and end_marker:
        extracted_code = _extract_by_markers(source_code_text, start_marker, end_marker)
        if extracted_code:
            extraction_strategy = "markers"
            locator = {"type": "markers", "start_marker": start_marker, "end_marker": end_marker}
        else:
            return {"status": "error", "message": f"Could not find content between markers '{start_marker}' and '{end_marker}'."}
    
    # Try function extraction if no markers or markers failed
    elif function_name:
        if detected_language == "python":
            try:
                tree = ast.parse(source_code_text)
                class FunctionFinder(ast.NodeVisitor):
                    def __init__(self, target_name):
                        self.target_name = target_name
                        self.found_node = None
                    def visit_FunctionDef(self, node):
                        if node.name == self.target_name:
                            self.found_node = node
                        self.generic_visit(node)
                finder = FunctionFinder(function_name)
                finder.visit(tree)
                if finder.found_node:
                    extracted_code = ast.get_source_segment(source_code_text, finder.found_node)
                    extraction_strategy = "python_ast"
                    locator = {"type": "function_name", "function_name": function_name}
            except Exception as e:
                logger.warning(f"[add_to_cookbook] Python AST parsing failed: {e}")
        
        elif detected_language in ("javascript", "typescript"):
            extracted_code = _extract_js_function(source_code_text, function_name)
            if extracted_code:
                extraction_strategy = "js_regex"
                locator = {"type": "function_name", "function_name": function_name}
        
        elif detected_language == "html":
            extracted_code = _extract_from_html(source_code_text, function_name, start_marker, end_marker)
            if extracted_code:
                extraction_strategy = "js_regex" if function_name else "markers"
                locator = {"type": "function_name", "function_name": function_name} if function_name else {"type": "markers"}
    
    # Fallback: whole file or provide guidance
    if not extracted_code:
        if not function_name and not start_marker:
            # Store whole file for non-Python languages without specific extraction
            if detected_language in ("json", "text"):
                extracted_code = source_code_text
                extraction_strategy = "whole_file"
                locator = {"type": "whole_file"}
            else:
                return {"status": "error", 
                       "message": f"Could not extract content. For {detected_language} files, provide either 'function_name', 'start_marker'+'end_marker', or 'code_snippet'."}
        else:
            error_msg = f"Could not find function '{function_name}' in {detected_language} file." if function_name else "Extraction failed."
            if detected_language not in ("python", "javascript", "typescript", "html"):
                error_msg += f" Try using 'start_marker'+'end_marker' for {detected_language} files."
            return {"status": "error", "message": error_msg}
    
    # Create pattern data with richer metadata
    pattern_data = {
        "pattern_name": pattern_name,
        "description": description,
        "language": detected_language,
        "file_path": file_path,
        "locator": locator,
        "extraction_strategy": extraction_strategy,
        "source_code": extracted_code,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        # Legacy fields for backward compatibility
        "source_file": file_path,
        "function_name": function_name,
        "added_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(pattern_data, f, indent=4)
        logger.info(f"[add_to_cookbook] Successfully saved {detected_language} pattern to {output_path}")
        return {"status": "success", "message": f"Pattern '{pattern_name}' was successfully added to the cookbook."}
    except Exception as e:
        logger.error(f"[add_to_cookbook] Failed to write pattern file {output_path}: {e}")
        return {"status": "error", "message": f"Failed to write pattern file: {e}"}

def _find_in_cookbook_impl(request) -> dict:
    """Enhanced find implementation supporting language filtering."""
    # Handle both request types
    if hasattr(request, 'mode'):  # CookbookMultitoolRequest
        query = request.query
        language_filter = getattr(request, 'language', None)
    else:  # Legacy FindInCookbookRequest
        query = request.query
        language_filter = None
    
    logger.info(f"[find_in_cookbook] Searching for pattern with query: '{query}' (language: {language_filter or 'any'})")
    
    base_dir = pathlib.Path(r"C:\Projects\MCP Server")
    cookbook_dir = base_dir / COOKBOOK_DIR_NAME
    if not cookbook_dir.is_dir():
        return {"status": "not_found", "message": "Cookbook directory does not exist. Add a pattern first."}
    
    matches = []
    query_lower = query.lower()
    
    for pattern_file in cookbook_dir.glob("*.json"):
        try:
            with open(pattern_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Language filtering
            if language_filter and language_filter.lower() != "any":
                pattern_language = data.get('language', 'unknown').lower()
                if pattern_language != language_filter.lower():
                    continue
            
            # Enhanced searchable text including new metadata
            searchable_text = " ".join([
                data.get('pattern_name', ''),
                data.get('description', ''),
                data.get('function_name', ''),
                data.get('language', ''),
                data.get('extraction_strategy', '')
            ]).lower()
            
            if query_lower in searchable_text:
                matches.append(data)
                
        except Exception as e:
            logger.warning(f"[find_in_cookbook] Could not read or parse pattern file {pattern_file}: {e}")
            continue
    
    if not matches:
        lang_msg = f" (language: {language_filter})" if language_filter else ""
        return {"status": "not_found", "message": f"No patterns found matching the query: '{query}'{lang_msg}"}
    
    return {"status": "success", "results": matches}


def _remove_from_cookbook_impl(pattern_name: str) -> dict:
    """
    Removes a pattern from the cookbook by pattern_name.
    Returns success or error dict.
    """
    base_dir = pathlib.Path(r"C:\Projects\MCP Server")
    cookbook_dir = base_dir / COOKBOOK_DIR_NAME
    safe_filename = re.sub(r'[^\w\-_\. ]', '_', pattern_name) + ".json"
    pattern_path = cookbook_dir / safe_filename
    if not pattern_path.exists():
        return {"status": "error", "message": f"Pattern '{pattern_name}' does not exist."}
    try:
        pattern_path.unlink()
        return {"status": "success", "message": f"Pattern '{pattern_name}' was removed from the cookbook."}
    except Exception as e:
        return {"status": "error", "message": f"Failed to remove pattern: {e}"}


def _update_cookbook_pattern_impl(request: CookbookMultitoolRequest) -> dict:
    """
    Updates a pattern in the cookbook by pattern_name. Only provided fields are updated.
    Returns success or error dict.
    """
    base_dir = pathlib.Path(r"C:\Projects\MCP Server")
    cookbook_dir = base_dir / COOKBOOK_DIR_NAME
    safe_filename = re.sub(r'[^\w\-_\. ]', '_', request.pattern_name) + ".json"
    pattern_path = cookbook_dir / safe_filename
    if not pattern_path.exists():
        return {"status": "error", "message": f"Pattern '{request.pattern_name}' does not exist."}
    try:
        with open(pattern_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Only update fields that are provided
        if request.file_path:
            data["source_file"] = request.file_path
        if request.function_name:
            data["function_name"] = request.function_name
        if request.description:
            data["description"] = request.description
        with open(pattern_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
        return {"status": "success", "message": f"Pattern '{request.pattern_name}' was updated in the cookbook."}
    except Exception as e:
        return {"status": "error", "message": f"Failed to update pattern: {e}"}

@mcp.tool()
@tool_process_timeout_and_errors(timeout=30)
def cookbook_multitool(request: CookbookMultitoolRequest) -> dict:
    """
    Unified Code Cookbook multitool for adding, searching, removing, and updating code patterns with multi-language support.

    Args:
        request (CookbookMultitoolRequest):
            - mode (str): 'add' to save a new pattern, 'find' to search for patterns, 'remove' to remove a pattern, 'update' to update a pattern.
            - pattern_name (str, required for 'add', 'remove', 'update'): Unique name for the pattern.
            - file_path (str, optional): Absolute path to the file containing the code.
            - function_name (str, optional): Name of the function to extract.
            - description (str, required for 'add', optional for 'update'): Description of the pattern.
            - query (str, required for 'find'): Search query for finding patterns.
            - language (str, optional): Language hint ('python', 'javascript', 'typescript', 'html', 'json', 'text', or 'auto').
            - code_snippet (str, optional): Raw code or text snippet to store directly.
            - start_marker (str, optional): Start marker for text extraction between markers.
            - end_marker (str, optional): End marker for text extraction between markers.

    Returns:
        dict: Status and results or error message.

    Multi-Language Support:
        - Python: AST-based function extraction (existing behavior)
        - JavaScript/TypeScript: Regex-based function extraction with brace balancing
        - HTML: JavaScript function extraction from <script> blocks or marker-based extraction
        - JSON/Text: Marker-based extraction or whole-file storage
        - Any language: Direct code snippet storage or marker-based extraction

    Usage:
        - Add Python function: mode='add', pattern_name, file_path, function_name, description
        - Add JS function: mode='add', pattern_name, file_path, function_name, description, language='javascript'
        - Add by markers: mode='add', pattern_name, file_path, start_marker, end_marker, description
        - Add snippet: mode='add', pattern_name, code_snippet, description, language='json'
        - Find patterns: mode='find', query, language='javascript' (optional filter)
        - Remove: mode='remove', pattern_name
        - Update: mode='update', pattern_name, [fields to update]
    """
    logger.info(f"[cookbook_multitool] mode={request.mode}")
    if request.mode == "add":
        # Validate required fields with enhanced logic
        missing = []
        if not request.pattern_name:
            missing.append("pattern_name")
        if not request.description:
            missing.append("description")
        
        # Flexible validation: need either (file_path + function_name) OR (file_path + markers) OR code_snippet
        has_file_and_function = request.file_path and request.function_name
        has_file_and_markers = request.file_path and request.start_marker and request.end_marker
        has_snippet = request.code_snippet
        
        if not (has_file_and_function or has_file_and_markers or has_snippet):
            return {
                "status": "error", 
                "message": "For 'add' mode, provide either: (file_path + function_name) OR (file_path + start_marker + end_marker) OR code_snippet"
            }
        
        if missing:
            return {"status": "error", "message": f"Missing required fields for 'add': {', '.join(missing)}"}
        
        return _add_to_cookbook_impl(request)
    elif request.mode == "find":
        if not request.query:
            return {"status": "error", "message": "Missing 'query' for 'find' mode."}
        return _find_in_cookbook_impl(request)
    elif request.mode == "remove":
        if not request.pattern_name:
            return {"status": "error", "message": "Missing 'pattern_name' for 'remove' mode."}
        return _remove_from_cookbook_impl(request.pattern_name)
    elif request.mode == "update":
        if not request.pattern_name:
            return {"status": "error", "message": "Missing 'pattern_name' for 'update' mode."}
        return _update_cookbook_pattern_impl(request)
    else:
        return {"status": "error", "message": f"Invalid mode: {request.mode}. Use 'add', 'find', 'remove', or 'update'."}

# ---------------------------------------------------------------------------
# Code Snippet Extraction Tool
# ---------------------------------------------------------------------------

def _detect_language_from_content(content: str, file_path: str) -> str:
    """Enhanced language detection prioritizing file extension over content analysis."""
    # File extension based detection (PRIORITY)
    ext = pathlib.Path(file_path).suffix.lower()
    ext_mapping = {
        '.py': 'python', '.js': 'javascript', '.ts': 'typescript', '.tsx': 'typescript',
        '.jsx': 'javascript', '.html': 'html', '.htm': 'html', '.json': 'json',
        '.yaml': 'yaml', '.yml': 'yaml', '.md': 'markdown', '.txt': 'text'
    }
    
    lang_from_ext = ext_mapping.get(ext, None)
    
    # If we have a confident extension match, use it
    if lang_from_ext:
        return lang_from_ext
    
    # Only use content-based detection for ambiguous/missing extensions
    content_lower = content.lower().strip()
    if 'def ' in content_lower and ':' in content_lower and ('import ' in content_lower or 'from ' in content_lower):
        return 'python'
    elif 'function ' in content_lower and ('const ' in content_lower or 'let ' in content_lower or '=>' in content_lower):
        return 'javascript'
    elif 'interface ' in content_lower or 'type ' in content_lower or ': string' in content_lower:
        return 'typescript'
    elif '<' in content_lower and '>' in content_lower and ('html' in content_lower or 'body' in content_lower):
        return 'html'
    
    return lang_from_ext

def _extract_python_snippet(content: str, mode: str, name: Optional[str] = None, 
                           start_line: Optional[int] = None, end_line: Optional[int] = None,
                           context_lines: int = 0) -> Dict[str, Any]:
    """Extract Python code using AST with fallback to line-based extraction."""
    try:
        tree = ast.parse(content)
        lines = content.splitlines()
        
        if mode in ['function', 'class'] and name:
            target_type = ast.FunctionDef if mode == 'function' else ast.ClassDef
            
            for node in ast.walk(tree):
                if isinstance(node, target_type) and node.name == name:
                    # Include decorators if present
                    actual_start = node.lineno
                    if hasattr(node, 'decorator_list') and node.decorator_list:
                        actual_start = node.decorator_list[0].lineno
                    
                    actual_end = getattr(node, 'end_lineno', node.lineno)
                    
                    # Apply context lines
                    ctx_start = max(1, actual_start - context_lines)
                    ctx_end = min(len(lines), actual_end + context_lines)
                    
                    snippet_lines = lines[ctx_start-1:ctx_end]
                    return {
                        'snippet': '\n'.join(snippet_lines),
                        'start_line': ctx_start,
                        'end_line': ctx_end,
                        'actual_start': actual_start,
                        'actual_end': actual_end
                    }
            
            return {'error': f'{mode.capitalize()} "{name}" not found'}
        
        elif mode == 'lines' and start_line and end_line:
            ctx_start = max(1, start_line - context_lines)
            ctx_end = min(len(lines), end_line + context_lines)
            
            snippet_lines = lines[ctx_start-1:ctx_end]
            return {
                'snippet': '\n'.join(snippet_lines),
                'start_line': ctx_start,
                'end_line': ctx_end,
                'actual_start': start_line,
                'actual_end': end_line
            }
        
    except SyntaxError as e:
        logger.warning(f"AST parsing failed, falling back to line extraction: {e}")
        # Fallback to simple line-based extraction
        pass
    
    # Fallback: simple text search
    lines = content.splitlines()
    if mode in ['function', 'class'] and name:
        pattern = f"{'def' if mode == 'function' else 'class'} {name}"
        for i, line in enumerate(lines):
            if pattern in line:
                # Simple heuristic: find next function/class or end of file
                end_idx = len(lines)
                for j in range(i + 1, len(lines)):
                    if lines[j].strip() and not lines[j].startswith((' ', '\t')):
                        if 'def ' in lines[j] or 'class ' in lines[j]:
                            end_idx = j
                            break
                
                ctx_start = max(1, i + 1 - context_lines)
                ctx_end = min(len(lines), end_idx + context_lines)
                
                snippet_lines = lines[ctx_start-1:ctx_end]
                return {
                    'snippet': '\n'.join(snippet_lines),
                    'start_line': ctx_start,
                    'end_line': ctx_end,
                    'actual_start': i + 1,
                    'actual_end': end_idx
                }
    
    return {'error': f'Could not extract {mode} snippet'}

def _extract_js_snippet(content: str, mode: str, name: Optional[str] = None,
                       start_line: Optional[int] = None, end_line: Optional[int] = None,
                       context_lines: int = 0) -> Dict[str, Any]:
    """Extract JavaScript/TypeScript snippets with brace balancing."""
    lines = content.splitlines()
    
    if mode == 'function' and name:
        # Multiple function declaration patterns
        patterns = [
            rf'function\s+{re.escape(name)}\s*\(',
            rf'const\s+{re.escape(name)}\s*=\s*\(',
            rf'let\s+{re.escape(name)}\s*=\s*\(',
            rf'var\s+{re.escape(name)}\s*=\s*\(',
            rf'{re.escape(name)}\s*:\s*function\s*\(',
            rf'{re.escape(name)}\s*=\s*\([^)]*\)\s*=>',
            rf'const\s+{re.escape(name)}\s*=\s*[^=]*=>'
        ]
        
        for i, line in enumerate(lines):
            for pattern in patterns:
                if re.search(pattern, line):
                    # Find the end using brace balancing
                    brace_count = 0
                    start_found = False
                    end_idx = len(lines)
                    
                    for j in range(i, len(lines)):
                        current_line = lines[j]
                        for char in current_line:
                            if char == '{':
                                brace_count += 1
                                start_found = True
                            elif char == '}' and start_found:
                                brace_count -= 1
                                if brace_count == 0:
                                    end_idx = j + 1
                                    break
                        if brace_count == 0 and start_found:
                            break
                    
                    ctx_start = max(1, i + 1 - context_lines)
                    ctx_end = min(len(lines), end_idx + context_lines)
                    
                    snippet_lines = lines[ctx_start-1:ctx_end]
                    return {
                        'snippet': '\n'.join(snippet_lines),
                        'start_line': ctx_start,
                        'end_line': ctx_end,
                        'actual_start': i + 1,
                        'actual_end': end_idx
                    }
    
    elif mode == 'lines' and start_line and end_line:
        ctx_start = max(1, start_line - context_lines)
        ctx_end = min(len(lines), end_line + context_lines)
        
        snippet_lines = lines[ctx_start-1:ctx_end]
        return {
            'snippet': '\n'.join(snippet_lines),
            'start_line': ctx_start,
            'end_line': ctx_end,
            'actual_start': start_line,
            'actual_end': end_line
        }
    
    return {'error': f'Could not extract {mode} snippet'}

def _extract_html_snippet(content: str, mode: str, name: Optional[str] = None,
                         start_line: Optional[int] = None, end_line: Optional[int] = None,
                         context_lines: int = 0) -> Dict[str, Any]:
    """Extract HTML snippets, including JavaScript from script tags."""
    lines = content.splitlines()
    
    if mode == 'function' and name:
        # Look for JavaScript functions within script tags
        in_script = False
        script_content = []
        script_start_line = 0
        
        for i, line in enumerate(lines):
            if '<script' in line.lower():
                in_script = True
                script_start_line = i + 1
                script_content = []
            elif '</script>' in line.lower():
                if in_script and script_content:
                    # Try to extract function from collected script content
                    script_text = '\n'.join(script_content)
                    js_result = _extract_js_snippet(script_text, 'function', name, context_lines=0)
                    if 'snippet' in js_result:
                        # Adjust line numbers to account for HTML context
                        actual_start = script_start_line + js_result['actual_start'] - 1
                        actual_end = script_start_line + js_result['actual_end'] - 1
                        
                        ctx_start = max(1, actual_start - context_lines)
                        ctx_end = min(len(lines), actual_end + context_lines)
                        
                        snippet_lines = lines[ctx_start-1:ctx_end]
                        return {
                            'snippet': '\n'.join(snippet_lines),
                            'start_line': ctx_start,
                            'end_line': ctx_end,
                            'actual_start': actual_start,
                            'actual_end': actual_end
                        }
                in_script = False
            elif in_script:
                script_content.append(line)
    
    elif mode == 'lines' and start_line and end_line:
        ctx_start = max(1, start_line - context_lines)
        ctx_end = min(len(lines), end_line + context_lines)
        
        snippet_lines = lines[ctx_start-1:ctx_end]
        return {
            'snippet': '\n'.join(snippet_lines),
            'start_line': ctx_start,
            'end_line': ctx_end,
            'actual_start': start_line,
            'actual_end': end_line
        }
    
    return {'error': f'Could not extract {mode} snippet'}# This will be appended to toolz.py to add get_snippet tool

@mcp.tool()
@tool_timeout_and_errors(timeout=30)
def get_snippet(request: SnippetRequest) -> Dict[str, Any]:
    """
    Extract code snippets from files with multi-language support and intelligent parsing.
    
    Args:
        request (SnippetRequest): Contains file_path, mode ('function'|'class'|'lines'), 
                                 name (for function/class), start_line/end_line (for lines).
    
    Returns:
        dict: Status, snippet content, line numbers, detected language, and message.
    """
    logger.info(f"[get_snippet] Extracting {request.mode} from {request.file_path}")
    
    # Validate file path safety
    file_path = pathlib.Path(request.file_path)
    if not _is_safe_path(file_path):
        return {'status': 'error', 'message': 'Access denied: Path is outside configured project roots.'}
    
    if not file_path.exists():
        return {'status': 'error', 'message': f'File not found: {request.file_path}'}
    
    # Validate mode-specific requirements
    if request.mode in ['function', 'class'] and not request.name:
        return {'status': 'error', 'message': f'Missing name for {request.mode} mode'}
    
    if request.mode == 'lines' and (not request.start_line or not request.end_line):
        return {'status': 'error', 'message': 'Missing start_line or end_line for lines mode'}
    
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        if not content.strip():
            return {'status': 'error', 'message': 'File is empty'}
        
        # Detect language and extract
        language = _detect_language_from_content(content, str(file_path))
        
        if language == 'python':
            result = _extract_python_snippet(content, request.mode, request.name, 
                                           request.start_line, request.end_line, 0)
        elif language in ['javascript', 'typescript']:
            result = _extract_js_snippet(content, request.mode, request.name,
                                       request.start_line, request.end_line, 0)
        elif language == 'html':
            result = _extract_html_snippet(content, request.mode, request.name,
                                         request.start_line, request.end_line, 0)
        else:
            # Generic line-based extraction
            lines = content.splitlines()
            if request.mode == 'lines' and request.start_line and request.end_line:
                snippet_lines = lines[request.start_line-1:request.end_line]
                result = {'snippet': '\n'.join(snippet_lines), 'start_line': request.start_line, 'end_line': request.end_line}
            else:
                result = {'error': f'Language {language} not supported for {request.mode} mode'}
        
        if 'error' in result:
            return {'status': 'error', 'message': result['error']}
        
        return {
            'status': 'success',
            'snippet': result['snippet'],
            'start_line': result['start_line'],
            'end_line': result['end_line'],
            'actual_start': result.get('actual_start', result['start_line']),
            'actual_end': result.get('actual_end', result['end_line']),
            'language': language,
            'message': f'Successfully extracted {request.mode} snippet'
        }
        
    except Exception as e:
        logger.error(f"[get_snippet] Failed to extract snippet: {e}", exc_info=True)
        return {'status': 'error', 'message': f'Failed to extract snippet: {e}'}
# Introspect tool implementation to be appended to toolz.py

@mcp.tool()
@tool_timeout_and_errors(timeout=30)
def introspect(request: IntrospectRequest) -> Dict[str, Any]:
    """
    Multi-modal code/project introspection tool for analysis of code and config files.
    
    Args:
        request (IntrospectRequest): Contains mode ('config'|'outline'|'stats'|'inspect'),
                                    file_path, and mode-specific parameters.
    
    Returns:
        dict: Status, analysis results, and message based on the selected mode.
    """
    logger.info(f"[introspect] Mode: {request.mode}, File: {request.file_path}")
    
    try:
        if request.mode == 'config':
            return _introspect_config(request)
        elif request.mode == 'outline':
            return _introspect_outline(request)  
        elif request.mode == 'stats':
            return _introspect_stats(request)
        elif request.mode == 'inspect':
            return _introspect_inspect(request)
        else:
            return {'status': 'error', 'message': f'Invalid introspect mode: {request.mode}'}
            
    except Exception as e:
        logger.error(f"[introspect] Failed: {e}", exc_info=True)
        return {'status': 'error', 'message': f'Introspection failed: {e}'}

def _introspect_config(request: IntrospectRequest) -> Dict[str, Any]:
    """Analyze configuration files (pyproject.toml, requirements.txt, package.json, etc.)."""
    if not request.file_path:
        return {'status': 'error', 'message': 'file_path required for config mode'}
    
    file_path = pathlib.Path(request.file_path)
    if not _is_safe_path(file_path):
        return {'status': 'error', 'message': 'Access denied: Path is outside configured project roots.'}
    
    if not file_path.exists():
        return {'status': 'error', 'message': f'File not found: {request.file_path}'}
    
    try:
        content = file_path.read_text(encoding='utf-8', errors='ignore')
        config_analysis = {}
        
        filename = file_path.name.lower()
        
        if filename == 'requirements.txt':
            lines = [line.strip() for line in content.splitlines() if line.strip() and not line.strip().startswith('#')]
            packages = []
            for line in lines:
                # Parse package specs like "requests>=2.25.1"
                if '==' in line:
                    name, version = line.split('==', 1)
                    packages.append({'name': name.strip(), 'version': version.strip(), 'operator': '=='})
                elif '>=' in line:
                    name, version = line.split('>=', 1)  
                    packages.append({'name': name.strip(), 'version': version.strip(), 'operator': '>='})
                else:
                    packages.append({'name': line.strip(), 'version': None, 'operator': None})
            
            config_analysis = {
                'type': 'requirements.txt',
                'packages': packages,
                'package_count': len(packages)
            }
            
        elif filename == 'package.json':
            import json
            try:
                data = json.loads(content)
                config_analysis = {
                    'type': 'package.json',
                    'name': data.get('name', 'unknown'),
                    'version': data.get('version', 'unknown'),
                    'dependencies': list(data.get('dependencies', {}).keys()),
                    'devDependencies': list(data.get('devDependencies', {}).keys()),
                    'scripts': list(data.get('scripts', {}).keys()),
                    'dependency_count': len(data.get('dependencies', {})),
                    'devDependency_count': len(data.get('devDependencies', {}))
                }
            except json.JSONDecodeError as e:
                config_analysis = {'type': 'package.json', 'error': f'Invalid JSON: {e}'}
                
        elif filename in ['pyproject.toml', 'setup.cfg']:
            config_analysis = {
                'type': filename,
                'content_preview': content[:500] + ('...' if len(content) > 500 else ''),
                'lines': len(content.splitlines()),
                'note': 'Full TOML parsing not implemented - showing preview'
            }
            
        else:
            config_analysis = {
                'type': 'unknown_config',
                'filename': filename,
                'content_preview': content[:300] + ('...' if len(content) > 300 else ''),
                'lines': len(content.splitlines())
            }
        
        return {'status': 'success', 'analysis': config_analysis, 'message': f'Analyzed {filename}'}
        
    except Exception as e:
        return {'status': 'error', 'message': f'Config analysis failed: {e}'}

def _introspect_outline(request: IntrospectRequest) -> Dict[str, Any]:
    """Generate outline of functions and classes in a file.""" 
    if not request.file_path:
        return {'status': 'error', 'message': 'file_path required for outline mode'}
    
    file_path = pathlib.Path(request.file_path)
    if not _is_safe_path(file_path):
        return {'status': 'error', 'message': 'Access denied: Path is outside configured project roots.'}
    
    if not file_path.exists():
        return {'status': 'error', 'message': f'File not found: {request.file_path}'}
    
    try:
        content = file_path.read_text(encoding='utf-8', errors='ignore')
        language = _detect_language_from_content(content, str(file_path))
        outline = []
        
        if language == 'python':
            try:
                tree = ast.parse(content)
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        outline.append({
                            'type': 'function',
                            'name': node.name,
                            'line': node.lineno,
                            'end_line': getattr(node, 'end_lineno', node.lineno),
                            'args': [arg.arg for arg in node.args.args] if node.args else []
                        })
                    elif isinstance(node, ast.ClassDef):
                        outline.append({
                            'type': 'class', 
                            'name': node.name,
                            'line': node.lineno,
                            'end_line': getattr(node, 'end_lineno', node.lineno),
                            'methods': [n.name for n in ast.walk(node) if isinstance(n, ast.FunctionDef)]
                        })
            except SyntaxError:
                # Fallback to regex for Python
                lines = content.splitlines()
                for i, line in enumerate(lines, 1):
                    if re.match(r'^\s*def\s+(\w+)', line):
                        match = re.match(r'^\s*def\s+(\w+)', line)
                        outline.append({'type': 'function', 'name': match.group(1), 'line': i})
                    elif re.match(r'^\s*class\s+(\w+)', line):
                        match = re.match(r'^\s*class\s+(\w+)', line)
                        outline.append({'type': 'class', 'name': match.group(1), 'line': i})
        
        elif language in ['javascript', 'typescript']:
            lines = content.splitlines()
            for i, line in enumerate(lines, 1):
                # Function patterns
                if re.search(r'function\s+(\w+)', line):
                    match = re.search(r'function\s+(\w+)', line)
                    outline.append({'type': 'function', 'name': match.group(1), 'line': i})
                elif re.search(r'const\s+(\w+)\s*=\s*\(.*\)\s*=>', line):
                    match = re.search(r'const\s+(\w+)\s*=', line)
                    outline.append({'type': 'function', 'name': match.group(1), 'line': i})
                elif re.search(r'(\w+)\s*:\s*function', line):
                    match = re.search(r'(\w+)\s*:\s*function', line)
                    outline.append({'type': 'method', 'name': match.group(1), 'line': i})
                # Class patterns
                elif re.search(r'class\s+(\w+)', line):
                    match = re.search(r'class\s+(\w+)', line)
                    outline.append({'type': 'class', 'name': match.group(1), 'line': i})
        
        return {
            'status': 'success', 
            'outline': outline, 
            'language': language,
            'total_items': len(outline),
            'message': f'Generated outline with {len(outline)} items'
        }
        
    except Exception as e:
        return {'status': 'error', 'message': f'Outline generation failed: {e}'}

def _introspect_stats(request: IntrospectRequest) -> Dict[str, Any]:
    """Generate code statistics for a file."""
    if not request.file_path:
        return {'status': 'error', 'message': 'file_path required for stats mode'}
    
    file_path = pathlib.Path(request.file_path)  
    if not _is_safe_path(file_path):
        return {'status': 'error', 'message': 'Access denied: Path is outside configured project roots.'}
    
    if not file_path.exists():
        return {'status': 'error', 'message': f'File not found: {request.file_path}'}
    
    try:
        content = file_path.read_text(encoding='utf-8', errors='ignore')
        language = _detect_language_from_content(content, str(file_path))
        
        lines = content.splitlines()
        total_lines = len(lines)
        blank_lines = sum(1 for line in lines if not line.strip())
        comment_lines = 0
        
        # Basic comment detection
        if language == 'python':
            comment_lines = sum(1 for line in lines if line.strip().startswith('#'))
        elif language in ['javascript', 'typescript']:
            comment_lines = sum(1 for line in lines if line.strip().startswith('//'))
        
        code_lines = total_lines - blank_lines - comment_lines
        
        # Function/class counts
        function_count = 0
        class_count = 0
        
        if language == 'python':
            function_count = len(re.findall(r'^\s*def\s+\w+', content, re.MULTILINE))
            class_count = len(re.findall(r'^\s*class\s+\w+', content, re.MULTILINE))
        elif language in ['javascript', 'typescript']:
            function_count = len(re.findall(r'function\s+\w+|const\s+\w+\s*=.*=>', content))
            class_count = len(re.findall(r'class\s+\w+', content))
        
        stats = {
            'file_size_bytes': file_path.stat().st_size,
            'total_lines': total_lines,
            'code_lines': code_lines,
            'blank_lines': blank_lines,
            'comment_lines': comment_lines,
            'function_count': function_count,
            'class_count': class_count,
            'language': language
        }
        
        return {'status': 'success', 'stats': stats, 'message': 'Generated code statistics'}
        
    except Exception as e:
        return {'status': 'error', 'message': f'Stats generation failed: {e}'}

def _introspect_inspect(request: IntrospectRequest) -> Dict[str, Any]:
    """Inspect a specific function or class in detail."""
    if not request.file_path:
        return {'status': 'error', 'message': 'file_path required for inspect mode'}
    
    if not (request.function_name or request.class_name):
        return {'status': 'error', 'message': 'function_name or class_name required for inspect mode'}
    
    file_path = pathlib.Path(request.file_path)
    if not _is_safe_path(file_path):
        return {'status': 'error', 'message': 'Access denied: Path is outside configured project roots.'}
    
    if not file_path.exists():
        return {'status': 'error', 'message': f'File not found: {request.file_path}'}
    
    try:
        content = file_path.read_text(encoding='utf-8', errors='ignore')
        language = _detect_language_from_content(content, str(file_path))
        target_name = request.function_name or request.class_name
        target_type = 'function' if request.function_name else 'class'
        
        inspection = {'name': target_name, 'type': target_type, 'language': language}
        
        if language == 'python':
            try:
                tree = ast.parse(content)
                target_ast_type = ast.FunctionDef if target_type == 'function' else ast.ClassDef
                
                for node in ast.walk(tree):
                    if isinstance(node, target_ast_type) and node.name == target_name:
                        inspection.update({
                            'line': node.lineno,
                            'end_line': getattr(node, 'end_lineno', node.lineno),
                            'docstring': ast.get_docstring(node) if hasattr(ast, 'get_docstring') else None
                        })
                        
                        if target_type == 'function' and hasattr(node, 'args'):
                            inspection['parameters'] = [arg.arg for arg in node.args.args]
                        elif target_type == 'class':
                            methods = [n.name for n in ast.walk(node) if isinstance(n, ast.FunctionDef)]
                            inspection['methods'] = methods
                        break
                        
            except SyntaxError:
                pass  # Will use regex fallback
        
        # Regex fallback for all languages
        if 'line' not in inspection:
            lines = content.splitlines()
            if target_type == 'function':
                pattern = rf"^\s*(?:function\s+{re.escape(target_name)}|def\s+{re.escape(target_name)}|const\s+{re.escape(target_name)}\s*=)"
            else:
                pattern = rf"^\s*class\s+{re.escape(target_name)}"
            
            for i, line in enumerate(lines, 1):
                if re.search(pattern, line):
                    inspection['line'] = i
                    inspection['signature'] = line.strip()
                    
                    # Look for docstring/comment in next few lines
                    for j in range(i, min(i + 5, len(lines))):
                        if lines[j].strip().startswith('"""') or lines[j].strip().startswith("'''"):
                            inspection['docstring'] = lines[j].strip()
                            break
                    break
        
        if 'line' in inspection:
            return {'status': 'success', 'inspection': inspection, 'message': f'Inspected {target_type} "{target_name}"'}
        else:
            return {'status': 'error', 'message': f'{target_type.capitalize()} "{target_name}" not found'}
            
    except Exception as e:
        return {'status': 'error', 'message': f'Inspection failed: {e}'}
