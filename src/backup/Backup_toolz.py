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

mcp = FastMCP(
    name="Project Search & Index",
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

@mcp.tool()
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

@mcp.tool()
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

@mcp.tool()
def index_project_files(project_name: str, subfolder: Optional[str] = None, max_file_size_mb: int = 5) -> Dict:
    """
    Index the project for semantic search. This tool scans source files, creates vector embeddings using sentence-transformers (with GPU acceleration if available), and builds a FAISS index for fast semantic/code search.

    Args:
        project_name (str): Name of the project as defined in PROJECT_ROOTS (e.g., "MCP-Server").
        subfolder (str, optional): If set, only index this subfolder within the project.
        max_file_size_mb (int, optional): Maximum file size (in MB) to include (default: 5).

    Returns:
        dict: {
            'status': 'success'|'error',
            'message': str,
            'files_scanned_and_included': int,
            'total_chunks_indexed': int,
            'indexing_duration_seconds': float
        }

    Usage:
        - Must be run before using 'semantic', 'similarity', or 'task_verification' search types.
        - Skips .venv, dependency folders, and binary files automatically.
        - Re-run if you add or change source files.
        - For large projects, index a subfolder for faster results.
    """
    if not LIBS_AVAILABLE:
        return {"status": "error", "message": "Indexing failed: Required libraries (faiss, numpy, sentence-transformers) are not installed."}

    # Configuration for intelligent file filtering
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

    start_time = time.monotonic()
    project_path = _get_project_path(project_name)
    if not project_path:
        return {"status": "error", "message": f"Project '{project_name}' not found."}

    scan_path = project_path
    if subfolder:
        scan_path = project_path / subfolder
        if not scan_path.is_dir():
            return {"status": "error", "message": f"Subfolder '{subfolder}' not found in project."}

    index_path = scan_path / INDEX_DIR_NAME
    index_path.mkdir(exist_ok=True)
    
    logger.info(f"Starting intelligent indexing for project '{project_name}'...")
    
    # 1. Collect relevant files
    relevant_files = [
        p for p in scan_path.rglob('*') if p.is_file() and
        not any(ignored in p.parts for ignored in directories_to_ignore) and
        p.suffix.lower() in text_extensions_to_include and
        p.stat().st_size <= max_file_size_bytes and
        ".windsurf_search_index" not in str(p) and not str(p).endswith(".json")
    ]

    if not relevant_files:
        return {"status": "error", "message": "No relevant text files found to index."}

    logger.info(f"Found {len(relevant_files)} relevant files to process.")

    # 2. Process files in batches: read, chunk, and embed
    all_vectors = []
    metadata = []
    CHUNK_SIZE = 512
    CHUNK_OVERLAP = 64
    
    for fp in relevant_files:
        try:
            text = fp.read_text("utf-8", errors="ignore")
            if not text.strip(): continue
            
            for i in range(0, len(text), CHUNK_SIZE - CHUNK_OVERLAP):
                chunk_content = text[i : i + CHUNK_SIZE]
                all_vectors.append(chunk_content) # Store text temporarily
                metadata.append({"path": str(fp.relative_to(project_path)), "content": chunk_content})
        except Exception as e:
            logger.warning(f"Could not read or chunk file {fp}: {e}")

    if not all_vectors:
        return {"status": "error", "message": "Could not extract any text content from the files."}

    # 3. Embed all chunks in batches for efficiency
    embedded_vectors = []
    PROCESSING_BATCH_SIZE = 128
    for i in range(0, len(all_vectors), PROCESSING_BATCH_SIZE):
        batch_texts = all_vectors[i : i + PROCESSING_BATCH_SIZE]
        embedded_vectors.extend(_embed_batch(batch_texts))
        logger.info(f"Processed batch: {len(batch_texts)} chunks. Total chunks so far: {len(embedded_vectors)}")

    # 4. Build and save the Faiss index and metadata
    try:
        dimension = len(embedded_vectors[0])
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(embedded_vectors, dtype=np.float32))

        faiss.write_index(index, str(index_path / "index.faiss"))
        with open(index_path / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f)
            
    except Exception as e:
        return {"status": "error", "message": f"Failed to build or save the index: {e}"}

    duration = time.monotonic() - start_time
    return {
        "status": "success",
        "message": f"Project '{project_name}' indexed successfully.",
        "files_scanned_and_included": len(relevant_files),
        "total_chunks_indexed": len(embedded_vectors),
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
@mcp.tool(name="search")
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
        logger.error("Please run: pip install faiss-cpu sentence-transformers torch numpy jedi")
    else:
        logger.info("Starting Project Search & Index MCP Server...")
        mcp.run()