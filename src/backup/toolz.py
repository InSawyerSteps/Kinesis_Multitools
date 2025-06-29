"""
General-purpose MCP tools for reading and searching project files.

This module provides four primary tools for interacting with project files:
  - list_files: Lists files within a project, with optional extension filters.
  - read_file: Safely reads the content of a single file.
  - keyword_search: Performs a literal substring search across files.
  - semantic_search: Performs similarity search using embeddings, with robust fallbacks.

The tools are designed to be general-purpose and operate on any file within the configured project roots.
"""
import base64
import json
import logging
import math
import os
import pathlib
import re
import time
from typing import Any, Callable, Dict, List, Optional

from fastmcp import FastMCP

# Attempt to import optional dependencies for different backends.
# These will be checked at runtime within the tools.
try:
    import requests
except ImportError:
    requests = None

try:
    from sentence_transformers import SentenceTransformer
    import torch
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    import numpy as np
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


# Cached Hugging Face embedding model (set at runtime)
_HF_MODEL: Optional[Any] = None

# ---------------------------------------------------------------------------
# FastMCP initialisation & logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("mcp_project_tools")

mcp = FastMCP(
    name="General Project Tools",
    title="General Project File Utilities",
    description="Read & search any project file using keyword or semantic similarity.",
    version="0.2.0", # Version bump for the rewrite
)

# ---------------------------------------------------------------------------
# Project root configuration
# ---------------------------------------------------------------------------
THIS_FILE = pathlib.Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parent.parent  # Assumes this file is in 'src/'
PROJECT_ROOTS: Dict[str, pathlib.Path] = {
    "MCP-Server": PROJECT_ROOT,
}
ALLOWED_PATHS: List[pathlib.Path] = list(PROJECT_ROOTS.values())

# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def _is_safe_path(path: pathlib.Path) -> bool:
    """Ensure *path* is inside one of the ALLOWED_PATHS roots."""
    try:
        resolved_path = path.resolve()
        for root in ALLOWED_PATHS:
            if resolved_path.is_relative_to(root.resolve()):
                return True
    except (OSError, ValueError): # Catches resolution errors or invalid paths
        return False
    return False

def _iter_files(root: pathlib.Path, extensions: Optional[List[str]] = None):
    """
    Yield all files under *root*, skipping common dependency/VCS directories.
    If *extensions* are provided, only files with matching extensions are yielded.
    """
    exclude_dirs = {".git", ".venv", "venv", "__pycache__", "node_modules", ".vscode", ".idea", "dist", "build"}
    
    # Normalize extensions to be like ".py"
    norm_exts = {f".{e.lower().lstrip('.')}" for e in extensions} if extensions else None

    for dirpath, dirnames, filenames in os.walk(root):
        # Modify dirnames in-place to prevent descending into excluded folders
        dirnames[:] = [d for d in dirnames if d not in exclude_dirs]
        
        for filename in filenames:
            fp = pathlib.Path(dirpath) / filename
            if norm_exts and fp.suffix.lower() not in norm_exts:
                continue
            yield fp

def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Helper to calculate cosine similarity between two vectors."""
    dot_product = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    # Avoid division by zero
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)

# ---------------------------------------------------------------------------
# Tool 1: List Files
# ---------------------------------------------------------------------------
@mcp.tool(name="list_files")
def list_project_files(
    project_name: str,
    extensions: Optional[List[str]] = None,
    max_items: int = 1000,
) -> List[str]:
    """
    Recursively list files for a given project.

    Args:
        project_name: The key for the project in PROJECT_ROOTS.
        extensions: Optional list of file extensions to include (e.g., ["py", "md"]).
        max_items: Safety cap on the number of paths returned.

    Returns:
        A list of absolute file paths as strings.
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

# ---------------------------------------------------------------------------
# Tool 2: Read File
# ---------------------------------------------------------------------------
@mcp.tool(name="read_file")
def read_project_file(
    absolute_file_path: str,
    max_bytes: int = 2_000_000,
) -> Dict[str, Any]:
    """
    Read a file from disk with path safety checks.

    Args:
        absolute_file_path: The full, absolute path to the file.
        max_bytes: A safety limit on the number of bytes to read.

    Returns:
        A dictionary with status, path, content, and a message.
        Content is text if decodable as UTF-8, otherwise a hex preview.
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
            content = data.hex()[:1000] # Return a hex preview for binary files
            message = f"Read {len(data)} bytes of binary data (showing hex preview)."
        
        return {"status": "success", "file_path": absolute_file_path, "content": content, "message": message}
    except Exception as e:
        logger.error("Failed to read file '%s': %s", absolute_file_path, e, exc_info=True)
        return {"status": "error", "file_path": absolute_file_path, "content": None, "message": str(e)}

# ---------------------------------------------------------------------------
# Tool 3: Keyword Search
# ---------------------------------------------------------------------------
@mcp.tool(name="keyword_search")
def keyword_search_in_files(
    query: str,
    project_name: Optional[str] = None,
    max_results: int = 50,
    extensions: Optional[List[str]] = None,
    case_insensitive: bool = True,
) -> Dict[str, Any]:
    """
    Scan files for a literal substring (keyword).

    Args:
        query: The text to search for.
        project_name: Restrict search to one project. If None, searches all projects.
        max_results: The maximum number of matching lines to return.
        extensions: Optional list of file extensions to search within.
        case_insensitive: If True, the search ignores case.

    Returns:
        A dictionary containing the results and the total number of files scanned.
    """
    logger.info("[keyword_search] q='%s' proj=%s ext=%s", query, project_name, extensions)
    roots = [PROJECT_ROOTS[project_name]] if project_name and project_name in PROJECT_ROOTS else PROJECT_ROOTS.values()
    
    flags = re.IGNORECASE if case_insensitive else 0
    try:
        pattern = re.compile(re.escape(query), flags)
    except re.error as e:
        return {"results": [], "total_scanned": 0, "error": f"Invalid query pattern: {e}"}

    findings = []
    scanned_count = 0
    for root in roots:
        for fp in _iter_files(root, extensions):
            if len(findings) >= max_results:
                break
            scanned_count += 1
            try:
                with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                    for line_num, line in enumerate(f, 1):
                        if pattern.search(line):
                            findings.append({"file_path": str(fp), "line_number": line_num, "line_content": line.strip()})
                            if len(findings) >= max_results:
                                break
            except Exception:
                continue # Ignore files that can't be opened/read
        if len(findings) >= max_results:
            break
            
    return {"results": findings, "total_scanned": scanned_count}

# ---------------------------------------------------------------------------
# Tool 4: Semantic Search
# ---------------------------------------------------------------------------

# --- Embedding Backend Implementations ---

def _embed_hf(texts: List[str], model_name: str) -> List[List[float]]:
    """Embedding using Hugging Face sentence-transformers."""
    if not HF_AVAILABLE:
        raise RuntimeError("Hugging Face libraries (sentence-transformers, torch) not installed.")
    global _HF_MODEL
    if _HF_MODEL is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _HF_MODEL = SentenceTransformer(model_name, device=device)
        logger.info("[semantic_search] Loaded HF model '%s' onto %s.", model_name, device)
    
    vectors = _HF_MODEL.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return [v.tolist() for v in vectors]

def _embed_ollama(texts: List[str]) -> List[List[float]]:
    """Embedding using a local Ollama server."""
    if requests is None:
        raise RuntimeError("The 'requests' library is not installed, which is required for the Ollama backend.")
    
    url = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/embeddings")
    model = os.environ.get("OLLAMA_MODEL", "nomic-embed-text")
    
    results = []
    for text in texts:
        if not text.strip():
            # Use a zero vector for empty text. Dimension should match the model.
            # nomic-embed-text has 768 dimensions.
            results.append([0.0] * 768)
            continue
        try:
            response = requests.post(url, json={"model": model, "prompt": text}, timeout=15)
            response.raise_for_status()
            embedding = response.json().get("embedding")
            if not isinstance(embedding, list):
                raise ValueError(f"Ollama API did not return a valid list for embedding. Response: {response.text}")
            results.append(embedding)
        except requests.RequestException as e:
            logger.error("Ollama API request failed: %s", e)
            raise  # Propagate error to trigger fallback
    return results

def _embed_tfidf(texts: List[str]) -> List[List[float]]:
    """Fallback embedding using TF-IDF."""
    if not SKLEARN_AVAILABLE:
        raise RuntimeError("Scikit-learn is not installed, which is required for the TF-IDF fallback.")
    # TfidfVectorizer needs non-empty input to build vocab
    if not any(t.strip() for t in texts):
        return [[0.0] * 1024 for _ in texts] # Return zero vectors of arbitrary dimension
        
    vectorizer = TfidfVectorizer(max_features=1024) # Smaller dimension for performance
    matrix = vectorizer.fit_transform(texts)
    return matrix.toarray().tolist()

def _embed_noop(texts: List[str]) -> List[List[float]]:
    """A no-op embedder that returns zero vectors as a last resort."""
    logger.warning("All embedding backends failed. Using no-op zero vectors.")
    return [[0.0] * 1 for _ in texts] # Minimal dimension

# --- Imports for Semantic Search and Indexing ---
import json
import pathlib
import time
import faiss
import numpy as np
try:
    from sentence_transformers import SentenceTransformer
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    _st_model = SentenceTransformer('all-MiniLM-L6-v2', device=DEVICE)
    logger.info(f"[INFO] SentenceTransformer loaded on {DEVICE}")
except ImportError:
    _st_model = None
    logger.error("[ERROR] sentence-transformers and torch must be installed for semantic search. Run: pip install sentence-transformers torch")
except Exception as e:
    _st_model = None
    logger.error(f"[ERROR] Failed to load SentenceTransformer: {e}")

INDEX_DIR_NAME = ".windsurf_search_index"

# --- Embedding Backend ---
def _embed_batch(texts: list[str]) -> list[list[float]]:
    if _st_model is None:
        raise RuntimeError("SentenceTransformer model is not loaded. Please ensure torch and sentence-transformers are installed and compatible with your GPU.")
    logger.info(f"Embedding a batch of {len(texts)} texts using SentenceTransformers on GPU...")
    with torch.no_grad():
        return _st_model.encode(texts, batch_size=32, show_progress_bar=False, device=DEVICE).tolist()

def _get_project_path(project_name: str) -> pathlib.Path:
    # Use PROJECT_ROOTS if possible, else fallback to cwd/project_name
    if project_name in PROJECT_ROOTS:
        return PROJECT_ROOTS[project_name]
    project_dir = pathlib.Path.cwd() / project_name
    project_dir.mkdir(exist_ok=True, parents=True)
    return project_dir

@mcp.tool()
def index_project_files(
    project_name: str,
    max_file_size_mb: int = 5,
) -> dict:
    """
    Scans a project, intelligently filtering for relevant source code files,
    and builds a searchable vector index. This version is designed to be fast
    and stable by avoiding large, binary, or irrelevant files.

    Args:
        project_name: The identifier for the project to index.
        max_file_size_mb: The maximum size in megabytes for a file to be considered.

    Returns:
        A dictionary summarizing the indexing operation.
    """
    directories_to_ignore = {
        'node_modules', '.git', '__pycache__', 'venv', '.venv', 'target',
        'build', 'dist', '.cache', '.idea', '.vscode', 'eggs', '.eggs'
    }
    text_extensions_to_include = {
        '.py', '.js', '.ts', '.jsx', '.tsx', '.md', '.json', '.yaml', '.yml',
        '.html', '.css', '.scss', '.txt', '.sh', '.bat', '.ps1', '.xml', '.rb',
        '.java', '.c', '.h', '.cpp', '.go', '.rs', '.php'
    }
    binary_extensions_to_ignore = {
        '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.ico', '.svg', '.pdf',
        '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx', '.zip', '.tar', '.gz',
        '.rar', '.7z', '.exe', '.dll', '.so', '.o', '.a', '.lib', '.jar', '.war',
        '.mp3', '.mp4', '.avi', '.mov', '.webm', '.db', '.sqlite', '.sqlite3'
    }
    max_file_size_bytes = max_file_size_mb * 1024 * 1024
    start_time = time.monotonic()
    try:
        project_path = _get_project_path(project_name)
        index_path = project_path / INDEX_DIR_NAME
        index_path.mkdir(exist_ok=True)
    except Exception as e:
        logger.error(f"Failed to access project path: {e}")
        return {"status": "error", "message": f"Failed to access project path: {e}"}

    logger.info(f"Starting intelligent indexing for project '{project_name}'...")
    logger.info(f"Ignoring files larger than {max_file_size_mb} MB.")
    relevant_files = []
    for p in project_path.rglob('*'):
        if p.is_dir():
            continue
        if any(ignored in p.parts for ignored in directories_to_ignore):
            continue
        if p.suffix.lower() in binary_extensions_to_ignore:
            continue
        if p.suffix.lower() not in text_extensions_to_include:
            continue
        try:
            if p.stat().st_size > max_file_size_bytes:
                logger.info(f"Skipping large file: {p.name} ({p.stat().st_size / 1024 / 1024:.2f} MB)")
                continue
        except FileNotFoundError:
            continue
        relevant_files.append(p)
    if not relevant_files:
        logger.warning("No relevant text files found to index with the given criteria.")
        return {"status": "error", "message": "No relevant text files found to index with the given criteria."}
    logger.info(f"Found {len(relevant_files)} relevant files to process.")
    all_vectors = []
    metadata = []
    total_chunks = 0
    PROCESSING_BATCH_SIZE = 128
    CHUNK_SIZE = 512
    CHUNK_OVERLAP = 64
    for i in range(0, len(relevant_files), PROCESSING_BATCH_SIZE):
        batch_files = relevant_files[i : i + PROCESSING_BATCH_SIZE]
        texts_to_embed = []
        batch_metadata = []
        for fp in batch_files:
            try:
                text = fp.read_text("utf-8", errors="ignore")
                if not text.strip():
                    continue
                for j in range(0, len(text), CHUNK_SIZE - CHUNK_OVERLAP):
                    chunk_content = text[j : j + CHUNK_SIZE]
                    texts_to_embed.append(chunk_content)
                    batch_metadata.append({
                        "path": str(fp.relative_to(project_path)),
                        "content": chunk_content
                    })
            except Exception as e:
                logger.warning(f"Could not read or chunk file {fp}: {e}")
        if not texts_to_embed:
            continue
        vectors = _embed_batch(texts_to_embed)
        all_vectors.extend(vectors)
        metadata.extend(batch_metadata)
        total_chunks += len(vectors)
        logger.info(f"Processed batch: {len(vectors)} chunks. Total chunks so far: {total_chunks}")
    if not all_vectors:
        logger.warning("Could not extract any text content from the filtered files.")
        return {"status": "error", "message": "Could not extract any text content from the filtered files."}
    try:
        dimension = len(all_vectors[0])
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(all_vectors, dtype=np.float32))
        faiss.write_index(index, str(index_path / "index.faiss"))
        with open(index_path / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f)
    except Exception as e:
        logger.error(f"Failed to build or save the index: {e}")
        return {"status": "error", "message": f"Failed to build or save the index: {e}"}
    duration = time.monotonic() - start_time
    logger.info(f"[INDEX COMPLETE] Project '{project_name}' indexed successfully.")
    logger.info(f"[INDEX SUMMARY] Files included: {len(relevant_files)} | Chunks indexed: {total_chunks} | Duration: {round(duration, 2)}s")
    return {
        "status": "success",
        "message": f"Project '{project_name}' indexed successfully.",
        "files_scanned_and_included": len(relevant_files),
        "total_chunks_indexed": total_chunks,
        "indexing_duration_seconds": round(duration, 2),
    }

@mcp.tool()
def search_project_index(project_name: str, query: str, max_results: int = 10) -> dict:
    """
    Performs a fast semantic search using a pre-built project index.
    The 'index_project_files' tool must be run on the project first.

    Args:
        project_name: The identifier for the project to search.
        query: The natural language query.
        max_results: The maximum number of results to return.

    Returns:
        A dictionary containing the search results.
    """
    try:
        logger.info(f"[SEARCH START] Attempting semantic search in project '{project_name}' for query: '{query}'")
        project_path = _get_project_path(project_name)
        index_path = project_path / INDEX_DIR_NAME
        faiss_index_file = index_path / "index.faiss"
        metadata_file = index_path / "metadata.json"
        if not (faiss_index_file.exists() and metadata_file.exists()):
            logger.error(f"[SEARCH ERROR] Index files not found for project '{project_name}'.")
            return {
                "error": "Index not found.",
                "message": f"Please run 'index_project_files' for project '{project_name}' first."
            }
    except Exception as e:
        logger.error(f"[SEARCH ERROR] Failed to access project path: {e}")
        return {"status": "error", "message": f"Failed to access project path: {e}"}
    try:
        logger.info(f"[SEARCH LOAD] Loading index and metadata...")
        index = faiss.read_index(str(faiss_index_file))
        with open(metadata_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        logger.info(f"[SEARCH LOAD COMPLETE] Index loaded. Chunks in index: {index.ntotal}")
    except Exception as e:
        logger.error(f"[SEARCH ERROR] Failed to load index or metadata: {e}")
        return {"status": "error", "message": f"Failed to load index or metadata: {e}"}
    try:
        logger.info(f"[SEARCH EMBED] Embedding query and performing similarity search...")
        query_vec = _embed_batch([query])[0]
        D, I = index.search(np.array([query_vec], dtype=np.float32), max_results)
        results = []
        for idx in I[0]:
            if idx == -1:
                continue
            meta = metadata[idx]
            results.append({
                "score": float(D[0][len(results)]),
                "path": meta["path"],
                "content": meta["content"]
            })
        logger.info(f"[SEARCH COMPLETE] Returned {len(results)} results for query: '{query}'")
        return {
            "query": query,
            "results": results,
            "total_chunks_in_index": index.ntotal
        }
    except Exception as e:
        logger.error(f"[SEARCH ERROR] Failed during search: {e}")
        return {"status": "error", "message": f"Failed during search: {e}"}

# ---------------------------------------------------------------------------
# Main execution block to run the server
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logger.info("Starting General Project Tools MCP Server...")
    # The `mcp dev` command is the typical way to run this for development,
    # but this block allows running it directly as a script.
    # Example: `python -m src.toolz` from the project root.
    mcp.run()
