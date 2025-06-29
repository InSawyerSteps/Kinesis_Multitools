from typing import Any, Callable, Dict, List, Optional
import pathlib
import time
import logging

# Assume logger, _embed_hf, _embed_ollama, _embed_tfidf, _embed_noop, _cosine_similarity, _is_safe_path, _iter_files, PROJECT_ROOTS, and SKLEARN_AVAILABLE are defined elsewhere in the actual project context

# Example decorator placeholder
def mcp_tool(*args, **kwargs):
    def decorator(func):
        return func
    return decorator

@mcp_tool(name="semantic_search")
def semantic_search_in_files(
    query: str,
    project_name: Optional[str] = None,
    max_results: int = 10,
    chunk_size: int = 512,
    backend: str = "auto",
    hf_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    file_paths: Optional[List[str]] = None,
    max_chunks: int = 2000,
) -> Dict[str, Any]:
    """
    Semantic similarity search over project files with robust fallbacks.

    Args:
        query: The natural language query.
        project_name: Project to search. If None, searches all. Ignored if file_paths is set.
        max_results: Number of top matching chunks to return.
        chunk_size: Size of text chunks for embedding.
        backend: Preferred embedding backend: 'auto', 'hf', 'ollama', 'tfidf'.
        hf_model_name: The Hugging Face model to use if backend is 'hf'.
        file_paths: An explicit list of absolute file paths to search.
        max_chunks: Safety cap on the number of chunks to process.

    Returns:
        A dictionary with results, backend used, and chunks scanned.
    """
    # --- 1. Select and test the embedding backend ---
    embed_batch: Callable[[List[str]], List[List[float]]] = _embed_noop
    chosen_backend = "noop"

    backend_preference = ["hf", "ollama", "tfidf"] if backend == "auto" else [backend, "hf", "ollama", "tfidf"]
    
    for backend_name in backend_preference:
        try:
            if backend_name == "hf":
                embed_batch = lambda t: _embed_hf(t, hf_model_name)
            elif backend_name == "ollama":
                embed_batch = _embed_ollama
            elif backend_name == "tfidf":
                embed_batch = _embed_tfidf
            else:
                continue # Skip unknown backends

            # Test the selected backend
            test_vector = embed_batch(["self-test"])
            if test_vector and test_vector[0]:
                chosen_backend = backend_name
                logging.info("[semantic_search] Using '%s' backend.", chosen_backend)
                break # Success, we have our embedder
        except Exception as e:
            logging.warning("[semantic_search] Backend '%s' failed: %s. Trying next.", backend_name, e)
            continue
    
    # --- 2. Collect and chunk text from files ---
    paths_to_scan: List[pathlib.Path] = []
    if file_paths:
        paths_to_scan = [pathlib.Path(p) for p in file_paths if _is_safe_path(pathlib.Path(p)) and pathlib.Path(p).is_file()]
    else:
        roots = [PROJECT_ROOTS[project_name]] if project_name and project_name in PROJECT_ROOTS else PROJECT_ROOTS.values()
        for root in roots:
            paths_to_scan.extend(_iter_files(root))

    all_chunks_text: List[str] = []
    all_chunks_meta: List[Dict[str, str]] = []

    for fp in paths_to_scan:
        if len(all_chunks_text) >= max_chunks:
            break
        try:
            text = fp.read_text("utf-8", errors="ignore")
            if not text.strip(): continue
            
            for i in range(0, len(text), chunk_size):
                chunk_content = text[i : i + chunk_size]
                all_chunks_text.append(chunk_content)
                all_chunks_meta.append({"path": str(fp), "content": chunk_content})
                if len(all_chunks_text) >= max_chunks:
                    break
        except Exception:
            continue

    if not all_chunks_text:
        return {"backend": chosen_backend, "results": [], "total_chunks_scanned": 0, "message": "No text content found to search."}

    # --- 3. Embed query and document chunks ---
    try:
        logging.info("Embedding %d chunks and query with '%s'...", len(all_chunks_text), chosen_backend)
        start_time = time.monotonic()
        
        # Embed query and documents in one batch if possible
        all_texts_to_embed = [query] + all_chunks_text
        all_vectors = embed_batch(all_texts_to_embed)
        
        query_vec = all_vectors[0]
        doc_vecs = all_vectors[1:]
        
        duration = time.monotonic() - start_time
        logging.info("Embedding completed in %.2f seconds.", duration)

    except Exception as e:
        logging.error("Fatal error during embedding with backend '%s': %s", chosen_backend, e, exc_info=True)
        return {"backend": chosen_backend, "results": [], "error": f"Embedding failed: {e}"}

    # --- 4. Calculate similarity and rank results ---
    scored_results = []
    for i, doc_vec in enumerate(doc_vecs):
        if not doc_vec or not query_vec: continue # Skip if embedding failed for a chunk
        score = _cosine_similarity(query_vec, doc_vec)
        scored_results.append({
            "score": score,
            "path": all_chunks_meta[i]["path"],
            "content": all_chunks_meta[i]["content"],
        })

    # Sort by score descending and take top results
    scored_results.sort(key=lambda x: x["score"], reverse=True)
    top_results = scored_results[:max_results]

    return {
        "backend": chosen_backend,
        "results": top_results,
        "total_chunks_scanned": len(all_chunks_text),
    }
