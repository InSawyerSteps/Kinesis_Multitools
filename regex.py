from __future__ import annotations

import os
import pathlib
import re
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Request model
# ---------------------------------------------------------------------------
class RegexSearchRequest(BaseModel):
    """Parameters for advanced regex search."""

    query: str = Field(..., description="Regular expression pattern to search for.")
    project_path: pathlib.Path = Field(..., description="Root path of the project to search.")
    includes: Optional[List[str]] = Field(None, description="Optional list of file paths to search.")
    extensions: Optional[List[str]] = Field(None, description="File extensions to include if 'includes' is not provided.")
    ignore_case: bool = Field(False, description="Use case-insensitive matching.")
    multiline: bool = Field(False, description="Use re.MULTILINE flag.")
    dotall: bool = Field(False, description="Use re.DOTALL flag to allow '.' to match newlines.")
    max_results: int = Field(1000, description="Maximum number of matches to return.")


# ---------------------------------------------------------------------------
# Helper for tests / standalone usage. The main server already defines a more
# robust version of this function, so when copying to toolz.py you can remove
# the implementation below and reuse the existing one.
# ---------------------------------------------------------------------------

def _iter_files(root: pathlib.Path, extensions: Optional[List[str]] = None):
    """Yield files under root optionally filtered by extension."""
    norm_exts = {f".{e.lstrip('.').lower()}" for e in extensions} if extensions else None
    for path in root.rglob("*"):
        if path.is_file() and (not norm_exts or path.suffix.lower() in norm_exts):
            yield path


# ---------------------------------------------------------------------------
# Advanced regex search implementation
# ---------------------------------------------------------------------------

def _search_by_regex(request: RegexSearchRequest) -> Dict[str, Any]:
    """Search project files using a compiled regular expression.

    Args:
        request: ``RegexSearchRequest`` defining the pattern, project path and
            optional filters.

    Returns:
        A dict with ``status``, ``results`` and ``files_scanned`` keys. ``results``
        is a list of matches with ``file_path``, ``line_number`` and ``match``.
    """

    logger = logging.getLogger("advanced_regex_search")
    flags = 0
    if request.ignore_case:
        flags |= re.IGNORECASE
    if request.multiline:
        flags |= re.MULTILINE
    if request.dotall:
        flags |= re.DOTALL

    try:
        pattern = re.compile(request.query, flags)
    except re.error as exc:
        return {"status": "error", "message": f"Invalid regex: {exc}"}

    if request.includes:
        files = [
            request.project_path / inc if not os.path.isabs(inc) else pathlib.Path(inc)
            for inc in request.includes
        ]
    else:
        files = list(_iter_files(request.project_path, request.extensions))

    results: List[Dict[str, Any]] = []
    files_scanned = 0

    for fp in files:
        if ".windsurf_search_index" in str(fp):
            continue
        try:
            with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                for lineno, line in enumerate(f, 1):
                    for match in pattern.finditer(line):
                        snippet = match.group(0)
                        if len(snippet) > 200:
                            snippet = snippet[:200] + "..."
                        results.append(
                            {
                                "file_path": str(fp),
                                "line_number": lineno,
                                "match": snippet,
                            }
                        )
                        if len(results) >= request.max_results:
                            return {
                                "status": "success",
                                "results": results,
                                "files_scanned": files_scanned + 1,
                            }
            files_scanned += 1
        except Exception:
            continue

    return {"status": "success", "results": results, "files_scanned": files_scanned}


__all__ = ["RegexSearchRequest", "_search_by_regex"]
