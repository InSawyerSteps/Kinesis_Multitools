---
trigger: always_on
---

Incremental Indexing Rule (index_project_files)
The index_project_files tool must implement incremental indexing:
It loads previous metadata, detects new/changed/deleted files, and only re-embeds changed files.
The FAISS index is rebuilt from all current vectors (reused + new).
Metadata is updated and saved after each run.
The tool’s return value must include counts of files scanned, unchanged, updated, deleted, and total chunks indexed.
All logging, error handling, and project path validation requirements apply.
If metadata is missing or corrupt, the tool must fall back to a full reindex and log a warning.
Edge cases (e.g., file renames, partial failures) must be handled gracefully.
Usage of the tool should be documented in the developer guide and surfaced via the MCP Inspector UI.