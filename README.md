# Kinesis Multitools

Kinesis Multitools is a powerful, local-first server providing a suite of advanced tools for AI-driven software development. Designed for integration with modern IDEs, it empowers AI agents with robust capabilities for code intelligence, analysis, and modification—allowing them to interact with and shape your codebase with precision.

## Key Features

- **Tool Reliability**: The supported MCP tools—`index_project_files`, `search` (all modes), and file read/list—are robust and respond within hard timeouts. The previous `analyze` and `edit` tools have been removed due to architectural limitations and instability. Only the tools documented below are supported.
- **Reliable by Design** – All tools run in isolated processes with hard timeouts, preventing hangs and ensuring the server remains responsive.
- **Multitool Architecture** – A single, unified `/search` endpoint provides access to multiple search modes. Only the documented search subtools are supported.
- **Advanced Search Modes**
  - `keyword` – fast literal text search.
  - `semantic` – natural language, concept-based code search.
  - `ast` – structural search for function and class definitions.
  - `references` – precise symbol usage and reference finding via Jedi.
  - `similarity` – find semantically similar code snippets.
  - `task_verification` – meta-search to assess task implementation status.
- **Incremental Vector Indexing** – The `index_project_files` tool detects file changes to avoid re-computing embeddings for unchanged files.
- **Secure & Sandboxed** – All file operations are validated to ensure they remain within the configured project root.
- **Extensible by Design** – A development guide provides a blueprint for adding new capabilities.

## Installation

```bash
git clone https://github.com/[your-username]/Kinesis-Multitools.git
cd Kinesis-Multitools
pip install -r requirements.txt
```

## Windsurf IDE Configuration

To use Kinesis Multitools within the Windsurf IDE, you need to configure it in your `mcp_config.json` file. This tells the IDE how to launch and communicate with the tool server.

Add the following entry to the `mcpServers` object in your configuration file:

```json
"kinesis_multitools": {
  "command": "C:\\Projects\\MCP Server\\.venv\\Scripts\\fastmcp.exe",
  "args": [
    "run",
    "C:\\Projects\\MCP Server\\src\\toolz.py",
    "--transport", "stdio"
  ],
  "env": {},
  "disabled": false
}
```

**Note:** Make sure the `command` path points to the `fastmcp.exe` executable inside your project's virtual environment (`.venv`). Adjust the path if your project is located elsewhere.

After saving the configuration, restart the Windsurf IDE for the changes to take effect.

## Usage

Run the server from the command line:

```bash
python src/toolz.py
```

The server will start on `http://localhost:8000`.

### Indexing the Project

Several of the `/search` subtools rely on a FAISS vector index generated from
your source files.  This index is stored in a hidden folder named
`.windsurf_search_index` inside the project root.  If the index does not exist,
embedding‑based modes such as `semantic`, `similarity` and `task_verification`
return an error:

```
Index not found for project. Please run 'index_project_files' first.
```

Run the indexing tool whenever you update your code so that these subtools have
up‑to‑date embeddings.  Indexing is incremental and will only reprocess files
that changed.

Start by indexing your project:

```http
POST http://localhost:8000/index_project_files
Content-Type: application/json

{
  "project_name": "MCP-Server"
}
```

Example response:

```json
{
  "status": "success",
  "message": "Project 'MCP-Server' indexed incrementally.",
  "files_scanned_and_included": 150,
  "unchanged_files": 140,
  "updated_files": 10,
  "deleted_files": 0,
  "total_chunks_indexed": 1250,
  "indexing_duration_seconds": 15.75
}
```

### Searching the Codebase

Send a request to `/search` with the desired `search_type` and query.

Example – semantic search:

```http
POST http://localhost:8000/search
Content-Type: application/json

{
  "search_type": "semantic",
  "query": "how does the server handle user authentication?",
  "project_name": "MCP-Server",
  "params": {
    "max_results": 5
  }
}
```

Example – AST search for a function definition:

```http
POST http://localhost:8000/search
Content-Type: application/json

{
  "search_type": "ast",
  "query": "unified_search",
  "project_name": "MCP-Server",
  "params": {
    "target_node_type": "function"
  }
}
```

## Roadmap

Kinesis Multitools is actively expanding. Future updates may introduce new tools for advanced static analysis and automated code modification, but only after robust, asynchronous, and library-based architectures are proven stable. The current supported toolset is:

- `index_project_files`: Incremental semantic indexing of project files for search (see `.windsurf/rules/indexing.md`).
- `search`: Multi-modal codebase search (keyword, semantic, ast, references, similarity, task_verification; see `.windsurf/rules/search.md`).
- `cookbook_multitool`: Unified tool for capturing and searching canonical code patterns (see below).
- File read/list utilities.

The `analyze` and `edit` tools are not present in this release.

---

## Code Cookbook Multitool

The `cookbook_multitool` is a unified endpoint for capturing and searching canonical code patterns ("golden patterns") in your project. It replaces the older `add_to_cookbook` and `find_in_cookbook` tools.

### Purpose
- **Add** a function's source and metadata to the project cookbook for future reuse and consistency.
- **Find** patterns by name, description, or function name for rapid code reuse and enforcement of best practices.

### Usage

Send a request to `/cookbook_multitool` with the desired `mode`:

#### Add a Pattern
```json
{
  "mode": "add",
  "pattern_name": "My Golden Pattern",
  "file_path": "C:/Projects/MCP Server/src/toolz.py",
  "function_name": "_is_safe_path",
  "description": "A canonical function for secure path validation."
}
```

#### Find a Pattern
```json
{
  "mode": "find",
  "query": "secure path"
}
```

### Response
- On success, returns a status and message (for add), or a list of matching patterns (for find).
- All cookbook patterns are stored as JSON files in `.project_cookbook/` in the project root.

See the developer guide for advanced usage and schema details.

## License

This project is licensed under the MIT License. See the [LICENSE.md](LICENSE.md) file for details.

