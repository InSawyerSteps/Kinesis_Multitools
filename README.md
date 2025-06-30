# Kinesis Multitools

Kinesis Multitools is a powerful, local-first server providing a suite of advanced tools for AI-driven software development. Designed for integration with modern IDEs, it empowers AI agents with robust capabilities for code intelligence, analysis, and modification—allowing them to interact with and shape your codebase with precision.

## Key Features

- **Multitool Architecture** – a single, unified `/search` endpoint provides access to multiple search modes.
- **Advanced Search Modes**
  - `keyword` – fast literal text search.
  - `semantic` – natural language, concept-based code search.
  - `ast` – structural search for function and class definitions.
  - `references` – precise symbol usage and reference finding via Jedi.
  - `similarity` – find semantically similar code snippets.
  - `task_verification` – meta-search to assess task implementation status.
- **Incremental Vector Indexing** – the `index_project_files` tool detects file changes to avoid re-computing embeddings for unchanged files.
- **Secure & Sandboxed** – all file operations are validated to ensure they remain within the configured project root.
- **Extensible by Design** – a development guide provides a blueprint for adding new capabilities.

## Installation

```bash
git clone https://github.com/[your-username]/Kinesis-Multitools.git
cd Kinesis-Multitools
pip install -r requirements.txt
```

## Usage

Run the server from the command line:

```bash
python src/toolz.py
```

The server will start on `http://localhost:8000`.

### Indexing the Project

Before using semantic search, index your project:

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

Kinesis Multitools is actively expanding. Future updates will introduce new tools for advanced static analysis, lossless code refactoring, automated code generation and project management.

## License

This project is licensed under the MIT License. See the [LICENSE.md](LICENSE.md) file for details.

