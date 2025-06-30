Kinesis Multitools
Kinesis Multitools is a powerful, local-first server providing a suite of advanced tools for AI-driven software development. Designed for integration with modern IDEs, it empowers AI agents with robust capabilities for code intelligence, analysis, and modification—allowing them to interact with and shape your codebase with precision.Key FeaturesMultitool Architecture: A single, unified /search endpoint provides access to multiple search modes, simplifying the interface for AI agents.Advanced Search Modes:keyword: Fast, literal text search.semantic: Natural language, concept-based code search.ast: Structural search for function and class definitions.references: Precise symbol usage and reference finding via Jedi.similarity: Find semantically similar code snippets.task_verification: A meta-search to assess task implementation status.Incremental Vector Indexing: The index_project_files tool is highly efficient, detecting file changes to avoid re-computing embeddings for the entire project on every run.Secure & Sandboxed: All file operations are validated to ensure they remain within the configured project root, preventing unintended file access.Extensible by Design: A clear development guide provides a blueprint for adding new capabilities.InstallationClone the repository:git clone https://github.com/[your-username]/Kinesis-Multitools.git
cd Kinesis-Multitools
Install the required Python packages:pip install -r requirements.txt
UsageRun the server from the command line:python toolz.py
The server will start on http://localhost:8000.1. Indexing the ProjectBefore using semantic search, you must index your project. Send a POST request to the /index_project_files endpoint.Request:POST http://localhost:8000/index_project_files
Content-Type: application/json

{
  "project_name": "MCP-Server"
}
Response:{
  "status": "success",
  "message": "Project 'MCP-Server' indexed incrementally.",
  "files_scanned_and_included": 150,
  "unchanged_files": 140,
  "updated_files": 10,
  "deleted_files": 0,
  "total_chunks_indexed": 1250,
  "indexing_duration_seconds": 15.75
}
2. Searching the CodebaseSend a POST request to the /search endpoint with the desired search_type and query.Example: Semantic SearchPOST http://localhost:8000/search
Content-Type: application/json

{
  "search_type": "semantic",
  "query": "how does the server handle user authentication?",
  "project_name": "MCP-Server",
  "params": {
    "max_results": 5
  }
}
Example: AST Search for a Function DefinitionPOST http://localhost:8000/search
Content-Type: application/json

{
  "search_type": "ast",
  "query": "unified_search",
  "project_name": "MCP-Server",
  "params": {
      "target_node_type": "function"
  }
}
RoadmapKinesis Multitools is actively being expanded. Future updates will introduce a wide range of new tools for advanced static analysis, lossless code refactoring, automated code generation, and project management, further enhancing the capabilities of your AI agent.LicenseThis project is licensed under the MIT License. See the LICENSE file for details.