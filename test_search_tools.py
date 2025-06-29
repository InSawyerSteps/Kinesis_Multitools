"""
Test script for search.py tools.

This script tests all search functionalities provided by search.py:
1. Index project files
2. Test each search type (keyword, semantic, ast, references, similarity, task_verification)
"""
import json
import time
from pathlib import Path
from typing import Dict, Any

# Import the tools we want to test
from src.search import index_project_files, SearchRequest, unified_search

def print_header(title: str) -> None:
    """Print a section header for better test output readability."""
    print(f"\n{'='*80}\n{title.upper():^80}\n{'='*80}")

def test_index_project() -> Dict[str, Any]:
    """Test the index_project_files tool."""
    print_header("1. Testing index_project_files")
    
    # Test with the current project
    project_name = "MCP-Server"
    max_file_size_mb = 5
    
    print(f"Indexing project '{project_name}' with max file size {max_file_size_mb}MB...")
    start_time = time.time()
    
    result = index_project_files(
        project_name=project_name,
        max_file_size_mb=max_file_size_mb
    )
    
    elapsed = time.time() - start_time
    print(f"Indexing completed in {elapsed:.2f} seconds")
    print("Result:", json.dumps(result, indent=2))
    
    return result

def test_search_tools() -> None:
    """Test all search tools after indexing is complete."""
    project_name = "MCP-Server"
    
    # 1. Test keyword search
    print_header("2. Testing Keyword Search")
    keyword_request = SearchRequest(
        search_type="keyword",
        query="def index_project_files",
        project_name=project_name,
        params={"case_sensitive": False}
    )
    keyword_results = unified_search(keyword_request)
    print("Keyword search results:")
    print(json.dumps(keyword_results, indent=2))
    
    # 2. Test semantic search (requires indexing first)
    print_header("3. Testing Semantic Search")
    semantic_request = SearchRequest(
        search_type="semantic",
        query="function that indexes project files",
        project_name=project_name,
        params={"top_k": 3}
    )
    semantic_results = unified_search(semantic_request)
    print("Semantic search results:")
    print(json.dumps(semantic_results, indent=2))
    
    # 3. Test AST search
    print_header("4. Testing AST Search")
    ast_request = SearchRequest(
        search_type="ast",
        query="index_project_files",
        project_name=project_name,
        params={"node_type": "function"}
    )
    ast_results = unified_search(ast_request)
    print("AST search results:")
    print(json.dumps(ast_results, indent=2))
    
    # 4. Test references search (requires a symbol that exists in the codebase)
    print_header("5. Testing References Search")
    refs_request = SearchRequest(
        search_type="references",
        query="index_project_files",
        project_name=project_name
    )
    refs_results = unified_search(refs_request)
    print("References search results:")
    print(json.dumps(refs_results, indent=2))
    
    # 5. Test similarity search
    print_header("6. Testing Similarity Search")
    code_snippet = """
    def example_function():
        # Example function for testing similarity search
        return "Hello, world!"
    """
    similarity_request = SearchRequest(
        search_type="similarity",
        query=code_snippet,
        project_name=project_name,
        params={"top_k": 2}
    )
    similarity_results = unified_search(similarity_request)
    print("Similarity search results:")
    print(json.dumps(similarity_results, indent=2))
    
    # 6. Test task verification
    print_header("7. Testing Task Verification")
    task_description = "Create a function that indexes project files for search"
    task_request = SearchRequest(
        search_type="task_verification",
        query=task_description,
        project_name=project_name,
        params={"confidence_threshold": 0.5}
    )
    task_results = unified_search(task_request)
    print("Task verification results:")
    print(json.dumps(task_results, indent=2))

def main():
    """Run all tests."""
    try:
        # First, index the project
        index_result = test_index_project()
        
        if index_result.get("status") != "success":
            print("Indexing failed, cannot proceed with search tests.")
            return
            
        # Then test all search tools
        test_search_tools()
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
