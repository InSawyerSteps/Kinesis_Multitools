"""
Focused test script for search.py tools.

This script tests the search functionality using a small test directory
with controlled input files.
"""
print("--- Script starting ---")
import json
import os
import sys
print("--- Standard imports successful ---")
from pathlib import Path
from typing import Dict, Any, List

# Add the src directory to the path so we can import search
print("--- Appending to sys.path ---")
sys.path.append(str(Path(__file__).parent / 'src'))
print(f"--- sys.path updated: {sys.path[-1]} ---")

print("--- Importing from src.search ---")
from search import index_project_files, SearchRequest, unified_search
print("--- Import from src.search successful ---")

# Configure logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("test_search")

def print_header(title: str) -> None:
    """Print a section header for better test output readability."""
    print(f"\n{'='*80}\n{title.upper():^80}\n{'='*80}")

def test_index_project(project_path: Path) -> Dict[str, Any]:
    """Test the index_project_files tool with a specific directory."""
    print_header(f"1. Testing index_project_files on {project_path}")
    
    # Create a test project in the PROJECT_ROOTS
    from search import PROJECT_ROOTS
    PROJECT_ROOTS["test_project"] = project_path
    
    try:
        # Index the test project
        print(f"Indexing test project at {project_path}...")
        start_time = time.time()
        
        result = index_project_files(
            project_name="test_project",
            max_file_size_mb=1  # Small size is fine for our test files
        )
        
        elapsed = time.time() - start_time
        print(f"Indexing completed in {elapsed:.2f} seconds")
        print("Result:", json.dumps(result, indent=2))
        
        return result
    except Exception as e:
        logger.error(f"Error during indexing: {e}")
        return {"status": "error", "message": str(e)}

def run_single_test(search_type: str, query: str, params: Dict[str, Any] = None):
    """Helper to run and print a single search test."""
    if params is None:
        params = {}
    print_header(f"Testing {search_type.upper()} Search")
    print(f"Query: {query[:100]}\n")
    try:
        request = SearchRequest(
            search_type=search_type,
            query=query,
            project_name="test_project",
            params=params
        )
        start_time = time.time()
        results = unified_search(request)
        elapsed = time.time() - start_time
        print(f"Search completed in {elapsed:.2f} seconds.")
        print("--- Results ---")
        print(json.dumps(results, indent=2))
        print("--- End Results ---")
    except Exception as e:
        logger.error(f"An error occurred during the '{search_type}' test.", exc_info=True)

def test_list_project_files(test_dir: Path):
    print_header("Testing list_project_files tool")
    from search import list_project_files
    files = list_project_files("test_project")
    print(f"Files found ({len(files)}):")
    for f in files:
        print(f"  {f}")
    return files

def test_read_project_file(file_path: str):
    print_header("Testing read_project_file tool")
    from search import read_project_file
    result = read_project_file(file_path)
    print(json.dumps(result, indent=2))
    return result

def main():
    """Run all tests with the test directory."""
    test_dir = Path(__file__).parent / "test_search_dir"

    index_result = test_index_project(test_dir)
    if index_result.get("status") != "success":
        print("\nIndexing failed. Aborting search tests.")
        return

    # Test list_project_files
    files = test_list_project_files(test_dir)

    # Test read_project_file (use the first .py file found)
    py_files = [f for f in files if f.endswith('.py')]
    if py_files:
        test_read_project_file(py_files[0])
    else:
        print("No .py files found for read_project_file test.")

    # Test Keyword Search
    run_single_test("keyword", "def add")

    # Test Semantic Search
    run_single_test("semantic", "a function that adds two numbers", {"top_k": 2})

    # Test AST Search
    run_single_test("ast", "Calculator", {"target_node_type": "class"})

    # Test References Search
    # Note: Jedi might not find references if the project structure is not standard.
    run_single_test("references", "math_operations.add")

    # Test Similarity Search
    similarity_query = "def subtract(x, y): return x - y"
    run_single_test("similarity", similarity_query, {"top_k": 2})

    # Test Task Verification
    task_query = "a tool to reverse a piece of text"
    run_single_test("task_verification", task_query)

if __name__ == "__main__":
    print("--- Entering main execution block ---")
    import time
    main()
    print("--- Script finished ---")
