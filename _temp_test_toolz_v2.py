import json
import logging
import sys
import os

# Define project and src directory paths
project_root = r'c:\Projects\MCP Server'
src_dir = os.path.join(project_root, 'src')

# Add the src directory to sys.path
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

print(f"Using sys.path: {sys.path}") # Diagnostic print

# Now import the function and logger from toolz.py
try:
    # Assuming logger is also defined/imported in toolz.py
    from toolz import hybrid_search_recruiting_data, logger
    print("Successfully imported from toolz") # Diagnostic print
except ImportError as e:
    print(f"Error importing from toolz: {e}")
    # Check if toolz.py exists
    toolz_path = os.path.join(src_dir, 'toolz.py')
    print(f"Checking for toolz.py at: {toolz_path}")
    print(f"toolz.py exists: {os.path.exists(toolz_path)}")
    # Also print contents of src_dir for debugging
    try:
        print(f"Contents of {src_dir}: {os.listdir(src_dir)}")
    except Exception as list_e:
        print(f"Could not list contents of {src_dir}: {list_e}")
    sys.exit(1)
except AttributeError as e:
    print(f"AttributeError: {e}. Does toolz.py define 'hybrid_search_recruiting_data' and 'logger'?")
    sys.exit(1)

# Set logging level
logger.setLevel(logging.DEBUG)

# Define the path to the arguments file (relative to project_root)
args_file_path = os.path.join(project_root, 'rag_recruiting_mvp', 'test_hybrid_args.json')
print(f"Looking for args file at: {args_file_path}") # Diagnostic print

# Check if the args file exists
if not os.path.exists(args_file_path):
    print(f"Error: Arguments file not found at {args_file_path}")
    sys.exit(1)

# Load arguments from the JSON file
try:
    with open(args_file_path, 'r') as f:
        args = json.load(f)
    print("Successfully loaded arguments from JSON") # Diagnostic print
except Exception as e:
    print(f"Error loading JSON from {args_file_path}: {e}")
    sys.exit(1)

# Call the function and print the result
try:
    print("Calling hybrid_search_recruiting_data...") # Diagnostic print
    result = hybrid_search_recruiting_data(**args)
    print("--- RESULT ---")
    print(json.dumps(result, indent=2))
except Exception as e:
    # Use basic print for traceback if logger itself failed
    import traceback
    print(f"Error executing hybrid_search_recruiting_data: {e}")
    traceback.print_exc()
    sys.exit(1)
