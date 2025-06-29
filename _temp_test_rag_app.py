import json
import logging
import sys
import os

# Define project and rag_app directory paths
project_root = r'c:\Projects\MCP Server'
rag_app_dir = os.path.join(project_root, 'rag_recruiting_mvp')

# Add the rag_app directory to sys.path
if rag_app_dir not in sys.path:
    sys.path.insert(0, rag_app_dir)

print(f"Using sys.path: {sys.path}") # Diagnostic print

# Now import the function and logger from main_app.py
try:
    # Assuming logger is also defined/imported in main_app.py
    # If logger is defined elsewhere (e.g., toolz.py), adjust import
    from main_app import hybrid_search_recruiting_data, logger
    print("Successfully imported from main_app") # Diagnostic print
except ImportError as e:
    print(f"Error importing from main_app: {e}")
    # Check if main_app.py exists
    main_app_path = os.path.join(rag_app_dir, 'main_app.py')
    print(f"Checking for main_app.py at: {main_app_path}")
    print(f"main_app.py exists: {os.path.exists(main_app_path)}")
    # Also print contents of rag_app_dir for debugging
    try:
        print(f"Contents of {rag_app_dir}: {os.listdir(rag_app_dir)}")
    except Exception as list_e:
        print(f"Could not list contents of {rag_app_dir}: {list_e}")
    sys.exit(1)
except AttributeError as e:
    print(f"AttributeError: {e}. Does main_app.py define 'hybrid_search_recruiting_data' and 'logger'?")
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
