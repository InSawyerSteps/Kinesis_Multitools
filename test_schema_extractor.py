"""Test script for the get_sqlite_schema MCP tool."""
import sys
import json
from pprint import pprint

# Add the src directory to the path
sys.path.insert(0, 'src')
import toolz

def main():
    print("=== Testing SQLite Schema Extraction Tool ===\n")
    
    # Path to the sample Drift file
    file_path = r'C:\Projects\ParentBuddy\lib\database\database.dart'
    
    # Call the tool
    result = toolz.get_sqlite_schema(project_name='ParentBuddy', file_path=file_path)
    
    # Print result summary
    print(f"Status: {result['status']}")
    print(f"Message: {result['message']}")
    print(f"Tables detected: {len(result['tables'])}")
    
    # Print detailed schema information
    print("\nDetailed Table Schema:")
    for table in result['tables']:
        print(f"\n- Table: {table['name']}")
        print("  Columns:")
        for col in table['columns']:
            constraints = ", ".join(col["constraints"]) if col["constraints"] else ""
            constraints_str = f" ({constraints})" if constraints else ""
            print(f"    - {col['name']} ({col['type']}){constraints_str}")
    
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    main()
