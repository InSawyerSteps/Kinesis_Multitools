import sys
import os

print(f'Initial sys.path: {sys.path}')

project_root = r'c:\Projects\MCP Server'
print(f'Attempting to add: {project_root}')

sys.path.insert(0, project_root)

print(f'Modified sys.path: {sys.path}')

mcp_file_path = os.path.join(project_root, 'mcp.py')
print(f'Checking for: {mcp_file_path}')
print(f'mcp.py exists: {os.path.exists(mcp_file_path)}')
