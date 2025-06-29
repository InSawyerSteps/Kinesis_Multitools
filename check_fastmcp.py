import sys
print('PYTHON EXECUTABLE:', sys.executable)
print('PYTHONPATH:', sys.path)
try:
    import fastmcp
    print('fastmcp import: OK')
except ImportError as e:
    print('fastmcp import: FAIL:', e)
