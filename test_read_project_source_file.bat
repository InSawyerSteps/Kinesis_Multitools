@echo off
REM Activate the virtual environment
call "C:\Projects\MCP Server\.venv\Scripts\activate.bat"
REM Change to the project directory
cd /d "C:\Projects\MCP Server"
REM Run the test with a non-existent Dart file (should return error: not found)
python -c "import sys; sys.path.insert(0, 'src'); import toolz; print(toolz.read_project_source_file(r'C:\\Projects\\ParentBuddy\\lib\\main.dart'))"
pause
