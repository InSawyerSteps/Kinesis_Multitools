@echo off
echo === Testing MCP Server with Dart File Reading Tool ===
echo.
echo Step 1: Creating dummy .dart file for testing...
echo.

if not exist "C:\Projects\ParentBuddy\lib\" mkdir "C:\Projects\ParentBuddy\lib"

echo // Test Dart file > "C:\Projects\ParentBuddy\lib\main.dart"
echo void main() { >> "C:\Projects\ParentBuddy\lib\main.dart"
echo   print('Hello from ParentBuddy!'); >> "C:\Projects\ParentBuddy\lib\main.dart"
echo } >> "C:\Projects\ParentBuddy\lib\main.dart"

echo Dummy Dart file created at C:\Projects\ParentBuddy\lib\main.dart
echo.
echo Step 2: Activating virtual environment...
call "C:\Projects\MCP Server\.venv\Scripts\activate.bat"

echo.
echo Step 3: Running test...
cd /d "C:\Projects\MCP Server"
python -c "import sys; sys.path.insert(0, 'src'); import toolz; result = toolz.read_project_source_file(r'C:\\Projects\\ParentBuddy\\lib\\main.dart'); print(f'Tool result: {result}')"

echo.
echo Testing complete!
pause
