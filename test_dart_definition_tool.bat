@echo off
echo === Testing MCP Server with Dart Definition Tool ===
echo.
echo Step 1: Creating dummy .dart file with sample class and method for testing...
echo.

if not exist "C:\Projects\ParentBuddy\lib\" mkdir "C:\Projects\ParentBuddy\lib"

echo // Test Dart file with UserProfile class > "C:\Projects\ParentBuddy\lib\user_profile.dart"
echo class UserProfile { >> "C:\Projects\ParentBuddy\lib\user_profile.dart"
echo   String name; >> "C:\Projects\ParentBuddy\lib\user_profile.dart"
echo   int age; >> "C:\Projects\ParentBuddy\lib\user_profile.dart"
echo. >> "C:\Projects\ParentBuddy\lib\user_profile.dart"
echo   UserProfile({required this.name, required this.age}); >> "C:\Projects\ParentBuddy\lib\user_profile.dart"
echo. >> "C:\Projects\ParentBuddy\lib\user_profile.dart"
echo   void printInfo() { >> "C:\Projects\ParentBuddy\lib\user_profile.dart"
echo     print('User: $name, Age: $age'); >> "C:\Projects\ParentBuddy\lib\user_profile.dart"
echo   } >> "C:\Projects\ParentBuddy\lib\user_profile.dart"
echo } >> "C:\Projects\ParentBuddy\lib\user_profile.dart"
echo. >> "C:\Projects\ParentBuddy\lib\user_profile.dart"
echo // Top-level function >> "C:\Projects\ParentBuddy\lib\user_profile.dart"
echo void displayProfile(UserProfile profile) { >> "C:\Projects\ParentBuddy\lib\user_profile.dart"
echo   profile.printInfo(); >> "C:\Projects\ParentBuddy\lib\user_profile.dart"
echo } >> "C:\Projects\ParentBuddy\lib\user_profile.dart"

echo Dummy Dart file created at C:\Projects\ParentBuddy\lib\user_profile.dart
echo.
echo Step 2: Activating virtual environment...
call "C:\Projects\MCP Server\.venv\Scripts\activate.bat"

echo.
echo Step 3: Running test to extract the UserProfile class definition...
cd /d "C:\Projects\MCP Server"
python -c "import sys; sys.path.insert(0, 'src'); import toolz; result = toolz.get_dart_definition(project_name='ParentBuddy', file_path=r'C:\\Projects\\ParentBuddy\\lib\\user_profile.dart', symbol_name='UserProfile'); print(f'Tool result: {result}')"

echo.
echo Step 4: Running test to extract the displayProfile function definition...
python -c "import sys; sys.path.insert(0, 'src'); import toolz; result = toolz.get_dart_definition(project_name='ParentBuddy', file_path=r'C:\\Projects\\ParentBuddy\\lib\\user_profile.dart', symbol_name='displayProfile'); print(f'Tool result: {result}')"

echo.
echo Testing complete!
pause
