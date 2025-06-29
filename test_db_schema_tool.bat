@echo off
echo === Testing MCP Server with SQLite Schema Tool ===
echo.
echo Step 1: Creating sample Drift database schema file for testing...
echo.

if not exist "C:\Projects\ParentBuddy\lib\database\" mkdir "C:\Projects\ParentBuddy\lib\database"

echo // Sample Drift database.dart file > "C:\Projects\ParentBuddy\lib\database\database.dart"
echo import 'package:drift/drift.dart'; >> "C:\Projects\ParentBuddy\lib\database\database.dart"
echo. >> "C:\Projects\ParentBuddy\lib\database\database.dart"
echo // User table definition >> "C:\Projects\ParentBuddy\lib\database\database.dart"
echo class Users extends Table { >> "C:\Projects\ParentBuddy\lib\database\database.dart"
echo   IntColumn get id =^> integer().autoIncrement(); >> "C:\Projects\ParentBuddy\lib\database\database.dart"
echo   TextColumn get name =^> text().withLength(min: 1, max: 50).required(); >> "C:\Projects\ParentBuddy\lib\database\database.dart"
echo   TextColumn get email =^> text().unique(); >> "C:\Projects\ParentBuddy\lib\database\database.dart"
echo   DateTimeColumn get createdAt =^> dateTime().withDefault(currentDateAndTime); >> "C:\Projects\ParentBuddy\lib\database\database.dart"
echo } >> "C:\Projects\ParentBuddy\lib\database\database.dart"
echo. >> "C:\Projects\ParentBuddy\lib\database\database.dart"
echo // Tasks table >> "C:\Projects\ParentBuddy\lib\database\database.dart"
echo class Tasks extends Table { >> "C:\Projects\ParentBuddy\lib\database\database.dart"
echo   IntColumn get id =^> integer().autoIncrement(); >> "C:\Projects\ParentBuddy\lib\database\database.dart"
echo   TextColumn get title =^> text().required(); >> "C:\Projects\ParentBuddy\lib\database\database.dart"
echo   TextColumn get description =^> text().nullable(); >> "C:\Projects\ParentBuddy\lib\database\database.dart"
echo   BoolColumn get isCompleted =^> boolean().withDefault(const Constant(false)); >> "C:\Projects\ParentBuddy\lib\database\database.dart"
echo   IntColumn get userId =^> integer().references(Users, #id); >> "C:\Projects\ParentBuddy\lib\database\database.dart"
echo   DateTimeColumn get dueDate =^> dateTime().nullable(); >> "C:\Projects\ParentBuddy\lib\database\database.dart"
echo } >> "C:\Projects\ParentBuddy\lib\database\database.dart"
echo. >> "C:\Projects\ParentBuddy\lib\database\database.dart"
echo // Example using drift SQL notation for categories >> "C:\Projects\ParentBuddy\lib\database\database.dart"
echo @DriftDatabase(tables: [Users, Tasks]) >> "C:\Projects\ParentBuddy\lib\database\database.dart"
echo class AppDatabase extends _$AppDatabase { >> "C:\Projects\ParentBuddy\lib\database\database.dart"
echo   AppDatabase(QueryExecutor e) : super(e); >> "C:\Projects\ParentBuddy\lib\database\database.dart"
echo. >> "C:\Projects\ParentBuddy\lib\database\database.dart"
echo   // For a drift: table example >> "C:\Projects\ParentBuddy\lib\database\database.dart"
echo   static const String categoriesTable = 'categories'; >> "C:\Projects\ParentBuddy\lib\database\database.dart"
echo   static const String categorySql = >> "C:\Projects\ParentBuddy\lib\database\database.dart"
echo       'drift: "CREATE TABLE categories (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL UNIQUE, color TEXT)"'; >> "C:\Projects\ParentBuddy\lib\database\database.dart"
echo } >> "C:\Projects\ParentBuddy\lib\database\database.dart"
echo. >> "C:\Projects\ParentBuddy\lib\database\database.dart"

echo Sample Drift database schema created at C:\Projects\ParentBuddy\lib\database\database.dart
echo.
echo Step 2: Activating virtual environment...
call "C:\Projects\MCP Server\.venv\Scripts\activate.bat"

echo.
echo Step 3: Running test to extract database schema...
cd /d "C:\Projects\MCP Server"
python -c "import sys; sys.path.insert(0, 'src'); import toolz; result = toolz.get_sqlite_schema(project_name='ParentBuddy', file_path=r'C:\\Projects\\ParentBuddy\\lib\\database\\database.dart'); print(f'Tool result: {result}')"

echo.
echo Testing complete!
pause
