@echo off
call "C:\Projects\MCP Server\.venv\Scripts\activate.bat"
cd /d "C:\Projects\MCP Server"
python test_schema_extractor.py
pause
