@echo off
call "C:\Projects\MCP Server\.venv\Scripts\activate.bat"
cd /d "C:\Projects\MCP Server"
python check_fastmcp.py
pause
