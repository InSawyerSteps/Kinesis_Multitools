"""
gemini_suite.py
A suite of Gemini-powered developer tools for code explanation, review, docstring generation, refactoring, bug finding, and more.
Follows the same pattern as test_suite_manager.py for MCP multitool integration.
"""

import os
import json
import time
import pathlib
from typing import Optional, Literal
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
from fastmcp import FastMCP

# Assume logger and PROJECT_ROOTS are provided by the MCP server context
logger = globals().get('logger', None)
PROJECT_ROOTS = globals().get('PROJECT_ROOTS', {})

mcp = FastMCP()


# ------------------------
# Pydantic Request Models
# ------------------------

class GeminiCodeExplainerRequest(BaseModel):
    source_file: Optional[str] = Field(None, description="Absolute or relative path to the source file.")
    function_name: Optional[str] = Field(None, description="Function to explain.")
    snippet: Optional[str] = Field(None, description="Direct code snippet to explain.")
    timeout: int = Field(default=60, description="Timeout in seconds.")

class GeminiCodeReviewRequest(BaseModel):
    source_file: Optional[str] = None
    diff: Optional[str] = None
    review_level: Literal["quick", "thorough"] = "quick"
    timeout: int = 60

class GeminiDocstringGenRequest(BaseModel):
    source_file: str
    function_name: Optional[str] = None
    docstring_style: Literal["google", "numpy", "rest"] = "google"
    timeout: int = 60

class GeminiRefactorRequest(BaseModel):
    source_file: str
    refactor_type: Literal["rename", "extract_method", "simplify"]
    target: str
    new_name: Optional[str] = None
    timeout: int = 60

class GeminiBugFinderRequest(BaseModel):
    source_file: str
    function_name: Optional[str] = None
    timeout: int = 60

class GeminiUsageAnalyzerRequest(BaseModel):
    period: Literal["day", "week", "month"] = "day"
    user: Optional[str] = None

class GeminiCommitSummarizerRequest(BaseModel):
    repo_path: str
    commit_range: Optional[str] = None
    timeout: int = 60

# ------------------------
# Helper Functions
# ------------------------

import subprocess

def _read_file(file_path: str) -> str:
    """Safely reads a file by absolute or relative path, restricting to the workspace root."""
    abs_path = pathlib.Path(file_path).resolve()
    # Restrict to workspace root for safety
    workspace_root = pathlib.Path(__file__).parent.parent.resolve()
    if not str(abs_path).startswith(str(workspace_root)):
        raise ValueError(f"Unsafe path: {file_path}")
    return abs_path.read_text(encoding="utf-8")

def call_gflash(prompt: str, timeout: int = 60) -> str:
    """Call the Gemini CLI with the Gemini-2.5-Flash model and return the output."""
    try:
        result = subprocess.run(
            ["gemini", "-m", "gemini-2.5-flash"],
            input=prompt.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.decode("utf-8"))
        return result.stdout.decode("utf-8")
    except Exception as e:
        if logger: logger.error(f"[gemini] {e}")
        return f"[gemini ERROR] {e}"

# ------------------------
# Tool Implementations
# ------------------------

@mcp.tool()
def gemini_code_explainer(request: GeminiCodeExplainerRequest) -> dict:
    """
    Explains a code file, function, or snippet using Gemini.
    Args:
        request: Details of the code to explain.
    Returns:
        dict: Explanation or error.
    """
    try:
        code = request.snippet
        if not code and request.source_file:
            code = _read_file(request.source_file)
            if request.function_name:
                import re
                pattern = rf"def {request.function_name}\\s*\\(.*?\\):[\\s\\S]*?^(?=def |class |$)"
                match = re.search(pattern, code, re.MULTILINE)
                code = match.group(0) if match else code
        prompt = f"Explain the following code:\n\n{code}"
        explanation = call_gflash(prompt, timeout=request.timeout)
        return {"status": "success", "explanation": explanation}
    except Exception as e:
        if logger: logger.error(f"[gemini_code_explainer] {e}")
        return {"status": "error", "message": str(e)}

@mcp.tool()
def gemini_code_reviewer(request: GeminiCodeReviewRequest) -> dict:
    """
    Reviews code or a diff using Gemini, returning suggestions and feedback.
    """
    try:
        code = request.diff
        if not code and request.source_file:
            code = _read_file(request.source_file)
        prompt = f"Review the following code for bugs, improvements, and best practices:\n\n{code}"
        review = call_gflash(prompt, timeout=request.timeout)
        return {"status": "success", "review": review}
    except Exception as e:
        if logger: logger.error(f"[gemini_code_reviewer] {e}")
        return {"status": "error", "message": str(e)}

@mcp.tool()
def gemini_docstring_generator(request: GeminiDocstringGenRequest) -> dict:
    """
    Generates or updates docstrings for functions/classes using Gemini.
    """
    try:
        code = _read_file(request.source_file)
        prompt = f"Generate a {request.docstring_style} style docstring for the following code:\n\n{code}"
        docstring = call_gflash(prompt, timeout=request.timeout)
        return {"status": "success", "docstring": docstring}
    except Exception as e:
        if logger: logger.error(f"[gemini_docstring_generator] {e}")
        return {"status": "error", "message": str(e)}

@mcp.tool()
def gemini_refactor_tool(request: GeminiRefactorRequest) -> dict:
    """
    Suggests or applies refactorings to code using Gemini.
    """
    try:
        code = _read_file(request.source_file)
        prompt = f"Refactor the following code. Refactor type: {request.refactor_type}. Target: {request.target}.\n\n{code}"
        refactor_msg = call_gflash(prompt, timeout=request.timeout)
        return {"status": "success", "refactor": refactor_msg}
    except Exception as e:
        if logger: logger.error(f"[gemini_refactor_tool] {e}")
        return {"status": "error", "message": str(e)}

@mcp.tool()
def gemini_bug_finder(request: GeminiBugFinderRequest) -> dict:
    """
    Finds bugs or anti-patterns in code using Gemini.
    """
    try:
        code = _read_file(request.source_file)
        if request.function_name:
            prompt = f"Find bugs or anti-patterns in the following function named '{request.function_name}':\n\n{code}"
        else:
            prompt = f"Find bugs or anti-patterns in the following code:\n\n{code}"
        bug_report = call_gflash(prompt, timeout=request.timeout)
        return {"status": "success", "bugs": bug_report}
    except Exception as e:
        if logger: logger.error(f"[gemini_bug_finder] {e}")
        return {"status": "error", "message": str(e)}

@mcp.tool()
def gemini_usage_analyzer(request: GeminiUsageAnalyzerRequest) -> dict:
    """
    Provides Gemini API usage analytics.
    """
    try:
        # Placeholder: Replace with real usage analytics
        usage = {"period": request.period, "user": request.user, "count": 42}
        return {"status": "success", "usage": usage}
    except Exception as e:
        if logger: logger.error(f"[gemini_usage_analyzer] {e}")
        return {"status": "error", "message": str(e)}

@mcp.tool()
def gemini_commit_summarizer(request: GeminiCommitSummarizerRequest) -> dict:
    """
    Summarizes git commits or PRs using Gemini.
    """
    try:
        # For commit summarizer, you might want to run 'git log' and pass the output to gflash
        import subprocess
        commit_range = request.commit_range or 'HEAD'
        try:
            git_log = subprocess.check_output(
                ["git", "log", commit_range, "--pretty=medium"],
                cwd=request.repo_path,
                timeout=request.timeout,
            ).decode("utf-8")
        except Exception as e:
            git_log = f"[Error getting git log: {e}]"
        prompt = f"Summarize the following git commit log:\n\n{git_log}"
        summary = call_gflash(prompt, timeout=request.timeout)
        return {"status": "success", "summary": summary}
    except Exception as e:
        if logger: logger.error(f"[gemini_commit_summarizer] {e}")
        return {"status": "error", "message": str(e)}

# ------------------------
# Gemini Test Generator
# ------------------------

class GeminiTestGeneratorRequest(BaseModel):
    mode: Literal["generate", "run", "status"] = Field(..., description="Operation mode.")
    # --- Args for 'run' mode ---
    command: Optional[str] = Field(None, description="[run mode] The shell command to execute.")
    report_file: Optional[str] = Field(None, description="[run mode] Optional report file to read.")
    # --- Args for 'generate' mode ---
    source_file: Optional[str] = Field(None, description="[generate mode] Absolute or relative path to the source file.")
    function_name: Optional[str] = Field(None, description="[generate mode] Name of the function to test.")
    test_file_path: Optional[str] = Field(None, description="[generate mode] Absolute or relative path to save the new test file.")
    test_type: Literal["unit", "parameterized", "integration"] = Field("unit", description="Type of test to generate.")
    coverage_goal: Optional[str] = Field(None, description="Coverage focus: function, branch, file, etc.")
    test_style: Literal["pytest", "unittest"] = Field("pytest", description="Test framework style.")
    timeout: int = Field(default=120, description="Timeout in seconds.")

USAGE_FILE_PATH = None
LOCK_FILE_PATH = None

def _initialize_usage_tracker(project_path: pathlib.Path):
    global USAGE_FILE_PATH, LOCK_FILE_PATH
    main_project_root = next(iter(PROJECT_ROOTS.values()))
    USAGE_FILE_PATH = main_project_root / ".gemini_usage.json"
    LOCK_FILE_PATH = main_project_root / ".gemini_usage.lock"

def _get_usage_data() -> dict:
    if not USAGE_FILE_PATH or not LOCK_FILE_PATH:
        raise RuntimeError("Usage tracker not initialized.")
    lock_acquired = False
    for _ in range(10):
        try:
            fd = os.open(str(LOCK_FILE_PATH), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            lock_acquired = True
            break
        except FileExistsError:
            time.sleep(0.1)
    if not lock_acquired:
        raise RuntimeError("Could not acquire lock for usage file.")
    try:
        data = {"count": 0, "last_reset_utc": datetime.utcnow().isoformat()}
        if USAGE_FILE_PATH.exists():
            with open(USAGE_FILE_PATH, 'r') as f:
                try:
                    data = json.load(f)
                    last_reset = datetime.fromisoformat(data["last_reset_utc"])
                    if datetime.utcnow() - last_reset > timedelta(days=1):
                        logger.info("Gemini usage counter is older than 24 hours. Resetting.")
                        data = {"count": 0, "last_reset_utc": datetime.utcnow().isoformat()}
                except (json.JSONDecodeError, KeyError, ValueError):
                    logger.warning("Usage file corrupted or has invalid format. Resetting.")
                    data = {"count": 0, "last_reset_utc": datetime.utcnow().isoformat()}
        return data
    finally:
        if os.path.exists(LOCK_FILE_PATH):
            os.remove(LOCK_FILE_PATH)

def _update_usage_data(data: dict):
    if not USAGE_FILE_PATH or not LOCK_FILE_PATH:
        raise RuntimeError("Usage tracker not initialized.")
    lock_acquired = False
    for _ in range(10):
        try:
            fd = os.open(str(LOCK_FILE_PATH), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            lock_acquired = True
            break
        except FileExistsError:
            time.sleep(0.1)
    if not lock_acquired:
        raise RuntimeError("Could not acquire lock for usage file.")
    try:
        with open(USAGE_FILE_PATH, 'w') as f:
            json.dump(data, f)
    finally:
        if os.path.exists(LOCK_FILE_PATH):
            os.remove(LOCK_FILE_PATH)

def _write_project_file(project_path: pathlib.Path, relative_path: str, content: str) -> dict:
    target_path = (project_path / relative_path).resolve()
    if not str(target_path).startswith(str(project_path.resolve())):
        return {"status": "error", "message": f"Path '{relative_path}' resolves outside the project root."}
    try:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(content, encoding="utf-8")
        logger.info(f"Successfully wrote {len(content)} bytes to {target_path}")
        return {"status": "success", "file_path": str(target_path)}
    except Exception as e:
        logger.error(f"Failed to write file at {target_path}: {e}", exc_info=True)
        return {"status": "error", "message": f"Failed to write file: {e}"}

@mcp.tool()
def gemini_test_generator(request: GeminiTestGeneratorRequest) -> dict:
    """
    Enhanced Gemini-powered test suite manager for generating, running, and tracking tests.
    Modes:
        - generate: Creates a test file (unit/parameterized/integration, pytest/unittest, coverage options).
        - run: Executes a shell command (like pytest).
        - status: Checks Gemini API usage count.
    Args:
        request: GeminiTestGeneratorRequest
    Returns:
        dict: Status, results, or error message.
    """
    import subprocess
    import shlex
    project_path = _get_project_path(request.project_name)
    _initialize_usage_tracker(project_path)
    logger.info(f"[gemini_test_generator] Mode: {request.mode}, Project: {request.project_name}")
    try:
        if request.mode == "status":
            usage_data = _get_usage_data()
            return {"status": "success", "usage": usage_data}
        elif request.mode == "generate":
            if not request.source_file or not request.function_name or not request.test_file_path:
                return {"status": "error", "message": "Missing required fields for 'generate' mode."}
            usage_data = _get_usage_data()
            usage_data["count"] += 1
            _update_usage_data(usage_data)
            # Enhanced test content (placeholder for Gemini API)
            header = f"# Auto-generated {request.test_type} test for {request.function_name} ({request.test_style})\n\n"
            if request.test_type == "unit":
                test_content = header + f"def test_{request.function_name}():\n    assert True\n"
            elif request.test_type == "parameterized":
                if request.test_style == "pytest":
                    test_content = header + "import pytest\n\n@pytest.mark.parametrize('input,expected', [(1, 1), (2, 2)])\ndef test_{0}(input, expected):\n    assert input == expected\n".format(request.function_name)
                else:
                    test_content = header + f"def test_{request.function_name}_param():\n    for input, expected in [(1, 1), (2, 2)]:\n        assert input == expected\n"
            elif request.test_type == "integration":
                test_content = header + f"def test_{request.function_name}_integration():\n    # Integration test placeholder\n    assert True\n"
            else:
                test_content = header + f"def test_{request.function_name}():\n    assert True\n"
            if request.coverage_goal:
                test_content += f"\n# Coverage goal: {request.coverage_goal}\n"
            write_result = _write_project_file(project_path, request.test_file_path, test_content)
            if write_result["status"] != "success":
                return write_result
            return {
                "status": "success",
                "message": f"Test file generated at {write_result['file_path']}",
                "usage": usage_data
            }
        elif request.mode == "run":
            if not request.command:
                return {"status": "error", "message": "No command provided for 'run' mode."}
            try:
                logger.info(f"[gemini_test_generator] Running command: {request.command}")
                proc = subprocess.run(
                    shlex.split(request.command),
                    cwd=str(project_path),
                    timeout=request.timeout,
                    capture_output=True,
                    text=True
                )
                output = proc.stdout
                error = proc.stderr
                result = {
                    "status": "success" if proc.returncode == 0 else "error",
                    "returncode": proc.returncode,
                    "output": output,
                    "error": error
                }
                if request.report_file:
                    report_path = (project_path / request.report_file).resolve()
                    if report_path.exists():
                        try:
                            with open(report_path, 'r', encoding='utf-8') as f:
                                result["report"] = f.read()
                        except Exception as e:
                            result["report_error"] = f"Failed to read report: {e}"
                return result
            except subprocess.TimeoutExpired:
                return {"status": "error", "message": "Test run timed out."}
            except Exception as e:
                logger.error(f"[gemini_test_generator] Exception during run: {e}", exc_info=True)
                return {"status": "error", "message": f"Exception during run: {e}"}
        else:
            return {"status": "error", "message": f"Unknown mode: {request.mode}"}
    except Exception as e:
        logger.error(f"[gemini_test_generator] Unexpected error: {e}", exc_info=True)
        return {"status": "error", "message": f"Unexpected error: {e}"}

# End of gemini_suite.py

