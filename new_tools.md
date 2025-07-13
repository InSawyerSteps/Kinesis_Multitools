## **New Tool Specifications**

This document outlines the functional requirements for two new tools to be added to the MCP server: /introspect and /get\_snippet.

### **1\. The /introspect Multitool**

* **High-Level Purpose:** To provide a fast, read-only, multi-modal tool for lightweight code and project introspection.  
* **Core Principle:** This tool **must** be extremely fast, with all operations completing in well under 30 seconds. It achieves this by using Python's built-in ast module and simple file I/O, avoiding slow external processes or heavy library imports. It acts as a developer's "magnifying glass" for quick analysis without the overhead of a full linter or static analysis suite.

#### **Sub-tool Functionality (Modes):**

* **mode: 'config'**  
  * **Functionality:** Reads and returns the content of key project configuration files.  
  * **Input:** Requires a config\_type of either 'pyproject' or 'requirements'.  
  * **Output:** For 'pyproject', it returns the full text content of pyproject.toml. For 'requirements', it parses requirements.txt and returns a JSON list of package strings.  
* **mode: 'outline'**  
  * **Functionality:** Generates a high-level structural map of a single Python source file.  
  * **Input:** Requires a file\_path.  
  * **Output:** Returns a JSON object containing two lists: one for top-level functions and one for classes, where each class object also contains a list of its methods.  
* **mode: 'stats'**  
  * **Functionality:** Calculates basic statistics for a given source file.  
  * **Input:** Requires a file\_path.  
  * **Output:** Returns a JSON object with metrics like total\_lines, code\_lines (non-empty, non-comment), comment\_lines, and file\_size\_bytes.  
* **mode: 'inspect'**  
  * **Functionality:** Provides specific details about a single function or class within a file.  
  * **Input:** Requires a file\_path and a function\_name or class\_name.  
  * **Output:** For a function, it returns its name, a list of its args, and its docstring.

### **2\. The /get\_snippet Tool ✂️**

* **High-Level Purpose:** To act as a precision instrument for extracting specific blocks of source code from a file.  
* **Core Principle:** This tool prioritizes accuracy. It uses Python's ast module to guarantee that it extracts the complete and exact source of a function or class, regardless of formatting. For line-based extraction, it provides a simple and fast alternative to reading an entire file.

#### **Sub-tool Functionality (Modes):**

* **mode: 'function' or mode: 'class'**  
  * **Functionality:** Extracts the full source code of a single, named function or class.  
  * **Input:** Requires a file\_path and the name of the function or class.  
  * **Output:** Returns a JSON object containing the complete source code snippet, including the signature and decorator lines.  
* **mode: 'lines'**  
  * **Functionality:** Extracts a specific range of lines from any text file.  
  * **Input:** Requires a file\_path, a start\_line (1-indexed), and an end\_line (inclusive).  
  * **Output:** Returns a JSON object containing the requested lines as a single string snippet.

---

### **A) The /introspect Multitool**

This tool combines several fast, read-only analysis functions into a single, unified endpoint. It's designed to give you immediate insights into your code's structure, configuration, and basic metrics without the performance overhead of full linting or analysis.  
Here is the Pydantic model and the tool implementation you can add to toolz.py:  
Python  
\# Add this with your other Pydantic models  
class IntrospectRequest(BaseModel):  
    mode: Literal\[  
        "config", "outline", "stats", "inspect"  
    \] \= Field(..., description="The introspection mode to use.")  
    file\_path: Optional\[str\] \= Field(None, description="The absolute path to the file for inspection.")  
    \# For 'inspect' mode  
    function\_name: Optional\[str\] \= Field(None, description="The name of the function to inspect.")  
    class\_name: Optional\[str\] \= Field(None, description="The name of the class to inspect.")  
    \# For 'config' mode  
    config\_type: Optional\[Literal\["pyproject", "requirements"\]\] \= Field(None, description="The type of config file to read.")

\# Add this with your other tool functions  
@mcp.tool()  
@tool\_timeout\_and\_errors(timeout=30)  
def introspect(request: IntrospectRequest) \-\> dict:  
    """  
    A fast, multi-modal tool for introspecting project files.

    Provides quick, read-only information about configuration, code structure,  
    file statistics, and specific code elements using lightweight methods.

    Args:  
        request (IntrospectRequest):  
            \- mode (str): 'config', 'outline', 'stats', or 'inspect'.  
            \- file\_path (str, optional): The target file for the operation.  
            \- function\_name (str, optional): For 'inspect' mode with a function.  
            \- class\_name (str, optional): For 'inspect' mode with a class.  
            \- config\_type (str, optional): For 'config' mode, 'pyproject' or 'requirements'.

    Returns:  
        dict: A dictionary containing the introspection results.  
    """  
    logger.info(f"\[introspect\] mode='{request.mode}' file='{request.file\_path}'")  
    project\_path \= \_get\_project\_path("MCP-Server")  
    if not project\_path:  
        return {"status": "error", "message": "Project 'MCP-Server' not found."}

    \# \--- Mode Dispatch \---  
    if request.mode \== "config":  
        if not request.config\_type:  
            return {"status": "error", "message": "Missing 'config\_type' for 'config' mode."}  
        target\_file \= "pyproject.toml" if request.config\_type \== "pyproject" else "requirements.txt"  
        config\_path \= project\_path / target\_file  
        if not config\_path.exists():  
            return {"status": "not\_found", "message": f"{target\_file} not found."}  
        try:  
            content \= config\_path.read\_text("utf-8")  
            if request.config\_type \== "pyproject":  
                \# A simple text return is fast and safe. For TOML parsing, ensure a library is available.  
                return {"status": "success", "config\_type": "pyproject", "content": content}  
            else:  
                packages \= \[line.strip() for line in content.splitlines() if line.strip() and not line.startswith('\#')\]  
                return {"status": "success", "config\_type": "requirements", "packages": packages}  
        except Exception as e:  
            return {"status": "error", "message": f"Failed to read {target\_file}: {e}"}

    \# All other modes require a file\_path  
    if not request.file\_path:  
        return {"status": "error", "message": f"Missing 'file\_path' for '{request.mode}' mode."}

    path \= pathlib.Path(request.file\_path)  
    if not \_is\_safe\_path(path) or not path.is\_file():  
        return {"status": "error", "message": "Invalid or unsafe file path."}

    if request.mode \== "outline":  
        try:  
            source \= path.read\_text("utf-8")  
            tree \= ast.parse(source)  
            outline \= {"classes": \[\], "functions": \[\]}  
            for node in ast.walk(tree):  
                if isinstance(node, ast.FunctionDef):  
                    \# To avoid nesting functions inside class methods in the top-level list  
                    if not any(isinstance(p, ast.ClassDef) for p in ast.iter\_parents(node)):  
                        outline\["functions"\].append(node.name)  
                elif isinstance(node, ast.ClassDef):  
                    methods \= \[m.name for m in node.body if isinstance(m, ast.FunctionDef)\]  
                    outline\["classes"\].append({"name": node.name, "methods": methods})  
            return {"status": "success", "results": outline}  
        except Exception as e:  
            return {"status": "error", "message": f"Failed to parse file: {e}"}

    if request.mode \== "stats":  
        try:  
            lines \= path.read\_text("utf-8").splitlines()  
            stats \= {  
                "total\_lines": len(lines),  
                "code\_lines": len(\[l for l in lines if l.strip() and not l.strip().startswith('\#')\]),  
                "comment\_lines": len(\[l for l in lines if l.strip().startswith('\#')\]),  
                "file\_size\_bytes": path.stat().st\_size,  
            }  
            return {"status": "success", "results": stats}  
        except Exception as e:  
            return {"status": "error", "message": f"Failed to read file stats: {e}"}

    if request.mode \== "inspect":  
        try:  
            source \= path.read\_text("utf-8")  
            tree \= ast.parse(source)  
            if request.function\_name:  
                for node in ast.walk(tree):  
                    if isinstance(node, ast.FunctionDef) and node.name \== request.function\_name:  
                        args \= \[a.arg for a in node.args.args\]  
                        docstring \= ast.get\_docstring(node)  
                        return {"status": "success", "type": "function", "name": node.name, "args": args, "docstring": docstring}  
                return {"status": "not\_found", "message": f"Function '{request.function\_name}' not found."}  
            \# Add class inspection logic here if needed  
            return {"status": "error", "message": "Inspection target (e.g., function\_name) not specified."}  
        except Exception as e:  
            return {"status": "error", "message": f"Failed to inspect file: {e}"}

    return {"status": "error", "message": f"Invalid mode '{request.mode}'."}

---

### **B) The /snippet Tool ✂️**

This tool is your high-precision instrument for extracting *exactly* the code you need to see, whether it's the body of a function or a specific range of lines.  
Here is the implementation:  
Python  
\# Add this with your other Pydantic models  
class SnippetRequest(BaseModel):  
    file\_path: str \= Field(..., description="The absolute path to the file.")  
    mode: Literal\["function", "class", "lines"\] \= Field(..., description="The extraction mode.")  
    \# For function/class mode  
    name: Optional\[str\] \= Field(None, description="The name of the function or class to extract.")  
    \# For lines mode  
    start\_line: Optional\[int\] \= Field(None, description="The starting line number (1-indexed).")  
    end\_line: Optional\[int\] \= Field(None, description="The ending line number (inclusive).")

\# Add this with your other tool functions  
@mcp.tool()  
@tool\_timeout\_and\_errors(timeout=10)  
def get\_snippet(request: SnippetRequest) \-\> dict:  
    """  
    Extracts specific code snippets from a file.

    Use this tool to get the exact source code of a function, class, or a  
    specific range of lines from a file.

    Args:  
        request (SnippetRequest):  
            \- file\_path (str): The absolute path to the file.  
            \- mode (str): 'function', 'class', or 'lines'.  
            \- name (str, optional): The name of the function/class to extract.  
            \- start\_line (int, optional): The starting line for 'lines' mode.  
            \- end\_line (int, optional): The ending line for 'lines' mode.

    Returns:  
        dict: A dictionary containing the requested code snippet.  
    """  
    logger.info(f"\[get\_snippet\] mode='{request.mode}' file='{request.file\_path}'")  
    path \= pathlib.Path(request.file\_path)  
    if not \_is\_safe\_path(path) or not path.is\_file():  
        return {"status": "error", "message": "Invalid or unsafe file path."}

    try:  
        source \= path.read\_text("utf-8")  
        if request.mode in \["function", "class"\]:  
            if not request.name:  
                return {"status": "error", "message": "Missing 'name' for function/class mode."}  
            tree \= ast.parse(source)  
            target\_node\_type \= ast.FunctionDef if request.mode \== "function" else ast.ClassDef  
            for node in ast.walk(tree):  
                if isinstance(node, target\_node\_type) and node.name \== request.name:  
                    snippet \= ast.get\_source\_segment(source, node)  
                    return {"status": "success", "snippet": snippet}  
            return {"status": "not\_found", "message": f"{request.mode.capitalize()} '{request.name}' not found."}

        if request.mode \== "lines":  
            if not request.start\_line or not request.end\_line:  
                return {"status": "error", "message": "Missing 'start\_line' or 'end\_line' for lines mode."}  
            lines \= source.splitlines()  
            \# Adjust for 0-based indexing and ensure slice is within bounds  
            start \= max(0, request.start\_line \- 1)  
            end \= min(len(lines), request.end\_line)  
            if start \>= end:  
                return {"status": "error", "message": "Invalid line range."}  
            snippet \= "\\n".join(lines\[start:end\])  
            return {"status": "success", "snippet": snippet}

    except Exception as e:  
        return {"status": "error", "message": f"Failed to extract snippet: {e}"}

    return {"status": "error", "message": f"Invalid mode '{request.mode}'."}

---

### **Regarding the /regex Tool**

You are correct to question this. The existing regex search within the search multitool is already quite capable. My suggestion for a separate tool was redundant.  
The primary difference in my proposal was an extract\_pattern mode that would return *only the captured groups* from the regex, rather than the full matching line. This can be useful for pulling out specific values (e.g., all URLs or version numbers) from files.  
Instead of a new tool, we can simply **enhance the existing regex search**. You could modify the \_search\_by\_regex function to accept an optional boolean parameter, like extract\_groups: bool \= False.

* If extract\_groups is False (the default), it behaves exactly as it does now.  
* If extract\_groups is True, it changes its output to return a list of the captured groups for each match.

This approach improves the existing tool without adding unnecessary complexity, keeping your toolset lean and powerful. It's a great idea to keep on the back burner for when you need that more precise extraction capability.  
