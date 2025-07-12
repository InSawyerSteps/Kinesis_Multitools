import pathlib

import importlib.util
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

def load_module():
    spec = importlib.util.spec_from_file_location("regex_local", REPO_ROOT / "regex.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.RegexSearchRequest.model_rebuild()
    return module


def test_simple_regex_search(tmp_path: pathlib.Path):
    file_py = tmp_path / "test.py"
    file_txt = tmp_path / "notes.txt"
    file_py.write_text("""\n# TODO fix this\ndef foo():\n    pass\n""")
    file_txt.write_text("numbers: 12345\n")

    import subprocess, sys, json, textwrap
    script = textwrap.dedent(
        f"""
        import pathlib, json, importlib.util, sys
        spec = importlib.util.spec_from_file_location('regex_local', r'{REPO_ROOT}/regex.py')
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        mod.RegexSearchRequest.model_rebuild()
        path = pathlib.Path(r'{tmp_path}')
        req = mod.RegexSearchRequest(query=r'TODO\\s+fix', project_path=path, extensions=['py'], ignore_case=True)
        result = mod._search_by_regex(req)
        print(json.dumps(result))
        """
    )
    proc = subprocess.run([sys.executable, '-'], input=script, text=True, capture_output=True, check=True)
    result = json.loads(proc.stdout.strip())
    assert result['status'] == 'success'
    assert len(result['results']) == 1
    match = result['results'][0]
    assert match['file_path'].endswith('test.py')
    assert match['line_number'] == 2


def test_invalid_pattern(tmp_path: pathlib.Path):
    regex = load_module()
    RegexSearchRequest = regex.RegexSearchRequest
    _search_by_regex = regex._search_by_regex
    req = RegexSearchRequest(
        query=r"(unclosed",
        project_path=tmp_path,
    )
    result = _search_by_regex(req)
    assert result["status"] == "error"
    assert "Invalid regex" in result["message"]
