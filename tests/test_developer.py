import io
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


from termux_mcp.handlers.developer import (
    _collect_project_source_files,
    _resolve_trace_import,
    _trace_imports,
)


class FakeHandler:
    def __init__(self):
        self.status = None
        self.headers_ended = False
        self.wfile = io.BytesIO()

    def send_response(self, status):
        self.status = status

    def end_headers(self):
        self.headers_ended = True


class PythonImportTracingTests(unittest.TestCase):

    def make_project(self):
        temp_dir = tempfile.TemporaryDirectory()
        root = Path(temp_dir.name)

        package = root / "package"
        tests = root / "tests"

        package.mkdir()
        tests.mkdir()

        (package / "__init__.py").write_text(
            "",
            encoding="utf-8",
        )

        (package / "target.py").write_text(
            "class Target:\n"
            "    pass\n",
            encoding="utf-8",
        )

        (package / "cli.py").write_text(
            "from .target import Target\n",
            encoding="utf-8",
        )

        (tests / "test_target.py").write_text(
            "from package.target import Target\n",
            encoding="utf-8",
        )

        return temp_dir, root

    def test_trace_imports_detects_absolute_python_import(self):
        source = (
            "from package.target import Target\n"
            "import package.target\n"
        )

        imports = _trace_imports(
            source,
            Path("tests/test_target.py"),
        )

        self.assertIn("package.target", imports)

    def test_trace_imports_detects_relative_python_import(self):
        source = (
            "from .target import Target\n"
            "from ..shared import helper\n"
        )

        imports = _trace_imports(
            source,
            Path("package/cli.py"),
        )

        self.assertIn(".target", imports)
        self.assertIn("..shared", imports)

    def test_python_module_resolves_to_py_file(self):
        temp_dir, root = self.make_project()

        try:
            source_file = root / "tests" / "test_target.py"

            resolved = _resolve_trace_import(
                root,
                source_file,
                "package.target",
            )

            self.assertEqual(
                resolved,
                root / "package" / "target.py",
            )
        finally:
            temp_dir.cleanup()

    def test_python_package_resolves_to_init_file(self):
        temp_dir, root = self.make_project()

        try:
            source_file = root / "tests" / "test_target.py"

            resolved = _resolve_trace_import(
                root,
                source_file,
                "package",
            )

            self.assertEqual(
                resolved,
                root / "package" / "__init__.py",
            )
        finally:
            temp_dir.cleanup()

    def test_collect_project_source_files_includes_python(self):
        temp_dir, root = self.make_project()

        try:
            files = _collect_project_source_files(root)

            relative = {
                file.relative_to(root).as_posix()
                for file in files
            }

            self.assertIn("package/target.py", relative)
            self.assertIn("package/cli.py", relative)
            self.assertIn("tests/test_target.py", relative)
        finally:
            temp_dir.cleanup()


class PythonImportRegressionTests(unittest.TestCase):

    def test_real_project_impact_finds_python_dependents(self):
        from termux_agent.mcp_client import MCPClient

        client = MCPClient()

        try:
            client.connect()

            result = client.call_tool(
                "impact_project",
                {
                    "path": "termux_agent/agent.py",
                },
            )

            payload = json.loads(
                result["content"][0]["text"]
            )

            self.assertEqual(
                payload["target"],
                "termux_agent/agent.py",
            )

            self.assertGreaterEqual(
                payload["dependent_count"],
                3,
            )

            dependent_files = {
                item["file"]
                for item in payload["dependents"]
            }

            self.assertIn(
                "tests/test_agent.py",
                dependent_files,
            )

            self.assertIn(
                "tests/test_provider.py",
                dependent_files,
            )

            self.assertIn(
                "tests/test_agent_integration.py",
                dependent_files,
            )

        finally:
            client.close()


if __name__ == "__main__":
    unittest.main()
