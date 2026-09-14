import json
import os
import re
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from http.server import BaseHTTPRequestHandler


MAX_FILES = 2000
MAX_DEPTH = 5
MAX_PROJECT_ROOTS = 20

IGNORED_DIRS = {
    ".git",
    ".next",
    ".nuxt",
    ".svelte-kit",
    "node_modules",
    "venv",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".tox",
    "dist",
    "build",
    "coverage",
    ".turbo",
    ".cache",
    ".vercel",
}

CONFIG_FILES = {
    "package.json": "Node.js / JavaScript project",
    "pyproject.toml": "Python project",
    "requirements.txt": "Python dependencies",
    "Pipfile": "Python project",
    "setup.py": "Python project",
    "Cargo.toml": "Rust project",
    "go.mod": "Go project",
    "pom.xml": "Java Maven project",
    "build.gradle": "Java/Gradle project",
    "composer.json": "PHP project",
    "Gemfile": "Ruby project",
    "pubspec.yaml": "Dart/Flutter project",
}

FRAMEWORK_FILES = {
    "next.config.js": "Next.js",
    "next.config.mjs": "Next.js",
    "next.config.ts": "Next.js",
    "vite.config.js": "Vite",
    "vite.config.ts": "Vite",
    "angular.json": "Angular",
    "nuxt.config.ts": "Nuxt",
    "nuxt.config.js": "Nuxt",
    "svelte.config.js": "SvelteKit",
    "astro.config.mjs": "Astro",
    "manage.py": "Django",
    "artisan": "Laravel",
}

ENTRY_FILES = {
    "main.py",
    "app.py",
    "server.py",
    "index.py",
    "main.js",
    "index.js",
    "server.js",
    "app.js",
    "main.ts",
    "index.ts",
    "server.ts",
    "app.ts",
    "main.tsx",
    "index.tsx",
    "App.tsx",
    "main.go",
    "main.rs",
}

PROJECT_MARKERS = set(CONFIG_FILES) | {
    "package-lock.json",
    "pnpm-lock.yaml",
    "yarn.lock",
    "bun.lock",
    "bun.lockb",
    "uv.lock",
    "poetry.lock",
    "Pipfile.lock",
    "Cargo.lock",
    "go.sum",
}

IMPORTANT_NAMES = {
    "tsconfig.json",
    "jsconfig.json",
    "tailwind.config.js",
    "tailwind.config.ts",
    "tailwind.config.mjs",
    "docker-compose.yml",
    "docker-compose.yaml",
    "Dockerfile",
    ".env.example",
    "README.md",
}


def _resolve_project(raw_path: str) -> Path:
    value = str(raw_path or ".").strip() or "."

    path = Path(value).expanduser()

    if not path.is_absolute():
        path = Path.cwd() / path

    return path.resolve()


def _safe_relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _run_git(root: Path, args: list[str]) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _git_info(root: Path) -> dict:
    inside = _run_git(
        root,
        ["rev-parse", "--is-inside-work-tree"],
    )

    if inside != "true":
        return {
            "is_repository": False,
            "branch": None,
            "clean": None,
            "status": [],
        }

    branch = _run_git(
        root,
        ["branch", "--show-current"],
    )

    status_raw = _run_git(
        root,
        ["status", "--short"],
    )

    status = [
        line
        for line in status_raw.splitlines()
        if line.strip()
    ]

    return {
        "is_repository": True,
        "branch": branch or None,
        "clean": not status,
        "status": status[:100],
    }


def _detect_languages(files: list[Path]) -> tuple[list[str], list[str]]:
    extensions = {
        ".py": "Python",
        ".ts": "TypeScript",
        ".tsx": "TypeScript/React",
        ".js": "JavaScript",
        ".jsx": "JavaScript/React",
        ".go": "Go",
        ".rs": "Rust",
        ".java": "Java",
        ".kt": "Kotlin",
        ".dart": "Dart",
        ".php": "PHP",
        ".rb": "Ruby",
        ".c": "C",
        ".cpp": "C++",
    }

    counts = {}

    for path in files:
        language = extensions.get(path.suffix.lower())

        if language:
            counts[language] = counts.get(language, 0) + 1

    if not counts:
        return [], []

    total = sum(counts.values())

    ordered = [
        language
        for language, _ in sorted(
            counts.items(),
            key=lambda item: (-item[1], item[0]),
        )
    ]

    primary = []
    auxiliary = []

    for language in ordered:
        count = counts[language]

        if count / total >= 0.10:
            primary.append(language)
        else:
            auxiliary.append(language)

    return primary, auxiliary


def _read_json_file(path: Path) -> dict:
    try:
        data = json.loads(
            path.read_text(
                encoding="utf-8",
                errors="ignore",
            )
        )

        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _detect_package_manager(root: Path) -> list[str]:
    managers = []

    if (root / "package-lock.json").exists():
        managers.append("npm")
    elif (root / "package.json").exists():
        managers.append("npm")

    if (root / "pnpm-lock.yaml").exists():
        managers.append("pnpm")

    if (root / "yarn.lock").exists():
        managers.append("yarn")

    if (root / "bun.lockb").exists() or (root / "bun.lock").exists():
        managers.append("bun")

    if (root / "uv.lock").exists():
        managers.append("uv")

    if (root / "poetry.lock").exists():
        managers.append("poetry")

    if (root / "Pipfile.lock").exists():
        managers.append("pipenv")

    if (
        (root / "requirements.txt").exists()
        or (root / "pyproject.toml").exists()
    ):
        if "uv" not in managers and "poetry" not in managers:
            managers.append("pip")

    if (root / "Cargo.lock").exists() or (root / "Cargo.toml").exists():
        managers.append("cargo")

    if (root / "go.sum").exists() or (root / "go.mod").exists():
        managers.append("go")

    return managers


def _read_package_scripts(root: Path) -> dict:
    package_file = root / "package.json"

    if not package_file.is_file():
        return {}

    data = _read_json_file(package_file)
    scripts = data.get("scripts", {})

    if not isinstance(scripts, dict):
        return {}

    return {
        str(name): str(command)
        for name, command in scripts.items()
    }


def _detect_framework(root: Path) -> list[str]:
    frameworks = []

    for filename, framework in FRAMEWORK_FILES.items():
        if (root / filename).exists():
            if framework not in frameworks:
                frameworks.append(framework)

    package_file = root / "package.json"

    if package_file.is_file():
        data = _read_json_file(package_file)

        dependencies = {}
        dependencies.update(data.get("dependencies", {}))
        dependencies.update(data.get("devDependencies", {}))

        known = {
            "next": "Next.js",
            "react": "React",
            "react-native": "React Native",
            "express": "Express",
            "fastify": "Fastify",
            "nestjs": "NestJS",
            "@nestjs/core": "NestJS",
            "vue": "Vue",
            "svelte": "Svelte",
            "@angular/core": "Angular",
            "angular": "Angular",
            "astro": "Astro",
        }

        for package_name, framework in known.items():
            if package_name in dependencies and framework not in frameworks:
                frameworks.append(framework)

    return frameworks


def _collect_files(root: Path) -> tuple[list[Path], bool]:
    files = []
    truncated = False

    for current, dirs, filenames in os.walk(root):
        current_path = Path(current)

        try:
            relative = current_path.relative_to(root)
            depth = len(relative.parts)
        except ValueError:
            continue

        if depth >= MAX_DEPTH:
            dirs[:] = []

        dirs[:] = [
            name
            for name in dirs
            if name not in IGNORED_DIRS
            and not name.startswith(".")
        ]

        for filename in filenames:
            path = current_path / filename

            if path.is_symlink():
                continue

            files.append(path)

            if len(files) >= MAX_FILES:
                truncated = True
                return files, truncated

    return files, truncated


def _find_project_roots(root: Path, files: list[Path]) -> list[Path]:
    candidates = []

    for path in files:
        if path.name not in PROJECT_MARKERS:
            continue

        project_root = path.parent

        try:
            relative = project_root.relative_to(root)
        except ValueError:
            continue

        if len(relative.parts) > 3:
            continue

        if project_root not in candidates:
            candidates.append(project_root)

        if len(candidates) >= MAX_PROJECT_ROOTS:
            break

    # If the supplied directory itself is a project, include it.
    root_is_project = any(
        (root / marker).exists()
        for marker in PROJECT_MARKERS
    )

    if root_is_project and root not in candidates:
        candidates.insert(0, root)

    return sorted(
        candidates,
        key=lambda path: (
            len(path.relative_to(root).parts),
            str(path),
        ),
    )




def _important_files(root: Path, files: list[Path]) -> list[str]:
    result = []

    for path in files:
        relative = _safe_relative(path, root)

        if path.name in CONFIG_FILES:
            result.append(relative)
            continue

        if path.name in FRAMEWORK_FILES:
            result.append(relative)
            continue

        if path.name in ENTRY_FILES:
            result.append(relative)
            continue

        if path.name in IMPORTANT_NAMES:
            result.append(relative)

    return sorted(set(result))[:200]


def _directories(root: Path) -> list[str]:
    result = []

    try:
        for child in root.iterdir():
            if not child.is_dir():
                continue

            if child.name in IGNORED_DIRS:
                continue

            if child.name.startswith("."):
                continue

            result.append(child.name)

    except OSError:
        pass

    return sorted(result)


def _project_summary(
    project_root: Path,
    repository_root: Path,
    files: list[Path],
) -> dict:
    relative_root = _safe_relative(
        project_root,
        repository_root,
    )

    # Nested projects need their own file scan so they do not inherit
    # languages or files from the parent repository.
    if project_root == repository_root:
        project_files = [
            path
            for path in files
            if path.is_file()
        ]
    else:
        project_files, _ = _collect_files(project_root)

    languages, auxiliary_languages = _detect_languages(project_files)
    frameworks = _detect_framework(project_root)
    package_managers = _detect_package_manager(project_root)

    config_files = []

    for filename in CONFIG_FILES:
        if (project_root / filename).exists():
            config_files.append(filename)

    for filename in FRAMEWORK_FILES:
        if (project_root / filename).exists():
            config_files.append(filename)

    for filename in IMPORTANT_NAMES:
        if (project_root / filename).exists():
            config_files.append(filename)

    entries = sorted(
        _safe_relative(path, project_root)
        for path in project_files
        if path.name in ENTRY_FILES
    )[:100]

    source_extensions = {
        ".py",
        ".ts",
        ".tsx",
        ".js",
        ".jsx",
        ".mjs",
        ".cjs",
    }

    source_directories = []

    for directory in _directories(project_root):
        directory_path = project_root / directory

        try:
            has_source = any(
                child.is_file()
                and child.suffix.lower() in source_extensions
                for child in directory_path.rglob("*")
                if not any(
                    part in IGNORED_DIRS
                    for part in child.relative_to(project_root).parts
                )
            )
        except (OSError, RuntimeError, ValueError):
            has_source = False

        if has_source:
            source_directories.append(directory)

    return {
        "path": str(project_root),
        "relative_path": relative_root,
        "languages": languages,
        "auxiliary_languages": auxiliary_languages,
        "frameworks": frameworks,
        "package_managers": package_managers,
        "config_files": sorted(set(config_files)),
        "source_directories": sorted(source_directories),
        "entry_points": entries,
        "package_scripts": _read_package_scripts(project_root),
        "file_count": len(project_files),
    }


SEARCH_MAX_RESULTS = 100
SEARCH_MAX_FILE_BYTES = 2 * 1024 * 1024


def _normalize_extensions(values) -> set[str]:
    if not values:
        return set()

    if isinstance(values, str):
        values = [values]

    result = set()

    for value in values:
        value = str(value).strip().lower()

        if not value:
            continue

        if not value.startswith("."):
            value = "." + value

        result.add(value)

    return result


def _search_project_files(
    root: Path,
    query: str,
    extensions: set[str],
    max_results: int,
) -> list[dict]:
    results = []
    query_lower = query.lower()

    files, _ = _collect_files(root)

    for path in files:
        if len(results) >= max_results:
            break

        if not path.is_file():
            continue

        if extensions and path.suffix.lower() not in extensions:
            continue

        try:
            if path.stat().st_size > SEARCH_MAX_FILE_BYTES:
                continue

            content = path.read_text(
                encoding="utf-8",
                errors="ignore",
            )
        except (OSError, UnicodeError):
            continue

        for line_number, line in enumerate(
            content.splitlines(),
            start=1,
        ):
            if query_lower not in line.lower():
                continue

            results.append({
                "file": _safe_relative(path, root),
                "line": line_number,
                "text": line.strip(),
            })

            if len(results) >= max_results:
                break

    return results



READ_PROJECT_MAX_FILE_BYTES = 2 * 1024 * 1024


TRACE_PROJECT_MAX_FILES = 100
TRACE_PROJECT_MAX_DEPTH = 5
TRACE_PROJECT_MAX_FILE_BYTES = 2 * 1024 * 1024


def _trace_imports(source: str, file_path: Path | None = None) -> list[str]:
    """Extract likely local/module import targets from source code."""
    patterns = [
        r'import\s+(?:[\s\S]*?\s+from\s+)?["\']([^"\']+)["\']',
        r'export\s+(?:[\s\S]*?\s+from\s+)?["\']([^"\']+)["\']',
        r'require\s*\(\s*["\']([^"\']+)["\']\s*\)',
        r'import\s*\(\s*["\']([^"\']+)["\']\s*\)',
    ]

    python_patterns = [
        r'^\s*from\s+(\.+[A-Za-z_][A-Za-z0-9_.]*|[A-Za-z_][A-Za-z0-9_.]*)\s+import\s+',
        r'^\s*import\s+([A-Za-z_][A-Za-z0-9_.]*)',
    ]

    imports = []

    for pattern in patterns:
        try:
            matches = re.findall(pattern, source)
        except Exception:
            matches = []

        for value in matches:
            value = value.strip()

            if value and value not in imports:
                imports.append(value)

    if file_path is None or file_path.suffix.lower() == ".py":
        for pattern in python_patterns:
            try:
                matches = re.findall(pattern, source, re.MULTILINE)
            except Exception:
                matches = []

            for value in matches:
                value = value.strip()

                if value and value not in imports:
                    imports.append(value)

    return imports


def _trace_tsconfig_paths(root: Path) -> tuple[str, dict[str, list[str]]]:
    """Read TypeScript path aliases when available."""
    candidates = [
        root / "tsconfig.json",
        root / "jsconfig.json",
    ]

    for config_path in candidates:
        if not config_path.is_file():
            continue

        data = _read_json_file(config_path)

        compiler = data.get("compilerOptions", {})
        if not isinstance(compiler, dict):
            continue

        base_url = str(compiler.get("baseUrl", "."))
        paths = compiler.get("paths", {})

        if isinstance(paths, dict):
            clean_paths = {}

            for key, values in paths.items():
                if isinstance(values, list):
                    clean_paths[str(key)] = [
                        str(value)
                        for value in values
                    ]

            return base_url, clean_paths

    return ".", {}


def _resolve_trace_import(
    root: Path,
    source_file: Path,
    import_name: str,
) -> Path | None:
    """Resolve a local JS/TS import to an actual file."""
    if not (
        import_name.startswith(".")
        or import_name.startswith("/")
        or import_name.startswith("@/")
        or (
            source_file.suffix.lower() == ".py"
            and re.fullmatch(
                r"[A-Za-z_][A-Za-z0-9_.]*",
                import_name,
            )
        )
    ):
        return None

    base_url, aliases = _trace_tsconfig_paths(root)

    candidates = []

    def add_candidates(base: Path) -> None:
        suffixes = [
            "",
            ".py",
            ".ts",
            ".tsx",
            ".js",
            ".jsx",
            ".mjs",
            ".cjs",
        ]

        for suffix in suffixes:
            candidates.append(Path(str(base) + suffix))

        for name in [
            "__init__.py",
            "index.ts",
            "index.tsx",
            "index.js",
            "index.jsx",
            "index.mjs",
            "index.cjs",
        ]:
            candidates.append(base / name)

    if import_name.startswith("@/"):
        add_candidates(root / base_url / import_name[2:])
    elif import_name.startswith("."):
        if source_file.suffix.lower() == ".py":
            current = source_file.parent

            leading_dots = len(import_name) - len(import_name.lstrip("."))
            relative_name = import_name[leading_dots:]

            # In Python, one leading dot means the current package,
            # two means the parent package, three means two levels up, etc.
            for _ in range(max(0, leading_dots - 1)):
                if current != root:
                    current = current.parent

            if relative_name:
                add_candidates(current / relative_name.replace(".", "/"))
        else:
            add_candidates(source_file.parent / import_name)
    elif import_name.startswith("/"):
        add_candidates(root / import_name.lstrip("/"))
    elif source_file.suffix.lower() == ".py":
        module_parts = import_name.split(".")

        for index in range(len(module_parts), 0, -1):
            module_root = root.joinpath(*module_parts[:index])

            remainder = module_parts[index:]

            if remainder:
                add_candidates(
                    module_root.joinpath(*remainder)
                )
            else:
                add_candidates(module_root)

            add_candidates(module_root)

    else:
        for alias, targets in aliases.items():
            if alias.endswith("/*"):
                prefix = alias[:-2]

                if import_name.startswith(prefix):
                    remainder = import_name[len(prefix):].lstrip("/")

                    for target in targets:
                        target = target.replace("/*", "")
                        add_candidates(
                            root / base_url / target / remainder
                        )

    for candidate in candidates:
        try:
            resolved = candidate.resolve()

            if resolved.is_file() and resolved.is_relative_to(root):
                return resolved

        except (OSError, RuntimeError, ValueError):
            continue

    return None


def _trace_project_file(
    root: Path,
    file_path: Path,
    depth: int,
    visited: set[Path],
    nodes: list[dict],
) -> None:
    if depth > TRACE_PROJECT_MAX_DEPTH:
        return

    if file_path in visited:
        return

    if len(nodes) >= TRACE_PROJECT_MAX_FILES:
        return

    visited.add(file_path)

    try:
        source = file_path.read_text(
            encoding="utf-8",
            errors="ignore",
        )
    except OSError:
        return

    imports = _trace_imports(source)

    relationships = []

    for import_name in imports:
        resolved = _resolve_trace_import(
            root,
            file_path,
            import_name,
        )

        relationship = {
            "import": import_name,
            "resolved": resolved is not None,
        }

        if resolved is not None:
            relationship["file"] = _safe_relative(
                resolved,
                root,
            )

        relationships.append(relationship)

    node = {
        "file": _safe_relative(file_path, root),
        "depth": depth,
        "imports": relationships,
    }

    nodes.append(node)

    for relationship in relationships:
        resolved_name = relationship.get("file")

        if not resolved_name:
            continue

        resolved_path = root / resolved_name

        _trace_project_file(
            root,
            resolved_path,
            depth + 1,
            visited,
            nodes,
        )


def handle_trace_project(
    handler: "BaseHTTPRequestHandler",
    data: dict,
) -> None:
    raw_path = data.get("path")

    if not raw_path:
        handler.send_response(400)
        handler.end_headers()
        handler.wfile.write(
            json.dumps({
                "error": "File path is required."
            }).encode()
        )
        return

    try:
        path = _resolve_project(raw_path)
    except (OSError, RuntimeError) as exc:
        handler.send_response(400)
        handler.end_headers()
        handler.wfile.write(
            json.dumps({
                "error": f"Invalid file path: {exc}"
            }).encode()
        )
        return

    if not path.exists():
        handler.send_response(404)
        handler.end_headers()
        handler.wfile.write(
            json.dumps({
                "error": f"Path does not exist: {path}"
            }).encode()
        )
        return

    if not path.is_file():
        handler.send_response(400)
        handler.end_headers()
        handler.wfile.write(
            json.dumps({
                "error": f"Path is not a file: {path}"
            }).encode()
        )
        return

    root = path.parent

    while root != root.parent:
        if any(
            (root / marker).exists()
            for marker in PROJECT_MARKERS
        ):
            break

        root = root.parent

    nodes = []
    visited = set()

    _trace_project_file(
        root,
        path,
        0,
        visited,
        nodes,
    )

    handler.send_response(200)
    handler.send_header(
        "Content-Type",
        "application/json; charset=utf-8",
    )
    handler.end_headers()
    handler.wfile.write(
        json.dumps({
            "root": str(root),
            "entry": _safe_relative(path, root),
            "nodes": nodes,
            "node_count": len(nodes),
            "truncated": len(nodes) >= TRACE_PROJECT_MAX_FILES,
            "max_depth": TRACE_PROJECT_MAX_DEPTH,
        }).encode()
    )


def _collect_project_source_files(root: Path) -> list[Path]:
    files = []

    source_extensions = {
        ".py",
        ".ts",
        ".tsx",
        ".js",
        ".jsx",
        ".mjs",
        ".cjs",
    }

    try:
        for path in root.rglob("*"):
            if not path.is_file():
                continue

            if any(
                part in IGNORED_DIRS
                for part in path.relative_to(root).parts
            ):
                continue

            if path.suffix.lower() not in source_extensions:
                continue

            try:
                if path.stat().st_size > TRACE_PROJECT_MAX_FILE_BYTES:
                    continue
            except OSError:
                continue

            files.append(path)

            if len(files) >= TRACE_PROJECT_MAX_FILES:
                break

    except OSError:
        pass

    return files


def _find_project_root(path: Path) -> Path:
    root = path.parent

    while root != root.parent:
        if any(
            (root / marker).exists()
            for marker in PROJECT_MARKERS
        ):
            return root

        root = root.parent

    return path.parent


def handle_impact_project(
    handler: "BaseHTTPRequestHandler",
    data: dict,
) -> None:
    raw_path = data.get("path")

    if not raw_path:
        handler.send_response(400)
        handler.end_headers()
        handler.wfile.write(
            json.dumps({
                "error": "File path is required."
            }).encode()
        )
        return

    try:
        target = _resolve_project(raw_path)
    except (OSError, RuntimeError) as exc:
        handler.send_response(400)
        handler.end_headers()
        handler.wfile.write(
            json.dumps({
                "error": f"Invalid file path: {exc}"
            }).encode()
        )
        return

    if not target.exists():
        handler.send_response(404)
        handler.end_headers()
        handler.wfile.write(
            json.dumps({
                "error": f"Path does not exist: {target}"
            }).encode()
        )
        return

    if not target.is_file():
        handler.send_response(400)
        handler.end_headers()
        handler.wfile.write(
            json.dumps({
                "error": f"Path is not a file: {target}"
            }).encode()
        )
        return

    root = _find_project_root(target)

    if not target.is_relative_to(root):
        handler.send_response(400)
        handler.end_headers()
        handler.wfile.write(
            json.dumps({
                "error": "Target file is outside the detected project root."
            }).encode()
        )
        return

    target_relative = _safe_relative(target, root)

    # ---------------------------------------------------------
    # Forward dependencies
    # ---------------------------------------------------------

    forward_nodes = []
    forward_visited = set()

    _trace_project_file(
        root,
        target,
        0,
        forward_visited,
        forward_nodes,
    )

    dependencies = []

    for node in forward_nodes:
        file_name = node.get("file")

        if not file_name or file_name == target_relative:
            continue

        dependencies.append({
            "file": file_name,
            "depth": node.get("depth", 0),
        })

    # ---------------------------------------------------------
    # Reverse dependencies
    #
    # Scan the project and resolve every local import.
    # ---------------------------------------------------------

    dependents = []
    scanned_files = 0
    scan_truncated = False

    source_files = _collect_project_source_files(root)

    if len(source_files) >= TRACE_PROJECT_MAX_FILES:
        scan_truncated = True

    for source_file in source_files:
        if source_file == target:
            continue

        scanned_files += 1

        try:
            source = source_file.read_text(
                encoding="utf-8",
                errors="ignore",
            )
        except OSError:
            continue

        relationships = _trace_imports(source, source_file)

        for relationship in relationships:
            if isinstance(relationship, str):
                import_name = relationship
            elif isinstance(relationship, dict):
                import_name = relationship.get("import")
            else:
                continue

            if not import_name:
                continue

            resolved = _resolve_trace_import(
                root,
                source_file,
                import_name,
            )

            if resolved is None:
                continue

            try:
                same_target = resolved.resolve() == target.resolve()
            except OSError:
                same_target = resolved == target

            if not same_target:
                continue

            dependents.append({
                "file": _safe_relative(source_file, root),
                "import": import_name,
            })

            break

    # Remove accidental duplicates while preserving order.
    seen = set()
    unique_dependents = []

    for item in dependents:
        key = item["file"]

        if key in seen:
            continue

        seen.add(key)
        unique_dependents.append(item)

    # ---------------------------------------------------------
    # Response
    # ---------------------------------------------------------

    result = {
        "root": str(root),
        "target": target_relative,

        "dependencies": dependencies,
        "dependency_count": len(dependencies),

        "dependents": unique_dependents,
        "dependent_count": len(unique_dependents),

        "scanned_files": scanned_files,
        "truncated": scan_truncated,

        "impact": {
            "direct_dependencies": len(dependencies),
            "direct_dependents": len(unique_dependents),
            "potentially_affected_files": len(unique_dependents),
        },
    }

    handler.send_response(200)
    handler.send_header(
        "Content-Type",
        "application/json; charset=utf-8",
    )
    handler.end_headers()
    handler.wfile.write(
        json.dumps(result).encode()
    )


EDIT_PROJECT_MAX_FILE_BYTES = 2 * 1024 * 1024
EDIT_PROJECT_MAX_OPERATIONS = 50
EDIT_PROJECT_MAX_CONTENT_BYTES = 4 * 1024 * 1024


def _edit_project_error(handler, status: int, message: str, **extra) -> None:
    result = {
        "success": False,
        "error": message,
        **extra,
    }

    body = json.dumps(
        result,
        indent=2,
        ensure_ascii=False,
    ).encode("utf-8")

    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _edit_project_response(handler, result: dict, status: int = 200) -> None:
    body = json.dumps(
        result,
        indent=2,
        ensure_ascii=False,
    ).encode("utf-8")

    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _edit_project_root(path: Path) -> Path:
    if path.is_dir():
        candidate = path
    else:
        candidate = path.parent

    return _find_project_root(candidate / "__placeholder__")


def _edit_project_resolve_inside(root: Path, raw_path) -> Path:
    value = str(raw_path or "").strip()

    if not value:
        raise ValueError("File path is required.")

    path = Path(value).expanduser()

    if not path.is_absolute():
        path = root / path

    resolved = path.resolve()

    if not resolved.is_relative_to(root.resolve()):
        raise ValueError(
            "Path is outside the detected project root."
        )

    return resolved


def _edit_project_read(path: Path) -> str:
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise OSError(
            f"Could not inspect file: {exc}"
        ) from exc

    if size > EDIT_PROJECT_MAX_FILE_BYTES:
        raise ValueError(
            f"File is too large to edit: {size} bytes "
            f"(maximum {EDIT_PROJECT_MAX_FILE_BYTES})."
        )

    try:
        return path.read_text(
            encoding="utf-8",
            errors="replace",
        )
    except OSError as exc:
        raise OSError(
            f"Could not read file: {exc}"
        ) from exc


def _edit_project_validate_content(content: str) -> None:
    size = len(content.encode("utf-8"))

    if size > EDIT_PROJECT_MAX_CONTENT_BYTES:
        raise ValueError(
            f"Resulting file is too large: {size} bytes "
            f"(maximum {EDIT_PROJECT_MAX_CONTENT_BYTES})."
        )


def _edit_project_apply_operation(
    root: Path,
    files: dict[Path, str | None],
    operation: dict,
) -> dict:
    if not isinstance(operation, dict):
        raise ValueError("Each operation must be an object.")

    kind = str(operation.get("operation", "")).strip().lower()

    if kind not in {
        "replace",
        "insert",
        "delete",
        "create",
        "delete_file",
        "move",
    }:
        raise ValueError(
            f"Unsupported operation: {kind or '<missing>'}."
        )

    raw_path = operation.get("path")

    if not raw_path:
        raise ValueError(
            f"Operation '{kind}' requires a path."
        )

    target = _edit_project_resolve_inside(root, raw_path)

    def current_content(file_path: Path) -> str:
        if file_path in files:
            value = files[file_path]

            if value is None:
                raise ValueError(
                    f"File is already deleted in this edit: "
                    f"{_safe_relative(file_path, root)}."
                )

            return value

        if not file_path.exists():
            raise FileNotFoundError(
                f"File does not exist: "
                f"{_safe_relative(file_path, root)}."
            )

        if not file_path.is_file():
            raise ValueError(
                f"Path is not a file: "
                f"{_safe_relative(file_path, root)}."
            )

        value = _edit_project_read(file_path)
        files[file_path] = value
        return value

    if kind == "replace":
        old_text = operation.get("old_text")
        new_text = operation.get("new_text")

        if not isinstance(old_text, str):
            raise ValueError(
                "Replace requires string 'old_text'."
            )

        if not isinstance(new_text, str):
            raise ValueError(
                "Replace requires string 'new_text'."
            )

        content = current_content(target)
        count = content.count(old_text)

        if count == 0:
            raise ValueError(
                "Replacement target was not found."
            )

        expected_count = operation.get("expected_count", 1)

        try:
            expected_count = int(expected_count)
        except (TypeError, ValueError):
            raise ValueError(
                "'expected_count' must be an integer."
            )

        if count != expected_count:
            raise ValueError(
                f"Replacement target occurs {count} times; "
                f"expected {expected_count}."
            )

        updated = content.replace(
            old_text,
            new_text,
            expected_count,
        )

        _edit_project_validate_content(updated)
        files[target] = updated

        return {
            "operation": kind,
            "file": _safe_relative(target, root),
            "replacements": count,
        }

    if kind in {"insert", "delete"}:
        target_text = operation.get("target")

        if not isinstance(target_text, str):
            raise ValueError(
                f"{kind} requires string 'target'."
            )

        content = current_content(target)
        count = content.count(target_text)

        if count == 0:
            raise ValueError(
                f"Target for {kind} was not found."
            )

        expected_count = operation.get("expected_count", 1)

        try:
            expected_count = int(expected_count)
        except (TypeError, ValueError):
            raise ValueError(
                "'expected_count' must be an integer."
            )

        if count != expected_count:
            raise ValueError(
                f"Target occurs {count} times; "
                f"expected {expected_count}."
            )

        if kind == "delete":
            updated = content.replace(
                target_text,
                "",
                expected_count,
            )
        else:
            insertion = operation.get("content")

            if not isinstance(insertion, str):
                raise ValueError(
                    "Insert requires string 'content'."
                )

            position = str(
                operation.get("position", "after")
            ).lower()

            if position == "before":
                replacement = insertion + target_text
            elif position == "after":
                replacement = target_text + insertion
            else:
                raise ValueError(
                    "Insert position must be 'before' or 'after'."
                )

            updated = content.replace(
                target_text,
                replacement,
                expected_count,
            )

        _edit_project_validate_content(updated)
        files[target] = updated

        return {
            "operation": kind,
            "file": _safe_relative(target, root),
            "matches": count,
        }

    if kind == "create":
        if target.exists() or target in files:
            raise ValueError(
                f"File already exists: "
                f"{_safe_relative(target, root)}."
            )

        content = operation.get("content")

        if not isinstance(content, str):
            raise ValueError(
                "Create requires string 'content'."
            )

        _edit_project_validate_content(content)

        files[target] = content

        return {
            "operation": kind,
            "file": _safe_relative(target, root),
        }

    if kind == "delete_file":
        if target in files and files[target] is None:
            raise ValueError(
                f"File is already deleted: "
                f"{_safe_relative(target, root)}."
            )

        if not target.exists() and target not in files:
            raise FileNotFoundError(
                f"File does not exist: "
                f"{_safe_relative(target, root)}."
            )

        if target.exists() and not target.is_file():
            raise ValueError(
                f"Path is not a file: "
                f"{_safe_relative(target, root)}."
            )

        files[target] = None

        return {
            "operation": kind,
            "file": _safe_relative(target, root),
        }

    if kind == "move":
        raw_destination = operation.get("destination")

        if not raw_destination:
            raise ValueError(
                "Move requires a destination."
            )

        destination = _edit_project_resolve_inside(
            root,
            raw_destination,
        )

        if destination == target:
            raise ValueError(
                "Move source and destination are identical."
            )

        if destination.exists() or destination in files:
            raise ValueError(
                f"Destination already exists: "
                f"{_safe_relative(destination, root)}."
            )

        content = current_content(target)

        files[target] = None
        files[destination] = content

        return {
            "operation": kind,
            "file": _safe_relative(target, root),
            "destination": _safe_relative(
                destination,
                root,
            ),
        }

    raise ValueError(
        f"Unsupported operation: {kind}."
    )


def _edit_project_diff(
    root: Path,
    before: dict[Path, str | None],
    after: dict[Path, str | None],
) -> str:
    import difflib

    diff = []

    paths = sorted(
        set(before) | set(after),
        key=lambda value: str(value),
    )

    for path in paths:
        old = before.get(path)
        new = after.get(path)

        if old == new:
            continue

        relative = _safe_relative(path, root)

        old_lines = (
            [] if old is None
            else old.splitlines(keepends=True)
        )
        new_lines = (
            [] if new is None
            else new.splitlines(keepends=True)
        )

        diff.extend(
            difflib.unified_diff(
                old_lines,
                new_lines,
                fromfile=relative,
                tofile=relative,
            )
        )

    return "".join(diff)


def handle_edit_project(
    handler: "BaseHTTPRequestHandler",
    data: dict,
) -> None:
    operations = data.get("operations")

    if operations is None:
        single = data.get("operation")

        if isinstance(single, dict):
            operations = [single]
        else:
            operations = []

    if not isinstance(operations, list) or not operations:
        _edit_project_error(
            handler,
            400,
            "At least one edit operation is required.",
        )
        return

    if len(operations) > EDIT_PROJECT_MAX_OPERATIONS:
        _edit_project_error(
            handler,
            400,
            "Too many edit operations.",
            max_operations=EDIT_PROJECT_MAX_OPERATIONS,
        )
        return

    first_operation = operations[0]

    if not isinstance(first_operation, dict):
        _edit_project_error(
            handler,
            400,
            "Each operation must be an object.",
        )
        return

    raw_path = data.get("path")

    if not raw_path:
        raw_path = first_operation.get("path")

    if not raw_path:
        _edit_project_error(
            handler,
            400,
            "A project path or operation path is required.",
        )
        return

    try:
        supplied_path = _resolve_project(raw_path)
    except (OSError, RuntimeError) as exc:
        _edit_project_error(
            handler,
            400,
            f"Invalid project path: {exc}",
        )
        return

    if supplied_path.exists() and supplied_path.is_file():
        root = _find_project_root(supplied_path)
    else:
        root = _find_project_root(
            supplied_path / "__placeholder__"
        )

    try:
        root = root.resolve()
    except (OSError, RuntimeError) as exc:
        _edit_project_error(
            handler,
            400,
            f"Could not resolve project root: {exc}",
        )
        return

    if not root.exists() or not root.is_dir():
        _edit_project_error(
            handler,
            400,
            "Detected project root is invalid.",
        )
        return

    files = {}
    before = {}
    summaries = []

    try:
        for operation in operations:
            if not isinstance(operation, dict):
                raise ValueError(
                    "Each operation must be an object."
                )

            kind = str(
                operation.get("operation", "")
            ).strip().lower()

            raw_operation_path = operation.get("path")

            if not raw_operation_path:
                raise ValueError(
                    f"Operation '{kind or '<missing>'}' "
                    "requires a path."
                )

            target = _edit_project_resolve_inside(
                root,
                raw_operation_path,
            )

            if target not in before and target.exists():
                if not target.is_file():
                    raise ValueError(
                        f"Path is not a file: "
                        f"{_safe_relative(target, root)}."
                    )

                before[target] = _edit_project_read(target)

            summary = _edit_project_apply_operation(
                root,
                files,
                operation,
            )

            destination_raw = operation.get("destination")

            if destination_raw:
                destination = _edit_project_resolve_inside(
                    root,
                    destination_raw,
                )

                if (
                    destination.exists()
                    and destination not in before
                ):
                    if not destination.is_file():
                        raise ValueError(
                            f"Path is not a file: "
                            f"{_safe_relative(destination, root)}."
                        )

                    before[destination] = _edit_project_read(
                        destination
                    )

            summaries.append(summary)

        # Capture the final state of files touched only by the
        # in-memory edit plan.
        for file_path, content in files.items():
            if file_path not in before:
                before[file_path] = None

        dry_run = bool(data.get("dry_run", False))

        if dry_run:
            after = dict(before)

            for file_path, content in files.items():
                after[file_path] = content

            diff = _edit_project_diff(
                root,
                before,
                after,
            )

            _edit_project_response(
                handler,
                {
                    "success": True,
                    "dry_run": True,
                    "project_root": str(root),
                    "operations": summaries,
                    "operation_count": len(summaries),
                    "diff": diff,
                    "files_changed": sum(
                        1
                        for file_path in set(before) | set(after)
                        if before.get(file_path)
                        != after.get(file_path)
                    ),
                },
            )
            return

        # Nothing has been written yet. All operations have passed
        # validation, so apply the complete edit plan.
        changed_files = []

        for file_path, content in files.items():
            if content is None:
                try:
                    if file_path.exists():
                        file_path.unlink()
                except OSError as exc:
                    raise OSError(
                        f"Could not delete "
                        f"{_safe_relative(file_path, root)}: {exc}"
                    ) from exc

                changed_files.append(
                    _safe_relative(file_path, root)
                )
                continue

            file_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            temp_path = file_path.with_name(
                f".{file_path.name}.vendora-edit.tmp"
            )

            try:
                temp_path.write_text(
                    content,
                    encoding="utf-8",
                )
                temp_path.replace(file_path)
            except OSError as exc:
                try:
                    if temp_path.exists():
                        temp_path.unlink()
                except OSError:
                    pass

                raise OSError(
                    f"Could not write "
                    f"{_safe_relative(file_path, root)}: {exc}"
                ) from exc

            changed_files.append(
                _safe_relative(file_path, root)
            )

        after = dict(before)

        for file_path, content in files.items():
            after[file_path] = content

        diff = _edit_project_diff(
            root,
            before,
            after,
        )

        _edit_project_response(
            handler,
            {
                "success": True,
                "dry_run": False,
                "project_root": str(root),
                "operations": summaries,
                "operation_count": len(summaries),
                "files_changed": changed_files,
                "diff": diff,
            },
        )

    except FileNotFoundError as exc:
        _edit_project_error(
            handler,
            404,
            str(exc),
        )
    except ValueError as exc:
        _edit_project_error(
            handler,
            400,
            str(exc),
        )
    except OSError as exc:
        _edit_project_error(
            handler,
            500,
            str(exc),
        )

def handle_read_project_file(
    handler: "BaseHTTPRequestHandler",
    data: dict,
) -> None:
    raw_path = data.get("path")

    if not raw_path:
        handler.send_response(400)
        handler.end_headers()
        handler.wfile.write(
            json.dumps({
                "error": "File path is required."
            }).encode()
        )
        return

    try:
        path = _resolve_project(raw_path)
    except (OSError, RuntimeError) as exc:
        handler.send_response(400)
        handler.end_headers()
        handler.wfile.write(
            json.dumps({
                "error": f"Invalid file path: {exc}"
            }).encode()
        )
        return

    if not path.exists():
        handler.send_response(404)
        handler.end_headers()
        handler.wfile.write(
            json.dumps({
                "error": f"File does not exist: {path}"
            }).encode()
        )
        return

    if not path.is_file():
        handler.send_response(400)
        handler.end_headers()
        handler.wfile.write(
            json.dumps({
                "error": f"Path is not a file: {path}"
            }).encode()
        )
        return

    try:
        size = path.stat().st_size
    except OSError as exc:
        handler.send_response(500)
        handler.end_headers()
        handler.wfile.write(
            json.dumps({
                "error": f"Could not inspect file: {exc}"
            }).encode()
        )
        return

    if size > READ_PROJECT_MAX_FILE_BYTES:
        handler.send_response(413)
        handler.end_headers()
        handler.wfile.write(
            json.dumps({
                "error": "File is too large to read.",
                "size_bytes": size,
                "max_bytes": READ_PROJECT_MAX_FILE_BYTES,
            }).encode()
        )
        return

    try:
        content = path.read_text(
            encoding="utf-8",
            errors="replace",
        )
    except OSError as exc:
        handler.send_response(500)
        handler.end_headers()
        handler.wfile.write(
            json.dumps({
                "error": f"Could not read file: {exc}"
            }).encode()
        )
        return

    lines = content.splitlines()

    start_line = data.get("start_line")
    end_line = data.get("end_line")

    try:
        start_line = int(start_line) if start_line is not None else 1
    except (TypeError, ValueError):
        start_line = 1

    try:
        end_line = int(end_line) if end_line is not None else len(lines)
    except (TypeError, ValueError):
        end_line = len(lines)

    start_line = max(1, start_line)
    end_line = min(len(lines), max(start_line, end_line))

    selected_lines = [
        {
            "line": number,
            "text": lines[number - 1],
        }
        for number in range(start_line, end_line + 1)
    ]

    result = {
        "file": str(path),
        "path": str(path),
        "file_name": path.name,
        "size_bytes": size,
        "total_lines": len(lines),
        "start_line": start_line,
        "end_line": end_line,
        "content": selected_lines,
    }

    body = json.dumps(
        result,
        indent=2,
        ensure_ascii=False,
    ).encode("utf-8")

    handler.send_response(200)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def handle_search_project(
    handler: "BaseHTTPRequestHandler",
    data: dict,
) -> None:
    raw_path = data.get("path", ".")
    query = str(data.get("query", "")).strip()

    if not query:
        handler.send_response(400)
        handler.end_headers()
        handler.wfile.write(
            json.dumps({
                "error": "Search query is required."
            }).encode()
        )
        return

    try:
        root = _resolve_project(raw_path)
    except (OSError, RuntimeError) as exc:
        handler.send_response(400)
        handler.end_headers()
        handler.wfile.write(
            json.dumps({
                "error": f"Invalid project path: {exc}"
            }).encode()
        )
        return

    if not root.exists():
        handler.send_response(404)
        handler.end_headers()
        handler.wfile.write(
            json.dumps({
                "error": f"Project path does not exist: {root}"
            }).encode()
        )
        return

    if not root.is_dir():
        handler.send_response(400)
        handler.end_headers()
        handler.wfile.write(
            json.dumps({
                "error": f"Project path is not a directory: {root}"
            }).encode()
        )
        return

    try:
        max_results = int(data.get(
            "max_results",
            SEARCH_MAX_RESULTS,
        ))
    except (TypeError, ValueError):
        max_results = SEARCH_MAX_RESULTS

    max_results = max(
        1,
        min(max_results, SEARCH_MAX_RESULTS),
    )

    extensions = _normalize_extensions(
        data.get("file_types")
    )

    results = _search_project_files(
        root,
        query,
        extensions,
        max_results,
    )

    body = json.dumps(
        {
            "project": root.name,
            "path": str(root),
            "query": query,
            "file_types": sorted(extensions),
            "results": results,
            "result_count": len(results),
            "truncated": len(results) >= max_results,
        },
        indent=2,
        ensure_ascii=False,
    ).encode("utf-8")

    handler.send_response(200)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def handle_inspect_project(
    handler: "BaseHTTPRequestHandler",
    data: dict,
) -> None:
    raw_path = data.get("path", ".")

    try:
        root = _resolve_project(raw_path)
    except (OSError, RuntimeError) as exc:
        handler.send_response(400)
        handler.end_headers()
        handler.wfile.write(
            json.dumps({
                "error": f"Invalid project path: {exc}"
            }).encode()
        )
        return

    if not root.exists():
        handler.send_response(404)
        handler.end_headers()
        handler.wfile.write(
            json.dumps({
                "error": f"Project path does not exist: {root}"
            }).encode()
        )
        return

    if not root.is_dir():
        handler.send_response(400)
        handler.end_headers()
        handler.wfile.write(
            json.dumps({
                "error": f"Project path is not a directory: {root}"
            }).encode()
        )
        return

    files, truncated = _collect_files(root)

    project_roots = _find_project_roots(root, files)

    projects = [
        _project_summary(
            project_root,
            root,
            files,
        )
        for project_root in project_roots
    ]

    all_languages = []
    all_auxiliary_languages = []
    all_frameworks = []
    all_package_managers = []

    for project in projects:
        for language in project["languages"]:
            if language not in all_languages:
                all_languages.append(language)

        for language in project.get("auxiliary_languages", []):
            if language not in all_auxiliary_languages:
                all_auxiliary_languages.append(language)

        for framework in project["frameworks"]:
            if framework not in all_frameworks:
                all_frameworks.append(framework)

        for manager in project["package_managers"]:
            if manager not in all_package_managers:
                all_package_managers.append(manager)

    result = {
        "project": root.name,
        "path": str(root),
        "languages": all_languages,
        "auxiliary_languages": all_auxiliary_languages,
        "frameworks": all_frameworks,
        "package_managers": all_package_managers,
        "config_files": sorted({
            item
            for project in projects
            for item in [
                f"{project['relative_path']}/{name}"
                if project["relative_path"] != "."
                else name
                for name in project["config_files"]
            ]
        }),
        "source_directories": _directories(root),
        "entry_points": sorted({
            f"{project['relative_path']}/{entry}"
            if project["relative_path"] != "."
            else entry
            for project in projects
            for entry in project["entry_points"]
        })[:100],
        "important_files": _important_files(root, files),
        "package_scripts": {
            project["relative_path"]: project["package_scripts"]
            for project in projects
            if project["package_scripts"]
        },
        "projects": projects,
        "git": _git_info(root),
        "statistics": {
            "files_scanned": len(files),
            "scan_truncated": truncated,
            "projects_detected": len(projects),
        },
    }

    body = json.dumps(
        result,
        indent=2,
        ensure_ascii=False,
    ).encode("utf-8")

    handler.send_response(200)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)
