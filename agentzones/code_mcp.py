"""Read-only MCP server over one folder of code: ``python -m agentzones.code_mcp`` (stdio).

Tools: ``list_files``, ``search`` and ``read_file``, all confined to ``CODE_ROOT``. Nothing can be written or
run. Secrets-looking files (.env, keys, certificates, credentials) and big or binary files are never read.
The ``code-assistant`` agent uses it; other zones get these tools only if ``LLM_REMOTE_TOOLS`` lists them.
"""

from __future__ import annotations

import fnmatch
import os
import re
from pathlib import Path

from .mcp_lite import MiniMCPServer

SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", ".mypy_cache", ".pytest_cache", "dist",
             "build", ".idea", ".vscode", ".tox", ".next", "target"}
SECRET = re.compile(r"(^\.env(\..*)?$|\.(pem|key|p12|pfx|crt|cer|jks|keystore|kdbx)$|^id_(rsa|ed25519|ecdsa)|"
                    r"credentials|secrets?\.(ya?ml|json|txt)$|\.tfstate$|^\.netrc$|^\.npmrc$|^\.pypirc$)", re.I)
MAX_BYTES = 1_000_000
MAX_LINES = 400


def make_server(root: str) -> MiniMCPServer:
    base = Path(root).resolve()
    srv = MiniMCPServer("code", instructions=f"Read-only access to the code in {base.name}/. "
                        "Use list_files and search to find things, then read_file to read them.")

    def resolve(path: str) -> Path:
        p = (base / (path or ".")).resolve()
        if p != base and base not in p.parents:
            raise ValueError("path is outside the code folder")
        return p

    def readable(p: Path) -> bool:
        real = p.resolve()  # a symlink must not lead out of the folder
        if real != base and base not in real.parents:
            return False
        if SECRET.search(p.name) or p.stat().st_size > MAX_BYTES:
            return False
        with p.open("rb") as f:
            return b"\0" not in f.read(4096)  # skip binary files

    def walk(start: Path):
        for dirpath, dirs, files in os.walk(start):
            dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS and not d.startswith("."))
            for name in sorted(files):
                yield Path(dirpath) / name

    @srv.tool()
    def list_files(path: str = ".", pattern: str = "") -> dict:
        """List files under `path` (relative to the code folder). `pattern` filters names, e.g. '*.py'."""
        out, more = [], False
        for f in walk(resolve(path)):
            if (pattern and not fnmatch.fnmatch(f.name, pattern)) or SECRET.search(f.name):
                continue
            if len(out) >= 300:
                more = True
                break
            out.append(str(f.relative_to(base)))
        return {"files": out, "truncated": more}

    @srv.tool()
    def search(pattern: str, path: str = ".", file_pattern: str = "") -> dict:
        """Search file contents for a regular expression (case-insensitive). Returns file:line: text."""
        try:
            rx = re.compile(pattern, re.I)
        except re.error as exc:
            raise ValueError(f"bad pattern: {exc}") from exc
        hits = []
        for f in walk(resolve(path)):
            if file_pattern and not fnmatch.fnmatch(f.name, file_pattern):
                continue
            try:
                if not readable(f):
                    continue
                for i, line in enumerate(f.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                    if rx.search(line):
                        hits.append(f"{f.relative_to(base)}:{i}: {line.strip()[:200]}")
                        if len(hits) >= 60:
                            return {"matches": hits, "truncated": True}
            except OSError:
                continue
        return {"matches": hits, "truncated": False}

    @srv.tool()
    def read_file(path: str, start_line: int = 1, max_lines: int = 200) -> str:
        """Read a text file, with line numbers. Use start_line to page through long files."""
        p = resolve(path)
        if not p.is_file():
            raise ValueError(f"{path} is not a file")
        if not readable(p):
            raise ValueError(f"{path} is not readable here (secret-like, binary or over 1 MB)")
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        start = max(1, int(start_line))
        end = min(len(lines), start - 1 + max(1, min(int(max_lines), MAX_LINES)))
        body = "\n".join(f"{i:5d}  {lines[i - 1]}" for i in range(start, end + 1))
        more = f"\n... {len(lines) - end} more lines (start_line={end + 1})" if end < len(lines) else ""
        return f"{p.relative_to(base)} (lines {start}-{end} of {len(lines)})\n{body}{more}"

    return srv


if __name__ == "__main__":
    make_server(os.environ.get("CODE_ROOT", ".")).run()
