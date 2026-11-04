"""Get a local model into an Ollama server: ``agentzone model setup``.

Two settings:

* ``MODEL_NAME``   - the name agents ask for (``model`` in the OpenAI API), e.g. ``qwen2.5:3b`` or ``proxy-doctor``.
* ``MODEL_SOURCE`` - where it comes from; defaults to ``MODEL_NAME``:

  ===============================================  ===============================================
  ``qwen2.5:3b``                                   Ollama library (registry.ollama.ai)
  ``hf.co/bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M``  Hugging Face GGUF repo, pulled by Ollama itself
  ``https://.../model.gguf``                       any HTTPS file: an internal mirror, an Azure blob
                                                   (SAS URL), a Hugging Face ``resolve`` URL
  ``/local-models/model.gguf``                     a file on disk, e.g. a model downloaded once from
                                                   the Azure ML registry or your own fine-tune
  ===============================================  ===============================================

It talks to Ollama's HTTP API only, so it works the same against the ``ollama`` container of a zone and
against Ollama installed on a laptop. Downloads honour HTTPS_PROXY and SSL_CERT_FILE. A model that is
already there is left alone (``--force`` to replace it). Other OpenAI-compatible servers (LM Studio,
llama.cpp's llama-server, vLLM) load models their own way; for those this command only checks the server.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

import httpx

log = logging.getLogger(__name__)


def ollama_root(model_url: str) -> str:
    """``http://ollama:11434/v1`` -> ``http://ollama:11434`` (the native API sits next to the OpenAI one)."""
    return re.sub(r"/v1/?$", "", model_url.rstrip("/"))


def source_kind(source: str) -> str:
    if source.startswith(("http://", "https://")):
        return "url"
    if source.endswith(".gguf") or source.startswith(("/", "./", "~")):
        return "file"
    return "ollama"  # library name or hf.co/... - Ollama pulls both


def _full(name: str) -> str:
    return name if ":" in name.rsplit("/", 1)[-1] else name + ":latest"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, cache_dir: Path, sha256: str = "") -> Path:
    name = re.sub(r"[^A-Za-z0-9._-]", "_", url.split("?", 1)[0].rstrip("/").rsplit("/", 1)[-1]) or "model.gguf"
    dest = cache_dir / name
    if dest.exists() and (not sha256 or _sha256(dest) == sha256):
        log.info("using cached %s", dest)
        return dest
    cache_dir.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    log.info("downloading %s", url.split("?", 1)[0])  # never log a SAS token
    with httpx.stream("GET", url, follow_redirects=True, timeout=httpx.Timeout(60, read=300)) as r:
        r.raise_for_status()
        total, done, last = int(r.headers.get("content-length") or 0), 0, time.time()
        with part.open("wb") as f:
            for chunk in r.iter_bytes(1 << 20):
                f.write(chunk)
                done += len(chunk)
                if time.time() - last > 10:
                    log.info("  %d MB%s", done >> 20, f" of {total >> 20} MB" if total else "")
                    last = time.time()
    if sha256 and _sha256(part) != sha256:
        part.unlink()
        raise ValueError(f"{name}: sha256 does not match MODEL_SHA256")
    part.replace(dest)
    return dest


class Ollama:
    def __init__(self, root: str):
        self.http = httpx.Client(base_url=root, timeout=httpx.Timeout(30, read=None), trust_env=False)

    def is_up(self) -> bool:
        try:
            return self.http.get("/api/version").status_code == 200
        except httpx.HTTPError:
            return False

    def has(self, name: str) -> bool:
        tags = self.http.get("/api/tags").json().get("models") or []
        return _full(name) in {_full(m.get("name") or m.get("model", "")) for m in tags}

    def _ok(self, r: httpx.Response, what: str) -> Any:
        if r.status_code >= 400:
            raise RuntimeError(f"ollama {what}: HTTP {r.status_code}: {r.text[:300]}")
        body = r.json() if r.content else {}
        if isinstance(body, dict) and body.get("error"):
            raise RuntimeError(f"ollama {what}: {body['error']}")
        return body

    def pull(self, source: str) -> None:
        self._ok(self.http.post("/api/pull", json={"model": source, "stream": False}), f"pull {source}")

    def copy(self, source: str, dest: str) -> None:
        self._ok(self.http.post("/api/copy", json={"source": source, "destination": dest}), "copy")

    def create_from_gguf(self, name: str, path: Path, system: str = "") -> None:
        digest = "sha256:" + _sha256(path)
        if self.http.head(f"/api/blobs/{digest}").status_code != 200:
            with path.open("rb") as f:
                self._ok(self.http.post(f"/api/blobs/{digest}", content=f), "upload")
        body: dict[str, Any] = {"model": name, "files": {path.name: digest}, "stream": False}
        if system:
            body["system"] = system
        self._ok(self.http.post("/api/create", json=body), f"create {name}")


def setup(name: str, source: str = "", model_url: str = "http://ollama:11434/v1", cache_dir: str = "/models-cache",
          sha256: str = "", system: str = "", force: bool = False, wait_s: float = 300) -> str:
    """Make ``name`` available on the server at ``model_url``. Returns what was done."""
    source = source or name
    root = ollama_root(model_url)
    ol = Ollama(root)
    t0, ids = time.time(), None
    while not ol.is_up():  # the server may still be starting
        try:  # or it is another OpenAI-compatible server (LM Studio, llama-server, vLLM)
            r = httpx.get(model_url.rstrip("/") + "/models", timeout=5, trust_env=False)
            if r.status_code == 200:
                ids = [m.get("id") for m in r.json().get("data", [])]
                break
        except (httpx.HTTPError, ValueError):
            pass
        if time.time() - t0 > wait_s:
            raise RuntimeError(f"no model server at {model_url} after {wait_s:.0f}s")
        time.sleep(2)
    if ids is not None:
        if name in ids:
            return f"{name} is served by {model_url} (not Ollama; nothing to set up)"
        raise RuntimeError(f"{model_url} is not Ollama and does not serve {name!r}; it serves {ids}. "
                           "Load the model there, or set MODEL_NAME to one of these.")
    if ol.has(name) and not force:
        return f"{name} is already on {root}"
    kind = source_kind(source)
    if kind == "ollama":
        log.info("pulling %s into %s", source, root)
        ol.pull(source)
        if _full(source) != _full(name):
            ol.copy(source, name)
        return f"pulled {source}" + (f" as {name}" if _full(source) != _full(name) else "")
    path = download(source, Path(cache_dir), sha256) if kind == "url" else Path(os.path.expanduser(source))
    if not path.exists():
        raise FileNotFoundError(f"{path} not found (MODEL_SOURCE)")
    log.info("creating %s from %s", name, path.name)
    ol.create_from_gguf(name, path, system)
    return f"created {name} from {path.name}"


def setup_from_env(env: dict[str, str] | None = None, force: bool = False) -> str:
    e = dict(os.environ if env is None else env)
    return setup(e.get("MODEL_NAME", "qwen2.5:1.5b"), e.get("MODEL_SOURCE", ""),
                 e.get("MODEL_URL", "http://ollama:11434/v1"), e.get("MODEL_CACHE", "/models-cache"),
                 e.get("MODEL_SHA256", ""), e.get("MODEL_SYSTEM", ""), force=force,
                 wait_s=float(e.get("MODEL_WAIT_S", "300")))
