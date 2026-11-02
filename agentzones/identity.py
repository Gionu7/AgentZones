"""Per-zone identity: an Ed25519 key pair, plus pinning of peers' public keys.

* Each zone creates ``<data_dir>/zone_key.pem`` on first start and publishes the
  public key in its card.
* Peers *pin* the first key they see for a zone name (trust on first use), or
  take it from ``security.trusted_keys`` in zone.yaml (pre-shared, strongest).
  A card presenting a different key for a pinned name is rejected.
* Zones sign *attestations* ("I applied this fix and my checks passed"), so
  ``verified_by`` can no longer be claimed on someone else's behalf.

The shared network secret still authenticates *requests*; these keys
authenticate *statements*. The same keys can later replace the shared
secret for request signing (see docs/SECURITY.md).
"""

from __future__ import annotations

import base64
import json
import os
import threading
from pathlib import Path

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey


def _b64(b: bytes) -> str:
    return base64.b64encode(b).decode()


class Identity:
    def __init__(self, key: Ed25519PrivateKey):
        self._key = key
        raw = key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
        self.public_b64 = _b64(raw)

    @classmethod
    def load_or_create(cls, path: Path) -> "Identity":
        if path.exists():
            key = serialization.load_pem_private_key(path.read_bytes(), password=None)
            assert isinstance(key, Ed25519PrivateKey)
            return cls(key)
        key = Ed25519PrivateKey.generate()
        path.parent.mkdir(parents=True, exist_ok=True)
        pem = key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                serialization.NoEncryption())
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "wb") as f:
            f.write(pem)
        return cls(key)

    def sign(self, data: bytes) -> str:
        return _b64(self._key.sign(data))


def verify_sig(public_b64: str, data: bytes, sig_b64: str) -> bool:
    try:
        Ed25519PublicKey.from_public_bytes(base64.b64decode(public_b64)).verify(base64.b64decode(sig_b64), data)
        return True
    except (InvalidSignature, ValueError, TypeError):
        return False


class KeyPins:
    """zone name -> public key. Pre-shared keys (config) win over first-seen pins."""

    def __init__(self, path: Path, trusted: dict[str, str] | None = None):
        self.path = path
        self.trusted = dict(trusted or {})
        self._lock = threading.Lock()
        self.pins: dict[str, str] = json.loads(path.read_text()) if path.exists() else {}

    def key_for(self, zone: str) -> str | None:
        return self.trusted.get(zone) or self.pins.get(zone)

    def check(self, zone: str | None, pubkey: str | None) -> bool:
        """True if (zone, pubkey) is acceptable; pins it if first seen."""
        if not zone or not pubkey:
            return False
        known = self.key_for(zone)
        if known:
            return known == pubkey
        with self._lock:
            self.pins[zone] = pubkey
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.pins, indent=2))
        return True
