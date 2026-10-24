"""Password hashing (scrypt) and password policy.

Hashes are self-describing: ``scrypt$<log2 n>$<r>$<p>$<salt b64>$<hash b64>``, so the
cost can be raised later and old hashes upgraded on the next successful login.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import secrets

_N_LOG2 = 15  # n = 32768: ~32 MB and roughly 50-100 ms per hash on a server CPU
_R = 8
_P = 1
_DKLEN = 32
_MAXMEM = 128 * 1024 * 1024

# A small deny-list of the most common choices; length rules do most of the work.
_COMMON = frozenset(
    (  # noqa: SIM905
        "password password1 password123 passw0rd 123456789012 1234567890123 qwertyuiopas "
        "qwerty123456 iloveyou1234 letmein12345 welcome12345 admin1234567 administrator "
        "changeme1234 fuzzyfuzzy12 sentientai12 trustno11234 football1234 baseball1234 "
        "monkey123456 dragon123456 sunshine1234 princess1234 superman1234 starwars1234"
    ).split()
)


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def hash_password(password: str, *, n_log2: int = _N_LOG2) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(
        password.encode("utf-8"), salt=salt, n=2**n_log2, r=_R, p=_P, maxmem=_MAXMEM, dklen=_DKLEN
    )
    return f"scrypt${n_log2}${_R}${_P}${_b64(salt)}${_b64(digest)}"


def verify_password(password: str, stored: str) -> bool:
    try:
        algo, n_log2, r, p, salt_b64, hash_b64 = stored.split("$")
        if algo != "scrypt":
            return False
        expected = base64.b64decode(hash_b64)
        digest = hashlib.scrypt(
            password.encode("utf-8"),
            salt=base64.b64decode(salt_b64),
            n=2 ** int(n_log2),
            r=int(r),
            p=int(p),
            maxmem=_MAXMEM,
            dklen=len(expected),
        )
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(digest, expected)


def needs_rehash(stored: str) -> bool:
    parts = stored.split("$")
    return len(parts) != 6 or parts[0] != "scrypt" or int(parts[1]) < _N_LOG2


# A valid hash of a random password: verifying unknown usernames against it keeps
# login timing the same whether or not the account exists.
DUMMY_HASH = hash_password(secrets.token_urlsafe(24))


def password_problems(password: str, *, username: str, min_length: int) -> list[str]:
    problems = []
    if len(password) < min_length:
        problems.append(f"use at least {min_length} characters")
    if len(password) > 256:
        problems.append("use at most 256 characters")
    lowered = password.lower()
    if username and username.lower() in lowered:
        problems.append("don't include your username")
    if lowered in _COMMON or len(set(password)) < 4:
        problems.append("this password is too easy to guess")
    return problems
