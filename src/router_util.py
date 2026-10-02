"""Small shared helpers for the hybrid router."""
from __future__ import annotations

from urllib.parse import urlparse


def worker_name_for(url: str) -> str:
    """Human-friendly worker name from its A2A base URL.

    http://worker-a:8001 -> "worker-a". Falls back to the full URL when
    no hostname can be parsed.
    """
    host = urlparse(url).hostname
    return host or url
