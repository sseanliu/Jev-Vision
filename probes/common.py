"""Shared helpers for Jev API probes."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
API = "https://api.typesafe.ai/v1/systemone"


def api_key() -> str:
    key = os.environ.get("TYPESAFE_API_KEY")
    if key:
        return key
    env = ROOT / ".env"
    for line in env.read_text().splitlines():
        if line.startswith("TYPESAFE_API_KEY="):
            return line.split("=", 1)[1].strip()
    raise RuntimeError("TYPESAFE_API_KEY not found")


_session = None


def session() -> requests.Session:
    """Lazy: importing this module must not require an API key (fingerprint.py
    imports the probe designs on machines without .env)."""
    global _session
    if _session is None:
        _session = requests.Session()
        _session.headers.update(
            {"Authorization": f"Bearer {api_key()}", "Content-Type": "application/json"}
        )
    return _session


def call(payload: dict, retries: int = 3) -> dict:
    """POST one request; return body plus request id, server time and wall time."""
    _session = session()
    last: dict = {"status": 0, "text": ""}
    for attempt in range(retries):
        t0 = time.perf_counter()
        r = _session.post(API, data=json.dumps(payload), timeout=60)
        wall_ms = (time.perf_counter() - t0) * 1000
        if r.status_code == 200:
            body = r.json()
            return {
                "status": 200,
                "request_id": r.headers.get("x-typesafe-request-id"),
                "server_ms": _int(r.headers.get("x-envoy-upstream-service-time")),
                "wall_ms": round(wall_ms, 1),
                "body": body,
            }
        last = {"status": r.status_code, "text": r.text[:300]}
        if r.status_code in (429, 500, 502, 503):
            time.sleep(1.5 * (attempt + 1))
            continue
        break
    return {"status": last["status"], "error": last["text"]}


def _int(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def save(name: str, obj) -> Path:
    out = ROOT / "results" / name
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(obj, ensure_ascii=False, indent=1))
    return out
