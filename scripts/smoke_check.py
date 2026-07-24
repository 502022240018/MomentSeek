from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request


class SmokeCheckError(Exception):
    pass


def get_json(url: str, endpoint: str) -> object:
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            body = response.read()
        return json.loads(body.decode("utf-8"))
    except TimeoutError as exc:
        raise SmokeCheckError(f"{endpoint} request timed out ({url}): {exc}") from exc
    except urllib.error.URLError as exc:
        raise SmokeCheckError(f"{endpoint} request failed ({url}): {exc}") from exc
    except UnicodeDecodeError as exc:
        raise SmokeCheckError(f"{endpoint} response is not valid UTF-8 ({url}): {exc}") from exc
    except json.JSONDecodeError as exc:
        raise SmokeCheckError(f"{endpoint} response is not valid JSON ({url}): {exc}") from exc
    except ValueError as exc:
        raise SmokeCheckError(f"{endpoint} response could not be parsed ({url}): {exc}") from exc


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--require-release", action="store_true")
    args = parser.parse_args()

    try:
        health = get_json(f"{args.base_url.rstrip('/')}/api/health", "/api/health")
        jobs = get_json(f"{args.base_url.rstrip('/')}/api/jobs", "/api/jobs")
    except SmokeCheckError as exc:
        print(f"smoke_check failed: {exc}", file=sys.stderr)
        return 1

    if not isinstance(health, dict) or health.get("status") != "ok":
        print(f"unexpected /api/health response: {health}", file=sys.stderr)
        return 1
    if args.require_release and not health.get("release_id"):
        print("/api/health response does not include release_id", file=sys.stderr)
        return 1
    if not isinstance(jobs, list):
        print(f"unexpected /api/jobs response: {jobs}", file=sys.stderr)
        return 1

    print(json.dumps({"health": health, "jobs_count": len(jobs)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
