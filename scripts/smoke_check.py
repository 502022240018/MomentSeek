from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request


def get_json(url: str) -> object:
    with urllib.request.urlopen(url, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--require-release", action="store_true")
    args = parser.parse_args()

    try:
        health = get_json(f"{args.base_url.rstrip('/')}/api/health")
        jobs = get_json(f"{args.base_url.rstrip('/')}/api/jobs")
    except urllib.error.URLError as exc:
        print(f"smoke_check failed: {exc}", file=sys.stderr)
        return 1

    if not isinstance(health, dict) or health.get("status") != "ok":
        print(f"unexpected health response: {health}", file=sys.stderr)
        return 1
    if args.require_release and not health.get("release_id"):
        print("health response does not include release_id", file=sys.stderr)
        return 1
    if not isinstance(jobs, list):
        print(f"unexpected jobs response: {jobs}", file=sys.stderr)
        return 1

    print(json.dumps({"health": health, "jobs_count": len(jobs)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
