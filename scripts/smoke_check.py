from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request


def get_json(base_url: str, endpoint: str, timeout: float) -> object:
    with urllib.request.urlopen(
        f"{base_url.rstrip('/')}{endpoint}", timeout=timeout
    ) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description="MomentSeek 部署后基础检查")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout", type=float, default=10)
    parser.add_argument("--wait", type=float, default=180)
    args = parser.parse_args()

    deadline = time.monotonic() + args.wait
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            health = get_json(args.base_url, "/api/health", args.timeout)
            if isinstance(health, dict) and health.get("status") == "ok":
                break
            last_error = RuntimeError(f"health返回异常: {health!r}")
        except (OSError, ValueError, urllib.error.URLError) as exc:
            last_error = exc
        time.sleep(2)
    else:
        print(f"[ERROR] 服务未在等待时间内就绪: {last_error}", file=sys.stderr)
        return 1

    checks = [
        ("health", "/api/health", dict),
        ("videos", "/api/videos", list),
        ("jobs", "/api/jobs", list),
    ]
    for name, endpoint, expected_type in checks:
        try:
            payload = get_json(args.base_url, endpoint, args.timeout)
        except Exception as exc:
            print(f"[ERROR] {name}: {exc}", file=sys.stderr)
            return 1
        if not isinstance(payload, expected_type):
            print(f"[ERROR] {name}: 返回类型错误 {type(payload).__name__}", file=sys.stderr)
            return 1
        print(f"[OK] {name}: {endpoint}")
    print("[OK] MomentSeek基础检查通过。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

