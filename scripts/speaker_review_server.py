from __future__ import annotations

import argparse
import functools
import http.server
import socketserver
import webbrowser
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Serve the local speaker diarization review UI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(project_root))
    url = f"http://{args.host}:{args.port}/eval/speaker/viewer/"
    with socketserver.ThreadingTCPServer((args.host, args.port), handler) as server:
        server.daemon_threads = True
        print(f"Speaker review UI: {url}")
        if not args.no_open:
            webbrowser.open(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
