from __future__ import annotations

import argparse
import base64
import json
import os
import random
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


IMAGE_PROMPT = """You are annotating frames for a video retrieval benchmark.

Return only valid JSON. Describe visible content, not guesses.
Focus on:
1. main scene
2. people and actions
3. small objects
4. objects near edges/corners
5. visible text/OCR
6. useful search queries

Keep the output compact:
- at most 5 objects
- at most 3 people
- at most 5 text_overlay items
- at most 5 suggested_queries
- no explanations outside JSON

Schema:
{
  "caption_en": string,
  "caption_zh": string,
  "scene": string,
  "query_types": string[],
  "objects": [{"name": string, "location": string, "visibility": "clear|small|partial|uncertain"}],
  "people": [{"description": string, "location": string, "action": string, "visibility": "clear|small|partial|uncertain"}],
  "actions": string[],
  "small_objects": string[],
  "edge_objects": [{"name": string, "location": string, "visibility": "clear|small|partial|uncertain"}],
  "text_overlay": [{"text": string, "location": string, "confidence": number}],
  "suggested_queries": [{"query": string, "query_type": string, "positive_scope": "image"}],
  "confidence": number,
  "needs_review": boolean
}

Use concise captions. If text is unreadable, use an empty text_overlay list.
"""


SEGMENT_PROMPT = """You are annotating a 5-second video segment represented as a contact sheet.
Each cell is labeled with a timestamp.

Return only valid JSON. Describe the whole segment and temporal changes.
Do not invent events that are not visible.
Pay special attention to small objects, edge/corner objects, text on screen, and actions.
Generate search queries that a user might use to find this segment.

Keep the output compact:
- at most 5 objects
- at most 3 people
- at most 5 temporal_events
- at most 5 text_overlay items
- at most 5 suggested_queries
- no explanations outside JSON

Schema:
{
  "caption_en": string,
  "caption_zh": string,
  "scene": string,
  "query_types": string[],
  "objects": [{"name": string, "location": string, "visibility": "clear|small|partial|uncertain"}],
  "people": [{"description": string, "location": string, "action": string, "visibility": "clear|small|partial|uncertain"}],
  "actions": string[],
  "small_objects": string[],
  "edge_objects": [{"name": string, "location": string, "visibility": "clear|small|partial|uncertain"}],
  "text_overlay": [{"text": string, "location": string, "confidence": number}],
  "temporal_events": [{"time": string, "event": string}],
  "suggested_queries": [{"query": string, "query_type": string, "positive_scope": "segment"}],
  "confidence": number,
  "needs_review": boolean
}

Use concise captions. If text is unreadable, use an empty text_overlay list.
"""


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _mime_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".png":
        return "image/png"
    if suffix == ".webp":
        return "image/webp"
    return "application/octet-stream"


def _data_url(path: Path) -> str:
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{_mime_type(path)};base64,{payload}"


def _load_items(manifest_path: Path, kind: str) -> tuple[str, list[dict[str, Any]]]:
    payload = _read_json(manifest_path)
    if kind == "auto":
        if "frames" in payload:
            kind = "image"
        elif "sheets" in payload:
            kind = "segment"
        else:
            raise ValueError("Cannot infer item kind. Use --kind image or --kind segment.")
    if kind == "image":
        return kind, list(payload.get("frames", []))
    if kind == "segment":
        return kind, list(payload.get("sheets", []))
    raise ValueError(f"Unsupported kind: {kind}")


def _item_id(item: dict[str, Any], kind: str) -> str:
    if kind == "image":
        return str(item["image_id"])
    return str(item.get("sheet_id") or item.get("segment_id"))


def _path_for_item(item: dict[str, Any]) -> Path:
    return Path(str(item["path"]))


def _build_messages(item: dict[str, Any], kind: str) -> list[dict[str, Any]]:
    prompt = IMAGE_PROMPT if kind == "image" else SEGMENT_PROMPT
    image_path = _path_for_item(item)
    metadata = {
        "item_id": _item_id(item, kind),
        "group_id": item.get("group_id"),
        "variant_id": item.get("variant_id"),
        "resolution_label": item.get("resolution_label"),
        "time": item.get("time"),
        "start": item.get("start"),
        "end": item.get("end"),
        "sample_times": item.get("sample_times"),
    }
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt + "\n\nItem metadata:\n" + json.dumps(metadata, ensure_ascii=False)},
                {"type": "image_url", "image_url": {"url": _data_url(image_path)}},
            ],
        }
    ]


def _request_payload(item: dict[str, Any], kind: str, model: str, temperature: float, max_tokens: int) -> dict[str, Any]:
    return {
        "model": model,
        "messages": _build_messages(item, kind),
        "temperature": temperature,
        "max_tokens": max_tokens,
    }


def _post_json(url: str, payload: dict[str, Any], api_key: str | None, timeout: int) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _extract_text(response: dict[str, Any]) -> str:
    choices = response.get("choices") or []
    if choices:
        message = choices[0].get("message") or {}
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            pieces = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    pieces.append(str(part.get("text", "")))
            return "\n".join(pieces)
    if "output_text" in response:
        return str(response["output_text"])
    return json.dumps(response, ensure_ascii=False)


def _parse_json_maybe(text: str) -> tuple[dict[str, Any] | None, str]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].strip()
    try:
        return json.loads(stripped), stripped
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start >= 0 and end > start:
            candidate = stripped[start:end + 1]
            try:
                return json.loads(candidate), candidate
            except json.JSONDecodeError:
                return None, stripped
    return None, stripped


def _existing_ids(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()
    ids = set()
    for line in output_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        # Resume should skip only successful model calls. Failed rows are kept as
        # diagnostics but retried on the next run after the key/endpoint/model is
        # fixed.
        if item.get("status") == "ok" and "item_id" in item:
            ids.add(str(item["item_id"]))
    return ids


def _envelope(
    item: dict[str, Any],
    kind: str,
    annotation: dict[str, Any] | None,
    raw_text: str,
    model: str,
    status: str,
    attempts: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    item_id = _item_id(item, kind)
    return {
        "schema_version": 1,
        "item_id": item_id,
        "item_type": kind,
        "group_id": item.get("group_id"),
        "variant_id": item.get("variant_id"),
        "resolution_label": item.get("resolution_label"),
        "source_path": item.get("path"),
        "time": item.get("time"),
        "start": item.get("start"),
        "end": item.get("end"),
        "annotation": annotation,
        "raw_text": raw_text,
        "review_status": "pending",
        "status": status,
        "annotator": {
            "type": "model",
            "name": model,
        },
        "attempts": attempts or [],
        "created_at_unix": time.time(),
    }


def _split_models(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.replace("\n", ",").split(",") if item.strip()]


def _request_error_record(item: dict[str, Any], kind: str, model: str, exc: urllib.error.HTTPError) -> dict[str, Any]:
    try:
        body = exc.read().decode("utf-8", errors="replace")
    except Exception:
        body = ""
    return _envelope(
        item,
        kind,
        None,
        f"HTTPError {exc.code}: {exc.reason}; body={body}",
        model,
        "request_error",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto-annotate visual eval frames/contact sheets with a VLM.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--kind", choices=["auto", "image", "segment"], default="auto")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--backend", choices=["openai-compatible"], default="openai-compatible")
    parser.add_argument("--base-url", default=os.environ.get("VLM_BASE_URL", "http://127.0.0.1:8000/v1"))
    parser.add_argument("--api-key-env", default="VLM_API_KEY")
    parser.add_argument("--model", default=os.environ.get("VLM_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct"))
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Fallback model list. If omitted, VLM_MODELS comma list is used, then --model.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-tokens", type=int, default=900)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--dry-run", action="store_true", help="Write request payloads instead of calling the model.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--max-new",
        type=int,
        default=None,
        help="Maximum number of newly annotated non-skipped items to write in this run. Useful with --resume.",
    )
    args = parser.parse_args()

    kind, items = _load_items(args.manifest, args.kind)
    if args.sample is not None and args.sample < len(items):
        rng = random.Random(args.seed)
        items = rng.sample(items, args.sample)
    if args.limit is not None:
        items = items[:args.limit]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    done_ids = _existing_ids(args.out) if args.resume else set()
    api_key = os.environ.get(args.api_key_env)
    url = args.base_url.rstrip("/") + "/chat/completions"
    models = args.models or _split_models(os.environ.get("VLM_MODELS")) or [args.model]

    written = 0
    skipped = 0
    with args.out.open("a" if args.resume else "w", encoding="utf-8") as file:
        for item in items:
            item_id = _item_id(item, kind)
            if item_id in done_ids:
                skipped += 1
                continue
            if args.max_new is not None and written >= args.max_new:
                break
            payload = _request_payload(item, kind, models[0], args.temperature, args.max_tokens)
            if args.dry_run:
                record = {
                    "item_id": item_id,
                    "item_type": kind,
                    "request": payload,
                    "target_url": url,
                    "models": models,
                }
            else:
                attempts: list[dict[str, Any]] = []
                record = None
                for model_index, model in enumerate(models):
                    payload = _request_payload(item, kind, model, args.temperature, args.max_tokens)
                    try:
                        response = _post_json(url, payload, api_key, args.timeout)
                        text = _extract_text(response)
                        annotation, raw = _parse_json_maybe(text)
                        status = "ok" if annotation is not None else "parse_error"
                        attempts.append({"model": model, "status": status})
                        record = _envelope(item, kind, annotation, raw, model, status, attempts)
                        # A parse error means the model was reachable and returned content.
                        # Do not spend fallback-model quota unless the request itself failed.
                        break
                    except urllib.error.HTTPError as exc:
                        record = _request_error_record(item, kind, model, exc)
                        attempts.append({
                            "model": model,
                            "status": "request_error",
                            "raw_text": record.get("raw_text", ""),
                        })
                        if model_index < len(models) - 1:
                            continue
                        record["attempts"] = attempts
                    except (urllib.error.URLError, TimeoutError, OSError) as exc:
                        record = _envelope(
                            item,
                            kind,
                            None,
                            f"{type(exc).__name__}: {exc}",
                            model,
                            "request_error",
                            attempts + [{"model": model, "status": "request_error", "raw_text": f"{type(exc).__name__}: {exc}"}],
                        )
                        if model_index < len(models) - 1:
                            attempts = record["attempts"]
                            continue
                    if record is not None:
                        break
                if record is None:
                    record = _envelope(item, kind, None, "no models attempted", models[0], "request_error", attempts)
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
            file.flush()
            written += 1
            if args.sleep:
                time.sleep(args.sleep)

    print(json.dumps({
        "manifest": str(args.manifest),
        "out": str(args.out),
        "kind": kind,
        "model": models[0],
        "models": models,
        "items_selected": len(items),
        "written": written,
        "skipped": skipped,
        "max_new": args.max_new,
        "dry_run": args.dry_run,
        "base_url": args.base_url,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
