from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _best_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for row in rows:
        item_id = str(row.get("item_id"))
        current = best.get(item_id)
        if current is None:
            best[item_id] = row
            continue
        # Prefer successful parsed annotations; otherwise keep the latest row.
        if current.get("status") != "ok" and row.get("status") == "ok":
            best[item_id] = row
        elif current.get("status") == row.get("status"):
            best[item_id] = row
    return list(best.values())


def _join_values(values: Any, limit: int = 5) -> str:
    if not values:
        return ""
    if isinstance(values, list):
        return "; ".join(str(item) for item in values[:limit])
    return str(values)


def _queries(annotation: dict[str, Any]) -> str:
    queries = annotation.get("suggested_queries") or []
    values = []
    for item in queries[:5]:
        if isinstance(item, dict):
            values.append(f"{item.get('query', '')} [{item.get('query_type', '')}]")
        else:
            values.append(str(item))
    return "; ".join(values)


def _objects(annotation: dict[str, Any]) -> str:
    values = []
    for key in ("objects", "people"):
        for item in (annotation.get(key) or [])[:5]:
            if isinstance(item, dict):
                name = item.get("name") or item.get("description") or ""
                loc = item.get("location") or ""
                action = item.get("action") or ""
                values.append(" / ".join(value for value in (name, loc, action) if value))
            else:
                values.append(str(item))
    return "; ".join(values[:8])


def export_review(input_path: Path, output_path: Path, title: str) -> dict[str, Any]:
    rows = _best_rows(_read_jsonl(input_path))
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    lines = [
        f"# {title}",
        "",
        f"- Source: `{input_path.as_posix()}`",
        f"- Unique items: {len(rows)}",
        f"- Parsed OK: {len(ok_rows)}",
        "",
        "| # | item_id | image/path | caption_zh | objects/actions | suggested_queries | review |",
        "|---:|---|---|---|---|---|---|",
    ]
    for index, row in enumerate(rows, start=1):
        annotation = row.get("annotation") or {}
        caption = (annotation.get("caption_zh") or annotation.get("caption_en") or "").replace("|", "\\|")
        objects = _objects(annotation).replace("|", "\\|")
        queries = _queries(annotation).replace("|", "\\|")
        source = str(row.get("source_path") or "").replace("\\", "/")
        if source:
            image_link = f"[open]({source})"
        else:
            image_link = ""
        review = "pending"
        if row.get("status") != "ok":
            caption = f"`{row.get('status')}`"
            queries = str(row.get("raw_text") or "")[:120].replace("|", "\\|").replace("\n", " ")
        lines.append(
            f"| {index} | `{row.get('item_id')}` | {image_link} | {caption} | {objects} | {queries} | {review} |"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"items": len(rows), "ok": len(ok_rows), "out": str(output_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Export visual auto annotations to a Markdown review table.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--title", default="Visual auto annotation review")
    args = parser.parse_args()
    print(json.dumps(export_review(args.input, args.out, args.title), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
