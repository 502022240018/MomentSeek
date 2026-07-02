from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def collect_queries(rows: list[dict[str, Any]]) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    for row in rows:
        annotation = row.get("annotation") or {}
        for item in annotation.get("suggested_queries") or []:
            text = str(item.get("query") or "").strip()
            if text and text not in seen:
                seen.add(text)
                values.append(text)
    return values


def request_translation(endpoint: str, model: str, queries: list[str], timeout: float) -> list[str]:
    prompt = (
        "Translate each visual search query into concise natural Chinese. "
        "Keep proper nouns, brands, and English acronyms when appropriate. "
        "Return ONLY a JSON array of strings with the same length and order.\n\n"
        f"Queries:\n{json.dumps(queries, ensure_ascii=False)}"
    )
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a precise bilingual translator for image/video retrieval queries."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "max_tokens": max(512, len(queries) * 24),
    }
    request = urllib.request.Request(
        endpoint.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = json.loads(response.read().decode("utf-8"))
    content = data["choices"][0]["message"]["content"].strip()
    if content.startswith("```"):
        content = content.strip("`")
        if content.startswith("json"):
            content = content[4:].strip()
    parsed = json.loads(content)
    if not isinstance(parsed, list) or len(parsed) != len(queries):
        raise ValueError(f"Expected {len(queries)} translations, got {type(parsed).__name__} len={len(parsed) if isinstance(parsed, list) else 'n/a'}")
    return [str(item).strip() for item in parsed]


def translate_all(
    queries: list[str],
    endpoint: str,
    model: str,
    batch_size: int,
    timeout: float,
    retries: int,
) -> dict[str, str]:
    translations: dict[str, str] = {}
    for start in range(0, len(queries), batch_size):
        batch = queries[start:start + batch_size]
        last_error: Exception | None = None
        for attempt in range(1, retries + 1):
            try:
                zh = request_translation(endpoint, model, batch, timeout)
                translations.update(dict(zip(batch, zh)))
                print(json.dumps({"event": "translated", "start": start, "count": len(batch)}, ensure_ascii=False))
                break
            except (urllib.error.URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
                last_error = exc
                print(json.dumps({"event": "retry", "start": start, "attempt": attempt, "error": str(exc)}, ensure_ascii=False))
                time.sleep(min(10, attempt * 2))
        else:
            if len(batch) == 1:
                raise RuntimeError(f"Failed to translate query {batch[0]!r}: {last_error}") from last_error
            half = max(1, len(batch) // 2)
            translations.update(translate_all(batch[:half], endpoint, model, half, timeout, retries))
            translations.update(translate_all(batch[half:], endpoint, model, half, timeout, retries))
    return translations


def apply_translations(rows: list[dict[str, Any]], translations: dict[str, str], model: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        new_row = json.loads(json.dumps(row, ensure_ascii=False))
        annotation = new_row.get("annotation") or {}
        for item in annotation.get("suggested_queries") or []:
            original = str(item.get("query") or "").strip()
            if original in translations:
                item["query_en"] = original
                item["query"] = translations[original]
                item["language"] = "zh"
        annotation["query_language"] = "zh"
        new_row["translator"] = {"type": "local_llm", "model": model}
        out.append(new_row)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Translate visual eval suggested queries to Chinese using a local OpenAI-compatible chat endpoint.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="qwen36")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    queries = collect_queries(rows)
    print(json.dumps({"event": "start", "rows": len(rows), "unique_queries": len(queries)}, ensure_ascii=False))
    translations = translate_all(queries, args.endpoint, args.model, args.batch_size, args.timeout, args.retries)
    out_rows = apply_translations(rows, translations, args.model)
    write_jsonl(args.output, out_rows)
    print(json.dumps({"event": "done", "output": str(args.output), "translations": len(translations)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
