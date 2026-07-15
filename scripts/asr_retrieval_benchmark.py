from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np


@dataclass(frozen=True)
class Chunk:
    global_id: int
    source_id: str
    source_group: str
    start_ms: int
    end_ms: int
    text: str
    base_semantic_eligible: bool


@dataclass
class Corpus:
    chunks: list[Chunk]
    source_indices: dict[str, np.ndarray]
    duration_hours: dict[str, float]


MODEL_SPECS = {
    "minilm": {
        "name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "query_prefix": "",
        "passage_prefix": "",
        "trust_remote_code": False,
    },
    "e5_small": {
        "name": "intfloat/multilingual-e5-small",
        "query_prefix": "query: ",
        "passage_prefix": "passage: ",
        "trust_remote_code": False,
    },
    "gte_base": {
        "name": "Alibaba-NLP/gte-multilingual-base",
        "query_prefix": "",
        "passage_prefix": "",
        "trust_remote_code": True,
    },
}

FILLER_ONLY = {
    "嗯", "啊", "哦", "呃", "唉", "哎", "诶", "对", "好", "好的", "是",
    "and", "but", "because", "so", "well", "okay", "ok", "yeah", "yes", "no",
    "但是", "所以", "然后", "就是", "那个", "这个",
}


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not row.get("id") or not row.get("query") or "targets" not in row:
                raise ValueError(f"Invalid query at {path}:{line_number}")
            rows.append(row)
    return rows


def _decode_texts(values: np.ndarray) -> list[str]:
    return [str(value) for value in values.tolist()]


def _load_platform_corpus(index_root: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for index_dir in sorted(path for path in index_root.iterdir() if path.is_dir()):
        index_path = index_dir / "asr.npz"
        if not index_path.exists():
            continue
        with np.load(index_path, allow_pickle=False) as data:
            if not {"chunk_times_ms", "texts"}.issubset(data.files):
                continue
            times = np.asarray(data["chunk_times_ms"], dtype=np.int32)
            texts = _decode_texts(data["texts"])
            if times.shape != (len(texts), 2):
                raise ValueError(f"Invalid ASR arrays: {index_path}")
            if "embedding_chunk_indices" in data.files:
                eligible_ids = set(np.asarray(data["embedding_chunk_indices"], dtype=np.int32).tolist())
            else:
                eligible_ids = set(range(len(texts)))
            for local_id, text in enumerate(texts):
                chunks.append(Chunk(
                    global_id=-1,
                    source_id=index_dir.name,
                    source_group="platform",
                    start_ms=int(times[local_id, 0]),
                    end_ms=int(times[local_id, 1]),
                    text=text,
                    base_semantic_eligible=local_id in eligible_ids,
                ))
    return chunks


def _load_open_corpus(eval_root: Path) -> tuple[list[Chunk], float]:
    run_root = eval_root / "runs" / "funasr_sensevoice_small_ts_silero_vad_zh" / "samples"
    manifest_path = eval_root / "input" / "eval" / "asr" / "internal_testset" / "manifest.jsonl"
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest_rows = [json.loads(line) for line in handle if line.strip()]
    durations = {
        row["sample_id"]: float(row.get("duration_seconds") or 0.0)
        for row in manifest_rows
    }
    chunks: list[Chunk] = []
    included_duration = 0.0
    for sample_dir in sorted(path for path in run_root.iterdir() if path.is_dir()):
        source_id = sample_dir.name
        if source_id == "asr_v1_platform_yesterday_ep04":
            continue
        processed_path = sample_dir / "processed.json"
        if not processed_path.exists():
            continue
        rows = json.loads(processed_path.read_text(encoding="utf-8"))
        included_duration += durations.get(source_id, 0.0)
        for row in rows:
            text = str(row.get("text") or "").strip()
            if not text:
                continue
            chunks.append(Chunk(
                global_id=-1,
                source_id=source_id,
                source_group="open",
                start_ms=int(row.get("start_ms") or round(float(row.get("start_time") or 0.0) * 1000)),
                end_ms=int(row.get("end_ms") or round(float(row.get("end_time") or 0.0) * 1000)),
                text=text,
                base_semantic_eligible=bool(row.get("semantic_eligible", True)),
            ))
    return chunks, included_duration / 3600.0


def load_corpus(index_root: Path, open_eval_root: Path) -> Corpus:
    platform = _load_platform_corpus(index_root)
    opened, open_hours = _load_open_corpus(open_eval_root)
    rows = platform + opened
    chunks = [
        Chunk(
            global_id=index,
            source_id=row.source_id,
            source_group=row.source_group,
            start_ms=row.start_ms,
            end_ms=row.end_ms,
            text=row.text,
            base_semantic_eligible=row.base_semantic_eligible,
        )
        for index, row in enumerate(rows)
    ]
    source_indices: dict[str, list[int]] = defaultdict(list)
    for chunk in chunks:
        source_indices[chunk.source_id].append(chunk.global_id)
    return Corpus(
        chunks=chunks,
        source_indices={key: np.asarray(value, dtype=np.int32) for key, value in source_indices.items()},
        duration_hours={"open": open_hours},
    )


def _target_indices(query: dict, corpus: Corpus) -> set[int]:
    targets: set[int] = set()
    for target in query.get("targets", []):
        source_id = str(target.get("source_id") or target.get("video_id") or "")
        indices = corpus.source_indices.get(source_id, np.empty(0, dtype=np.int32))
        if "text_contains" in target:
            needle = str(target["text_contains"]).casefold()
            targets.update(
                int(index)
                for index in indices
                if needle in corpus.chunks[int(index)].text.casefold()
            )
            continue
        start_ms = int(target["start_ms"])
        end_ms = int(target["end_ms"])
        overlapping = [
            int(index)
            for index in indices
            if corpus.chunks[int(index)].end_ms > start_ms
            and corpus.chunks[int(index)].start_ms < end_ms
        ]
        if not overlapping and len(indices):
            midpoint = (start_ms + end_ms) / 2.0
            nearest = min(
                (int(index) for index in indices),
                key=lambda index: abs(
                    (corpus.chunks[index].start_ms + corpus.chunks[index].end_ms) / 2.0 - midpoint
                ),
            )
            distance = abs(
                (corpus.chunks[nearest].start_ms + corpus.chunks[nearest].end_ms) / 2.0 - midpoint
            )
            if distance <= 8000:
                overlapping = [nearest]
        targets.update(overlapping)
    if query.get("targets") and not targets:
        raise ValueError(f"Query {query['id']} has no target chunk")
    return targets


def _query_domain(query: dict) -> str:
    return "open" if str(query["id"]).startswith(("o", "n")) else "platform"


def prepare_queries(query_paths: Iterable[Path], corpus: Corpus) -> list[dict]:
    queries = []
    seen = set()
    for path in query_paths:
        for row in _read_jsonl(path):
            if row["id"] in seen:
                raise ValueError(f"Duplicate query id: {row['id']}")
            seen.add(row["id"])
            item = dict(row)
            item["domain"] = _query_domain(item)
            item["target_indices"] = _target_indices(item, corpus)
            queries.append(item)
    return queries


def _normalize_text(text: str) -> str:
    value = unicodedata.normalize("NFKC", text).casefold()
    return "".join(character for character in value if character.isalnum() or _is_cjk(character))


def _is_cjk(character: str) -> bool:
    return (
        "\u3400" <= character <= "\u4dbf"
        or "\u4e00" <= character <= "\u9fff"
        or "\uf900" <= character <= "\ufaff"
    )


def _quality_profile(chunk: Chunk) -> tuple[bool, float, list[str]]:
    compact = _normalize_text(chunk.text)
    effective = len(compact)
    duration_ms = max(1, chunk.end_ms - chunk.start_ms)
    flags: list[str] = []
    hard_reject = False
    if effective <= 1:
        flags.append("too_short")
        hard_reject = True
    if compact in FILLER_ONLY:
        flags.append("filler_only")
        hard_reject = True
    if duration_ms < 500 and effective >= 8:
        flags.append("impossible_text_rate")
        hard_reject = True
    grams = [compact[index:index + 2] for index in range(max(0, len(compact) - 1))]
    if len(grams) >= 8 and len(set(grams)) / len(grams) <= 0.25:
        flags.append("high_repetition")
        hard_reject = True
    weight = 1.0
    if effective <= 4:
        flags.append("low_context")
        weight *= 0.75
    if len(grams) >= 5 and len(set(grams)) / len(grams) <= 0.5:
        flags.append("repetitive")
        weight *= 0.75
    return hard_reject, weight, flags


def semantic_policy(corpus: Corpus, name: str) -> tuple[np.ndarray, np.ndarray, dict]:
    base = np.asarray([chunk.base_semantic_eligible for chunk in corpus.chunks], dtype=bool)
    weights = np.ones(len(corpus.chunks), dtype=np.float32)
    profile = [_quality_profile(chunk) for chunk in corpus.chunks]
    hard = np.asarray([item[0] for item in profile], dtype=bool)
    profile_weights = np.asarray([item[1] for item in profile], dtype=np.float32)
    if name == "current":
        mask = base
    elif name == "obvious_hard_filter":
        mask = base & ~hard
    elif name == "hard_filter_soft_penalty":
        mask = base & ~hard
        weights = profile_weights
    else:
        raise ValueError(f"Unknown semantic policy: {name}")
    flag_counts = Counter(flag for item in profile for flag in item[2])
    return mask, weights, {
        "eligible": int(mask.sum()),
        "rejected_from_current": int((base & ~mask).sum()),
        "downweighted": int(np.sum((weights < 1.0) & mask)),
        "flag_counts": dict(sorted(flag_counts.items())),
    }


def _corpus_fingerprint(corpus: Corpus) -> str:
    digest = hashlib.sha256()
    for chunk in corpus.chunks:
        digest.update(
            f"{chunk.source_id}\0{chunk.start_ms}\0{chunk.end_ms}\0{chunk.text}\n".encode("utf-8")
        )
    return digest.hexdigest()


def _model_source(model_root: Path, model_name: str) -> Path | str:
    cache_dir = model_root / ("models--" + model_name.replace("/", "--")) / "snapshots"
    if cache_dir.exists():
        snapshots = sorted(path for path in cache_dir.iterdir() if path.is_dir())
        for snapshot in reversed(snapshots):
            has_weights = any((snapshot / filename).exists() for filename in (
                "model.safetensors",
                "pytorch_model.bin",
                "model.safetensors.index.json",
                "pytorch_model.bin.index.json",
            ))
            if has_weights:
                return snapshot
    return model_name


def encode_model(
    corpus: Corpus,
    queries: list[dict],
    model_key: str,
    model_root: Path,
    cache_dir: Path,
    device: str,
    allow_download: bool,
) -> tuple[np.ndarray, np.ndarray, dict]:
    spec = MODEL_SPECS[model_key]
    fingerprint = _corpus_fingerprint(corpus)
    cache_path = cache_dir / f"{model_key}_corpus.npz"
    meta_path = cache_dir / f"{model_key}_corpus.meta.json"
    corpus_embeddings = None
    cache_hit = False
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as data:
            cached_fingerprint = str(data["fingerprint"].tolist()[0])
            if cached_fingerprint == fingerprint:
                corpus_embeddings = np.asarray(data["embeddings"], dtype=np.float32)
                cache_hit = True

    from sentence_transformers import SentenceTransformer

    source = _model_source(model_root, spec["name"])
    local_only = isinstance(source, Path) or not allow_download
    load_started = time.perf_counter()
    model = SentenceTransformer(
        str(source),
        cache_folder=str(model_root),
        device=device,
        local_files_only=local_only,
        trust_remote_code=bool(spec["trust_remote_code"]),
    )
    load_seconds = time.perf_counter() - load_started

    corpus_encode_seconds = 0.0
    if corpus_embeddings is None:
        texts = [spec["passage_prefix"] + chunk.text for chunk in corpus.chunks]
        started = time.perf_counter()
        corpus_embeddings = np.asarray(model.encode(
            texts,
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        ), dtype=np.float32)
        corpus_encode_seconds = time.perf_counter() - started
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            fingerprint=np.asarray([fingerprint]),
            embeddings=corpus_embeddings.astype(np.float16),
        )
        meta_path.write_text(json.dumps({
            "model_key": model_key,
            "model_name": spec["name"],
            "corpus_chunks": len(corpus.chunks),
            "embedding_dim": int(corpus_embeddings.shape[1]),
            "encode_seconds": corpus_encode_seconds,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
    elif meta_path.exists():
        cached_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        corpus_encode_seconds = float(cached_meta.get("encode_seconds") or 0.0)

    query_texts = [spec["query_prefix"] + item["query"] for item in queries]
    started = time.perf_counter()
    query_embeddings = np.asarray(model.encode(
        query_texts,
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ), dtype=np.float32)
    query_encode_seconds = time.perf_counter() - started
    dim = int(corpus_embeddings.shape[1])
    stats = {
        "model_key": model_key,
        "model_name": spec["name"],
        "embedding_dim": dim,
        "cache_hit": cache_hit,
        "model_load_seconds": load_seconds,
        "corpus_encode_seconds": corpus_encode_seconds,
        "query_encode_seconds": query_encode_seconds,
        "float16_index_mb": len(corpus.chunks) * dim * 2 / (1024 * 1024),
    }
    del model
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    return corpus_embeddings, query_embeddings, stats


def _percentiles(values: np.ndarray) -> np.ndarray:
    if not len(values):
        return np.empty(0, dtype=np.float32)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float32)
    ranks[order] = np.arange(1, len(values) + 1, dtype=np.float32)
    return ranks / max(1, len(values))


def _confidence(cosines: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-10.0 * (cosines - 0.35)))


def semantic_scores(
    corpus: Corpus,
    corpus_embeddings: np.ndarray,
    query_embeddings: np.ndarray,
    eligible: np.ndarray,
    weights: np.ndarray,
    calibration: dict,
) -> np.ndarray:
    cosines = query_embeddings @ corpus_embeddings.T
    scores = np.full(cosines.shape, -np.inf, dtype=np.float32)
    kind = calibration["kind"]
    for query_index in range(len(query_embeddings)):
        for indices in corpus.source_indices.values():
            selected = indices[eligible[indices]]
            if not len(selected):
                continue
            values = cosines[query_index, selected]
            absolute = _confidence(values)
            percentile = _percentiles(values)
            if kind == "absolute":
                calibrated = absolute
            elif kind == "percentile":
                calibrated = percentile
            elif kind == "current":
                calibrated = 0.7 * absolute + 0.3 * percentile
            elif kind == "gated":
                percentile_weight = float(calibration["percentile_weight"])
                floor = float(calibration["cosine_floor"])
                calibrated = absolute.copy()
                passed = values >= floor
                calibrated[passed] = (
                    (1.0 - percentile_weight) * absolute[passed]
                    + percentile_weight * percentile[passed]
                )
            else:
                raise ValueError(f"Unknown calibration: {kind}")
            scores[query_index, selected] = calibrated * weights[selected]
    return scores


def _rank(scores: np.ndarray, candidates: np.ndarray | None = None) -> list[int]:
    if candidates is None:
        candidates = np.flatnonzero(np.isfinite(scores))
    else:
        candidates = candidates[np.isfinite(scores[candidates])]
    return sorted((int(index) for index in candidates), key=lambda index: (-float(scores[index]), index))


def _rank_of_targets(ranking: list[int], targets: set[int]) -> int | None:
    for rank, index in enumerate(ranking, start=1):
        if index in targets:
            return rank
    return None


def evaluate_rankings(
    queries: list[dict],
    rankings: list[list[int]],
    top_scores: list[float],
) -> list[dict]:
    rows = []
    for query, ranking, top_score in zip(queries, rankings, top_scores):
        targets = query["target_indices"]
        rows.append({
            "id": query["id"],
            "split": query["split"],
            "category": query["category"],
            "domain": query["domain"],
            "query": query["query"],
            "answerable": bool(targets),
            "rank": _rank_of_targets(ranking, targets) if targets else None,
            "top_score": float(top_score),
            "top_source_id": corpus_source_id(ranking[0]) if ranking else None,
        })
    return rows


_ACTIVE_CORPUS: Corpus | None = None


def corpus_source_id(index: int) -> str:
    assert _ACTIVE_CORPUS is not None
    return _ACTIVE_CORPUS.chunks[index].source_id


def _metric_rows(rows: list[dict], split: str = "all", domain: str = "all") -> dict:
    selected = [
        row for row in rows
        if (split == "all" or row["split"] == split)
        and (domain == "all" or row["domain"] == domain)
    ]
    answerable = [row for row in selected if row["answerable"]]
    ranks = [row["rank"] for row in answerable]
    finite = [rank for rank in ranks if rank is not None]
    result = {
        "queries": len(selected),
        "answerable": len(answerable),
        "no_answer": len(selected) - len(answerable),
        "mrr": sum(1.0 / rank for rank in finite) / max(1, len(answerable)),
    }
    for cutoff in (1, 5, 10, 20, 50):
        result[f"hit@{cutoff}"] = sum(
            rank is not None and rank <= cutoff for rank in ranks
        ) / max(1, len(answerable))
    no_answer_scores = [row["top_score"] for row in selected if not row["answerable"]]
    result["no_answer_mean_top_score"] = (
        float(np.mean(no_answer_scores)) if no_answer_scores else None
    )
    return result


def _objective(metrics: dict) -> float:
    return (
        0.50 * metrics["mrr"]
        + 0.25 * metrics["hit@5"]
        + 0.15 * metrics["hit@10"]
        + 0.10 * metrics["hit@50"]
    )


def summarize(rows: list[dict]) -> dict:
    return {
        split: {
            domain: _metric_rows(rows, split, domain)
            for domain in ("all", "platform", "open")
        }
        for split in ("tune", "dev", "holdout", "all")
    }


def evaluate_score_matrix(queries: list[dict], matrix: np.ndarray) -> tuple[list[dict], dict]:
    rankings = [_rank(matrix[index]) for index in range(len(queries))]
    top_scores = [float(matrix[index, ranking[0]]) if ranking else -math.inf for index, ranking in enumerate(rankings)]
    rows = evaluate_rankings(queries, rankings, top_scores)
    return rows, summarize(rows)


def _lexical_normalized(text: str) -> str:
    return _normalize_text(text)


def legacy_lexical_matrix(corpus: Corpus, queries: list[dict]) -> np.ndarray:
    texts = [_lexical_normalized(chunk.text) for chunk in corpus.chunks]
    matrix = np.zeros((len(queries), len(corpus.chunks)), dtype=np.float32)
    for query_index, query in enumerate(queries):
        query_value = _lexical_normalized(query["query"])
        if not query_value:
            continue
        size = 2 if len(query_value) > 1 else 1
        query_grams = {query_value[index:index + size] for index in range(max(1, len(query_value) - size + 1))}
        for chunk_index, text_value in enumerate(texts):
            if query_value in text_value:
                matrix[query_index, chunk_index] = 1.0
                continue
            text_grams = {text_value[index:index + size] for index in range(max(1, len(text_value) - size + 1))}
            matrix[query_index, chunk_index] = len(query_grams & text_grams) / max(1, len(query_grams))
    return matrix


TOKEN_PATTERN = re.compile(r"[a-z0-9]+|[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+", re.IGNORECASE)


def _lexical_tokens(text: str) -> list[str]:
    value = unicodedata.normalize("NFKC", text).casefold()
    tokens: list[str] = []
    for match in TOKEN_PATTERN.finditer(value):
        part = match.group(0)
        if part and _is_cjk(part[0]):
            if len(part) == 1:
                tokens.append(part)
            else:
                tokens.extend(part[index:index + 2] for index in range(len(part) - 1))
        else:
            tokens.append(part)
    return tokens


def advanced_lexical_matrices(corpus: Corpus, queries: list[dict]) -> tuple[np.ndarray, np.ndarray, dict]:
    documents = [_lexical_tokens(chunk.text) for chunk in corpus.chunks]
    counters = [Counter(tokens) for tokens in documents]
    document_frequency = Counter(token for counter in counters for token in counter)
    document_count = len(documents)
    idf = {
        token: math.log((document_count - frequency + 0.5) / (frequency + 0.5) + 1.0)
        for token, frequency in document_frequency.items()
    }
    lengths = np.asarray([len(tokens) for tokens in documents], dtype=np.float32)
    average_length = float(lengths.mean()) if len(lengths) else 1.0
    weighted = np.zeros((len(queries), document_count), dtype=np.float32)
    bm25 = np.zeros_like(weighted)
    k1, b = 1.2, 0.75
    for query_index, query in enumerate(queries):
        query_tokens = list(dict.fromkeys(_lexical_tokens(query["query"])))
        denominator = sum(idf.get(token, math.log(document_count + 1.0)) for token in query_tokens)
        for document_index, counter in enumerate(counters):
            matched = sum(idf.get(token, 0.0) for token in query_tokens if token in counter)
            weighted[query_index, document_index] = matched / max(1e-9, denominator)
            score = 0.0
            for token in query_tokens:
                frequency = counter.get(token, 0)
                if not frequency:
                    continue
                token_idf = idf.get(token, 0.0)
                norm = frequency + k1 * (1.0 - b + b * lengths[document_index] / max(1.0, average_length))
                score += token_idf * frequency * (k1 + 1.0) / norm
            bm25[query_index, document_index] = score
        max_score = float(bm25[query_index].max())
        if max_score > 0:
            bm25[query_index] /= max_score
    return weighted, bm25, {
        "vocabulary": len(document_frequency),
        "average_document_tokens": average_length,
    }


def _write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _report_table(results: dict[str, dict]) -> str:
    lines = [
        "| strategy | split | domain | MRR | H@1 | H@5 | H@10 | H@50 |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for strategy, summary in results.items():
        for split in ("tune", "dev", "holdout", "all"):
            for domain in ("all", "platform", "open"):
                metrics = summary[split][domain]
                if not metrics["answerable"]:
                    continue
                lines.append(
                    f"| {strategy} | {split} | {domain} | {metrics['mrr']:.3f} | "
                    f"{metrics['hit@1']:.3f} | {metrics['hit@5']:.3f} | "
                    f"{metrics['hit@10']:.3f} | {metrics['hit@50']:.3f} |"
                )
    return "\n".join(lines)


def stage1(corpus: Corpus, queries: list[dict], output: Path) -> None:
    target_counts = [len(query["target_indices"]) for query in queries if query["target_indices"]]
    open_source_splits: dict[str, set[str]] = defaultdict(set)
    review_rows = []
    for query in queries:
        target_chunks = [corpus.chunks[index] for index in sorted(query["target_indices"])]
        for chunk in target_chunks:
            if chunk.source_group == "open":
                open_source_splits[chunk.source_id].add(query["split"])
        review_rows.append({
            "id": query["id"],
            "split": query["split"],
            "category": query["category"],
            "query": query["query"],
            "targets": [
                {
                    "source_id": chunk.source_id,
                    "source_group": chunk.source_group,
                    "start_ms": chunk.start_ms,
                    "end_ms": chunk.end_ms,
                    "asr_text": chunk.text,
                }
                for chunk in target_chunks
            ],
        })
    leaking_sources = {
        source_id: sorted(splits)
        for source_id, splits in open_source_splits.items()
        if len(splits) > 1
    }
    if leaking_sources:
        raise ValueError(f"Open source split leakage: {leaking_sources}")
    payload = {
        "corpus": {
            "chunks": len(corpus.chunks),
            "sources": len(corpus.source_indices),
            "platform_sources": len({chunk.source_id for chunk in corpus.chunks if chunk.source_group == "platform"}),
            "open_sources": len({chunk.source_id for chunk in corpus.chunks if chunk.source_group == "open"}),
            "open_audio_hours": corpus.duration_hours["open"],
        },
        "queries": {
            "total": len(queries),
            "answerable": sum(bool(query["target_indices"]) for query in queries),
            "no_answer": sum(not query["target_indices"] for query in queries),
            "splits": dict(Counter(query["split"] for query in queries)),
            "domains": dict(Counter(query["domain"] for query in queries)),
            "categories": dict(Counter(query["category"] for query in queries)),
            "min_target_chunks": min(target_counts),
            "max_target_chunks": max(target_counts),
        },
        "validation": {
            "qrels_resolved": "passed",
            "open_source_split_isolation": "passed",
        },
    }
    stage_dir = output / "stage1_dataset"
    _write_json(stage_dir / "summary.json", payload)
    with (stage_dir / "query_target_review.jsonl").open("w", encoding="utf-8") as handle:
        for row in review_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (output / "stage1_dataset" / "REPORT.md").write_text(
        "# Stage 1 - 联合 ASR 检索评测集\n\n"
        f"- corpus: {payload['corpus']['sources']} sources / {payload['corpus']['chunks']} chunks\n"
        f"- open audio: {payload['corpus']['open_audio_hours']:.3f} hours\n"
        f"- queries: {payload['queries']['total']} total / {payload['queries']['answerable']} answerable / {payload['queries']['no_answer']} no-answer\n"
        f"- splits: `{payload['queries']['splits']}`\n"
        f"- validation: every answerable query resolved to actual ASR chunks; open sources are split-isolated\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def stage2(args, corpus: Corpus, queries: list[dict], output: Path) -> None:
    corpus_embeddings, query_embeddings, encoding = encode_model(
        corpus, queries, "minilm", args.model_root, output / "cache", args.device, args.allow_download
    )
    variants = ["current", "obvious_hard_filter", "hard_filter_soft_penalty"]
    summaries, rows_by_variant, policy_stats = {}, {}, {}
    for variant in variants:
        mask, weights, stats = semantic_policy(corpus, variant)
        matrix = semantic_scores(
            corpus, corpus_embeddings, query_embeddings, mask, weights, {"kind": "current"}
        )
        rows, summary = evaluate_score_matrix(queries, matrix)
        summaries[variant] = summary
        rows_by_variant[variant] = rows
        policy_stats[variant] = stats
    objectives = {
        name: _objective(summaries[name]["tune"]["all"])
        for name in variants
    }
    best_variant = max(variants, key=lambda name: objectives[name])
    # Do not add a heuristic policy for tiny rank-only movement. A new policy
    # must improve the weighted tuning objective by at least 0.5 percentage
    # points; otherwise the existing eligibility contract remains selected.
    minimum_material_gain = 0.005
    selected = (
        best_variant
        if objectives[best_variant] >= objectives["current"] + minimum_material_gain
        else "current"
    )
    payload = {
        "selected": selected,
        "best_raw_variant": best_variant,
        "minimum_material_gain": minimum_material_gain,
        "objectives": objectives,
        "selection_split": "tune",
        "encoding": encoding,
        "policy_stats": policy_stats,
        "metrics": summaries,
        "rows": rows_by_variant,
    }
    _write_json(output / "stage2_low_information" / "results.json", payload)
    (output / "stage2_low_information" / "REPORT.md").write_text(
        "# Stage 2 - 低信息 semantic 过滤 A/B\n\n"
        f"selected on tune: `{selected}`\n\n" + _report_table(summaries) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"selected": selected, "metrics": summaries}, ensure_ascii=False, indent=2))


def _load_selected(output: Path, stage_dir: str, key: str = "selected") -> str:
    payload = json.loads((output / stage_dir / "results.json").read_text(encoding="utf-8"))
    return str(payload[key])


def stage3(args, corpus: Corpus, queries: list[dict], output: Path) -> None:
    policy = _load_selected(output, "stage2_low_information")
    mask, weights, _ = semantic_policy(corpus, policy)
    corpus_embeddings, query_embeddings, encoding = encode_model(
        corpus, queries, "minilm", args.model_root, output / "cache", args.device, args.allow_download
    )
    calibrations = {
        "absolute": {"kind": "absolute"},
        "percentile_only": {"kind": "percentile"},
        "current_70_30": {"kind": "current"},
    }
    for floor in (0.25, 0.35, 0.45):
        for percentile_weight in (0.10, 0.20, 0.30):
            calibrations[f"gated_f{floor:.2f}_p{percentile_weight:.2f}"] = {
                "kind": "gated",
                "cosine_floor": floor,
                "percentile_weight": percentile_weight,
            }
    summaries, rows_by_variant = {}, {}
    for name, calibration in calibrations.items():
        matrix = semantic_scores(corpus, corpus_embeddings, query_embeddings, mask, weights, calibration)
        rows, summary = evaluate_score_matrix(queries, matrix)
        summaries[name] = summary
        rows_by_variant[name] = rows
    selected = max(
        calibrations,
        key=lambda name: (_objective(summaries[name]["tune"]["all"]), name == "current_70_30"),
    )
    payload = {
        "selected": selected,
        "selected_config": calibrations[selected],
        "semantic_policy": policy,
        "encoding": encoding,
        "metrics": summaries,
        "rows": rows_by_variant,
    }
    _write_json(output / "stage3_calibration" / "results.json", payload)
    (output / "stage3_calibration" / "REPORT.md").write_text(
        "# Stage 3 - semantic 分数校准 A/B\n\n"
        f"selected on tune: `{selected}` `{calibrations[selected]}`\n\n" + _report_table(summaries) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"selected": selected, "config": calibrations[selected], "metrics": summaries}, ensure_ascii=False, indent=2))


def stage4(args, corpus: Corpus, queries: list[dict], output: Path) -> None:
    policy = _load_selected(output, "stage2_low_information")
    stage3_payload = json.loads((output / "stage3_calibration" / "results.json").read_text(encoding="utf-8"))
    calibration = stage3_payload["selected_config"]
    mask, weights, _ = semantic_policy(corpus, policy)
    summaries, rows_by_model, timings = {}, {}, {}
    for model_key in MODEL_SPECS:
        corpus_embeddings, query_embeddings, stats = encode_model(
            corpus, queries, model_key, args.model_root, output / "cache", args.device, args.allow_download
        )
        matrix = semantic_scores(corpus, corpus_embeddings, query_embeddings, mask, weights, calibration)
        rows, summary = evaluate_score_matrix(queries, matrix)
        summaries[model_key] = summary
        rows_by_model[model_key] = rows
        timings[model_key] = stats
    selected = max(
        MODEL_SPECS,
        key=lambda name: (_objective(summaries[name]["tune"]["all"]), name == "minilm"),
    )
    payload = {
        "selected": selected,
        "semantic_policy": policy,
        "calibration": calibration,
        "timings": timings,
        "metrics": summaries,
        "rows": rows_by_model,
    }
    _write_json(output / "stage4_embedding_models" / "results.json", payload)
    (output / "stage4_embedding_models" / "REPORT.md").write_text(
        "# Stage 4 - 文本 embedding 模型对比\n\n"
        f"selected on tune: `{selected}`\n\n" + _report_table(summaries) + "\n\n"
        + "```json\n" + json.dumps(timings, ensure_ascii=False, indent=2) + "\n```\n",
        encoding="utf-8",
    )
    print(json.dumps({"selected": selected, "timings": timings, "metrics": summaries}, ensure_ascii=False, indent=2))


def stage5(corpus: Corpus, queries: list[dict], output: Path) -> None:
    legacy = legacy_lexical_matrix(corpus, queries)
    weighted, bm25, lexical_stats = advanced_lexical_matrices(corpus, queries)
    matrices = {"legacy_bigram": legacy, "idf_weighted": weighted, "bm25": bm25}
    summaries, rows_by_method = {}, {}
    for name, matrix in matrices.items():
        rows, summary = evaluate_score_matrix(queries, np.where(matrix > 0, matrix, -np.inf))
        summaries[name] = summary
        rows_by_method[name] = rows
    selected = max(
        matrices,
        key=lambda name: (_objective(summaries[name]["tune"]["all"]), name == "legacy_bigram"),
    )
    payload = {
        "selected": selected,
        "lexical_stats": lexical_stats,
        "metrics": summaries,
        "rows": rows_by_method,
    }
    _write_json(output / "stage5_lexical" / "results.json", payload)
    (output / "stage5_lexical" / "REPORT.md").write_text(
        "# Stage 5 - lexical 检索对比\n\n"
        f"selected on tune: `{selected}`\n\n" + _report_table(summaries) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"selected": selected, "metrics": summaries}, ensure_ascii=False, indent=2))


def _rrf_ranking(semantic: np.ndarray, lexical: np.ndarray, pool: int, k: int, lexical_weight: float) -> list[int]:
    semantic_order = _rank(semantic)[:pool]
    lexical_order = _rank(np.where(lexical > 0, lexical, -np.inf))[:pool]
    semantic_ranks = {index: rank for rank, index in enumerate(semantic_order, start=1)}
    lexical_ranks = {index: rank for rank, index in enumerate(lexical_order, start=1)}
    candidates = set(semantic_ranks) | set(lexical_ranks)
    return sorted(candidates, key=lambda index: (
        -(1.0 / (k + semantic_ranks[index]) if index in semantic_ranks else 0.0)
        -(lexical_weight / (k + lexical_ranks[index]) if index in lexical_ranks else 0.0),
        -float(lexical[index]),
        -float(semantic[index]),
        index,
    ))


def _reserved_ranking(primary_scores: np.ndarray, lexical: np.ndarray, threshold: float, initial: int, run: int) -> list[int]:
    primary = _rank(primary_scores)
    lexical_order = _rank(np.where(lexical >= threshold, lexical, -np.inf))
    ranking: list[int] = []
    emitted: set[int] = set()
    primary_pos = lexical_pos = 0

    def emit(index: int) -> None:
        if index not in emitted:
            emitted.add(index)
            ranking.append(index)

    while primary_pos < min(initial, len(primary)):
        emit(primary[primary_pos])
        primary_pos += 1
    while primary_pos < len(primary) or lexical_pos < len(lexical_order):
        while lexical_pos < len(lexical_order):
            index = lexical_order[lexical_pos]
            lexical_pos += 1
            if index not in emitted:
                emit(index)
                break
        taken = 0
        while primary_pos < len(primary) and taken < run:
            index = primary[primary_pos]
            primary_pos += 1
            if index in emitted:
                continue
            emit(index)
            taken += 1
        if primary_pos >= len(primary) and lexical_pos >= len(lexical_order):
            break
    return ranking


def _evaluate_custom_rankings(queries: list[dict], rankings: list[list[int]], top_score_matrix: np.ndarray) -> tuple[list[dict], dict]:
    top_scores = [
        float(top_score_matrix[index, ranking[0]]) if ranking else -math.inf
        for index, ranking in enumerate(rankings)
    ]
    rows = evaluate_rankings(queries, rankings, top_scores)
    return rows, summarize(rows)


def _strong_lexical_safety_failures(
    queries: list[dict],
    rows: list[dict],
    lexical: np.ndarray,
    splits: tuple[str, ...] = ("tune", "dev"),
    minimum_score: float = 0.50,
    maximum_rank: int = 5,
) -> list[str]:
    """Protect labeled targets that are already the strongest lexical hit."""
    failures: list[str] = []
    rows_by_id = {row["id"]: row for row in rows}
    for query_index, query in enumerate(queries):
        if query["split"] not in splits or not query["target_indices"]:
            continue
        target_score = max(float(lexical[query_index, index]) for index in query["target_indices"])
        best_score = float(lexical[query_index].max())
        if target_score < minimum_score or target_score < best_score - 1e-6:
            continue
        rank = rows_by_id[query["id"]]["rank"]
        if rank is None or rank > maximum_rank:
            failures.append(query["id"])
    return failures


def stage6(args, corpus: Corpus, queries: list[dict], output: Path) -> None:
    policy = _load_selected(output, "stage2_low_information")
    calibration = json.loads((output / "stage3_calibration" / "results.json").read_text(encoding="utf-8"))["selected_config"]
    model_key = _load_selected(output, "stage4_embedding_models")
    lexical_method = _load_selected(output, "stage5_lexical")
    mask, weights, _ = semantic_policy(corpus, policy)
    corpus_embeddings, query_embeddings, encoding = encode_model(
        corpus, queries, model_key, args.model_root, output / "cache", args.device, args.allow_download
    )
    semantic = semantic_scores(corpus, corpus_embeddings, query_embeddings, mask, weights, calibration)
    legacy = legacy_lexical_matrix(corpus, queries)
    weighted, bm25, _ = advanced_lexical_matrices(corpus, queries)
    lexical = {"legacy_bigram": legacy, "idf_weighted": weighted, "bm25": bm25}[lexical_method]

    summaries: dict[str, dict] = {}
    rows_by_strategy: dict[str, list[dict]] = {}

    # Same-corpus reference for the currently deployed MiniLM scoring formula.
    current_mask, current_weights, _ = semantic_policy(corpus, "current")
    current_corpus_embeddings, current_query_embeddings, current_encoding = encode_model(
        corpus, queries, "minilm", args.model_root, output / "cache", args.device, args.allow_download
    )
    current_semantic = semantic_scores(
        corpus,
        current_corpus_embeddings,
        current_query_embeddings,
        current_mask,
        current_weights,
        {"kind": "current"},
    )
    current_semantic_available = np.isfinite(current_semantic)
    current_union = current_semantic_available | (legacy > 0)
    current_semantic_values = np.where(current_semantic_available, current_semantic, 0.0)
    production_matrix = np.maximum(
        legacy,
        0.65 * current_semantic_values + 0.35 * legacy,
    )
    production_matrix = np.where(current_union, production_matrix, -np.inf)
    rows, summary = evaluate_score_matrix(queries, production_matrix)
    summaries["reference_minilm_max_blend"] = summary
    rows_by_strategy["reference_minilm_max_blend"] = rows
    production_rankings = [
        _reserved_ranking(production_matrix[index], legacy[index], 0.50, 3, 8)
        for index in range(len(queries))
    ]
    rows, summary = _evaluate_custom_rankings(queries, production_rankings, production_matrix)
    summaries["reference_minilm_reserve"] = summary
    rows_by_strategy["reference_minilm_reserve"] = rows

    minilm_absolute = semantic_scores(
        corpus,
        current_corpus_embeddings,
        current_query_embeddings,
        current_mask,
        current_weights,
        {"kind": "absolute"},
    )
    minilm_available = np.isfinite(minilm_absolute)
    minilm_union = minilm_available | (legacy > 0)
    minilm_values = np.where(minilm_available, minilm_absolute, 0.0)
    minilm_priority = 0.90 * minilm_values + 0.10 * legacy
    minilm_priority = np.where(
        legacy >= 0.50,
        1.0 + 0.01 * legacy,
        minilm_priority,
    )
    minilm_priority = np.where(minilm_union, minilm_priority, -np.inf)
    rows, summary = evaluate_score_matrix(queries, minilm_priority)
    summaries["minilm_absolute_priority_90_10"] = summary
    rows_by_strategy["minilm_absolute_priority_90_10"] = rows

    rows, summary = evaluate_score_matrix(queries, semantic)
    summaries["semantic_only"] = summary
    rows_by_strategy["semantic_only"] = rows
    rows, summary = evaluate_score_matrix(queries, np.where(lexical > 0, lexical, -np.inf))
    summaries["lexical_only"] = summary
    rows_by_strategy["lexical_only"] = rows

    semantic_available = np.isfinite(semantic)
    candidate_union = semantic_available | (lexical > 0)
    semantic_values = np.where(semantic_available, semantic, 0.0)
    linear_trials = []
    for lexical_weight in (0.10, 0.20, 0.30, 0.40, 0.50, 0.60):
        matrix = (1.0 - lexical_weight) * semantic_values + lexical_weight * lexical
        matrix = np.where(candidate_union, matrix, -np.inf)
        rows, summary = evaluate_score_matrix(queries, matrix)
        linear_trials.append((lexical_weight, rows, summary, matrix))
    best_linear = max(linear_trials, key=lambda item: _objective(item[2]["tune"]["all"]))
    linear_name = f"linear_l{best_linear[0]:.2f}"
    summaries[linear_name] = best_linear[2]
    rows_by_strategy[linear_name] = best_linear[1]

    guarded_trials = []
    for lexical_weight in (0.10, 0.20, 0.30):
        base_matrix = (1.0 - lexical_weight) * semantic_values + lexical_weight * lexical
        for lexical_threshold in (0.50, 0.65, 0.90):
            matrix = np.where(
                lexical >= lexical_threshold,
                np.maximum(base_matrix, lexical),
                base_matrix,
            )
            matrix = np.where(candidate_union, matrix, -np.inf)
            rows, summary = evaluate_score_matrix(queries, matrix)
            guarded_trials.append((lexical_weight, lexical_threshold, rows, summary, matrix))
    best_guarded = max(guarded_trials, key=lambda item: _objective(item[3]["tune"]["all"]))
    guarded_name = f"guarded_linear_l{best_guarded[0]:.2f}_t{best_guarded[1]:.2f}"
    summaries[guarded_name] = best_guarded[3]
    rows_by_strategy[guarded_name] = best_guarded[2]

    priority_trials = []
    for lexical_weight in (0.10, 0.20):
        base_matrix = (1.0 - lexical_weight) * semantic_values + lexical_weight * lexical
        for lexical_threshold in (0.50, 0.65, 0.80, 0.90):
            # A strong lexical hit belongs to an independent priority band.
            # Scores inside that band retain lexical ordering; everything else
            # keeps the tuned semantic-primary linear score.
            matrix = np.where(
                lexical >= lexical_threshold,
                1.0 + 0.01 * lexical,
                base_matrix,
            )
            matrix = np.where(candidate_union, matrix, -np.inf)
            rows, summary = evaluate_score_matrix(queries, matrix)
            priority_trials.append((lexical_weight, lexical_threshold, rows, summary, matrix))
    safe_priority_trials = [
        item
        for item in priority_trials
        if not _strong_lexical_safety_failures(queries, item[2], lexical)
    ]
    best_priority = max(
        safe_priority_trials or priority_trials,
        key=lambda item: _objective(item[3]["tune"]["all"]),
    )
    priority_name = f"priority_linear_l{best_priority[0]:.2f}_t{best_priority[1]:.2f}"
    summaries[priority_name] = best_priority[3]
    rows_by_strategy[priority_name] = best_priority[2]

    current_matrix = np.maximum(lexical, 0.65 * semantic_values + 0.35 * lexical)
    current_matrix = np.where(candidate_union, current_matrix, -np.inf)
    rows, summary = evaluate_score_matrix(queries, current_matrix)
    summaries["max_blend_65_35"] = summary
    rows_by_strategy["max_blend_65_35"] = rows

    rrf_trials = []
    for pool in (50, 100):
        for k in (20, 60):
            for lexical_weight in (0.75, 1.0, 1.25):
                rankings = [
                    _rrf_ranking(semantic[index], lexical[index], pool, k, lexical_weight)
                    for index in range(len(queries))
                ]
                rows, summary = _evaluate_custom_rankings(queries, rankings, current_matrix)
                rrf_trials.append((pool, k, lexical_weight, rows, summary))
    best_rrf = max(rrf_trials, key=lambda item: _objective(item[4]["tune"]["all"]))
    rrf_name = f"rrf_p{best_rrf[0]}_k{best_rrf[1]}_l{best_rrf[2]:.2f}"
    summaries[rrf_name] = best_rrf[4]
    rows_by_strategy[rrf_name] = best_rrf[3]

    reserve_trials = []
    for threshold in (0.35, 0.50, 0.65):
        for initial in (1, 3, 5):
            for run in (4, 8):
                rankings = [
                    _reserved_ranking(current_matrix[index], lexical[index], threshold, initial, run)
                    for index in range(len(queries))
                ]
                rows, summary = _evaluate_custom_rankings(queries, rankings, current_matrix)
                reserve_trials.append((threshold, initial, run, rows, summary))
    best_reserve = max(reserve_trials, key=lambda item: _objective(item[4]["tune"]["all"]))
    reserve_name = f"reserve_t{best_reserve[0]:.2f}_i{best_reserve[1]}_r{best_reserve[2]}"
    summaries[reserve_name] = best_reserve[4]
    rows_by_strategy[reserve_name] = best_reserve[3]

    safety_failures = {
        name: _strong_lexical_safety_failures(queries, rows_by_strategy[name], lexical)
        for name in summaries
    }
    safe_strategies = [name for name in summaries if not safety_failures[name]]
    selected = max(
        safe_strategies or list(summaries),
        key=lambda name: _objective(summaries[name]["tune"]["all"]),
    )
    payload = {
        "selected": selected,
        "inputs": {
            "semantic_policy": policy,
            "calibration": calibration,
            "embedding_model": model_key,
            "lexical_method": lexical_method,
        },
        "selected_linear_weight": best_linear[0],
        "selected_guarded_linear": {
            "lexical_weight": best_guarded[0],
            "lexical_threshold": best_guarded[1],
        },
        "selected_priority_linear": {
            "lexical_weight": best_priority[0],
            "lexical_threshold": best_priority[1],
        },
        "strong_lexical_safety": {
            "splits": ["tune", "dev"],
            "minimum_lexical_score": 0.50,
            "maximum_rank": 5,
            "failures": safety_failures,
        },
        "selected_rrf": {"pool": best_rrf[0], "k": best_rrf[1], "lexical_weight": best_rrf[2]},
        "selected_reserve": {"threshold": best_reserve[0], "initial": best_reserve[1], "run": best_reserve[2]},
        "encoding": encoding,
        "reference_encoding": current_encoding,
        "metrics": summaries,
        "rows": rows_by_strategy,
    }
    _write_json(output / "stage6_fusion" / "results.json", payload)
    (output / "stage6_fusion" / "REPORT.md").write_text(
        "# Stage 6 - semantic / lexical 融合\n\n"
        f"selected on tune: `{selected}`\n\n" + _report_table(summaries) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"selected": selected, "inputs": payload["inputs"], "metrics": summaries}, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, choices=range(1, 7), required=True)
    parser.add_argument("--index-root", type=Path, required=True)
    parser.add_argument("--open-eval-root", type=Path, required=True)
    parser.add_argument("--platform-queries", type=Path, required=True)
    parser.add_argument("--open-queries", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args()

    global _ACTIVE_CORPUS
    corpus = load_corpus(args.index_root, args.open_eval_root)
    _ACTIVE_CORPUS = corpus
    queries = prepare_queries([args.platform_queries, args.open_queries], corpus)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stages: dict[int, Callable] = {
        1: lambda: stage1(corpus, queries, args.output_dir),
        2: lambda: stage2(args, corpus, queries, args.output_dir),
        3: lambda: stage3(args, corpus, queries, args.output_dir),
        4: lambda: stage4(args, corpus, queries, args.output_dir),
        5: lambda: stage5(corpus, queries, args.output_dir),
        6: lambda: stage6(args, corpus, queries, args.output_dir),
    }
    stages[args.stage]()


if __name__ == "__main__":
    main()
