from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from app.indexing.common import normalize
from app.indexing.text_semantic import TextEmbeddingEncoder
from app.search import asr_semantic_confidence, normalize_text, robust_distribution


@dataclass(frozen=True)
class Chunk:
    global_id: int
    video_id: str
    chunk_id: int
    start_ms: int
    end_ms: int
    text: str


@dataclass
class Corpus:
    chunks: list[Chunk]
    video_indices: dict[str, np.ndarray]
    semantic_embeddings: np.ndarray
    semantic_available: np.ndarray


def _load_queries(path: Path) -> list[dict]:
    queries = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            if not item.get("id") or not item.get("query") or not item.get("targets"):
                raise ValueError(f"Invalid query at {path}:{line_number}")
            queries.append(item)
    return queries


def _decode_texts(values: np.ndarray) -> list[str]:
    return [str(value) for value in values.tolist()]


def _load_corpus(index_root: Path) -> Corpus:
    rows: list[tuple[Chunk, np.ndarray | None]] = []
    video_indices: dict[str, list[int]] = {}
    embedding_dim = 0

    for index_dir in sorted(path for path in index_root.iterdir() if path.is_dir()):
        index_path = index_dir / "asr.npz"
        if not index_path.exists():
            continue
        with np.load(index_path, allow_pickle=False) as data:
            required = {"chunk_times_ms", "texts"}
            if not required.issubset(data.files):
                continue
            times = np.asarray(data["chunk_times_ms"], dtype=np.int32)
            texts = _decode_texts(data["texts"])
            if times.shape != (len(texts), 2):
                raise ValueError(f"Invalid ASR arrays: {index_path}")

            local_embeddings: dict[int, np.ndarray] = {}
            if {"embeddings", "embedding_chunk_indices"}.issubset(data.files):
                embeddings = np.asarray(data["embeddings"], dtype=np.float32)
                embedding_indices = np.asarray(data["embedding_chunk_indices"], dtype=np.int32)
                if len(embeddings) != len(embedding_indices):
                    raise ValueError(f"Invalid semantic arrays: {index_path}")
                if embeddings.ndim == 2 and embeddings.shape[1]:
                    embedding_dim = embedding_dim or int(embeddings.shape[1])
                    if embedding_dim != int(embeddings.shape[1]):
                        raise ValueError("Mixed semantic embedding dimensions are not supported")
                    for local_index, chunk_index in enumerate(embedding_indices):
                        local_embeddings[int(chunk_index)] = embeddings[local_index]

            for chunk_id, text in enumerate(texts):
                global_id = len(rows)
                chunk = Chunk(
                    global_id=global_id,
                    video_id=index_dir.name,
                    chunk_id=chunk_id,
                    start_ms=int(times[chunk_id, 0]),
                    end_ms=int(times[chunk_id, 1]),
                    text=text,
                )
                rows.append((chunk, local_embeddings.get(chunk_id)))
                video_indices.setdefault(index_dir.name, []).append(global_id)

    if not rows or not embedding_dim:
        raise ValueError(f"No ASR semantic corpus found under {index_root}")

    semantic_embeddings = np.zeros((len(rows), embedding_dim), dtype=np.float32)
    semantic_available = np.zeros(len(rows), dtype=bool)
    for chunk, embedding in rows:
        if embedding is None:
            continue
        semantic_embeddings[chunk.global_id] = normalize(embedding)
        semantic_available[chunk.global_id] = True

    return Corpus(
        chunks=[row[0] for row in rows],
        video_indices={key: np.asarray(value, dtype=np.int32) for key, value in video_indices.items()},
        semantic_embeddings=semantic_embeddings,
        semantic_available=semantic_available,
    )


def _legacy_lexical_score(query: str, text: str) -> float:
    query_value, text_value = normalize_text(query), normalize_text(text)
    if not query_value or not text_value:
        return 0.0
    if query_value in text_value:
        return 1.0
    size = 2 if len(query_value) > 1 else 1
    query_grams = {
        query_value[index:index + size]
        for index in range(max(1, len(query_value) - size + 1))
    }
    text_grams = {
        text_value[index:index + size]
        for index in range(max(1, len(text_value) - size + 1))
    }
    return float(len(query_grams & text_grams) / max(1, len(query_grams)))


def _is_cjk_character(character: str) -> bool:
    return (
        "\u3400" <= character <= "\u4dbf"
        or "\u4e00" <= character <= "\u9fff"
        or "\uf900" <= character <= "\ufaff"
    )


def _cjk_patch_lexical_score(query: str, text: str) -> float:
    legacy = _legacy_lexical_score(query, text)
    query_value, text_value = normalize_text(query), normalize_text(text)
    if len(query_value) < 3 or not all(_is_cjk_character(value) for value in query_value):
        return legacy
    longest = 0
    for start in range(len(query_value) - 1):
        size = max(2, longest + 1)
        while start + size <= len(query_value) and query_value[start:start + size] in text_value:
            longest = size
            size += 1
    coverage = float(longest / len(query_value)) if longest >= 2 else 0.0
    return max(legacy, coverage)


def _semantic_scores(corpus: Corpus, query_embedding: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    query = normalize(query_embedding)
    cosines = np.full(len(corpus.chunks), np.nan, dtype=np.float32)
    scores = np.zeros(len(corpus.chunks), dtype=np.float32)
    for indices in corpus.video_indices.values():
        eligible = indices[corpus.semantic_available[indices]]
        if not len(eligible):
            continue
        local_cosines = corpus.semantic_embeddings[eligible] @ query
        percentiles = robust_distribution(local_cosines)["percentiles"]
        cosines[eligible] = local_cosines
        scores[eligible] = np.asarray([
            0.7 * asr_semantic_confidence(float(cosine)) + 0.3 * float(percentile)
            for cosine, percentile in zip(local_cosines, percentiles)
        ], dtype=np.float32)
    return scores, cosines


def _sorted_indices(indices: np.ndarray, *descending_values: np.ndarray) -> list[int]:
    return sorted(
        (int(index) for index in indices),
        key=lambda index: tuple(-float(values[index]) for values in descending_values) + (index,),
    )


def _combined_ranking(
    lexical_scores: np.ndarray,
    semantic_scores: np.ndarray,
    semantic_available: np.ndarray,
) -> list[int]:
    combined = np.maximum(lexical_scores, 0.65 * semantic_scores + 0.35 * lexical_scores)
    candidates = np.flatnonzero((lexical_scores > 0) | semantic_available)
    return _sorted_indices(candidates, combined, lexical_scores, semantic_scores)


def _dual_rrf_ranking(
    lexical_scores: np.ndarray,
    semantic_scores: np.ndarray,
    semantic_available: np.ndarray,
    pool_size: int,
    rrf_k: int,
    lexical_weight: float,
) -> list[int]:
    lexical_indices = np.flatnonzero(lexical_scores > 0)
    semantic_indices = np.flatnonzero(semantic_available)
    lexical_order = _sorted_indices(lexical_indices, lexical_scores, semantic_scores)[:pool_size]
    semantic_order = _sorted_indices(semantic_indices, semantic_scores, lexical_scores)[:pool_size]
    lexical_ranks = {index: rank for rank, index in enumerate(lexical_order, start=1)}
    semantic_ranks = {index: rank for rank, index in enumerate(semantic_order, start=1)}
    candidates = set(lexical_ranks) | set(semantic_ranks)

    def score(index: int) -> float:
        value = 0.0
        if index in lexical_ranks:
            value += lexical_weight / (rrf_k + lexical_ranks[index])
        if index in semantic_ranks:
            value += 1.0 / (rrf_k + semantic_ranks[index])
        return value

    return sorted(
        candidates,
        key=lambda index: (
            -score(index),
            -float(lexical_scores[index]),
            -float(semantic_scores[index]),
            index,
        ),
    )


def _dual_reserved_ranking(
    lexical_scores: np.ndarray,
    semantic_scores: np.ndarray,
    semantic_available: np.ndarray,
    pool_size: int,
    lexical_min_score: float,
    semantic_run: int,
) -> list[int]:
    """Keep semantic order primary while reserving sparse slots for strong lexical hits."""
    semantic_indices = np.flatnonzero(semantic_available)
    lexical_indices = np.flatnonzero(lexical_scores >= lexical_min_score)
    semantic_order = _sorted_indices(semantic_indices, semantic_scores, lexical_scores)[:pool_size]
    lexical_order = _sorted_indices(lexical_indices, lexical_scores, semantic_scores)[:pool_size]

    return _reserved_interleave(semantic_order, lexical_order, 1, semantic_run)


def _reserved_interleave(
    primary_order: list[int],
    lexical_order: list[int],
    initial_primary: int,
    primary_run: int,
) -> list[int]:
    ranking: list[int] = []
    emitted: set[int] = set()
    primary_position = 0
    lexical_position = 0

    def emit(index: int) -> None:
        if index not in emitted:
            emitted.add(index)
            ranking.append(index)

    while primary_position < len(primary_order) and primary_position < initial_primary:
        emit(primary_order[primary_position])
        primary_position += 1

    while primary_position < len(primary_order) or lexical_position < len(lexical_order):
        while lexical_position < len(lexical_order):
            lexical_index = lexical_order[lexical_position]
            lexical_position += 1
            if lexical_index not in emitted:
                emit(lexical_index)
                break
        taken = 0
        while primary_position < len(primary_order) and taken < primary_run:
            primary_index = primary_order[primary_position]
            primary_position += 1
            if primary_index in emitted:
                continue
            emit(primary_index)
            taken += 1

        if primary_position >= len(primary_order) and lexical_position >= len(lexical_order):
            break
    return ranking


def _dual_rescored_reserved_ranking(
    lexical_scores: np.ndarray,
    semantic_scores: np.ndarray,
    semantic_available: np.ndarray,
    pool_size: int,
    lexical_min_score: float,
    initial_primary: int,
    primary_run: int,
) -> list[int]:
    """Union independent pools, preserve calibrated ranking, and reserve lexical recall slots."""
    primary_order = _combined_ranking(
        lexical_scores, semantic_scores, semantic_available
    )[:pool_size]
    lexical_indices = np.flatnonzero(lexical_scores >= lexical_min_score)
    lexical_order = _sorted_indices(
        lexical_indices, lexical_scores, semantic_scores
    )[:pool_size]
    return _reserved_interleave(
        primary_order, lexical_order, initial_primary, primary_run
    )


def _target_indices(query: dict, corpus: Corpus) -> set[int]:
    targets: set[int] = set()
    for target in query["targets"]:
        video_id = target["video_id"]
        needle = str(target["text_contains"]).casefold()
        for index in corpus.video_indices.get(video_id, np.empty(0, dtype=np.int32)):
            if needle in corpus.chunks[int(index)].text.casefold():
                targets.add(int(index))
    if not targets:
        raise ValueError(f"Query {query['id']} has no matching target chunks")
    return targets


def _rank_of_targets(ranking: list[int], targets: set[int]) -> int | None:
    for rank, index in enumerate(ranking, start=1):
        if index in targets:
            return rank
    return None


def _metrics(results: list[dict], strategy: str, split: str) -> dict:
    selected = [row for row in results if split == "all" or row["split"] == split]
    ranks = [row["ranks"].get(strategy) for row in selected]
    finite = [rank for rank in ranks if rank is not None]
    metrics = {
        "queries": len(selected),
        "mrr": float(sum(1.0 / rank for rank in finite) / max(1, len(selected))),
    }
    for cutoff in (1, 3, 5, 10, 20, 50):
        metrics[f"hit@{cutoff}"] = float(sum(rank is not None and rank <= cutoff for rank in ranks) / max(1, len(selected)))
    return metrics


def _objective(metrics: dict) -> float:
    return (
        0.50 * metrics["mrr"]
        + 0.25 * metrics["hit@5"]
        + 0.15 * metrics["hit@10"]
        + 0.10 * metrics["hit@50"]
    )


def _prepare_queries(
    queries: list[dict],
    corpus: Corpus,
    query_embeddings: np.ndarray,
) -> list[dict]:
    prepared = []
    texts = [chunk.text for chunk in corpus.chunks]
    for query, query_embedding in zip(queries, query_embeddings):
        legacy_lexical = np.asarray([
            _legacy_lexical_score(query["query"], text) for text in texts
        ], dtype=np.float32)
        cjk_lexical = np.asarray([
            _cjk_patch_lexical_score(query["query"], text) for text in texts
        ], dtype=np.float32)
        semantic_scores, semantic_cosines = _semantic_scores(corpus, query_embedding)
        targets = _target_indices(query, corpus)
        prepared.append({
            "query": query,
            "legacy_lexical": legacy_lexical,
            "cjk_lexical": cjk_lexical,
            "semantic_scores": semantic_scores,
            "semantic_cosines": semantic_cosines,
            "targets": targets,
        })
    return prepared


def _evaluate_prepared(
    prepared: list[dict],
    corpus: Corpus,
    strategies: dict[str, Callable[[np.ndarray, np.ndarray], list[int]]],
) -> list[dict]:
    results = []
    for item in prepared:
        query = item["query"]
        semantic_scores = item["semantic_scores"]
        semantic_cosines = item["semantic_cosines"]
        targets = item["targets"]
        row = {
            "id": query["id"],
            "split": query["split"],
            "category": query["category"],
            "query": query["query"],
            "targets": sorted(targets),
            "ranks": {},
            "top5": {},
        }
        for name, strategy in strategies.items():
            lexical_values = item["cjk_lexical"] if name.endswith("_cjk") else item["legacy_lexical"]
            ranking = strategy(lexical_values, semantic_scores)
            row["ranks"][name] = _rank_of_targets(ranking, targets)
            row["top5"][name] = [
                {
                    "video_id": corpus.chunks[index].video_id,
                    "chunk_id": corpus.chunks[index].chunk_id,
                    "text": corpus.chunks[index].text,
                    "lexical": round(float(lexical_values[index]), 6),
                    "semantic": round(float(semantic_scores[index]), 6),
                    "cosine": None if np.isnan(semantic_cosines[index]) else round(float(semantic_cosines[index]), 6),
                    "target": index in targets,
                }
                for index in ranking[:5]
            ]
        results.append(row)
    return results


def _markdown_report(payload: dict) -> str:
    selected = payload["selected_rrf"]
    reserved = payload["selected_reserved"]
    rescored = payload["selected_rescored"]
    strategies = payload["strategies"]
    lines = [
        "# ASR lexical / semantic dual-pool retrieval A/B",
        "",
        f"- corpus chunks: {payload['corpus_chunks']}",
        f"- semantic chunks: {payload['semantic_chunks']}",
        f"- queries: {payload['query_count']} (tune={payload['split_counts']['tune']}, holdout={payload['split_counts']['holdout']})",
        f"- selected RRF: pool={selected['pool_size']}, k={selected['rrf_k']}, lexical_weight={selected['lexical_weight']}",
        f"- selected semantic-primary reserve: pool={reserved['pool_size']}, lexical_min={reserved['lexical_min_score']}, run={reserved['semantic_run']}",
        f"- selected rescored reserve: pool={rescored['pool_size']}, lexical_min={rescored['lexical_min_score']}, initial={rescored['initial_primary']}, run={rescored['primary_run']}",
        "",
        "## Metrics",
        "",
        "| strategy | split | MRR | H@1 | H@3 | H@5 | H@10 | H@50 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for strategy in strategies:
        for split in ("tune", "holdout", "all"):
            metrics = payload["metrics"][strategy][split]
            lines.append(
                f"| {strategy} | {split} | {metrics['mrr']:.3f} | "
                f"{metrics['hit@1']:.3f} | {metrics['hit@3']:.3f} | {metrics['hit@5']:.3f} | "
                f"{metrics['hit@10']:.3f} | {metrics['hit@50']:.3f} |"
            )

    lines.extend([
        "",
        "## Per-query target rank",
        "",
        "| id | split | category | query | legacy | CJK patch | RRF | semantic reserve | rescored reserve | rescored CJK |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in payload["results"]:
        ranks = row["ranks"]
        value = lambda name: str(ranks.get(name)) if ranks.get(name) is not None else "miss"
        lines.append(
            f"| {row['id']} | {row['split']} | {row['category']} | {row['query']} | "
            f"{value('combined_legacy')} | {value('combined_cjk')} | "
            f"{value('dual_rrf_legacy')} | {value('dual_reserved_legacy')} | "
            f"{value('dual_rescored_legacy')} | {value('dual_rescored_cjk')} |"
        )

    lines.extend(["", "## Reserved dual-pool regressions and wins", ""])
    for row in payload["results"]:
        baseline = row["ranks"]["combined_legacy"] or 999999
        dual = row["ranks"]["dual_rescored_legacy"] or 999999
        if dual == baseline:
            continue
        direction = "improved" if dual < baseline else "regressed"
        lines.append(
            f"- {row['id']} {direction}: {baseline if baseline < 999999 else 'miss'} -> "
            f"{dual if dual < 999999 else 'miss'}; `{row['query']}`"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-root", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    queries = _load_queries(args.queries)
    corpus = _load_corpus(args.index_root)
    encoder = TextEmbeddingEncoder(
        args.model_name,
        args.model_dir,
        args.device,
        local_files_only=True,
    )
    query_embeddings = encoder.encode([item["query"] for item in queries], batch_size=32)
    prepared = _prepare_queries(queries, corpus, query_embeddings)
    prepared_tune = [item for item in prepared if item["query"]["split"] == "tune"]

    fixed_strategies: dict[str, Callable[[np.ndarray, np.ndarray], list[int]]] = {
        "combined_legacy": lambda lexical, semantic: _combined_ranking(
            lexical, semantic, corpus.semantic_available
        ),
        "combined_cjk": lambda lexical, semantic: _combined_ranking(
            lexical, semantic, corpus.semantic_available
        ),
    }

    grid_results = []
    best_config = None
    best_score = -1.0
    for pool_size in (25, 50, 100, 150):
        for rrf_k in (10, 30, 60):
            for lexical_weight in (0.50, 0.75, 1.00, 1.25, 1.50):
                strategy_name = "grid"
                strategy = lambda lexical, semantic, p=pool_size, k=rrf_k, w=lexical_weight: _dual_rrf_ranking(
                    lexical, semantic, corpus.semantic_available, p, k, w
                )
                trial = _evaluate_prepared(
                    prepared_tune,
                    corpus,
                    {strategy_name: strategy},
                )
                metrics = _metrics(trial, strategy_name, "all")
                score = _objective(metrics)
                config = {
                    "pool_size": pool_size,
                    "rrf_k": rrf_k,
                    "lexical_weight": lexical_weight,
                    "objective": score,
                    "metrics": metrics,
                }
                grid_results.append(config)
                if score > best_score:
                    best_score = score
                    best_config = config

    assert best_config is not None
    pool_size = int(best_config["pool_size"])
    rrf_k = int(best_config["rrf_k"])
    lexical_weight = float(best_config["lexical_weight"])
    fixed_strategies.update({
        "dual_rrf_legacy": lambda lexical, semantic: _dual_rrf_ranking(
            lexical, semantic, corpus.semantic_available, pool_size, rrf_k, lexical_weight
        ),
        "dual_rrf_cjk": lambda lexical, semantic: _dual_rrf_ranking(
            lexical, semantic, corpus.semantic_available, pool_size, rrf_k, lexical_weight
        ),
    })

    reserved_grid_results = []
    best_reserved_config = None
    best_reserved_score = -1.0
    for reserved_pool_size in (50, 100, 150):
        for lexical_min_score in (0.25, 0.40, 0.50, 0.60):
            for semantic_run in (2, 4, 8):
                strategy_name = "reserved_grid"
                strategy = lambda lexical, semantic, p=reserved_pool_size, m=lexical_min_score, r=semantic_run: _dual_reserved_ranking(
                    lexical, semantic, corpus.semantic_available, p, m, r
                )
                trial = _evaluate_prepared(
                    prepared_tune,
                    corpus,
                    {strategy_name: strategy},
                )
                metrics = _metrics(trial, strategy_name, "all")
                score = _objective(metrics)
                config = {
                    "pool_size": reserved_pool_size,
                    "lexical_min_score": lexical_min_score,
                    "semantic_run": semantic_run,
                    "objective": score,
                    "metrics": metrics,
                }
                reserved_grid_results.append(config)
                if score > best_reserved_score:
                    best_reserved_score = score
                    best_reserved_config = config

    assert best_reserved_config is not None
    reserved_pool_size = int(best_reserved_config["pool_size"])
    lexical_min_score = float(best_reserved_config["lexical_min_score"])
    semantic_run = int(best_reserved_config["semantic_run"])
    fixed_strategies.update({
        "dual_reserved_legacy": lambda lexical, semantic: _dual_reserved_ranking(
            lexical,
            semantic,
            corpus.semantic_available,
            reserved_pool_size,
            lexical_min_score,
            semantic_run,
        ),
        "dual_reserved_cjk": lambda lexical, semantic: _dual_reserved_ranking(
            lexical,
            semantic,
            corpus.semantic_available,
            reserved_pool_size,
            lexical_min_score,
            semantic_run,
        ),
    })

    rescored_grid_results = []
    best_rescored_config = None
    best_rescored_score = -1.0
    for rescored_pool_size in (50, 100, 150):
        for rescored_lexical_min in (0.25, 0.40, 0.50, 0.60):
            for initial_primary in (1, 3, 5):
                for primary_run in (4, 8):
                    strategy_name = "rescored_grid"
                    strategy = lambda lexical, semantic, p=rescored_pool_size, m=rescored_lexical_min, i=initial_primary, r=primary_run: _dual_rescored_reserved_ranking(
                        lexical,
                        semantic,
                        corpus.semantic_available,
                        p,
                        m,
                        i,
                        r,
                    )
                    trial = _evaluate_prepared(
                        prepared_tune,
                        corpus,
                        {strategy_name: strategy},
                    )
                    metrics = _metrics(trial, strategy_name, "all")
                    score = _objective(metrics)
                    config = {
                        "pool_size": rescored_pool_size,
                        "lexical_min_score": rescored_lexical_min,
                        "initial_primary": initial_primary,
                        "primary_run": primary_run,
                        "objective": score,
                        "metrics": metrics,
                    }
                    rescored_grid_results.append(config)
                    if score > best_rescored_score:
                        best_rescored_score = score
                        best_rescored_config = config

    # Several settings tie on the tuning split. Prefer the conservative member:
    # preserve the first three primary hits, require 50% lexical coverage, and
    # reserve only one lexical slot per eight primary results.
    conservative = next(
        config
        for config in rescored_grid_results
        if config["pool_size"] == 50
        and config["lexical_min_score"] == 0.50
        and config["initial_primary"] == 3
        and config["primary_run"] == 8
    )
    if conservative["objective"] >= best_rescored_score - 1e-12:
        best_rescored_config = conservative

    assert best_rescored_config is not None
    rescored_pool_size = int(best_rescored_config["pool_size"])
    rescored_lexical_min = float(best_rescored_config["lexical_min_score"])
    initial_primary = int(best_rescored_config["initial_primary"])
    primary_run = int(best_rescored_config["primary_run"])
    fixed_strategies.update({
        "dual_rescored_legacy": lambda lexical, semantic: _dual_rescored_reserved_ranking(
            lexical,
            semantic,
            corpus.semantic_available,
            rescored_pool_size,
            rescored_lexical_min,
            initial_primary,
            primary_run,
        ),
        "dual_rescored_cjk": lambda lexical, semantic: _dual_rescored_reserved_ranking(
            lexical,
            semantic,
            corpus.semantic_available,
            rescored_pool_size,
            rescored_lexical_min,
            initial_primary,
            primary_run,
        ),
    })
    results = _evaluate_prepared(prepared, corpus, fixed_strategies)
    strategy_names = list(fixed_strategies)
    payload = {
        "corpus_chunks": len(corpus.chunks),
        "semantic_chunks": int(corpus.semantic_available.sum()),
        "query_count": len(queries),
        "split_counts": {
            split: sum(item["split"] == split for item in queries)
            for split in ("tune", "holdout")
        },
        "strategies": strategy_names,
        "selected_rrf": best_config,
        "selected_reserved": best_reserved_config,
        "selected_rescored": best_rescored_config,
        "rrf_grid": sorted(grid_results, key=lambda item: item["objective"], reverse=True),
        "reserved_grid": sorted(reserved_grid_results, key=lambda item: item["objective"], reverse=True),
        "rescored_grid": sorted(rescored_grid_results, key=lambda item: item["objective"], reverse=True),
        "metrics": {
            strategy: {
                split: _metrics(results, strategy, split)
                for split in ("tune", "holdout", "all")
            }
            for strategy in strategy_names
        },
        "results": results,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "REPORT.md").write_text(_markdown_report(payload), encoding="utf-8")
    print(json.dumps({
        "selected_rrf": best_config,
        "selected_reserved": best_reserved_config,
        "selected_rescored": best_rescored_config,
        "metrics": payload["metrics"],
        "output_dir": str(args.output_dir),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
