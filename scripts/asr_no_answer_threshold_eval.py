from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

import asr_retrieval_benchmark as benchmark


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _round_grid(start: float, stop: float, step: float) -> list[float]:
    count = int(round((stop - start) / step))
    return [round(start + index * step, 6) for index in range(count + 1)]


def _max_or_default(values: np.ndarray, default: float = -math.inf) -> float:
    return float(np.max(values)) if len(values) else default


def _split_metrics(
    queries: list[dict],
    accepted: np.ndarray,
    target_passed: np.ndarray,
    split: str,
) -> dict:
    selected = np.asarray([
        split == "all" or query["split"] == split
        for query in queries
    ], dtype=bool)
    answerable = selected & np.asarray([bool(query["target_indices"]) for query in queries])
    no_answer = selected & ~answerable
    answer_count = int(answerable.sum())
    no_answer_count = int(no_answer.sum())
    categories: dict[str, dict] = {}
    for category in sorted({query["category"] for query in queries if not query["target_indices"]}):
        category_mask = no_answer & np.asarray([
            query["category"] == category for query in queries
        ], dtype=bool)
        count = int(category_mask.sum())
        if count:
            categories[category] = {
                "queries": count,
                "false_accepts": int(accepted[category_mask].sum()),
                "false_accept_rate": float(accepted[category_mask].mean()),
            }
    return {
        "answerable": answer_count,
        "no_answer": no_answer_count,
        "answerable_accept_rate": float(accepted[answerable].mean()) if answer_count else None,
        "target_gate_recall": float(target_passed[answerable].mean()) if answer_count else None,
        "no_answer_false_accept_rate": float(accepted[no_answer].mean()) if no_answer_count else None,
        "no_answer_false_accepts": int(accepted[no_answer].sum()),
        "no_answer_categories": categories,
    }


def _config_metrics(
    queries: list[dict],
    max_lexical: np.ndarray,
    max_target_lexical: np.ndarray,
    max_semantic: np.ndarray,
    max_target_semantic: np.ndarray,
    config: dict,
) -> dict:
    lexical_threshold = float(config["lexical_threshold"])
    semantic_threshold = float(config["semantic_threshold"])
    accepted = (max_lexical >= lexical_threshold) | (max_semantic >= semantic_threshold)
    target_passed = (
        (max_target_lexical >= lexical_threshold)
        | (max_target_semantic >= semantic_threshold)
    )
    return {
        "config": config,
        "metrics": {
            split: _split_metrics(queries, accepted, target_passed, split)
            for split in ("tune", "dev", "holdout", "all")
        },
    }


def _candidate_key(row: dict) -> tuple:
    tune = row["metrics"]["tune"]
    recall = float(tune["target_gate_recall"] or 0.0)
    false_accept = float(tune["no_answer_false_accept_rate"] or 1.0)
    mode_order = {"cosine": 0, "absolute": 1, "dual": 2, "current": 3}
    lexical_order = {"idf_weighted": 0, "legacy_bigram": 1}
    config = row["config"]
    return (
        recall - false_accept,
        -false_accept,
        recall,
        -mode_order[config["semantic_mode"]],
        -lexical_order[config["lexical_mode"]],
        -float(config["lexical_threshold"]),
        -float(config["semantic_threshold"]),
    )


def _operating_point_key(row: dict) -> tuple:
    tune = row["metrics"]["tune"]
    recall = float(tune["target_gate_recall"] or 0.0)
    false_accept = float(tune["no_answer_false_accept_rate"] or 1.0)
    mode_order = {"cosine": 0, "absolute": 1, "dual": 2, "current": 3}
    lexical_order = {"idf_weighted": 0, "legacy_bigram": 1}
    config = row["config"]
    return (
        -false_accept,
        recall,
        -mode_order[config["semantic_mode"]],
        -lexical_order[config["lexical_mode"]],
        -float(config["lexical_threshold"]),
        -float(config["semantic_threshold"]),
    )


def _distribution(values: np.ndarray) -> dict:
    finite = values[np.isfinite(values)]
    if not len(finite):
        return {"count": 0}
    quantiles = np.quantile(finite, (0.0, 0.10, 0.25, 0.50, 0.75, 0.90, 1.0))
    return {
        "count": int(len(finite)),
        "min": float(quantiles[0]),
        "p10": float(quantiles[1]),
        "p25": float(quantiles[2]),
        "median": float(quantiles[3]),
        "p75": float(quantiles[4]),
        "p90": float(quantiles[5]),
        "max": float(quantiles[6]),
    }


def _semantic_maxima(
    queries: list[dict],
    matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    maximum = np.max(matrix, axis=1)
    target_maximum = np.full(len(queries), -math.inf, dtype=np.float32)
    for index, query in enumerate(queries):
        targets = sorted(query["target_indices"])
        if targets:
            target_maximum[index] = _max_or_default(matrix[index, targets])
    return maximum, target_maximum


def _gate_matrix(
    mode: str,
    current: np.ndarray,
    absolute: np.ndarray,
    cosines: np.ndarray,
    cosine_floor: float | None = None,
) -> np.ndarray:
    if mode == "current":
        return current
    if mode == "absolute":
        return absolute
    if mode == "cosine":
        return cosines
    if mode == "dual":
        if cosine_floor is None:
            raise ValueError("dual mode requires cosine_floor")
        return np.where(cosines >= cosine_floor, current, -np.inf)
    raise ValueError(f"Unknown semantic mode: {mode}")


def _config_id(config: dict) -> str:
    value = (
        f"{config['semantic_mode']}_{config['lexical_mode']}"
        f"_lex{config['lexical_threshold']:.2f}"
        f"_sem{config['semantic_threshold']:.2f}"
    )
    if config.get("cosine_floor") is not None:
        value += f"_cos{config['cosine_floor']:.2f}"
    return value


def _detail_rows(
    corpus: benchmark.Corpus,
    queries: list[dict],
    lexical_matrices: dict[str, np.ndarray],
    current: np.ndarray,
    absolute: np.ndarray,
    cosines: np.ndarray,
    combined: np.ndarray,
    config: dict,
) -> list[dict]:
    lexical = lexical_matrices[config["lexical_mode"]]
    semantic = _gate_matrix(
        config["semantic_mode"],
        current,
        absolute,
        cosines,
        config.get("cosine_floor"),
    )
    rows: list[dict] = []
    for query_index, query in enumerate(queries):
        passed = (
            (lexical[query_index] >= config["lexical_threshold"])
            | (semantic[query_index] >= config["semantic_threshold"])
        )
        accepted_indices = np.flatnonzero(passed)
        if len(accepted_indices):
            top_index = max(
                (int(index) for index in accepted_indices),
                key=lambda index: (float(combined[query_index, index]), -index),
            )
            chunk = corpus.chunks[top_index]
            top = {
                "source_id": chunk.source_id,
                "start_ms": chunk.start_ms,
                "end_ms": chunk.end_ms,
                "text": chunk.text,
                "lexical_score": float(lexical[query_index, top_index]),
                "semantic_score": float(current[query_index, top_index]),
                "semantic_absolute": float(absolute[query_index, top_index]),
                "semantic_cosine": float(cosines[query_index, top_index]),
                "rank_score": float(combined[query_index, top_index]),
            }
        else:
            top = None
        targets = sorted(query["target_indices"])
        target_pass = bool(np.any(passed[targets])) if targets else None
        rows.append({
            "id": query["id"],
            "split": query["split"],
            "category": query["category"],
            "query": query["query"],
            "answerable": bool(targets),
            "accepted": bool(len(accepted_indices)),
            "target_passed": target_pass,
            "top_accepted": top,
        })
    return rows


def _markdown(payload: dict) -> str:
    selected = payload["selected"]
    production = payload["production"]
    lines = [
        "# ASR 无答案阈值校准",
        "",
        "## 数据",
        "",
        f"- corpus：{payload['dataset']['sources']} sources / {payload['dataset']['chunks']} chunks",
        f"- 查询：{payload['dataset']['answerable']} 条有答案 / {payload['dataset']['no_answer']} 条无答案",
        "- 参数只在 tune 选择；dev 和 holdout 仅验证。",
        "",
        "## 结论",
        "",
        f"- 现行门槛：`{_config_id(production['config'])}`。",
        f"- tune 选中：`{_config_id(selected['config'])}`。",
        "- `target gate recall` 表示正确答案 chunk 没有被阈值挡掉；`false accept rate` 表示无答案查询仍被判为有答案。",
        "",
        "| config | split | target gate recall | no-answer FAR | false accepts |",
        "|---|---|---:|---:|---:|",
    ]
    for name, row in (("production", production), ("selected", selected)):
        for split in ("tune", "dev", "holdout", "all"):
            metric = row["metrics"][split]
            lines.append(
                f"| {name} | {split} | {metric['target_gate_recall']:.3f} | "
                f"{metric['no_answer_false_accept_rate']:.3f} | {metric['no_answer_false_accepts']} |"
            )
    lines.extend([
        "",
        "## 无答案分层",
        "",
        "| config | category | queries | false accepts | FAR |",
        "|---|---|---:|---:|---:|",
    ])
    for name, row in (("production", production), ("selected", selected)):
        for category, metric in row["metrics"]["all"]["no_answer_categories"].items():
            lines.append(
                f"| {name} | {category} | {metric['queries']} | "
                f"{metric['false_accepts']} | {metric['false_accept_rate']:.3f} |"
            )
    lines.extend([
        "",
        "## Recall / 拒答代价",
        "",
        "以下 operating point 均只按 tune 选择；recall floor 越低，允许被阈值挡掉的正确答案越多。",
        "",
        "| tune recall floor | config | tune recall/FAR | dev recall/FAR | holdout recall/FAR |",
        "|---:|---|---:|---:|---:|",
    ])
    for floor, row in payload["operating_points"].items():
        config = _config_id(row["config"])
        tune = row["metrics"]["tune"]
        dev = row["metrics"]["dev"]
        holdout = row["metrics"]["holdout"]
        lines.append(
            f"| {float(floor):.2f} | `{config}` | "
            f"{tune['target_gate_recall']:.3f}/{tune['no_answer_false_accept_rate']:.3f} | "
            f"{dev['target_gate_recall']:.3f}/{dev['no_answer_false_accept_rate']:.3f} | "
            f"{holdout['target_gate_recall']:.3f}/{holdout['no_answer_false_accept_rate']:.3f} |"
        )
    lines.extend([
        "",
        "## 解释限制",
        "",
        "- 阈值只决定结果是否进入‘有效结果’，不改变现有 combined 排序。",
        "- 同领域反事实负例可能与真实片段高度相似；单一相似度阈值不能可靠理解事实否定或实体冲突。",
        "- 逐查询命中详情见 `production_rows.jsonl` 与 `selected_rows.jsonl`。",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-root", type=Path, required=True)
    parser.add_argument("--open-eval-root", type=Path, required=True)
    parser.add_argument("--platform-queries", type=Path, required=True)
    parser.add_argument("--open-queries", type=Path, required=True)
    parser.add_argument("--no-answer-queries", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--embedding-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    corpus = benchmark.load_corpus(args.index_root, args.open_eval_root)
    queries = benchmark.prepare_queries(
        [args.platform_queries, args.open_queries, args.no_answer_queries], corpus
    )
    eligible, weights, policy = benchmark.semantic_policy(corpus, "current")
    corpus_embeddings, query_embeddings, encoding = benchmark.encode_model(
        corpus,
        queries,
        "minilm",
        args.model_root,
        args.embedding_cache,
        args.device,
        False,
    )
    cosines = query_embeddings @ corpus_embeddings.T
    cosines[:, ~eligible] = -np.inf
    absolute = benchmark._confidence(cosines)
    current = benchmark.semantic_scores(
        corpus,
        corpus_embeddings,
        query_embeddings,
        eligible,
        weights,
        {"kind": "current"},
    )
    legacy_lexical = benchmark.legacy_lexical_matrix(corpus, queries)
    idf_weighted, _, lexical_stats = benchmark.advanced_lexical_matrices(corpus, queries)
    lexical_matrices = {
        "legacy_bigram": legacy_lexical,
        "idf_weighted": idf_weighted,
    }
    lexical_maxima = {
        name: _semantic_maxima(queries, matrix)
        for name, matrix in lexical_matrices.items()
    }
    combined = np.maximum(legacy_lexical, 0.65 * current + 0.35 * legacy_lexical)
    semantic_cache: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}

    def maxima(mode: str, cosine_floor: float | None = None) -> tuple[np.ndarray, np.ndarray]:
        key = (mode, cosine_floor)
        if key not in semantic_cache:
            semantic_cache[key] = _semantic_maxima(
                queries,
                _gate_matrix(mode, current, absolute, cosines, cosine_floor),
            )
        return semantic_cache[key]

    trials: list[dict] = []
    lexical_thresholds = {
        "legacy_bigram": (0.25, 0.35, 0.50),
        "idf_weighted": (0.35, 0.50, 0.65, 0.80),
    }
    for lexical_mode, thresholds in lexical_thresholds.items():
        max_lexical, max_target_lexical = lexical_maxima[lexical_mode]
        for mode, semantic_thresholds in (
            ("current", _round_grid(0.55, 0.95, 0.01)),
            ("absolute", _round_grid(0.55, 0.95, 0.01)),
            ("cosine", _round_grid(0.25, 0.70, 0.01)),
        ):
            max_semantic, max_target_semantic = maxima(mode)
            for lexical_threshold in thresholds:
                for semantic_threshold in semantic_thresholds:
                    config = {
                        "semantic_mode": mode,
                        "lexical_mode": lexical_mode,
                        "lexical_threshold": lexical_threshold,
                        "semantic_threshold": semantic_threshold,
                        "cosine_floor": None,
                    }
                    trials.append(_config_metrics(
                        queries,
                        max_lexical,
                        max_target_lexical,
                        max_semantic,
                        max_target_semantic,
                        config,
                    ))
        for cosine_floor in _round_grid(0.30, 0.65, 0.05):
            max_semantic, max_target_semantic = maxima("dual", cosine_floor)
            for lexical_threshold in thresholds:
                for semantic_threshold in _round_grid(0.55, 0.95, 0.05):
                    config = {
                        "semantic_mode": "dual",
                        "lexical_mode": lexical_mode,
                        "lexical_threshold": lexical_threshold,
                        "semantic_threshold": semantic_threshold,
                        "cosine_floor": cosine_floor,
                    }
                    trials.append(_config_metrics(
                        queries,
                        max_lexical,
                        max_target_lexical,
                        max_semantic,
                        max_target_semantic,
                        config,
                    ))

    production_config = {
        "semantic_mode": "current",
        "lexical_mode": "legacy_bigram",
        "lexical_threshold": 0.25,
        "semantic_threshold": 0.55,
        "cosine_floor": None,
    }
    production = next(row for row in trials if row["config"] == production_config)
    feasible = [
        row for row in trials
        if float(row["metrics"]["tune"]["target_gate_recall"] or 0.0) >= 0.95
    ]
    if not feasible:
        raise RuntimeError("No threshold configuration preserved 95% tune target recall")
    selected = max(feasible, key=_candidate_key)
    operating_points = {}
    for recall_floor in (1.0, 0.98, 0.95, 0.90, 0.85):
        candidates = [
            row for row in trials
            if float(row["metrics"]["tune"]["target_gate_recall"] or 0.0) >= recall_floor
        ]
        if candidates:
            operating_points[f"{recall_floor:.2f}"] = max(candidates, key=_operating_point_key)

    production_rows = _detail_rows(
        corpus, queries, lexical_matrices, current, absolute, cosines, combined, production_config
    )
    selected_rows = _detail_rows(
        corpus, queries, lexical_matrices, current, absolute, cosines, combined, selected["config"]
    )
    max_cosine, max_target_cosine = maxima("cosine")
    answerable_mask = np.asarray([bool(query["target_indices"]) for query in queries], dtype=bool)
    no_answer_mask = ~answerable_mask
    payload = {
        "dataset": {
            "sources": len(corpus.source_indices),
            "chunks": len(corpus.chunks),
            "queries": len(queries),
            "answerable": sum(bool(query["target_indices"]) for query in queries),
            "no_answer": sum(not query["target_indices"] for query in queries),
            "splits": dict(Counter(query["split"] for query in queries)),
            "categories": dict(Counter(query["category"] for query in queries)),
        },
        "selection": {
            "split": "tune",
            "minimum_target_gate_recall": 0.95,
            "objective": "maximize target_gate_recall - no_answer_false_accept_rate",
            "trials": len(trials),
        },
        "semantic_policy": policy,
        "lexical_stats": lexical_stats,
        "encoding": encoding,
        "production": production,
        "selected": selected,
        "operating_points": operating_points,
        "score_distributions": {
            "answerable_target_cosine": _distribution(
                max_target_cosine[answerable_mask]
            ),
            "no_answer_max_cosine": _distribution(
                max_cosine[no_answer_mask]
            ),
            "answerable_target_lexical": _distribution(
                lexical_maxima["legacy_bigram"][1][answerable_mask]
            ),
            "no_answer_max_lexical": _distribution(
                lexical_maxima["legacy_bigram"][0][no_answer_mask]
            ),
            "answerable_target_idf_weighted": _distribution(
                lexical_maxima["idf_weighted"][1][answerable_mask]
            ),
            "no_answer_max_idf_weighted": _distribution(
                lexical_maxima["idf_weighted"][0][no_answer_mask]
            ),
        },
        "top_tune_candidates": sorted(feasible, key=_candidate_key, reverse=True)[:25],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(args.output_dir / "results.json", payload)
    _write_jsonl(args.output_dir / "production_rows.jsonl", production_rows)
    _write_jsonl(args.output_dir / "selected_rows.jsonl", selected_rows)
    (args.output_dir / "REPORT.md").write_text(_markdown(payload), encoding="utf-8")
    print(json.dumps({
        "dataset": payload["dataset"],
        "production": production,
        "selected": selected,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
