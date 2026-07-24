"""人脸模态单独评测脚本（VPCD / Sherlock）。

这是一个**手动运行的评测脚本**，不是 pytest 单元测试（依赖本地大视频 + 模型 +
数分钟推理，故意不加 test_ 前缀，避免被 pytest 自动收集）。

流程：
  1. 对选定的剧集视频跑人脸索引（build_face_index），得到每个人脸 track 的
     平均身份向量与时间段，缓存到磁盘（换阈值重跑无需重新抽帧）。
  2. 用参考图 sherlock.png 编码出查询身份向量。
  3. 按余弦阈值检索命中的 track，合并成一个个预测时间段（与项目检索输出一致）。
  4. 对照 GT（sherlock_face_segments_flat.csv 里 person=sherlock 的片段）计算：
       - 段级查全率（主指标）
       - 时间覆盖召回 / 时间精确率 / 时间 F1
       - duration-命中率直方图（PNG + 文本）

用法示例（在 backend 目录下）：
    python -m tests.face_recall_eval --episodes e01
    python -m tests.face_recall_eval --episodes e01,e02,e03 --fps 2 --threshold 0.35
    python -m tests.face_recall_eval --episodes e01 --reindex   # 强制重建索引

也可直接运行： python tests/face_recall_eval.py --episodes e01
"""
from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# 允许从 backend/tests 直接运行时导入 app.*
BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.indexing.faces import build_face_index, encode_face_reference  # noqa: E402
from app.search import _face_candidates, _seconds  # noqa: E402  (复用项目检索逻辑)

# ----------------------------- 默认路径配置 -----------------------------
DEFAULT_VIDEO_DIR = Path(r"D:\Data\VPCD\sherlock")
DEFAULT_GT_CSV = Path(r"D:\Data\VPCD\sherlock_simple_gt\sherlock_face_segments_flat.csv")
DEFAULT_QUERY_IMAGE = Path(r"D:\Data\VPCD\query_pic\sherlock.png")
DEFAULT_PERSON = "sherlock"
DEFAULT_MODEL_ROOT = BACKEND_DIR / "models" / "insightface"
DEFAULT_CACHE_DIR = BACKEND_DIR / "tests" / ".face_eval_cache"

# 视频文件名模板：episode "e01" -> sherlock_s01_e01_main.mp4
VIDEO_NAME_TEMPLATE = "sherlock_s01_{episode}_main.mp4"


@dataclass
class Interval:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


# ----------------------------- GT 加载 -----------------------------
def load_ground_truth(csv_path: Path, person: str, episodes: list[str]) -> dict[str, list[Interval]]:
    """读取 GT csv，返回 {episode: [Interval, ...]}，只保留指定 person 与剧集。"""
    person_key = person.strip().casefold()
    wanted = {ep.strip().casefold() for ep in episodes}
    result: dict[str, list[Interval]] = {ep: [] for ep in episodes}
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if str(row.get("person", "")).strip().casefold() != person_key:
                continue
            episode = str(row.get("episode", "")).strip()
            if episode.casefold() not in wanted:
                continue
            try:
                start = float(row["start"])
                end = float(row["end"])
            except (KeyError, ValueError):
                continue
            if end > start:
                result.setdefault(episode, []).append(Interval(start, end))
    for episode in result:
        result[episode].sort(key=lambda item: item.start)
    return result


# ----------------------------- 区间工具 -----------------------------
def merge_intervals(intervals: list[Interval], gap: float = 0.0) -> list[Interval]:
    """合并重叠/相邻（间隔<=gap）的区间，得到不重叠的预测覆盖。"""
    if not intervals:
        return []
    ordered = sorted(intervals, key=lambda item: item.start)
    merged = [Interval(ordered[0].start, ordered[0].end)]
    for current in ordered[1:]:
        last = merged[-1]
        if current.start <= last.end + gap:
            last.end = max(last.end, current.end)
        else:
            merged.append(Interval(current.start, current.end))
    return merged


def overlap_seconds(segment: Interval, predicted: list[Interval]) -> float:
    """segment 与预测区间集合的交集总秒数。predicted 需已排序且不重叠。"""
    total = 0.0
    for pred in predicted:
        lo = max(segment.start, pred.start)
        hi = min(segment.end, pred.end)
        if hi > lo:
            total += hi - lo
    return total


def total_duration(intervals: list[Interval]) -> float:
    return float(sum(item.duration for item in intervals))


# ----------------------------- 索引 + 检索 -----------------------------
def build_or_load_index(
    episode: str,
    video_path: Path,
    cache_dir: Path,
    fps: float,
    model_root: Path,
    reindex: bool = False,
) -> Path:
    """对某集视频跑人脸索引并缓存 face.npz；缓存按 (episode, fps) 区分。"""
    cache_dir.mkdir(parents=True, exist_ok=True)
    npz_path = cache_dir / f"{episode}_fps{fps:g}_face.npz"
    if npz_path.exists() and not reindex:
        print(f"  [{episode}] 复用缓存索引: {npz_path.name}")
        return npz_path
    if not video_path.exists():
        raise FileNotFoundError(f"视频文件不存在: {video_path}")
    print(f"  [{episode}] 建立人脸索引 (fps={fps:g})，视频较大请耐心等待…")
    result = build_face_index(
        video_path=str(video_path),
        output_path=str(npz_path),
        model_name="buffalo_l",
        sample_fps=fps,
        provider="cpu",
        device_id=0,
        model_root=str(model_root),
    )
    print(f"  [{episode}] 索引完成: tracks={result['tracks']} detections={result['detections']}")
    return npz_path


def predict_intervals(
    npz_path: Path,
    query: np.ndarray,
    threshold: float,
    merge_gap: float,
) -> tuple[list[Interval], list[float]]:
    """从 face.npz 检索命中 track，返回 (合并后的预测区间, 命中 track 的余弦分数)。"""
    with np.load(npz_path, allow_pickle=False) as data:
        embeddings = np.asarray(data["embeddings"], dtype=np.float32)
        limit = int(len(embeddings)) or 1
        candidates = _face_candidates(data, query, video_id=npz_path.stem, limit=limit, threshold=threshold)
    hits = [c for c in candidates if c.above_threshold]
    raw_intervals = [Interval(c.start_time, c.end_time) for c in hits]
    cosines = [float(c.raw_score) for c in hits if c.raw_score is not None]
    return merge_intervals(raw_intervals, gap=merge_gap), cosines


# ----------------------------- 指标计算 -----------------------------
@dataclass
class SegmentEval:
    episode: str
    segment: Interval
    covered_seconds: float

    @property
    def coverage_ratio(self) -> float:
        return self.covered_seconds / self.segment.duration if self.segment.duration > 0 else 0.0


def evaluate(
    gt: dict[str, list[Interval]],
    predictions: dict[str, list[Interval]],
    hit_ratio: float,
) -> dict:
    """计算段级查全率、时间覆盖召回、时间精确率与 F1。

    hit_ratio: GT 片段被判定为“命中”所需的最小覆盖比例（默认 0.5）。
    """
    seg_evals: list[SegmentEval] = []
    for episode, segments in gt.items():
        preds = predictions.get(episode, [])
        for seg in segments:
            covered = overlap_seconds(seg, preds)
            seg_evals.append(SegmentEval(episode, seg, covered))

    num_gt = len(seg_evals)
    # 段级查全率（两种口径）
    hit_ratio_count = sum(1 for s in seg_evals if s.coverage_ratio >= hit_ratio)
    any_overlap_count = sum(1 for s in seg_evals if s.covered_seconds > 0)
    segment_recall = hit_ratio_count / num_gt if num_gt else 0.0
    any_overlap_recall = any_overlap_count / num_gt if num_gt else 0.0

    # 时间覆盖召回：命中 GT 秒数 / GT 总秒数
    gt_seconds = sum(s.segment.duration for s in seg_evals)
    covered_seconds = sum(s.covered_seconds for s in seg_evals)
    temporal_recall = covered_seconds / gt_seconds if gt_seconds else 0.0

    # 时间精确率：预测落在 GT 内的秒数 / 预测总秒数
    predicted_seconds = 0.0
    predicted_inside_gt = 0.0
    for episode, preds in predictions.items():
        predicted_seconds += total_duration(preds)
        for pred in preds:
            predicted_inside_gt += overlap_seconds(pred, gt.get(episode, []))
    temporal_precision = predicted_inside_gt / predicted_seconds if predicted_seconds else 0.0
    denom = temporal_precision + temporal_recall
    temporal_f1 = (2 * temporal_precision * temporal_recall / denom) if denom else 0.0

    return {
        "num_gt_segments": num_gt,
        "segment_recall": segment_recall,
        "segment_hits": hit_ratio_count,
        "any_overlap_recall": any_overlap_recall,
        "any_overlap_hits": any_overlap_count,
        "hit_ratio": hit_ratio,
        "gt_seconds": gt_seconds,
        "covered_seconds": covered_seconds,
        "temporal_recall": temporal_recall,
        "predicted_seconds": predicted_seconds,
        "predicted_inside_gt": predicted_inside_gt,
        "temporal_precision": temporal_precision,
        "temporal_f1": temporal_f1,
        "seg_evals": seg_evals,
    }


# ----------------------------- duration-命中率直方图 -----------------------------
DEFAULT_DURATION_BINS = [0.0, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0, float("inf")]


def duration_hit_histogram(
    seg_evals: list[SegmentEval],
    hit_ratio: float,
    bins: list[float],
) -> list[dict]:
    """按 GT 片段时长分箱，统计每箱的命中率（覆盖>=hit_ratio 视为命中）。"""
    stats = []
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        in_bin = [s for s in seg_evals if lo <= s.segment.duration < hi]
        hits = sum(1 for s in in_bin if s.coverage_ratio >= hit_ratio)
        label = f"{lo:g}-{'∞' if hi == float('inf') else f'{hi:g}'}s"
        stats.append({
            "label": label,
            "lo": lo,
            "hi": hi,
            "count": len(in_bin),
            "hits": hits,
            "hit_rate": (hits / len(in_bin)) if in_bin else 0.0,
        })
    return stats


def print_text_histogram(stats: list[dict]) -> None:
    print("\nduration-命中率直方图（文本）:")
    print(f"  {'时长区间':<12}{'片段数':>6}{'命中':>6}{'命中率':>9}   条形")
    for row in stats:
        bar = "█" * int(round(row["hit_rate"] * 30))
        print(f"  {row['label']:<12}{row['count']:>6}{row['hits']:>6}{row['hit_rate'] * 100:>8.1f}%   {bar}")


def save_histogram_png(stats: list[dict], output_path: Path, title: str) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # matplotlib 不可用时退回文本
        print(f"  (matplotlib 不可用，跳过 PNG: {exc})")
        return False

    labels = [row["label"] for row in stats]
    hit_rates = [row["hit_rate"] * 100 for row in stats]
    counts = [row["count"] for row in stats]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, hit_rates, color="#4C72B0", edgecolor="black")
    ax.set_xlabel("GT 片段时长区间 (duration)")
    ax.set_ylabel("命中率 hit rate (%)")
    ax.set_ylim(0, 105)
    ax.set_title(title)
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f"n={count}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=130)
    plt.close(fig)
    print(f"  直方图已保存: {output_path}")
    return True


# ----------------------------- CLI / main -----------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="人脸模态单独评测（VPCD / Sherlock）")
    parser.add_argument("--episodes", default="e01",
                        help="要评测的剧集，逗号分隔，如 e01 或 e01,e02,e03")
    parser.add_argument("--fps", type=float, default=2.0, help="人脸检测抽帧 fps（默认 2.0）")
    parser.add_argument("--threshold", type=float, default=0.35, help="人脸余弦命中阈值（默认 0.35）")
    parser.add_argument("--hit-ratio", type=float, default=0.5,
                        help="GT 片段被判命中所需的最小覆盖比例（默认 0.5）")
    parser.add_argument("--merge-gap", type=float, default=2.0,
                        help="预测区间合并的最大间隔秒数（默认 2.0）")
    parser.add_argument("--person", default=DEFAULT_PERSON, help="GT 中的人物名（默认 sherlock）")
    parser.add_argument("--video-dir", type=Path, default=DEFAULT_VIDEO_DIR)
    parser.add_argument("--gt-csv", type=Path, default=DEFAULT_GT_CSV)
    parser.add_argument("--query-image", type=Path, default=DEFAULT_QUERY_IMAGE)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--reindex", action="store_true", help="强制重建索引，忽略缓存")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_CACHE_DIR,
                        help="直方图 PNG 输出目录")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    episodes = [ep.strip() for ep in args.episodes.split(",") if ep.strip()]
    if not episodes:
        print("未指定有效剧集。")
        return 1

    print("=" * 68)
    print(f"人脸模态评测 | person={args.person} | episodes={','.join(episodes)}")
    print(f"fps={args.fps:g} threshold={args.threshold:g} hit_ratio={args.hit_ratio:g} merge_gap={args.merge_gap:g}")
    print("=" * 68)

    # 1. 参考图编码
    print(f"\n[1] 编码参考图: {args.query_image}")
    query = encode_face_reference(
        str(args.query_image), model_name="buffalo_l", provider="cpu",
        device_id=0, model_root=str(args.model_root),
    )

    # 2. 逐集索引 + 检索
    print("\n[2] 索引与检索")
    predictions: dict[str, list[Interval]] = {}
    cosine_summary: list[float] = []
    for episode in episodes:
        video_path = args.video_dir / VIDEO_NAME_TEMPLATE.format(episode=episode)
        npz_path = build_or_load_index(
            episode, video_path, args.cache_dir, args.fps, args.model_root, args.reindex,
        )
        preds, cosines = predict_intervals(npz_path, query, args.threshold, args.merge_gap)
        predictions[episode] = preds
        cosine_summary.extend(cosines)
        print(f"  [{episode}] 命中 track={len(cosines)} 合并后预测段={len(preds)} "
              f"覆盖={total_duration(preds):.1f}s")

    # 3. 加载 GT
    print(f"\n[3] 加载 GT: {args.gt_csv}")
    gt = load_ground_truth(args.gt_csv, args.person, episodes)
    for episode in episodes:
        print(f"  [{episode}] GT 片段数={len(gt.get(episode, []))} "
              f"总时长={total_duration(gt.get(episode, [])):.1f}s")

    # 4. 计算指标
    print("\n[4] 指标")
    metrics = evaluate(gt, predictions, args.hit_ratio)
    print("-" * 68)
    print(f"  GT 片段总数              : {metrics['num_gt_segments']}")
    print(f"  段级查全率(主) [覆盖>={args.hit_ratio:g}]: "
          f"{metrics['segment_recall'] * 100:.1f}%  ({metrics['segment_hits']}/{metrics['num_gt_segments']})")
    print(f"  段级查全率 [任意重叠]    : "
          f"{metrics['any_overlap_recall'] * 100:.1f}%  ({metrics['any_overlap_hits']}/{metrics['num_gt_segments']})")
    print(f"  时间覆盖召回             : {metrics['temporal_recall'] * 100:.1f}%  "
          f"({metrics['covered_seconds']:.1f}/{metrics['gt_seconds']:.1f}s)")
    print(f"  时间精确率               : {metrics['temporal_precision'] * 100:.1f}%  "
          f"({metrics['predicted_inside_gt']:.1f}/{metrics['predicted_seconds']:.1f}s)")
    print(f"  时间 F1                  : {metrics['temporal_f1'] * 100:.1f}%")
    if cosine_summary:
        arr = np.asarray(cosine_summary, dtype=np.float32)
        print(f"  命中 track 余弦: min={arr.min():.3f} mean={arr.mean():.3f} max={arr.max():.3f}")
    print("-" * 68)

    # 5. duration-命中率直方图
    print("\n[5] duration-命中率直方图")
    stats = duration_hit_histogram(metrics["seg_evals"], args.hit_ratio, DEFAULT_DURATION_BINS)
    print_text_histogram(stats)
    png_name = f"duration_hitrate_{'_'.join(episodes)}_fps{args.fps:g}_th{args.threshold:g}.png"
    save_histogram_png(
        stats, args.out_dir / png_name,
        title=f"Face recall by GT duration ({args.person}, {','.join(episodes)}, fps={args.fps:g}, th={args.threshold:g})",
    )
    print("\n完成。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
