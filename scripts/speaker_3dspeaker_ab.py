from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
import wave
from pathlib import Path

import torch
import numpy as np


MODEL_CONFIGS = {
    "campplus": {
        "model_id": "iic/speech_campplus_sv_zh_en_16k-common_advanced",
        "revision": "v1.0.0",
        "model_ckpt": "campplus_cn_en_common.pt",
        "model_obj": "speakerlab.models.campplus.DTDNN.CAMPPlus",
        "model_args": {"feat_dim": 80, "embedding_size": 192},
        "batch_size": 64,
    },
    "eres2netv2": {
        "model_id": "iic/speech_eres2netv2_sv_zh-cn_16k-common",
        "revision": "v1.0.1",
        "model_ckpt": "pretrained_eres2netv2.ckpt",
        "model_obj": "speakerlab.models.eres2net.ERes2NetV2.ERes2NetV2",
        "model_args": {"feat_dim": 80, "embedding_size": 192},
        "batch_size": 16,
    },
}


def load_module(repo: Path):
    script = repo / "speakerlab" / "bin" / "infer_diarization.py"
    sys.path.insert(0, str(repo))
    spec = importlib.util.spec_from_file_location("speakerlab_diarization", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def install_embedding_factory(module, key: str):
    selected = MODEL_CONFIGS[key]

    def factory(device=None, cache_dir=None):
        conf = {
            "model_id": selected["model_id"],
            "revision": selected["revision"],
            "model_ckpt": selected["model_ckpt"],
            "embedding_model": {
                "obj": selected["model_obj"],
                "args": selected["model_args"],
            },
            "feature_extractor": {
                "obj": "speakerlab.process.processor.FBank",
                "args": {"n_mels": 80, "sample_rate": 16000, "mean_nor": True},
            },
        }
        model_dir = module.download_model_from_modelscope(
            conf["model_id"], conf["revision"], cache_dir
        )
        config = module.Config(conf)
        feature_extractor = module.build("feature_extractor", config)
        embedding_model = module.build("embedding_model", config)
        checkpoint = torch.load(
            str(Path(model_dir) / conf["model_ckpt"]), map_location="cpu"
        )
        embedding_model.load_state_dict(checkpoint)
        embedding_model.eval()
        if device is not None:
            embedding_model.to(device)
        return embedding_model, feature_extractor

    module.get_speaker_embedding_model = factory


def audio_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as wav:
        return wav.getnframes() / wav.getframerate()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Controlled 3D-Speaker embedding/oracle-count experiment."
    )
    parser.add_argument("audio", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--repo", type=Path, default=Path("/opt/3D-Speaker"))
    parser.add_argument("--cache-dir", type=Path, default=Path("/app/models/3dspeaker-cache"))
    parser.add_argument("--embedding", choices=MODEL_CONFIGS, default="campplus")
    parser.add_argument("--speaker-num", type=int)
    parser.add_argument(
        "--debug-npz", type=Path,
        help="Optional transient chunk/embedding diagnostics; never a production index.",
    )
    parser.add_argument(
        "--segmentation-json", type=Path,
        help="Use model-derived exclusive speech segments as clean embedding regions.",
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    module = load_module(args.repo)
    install_embedding_factory(module, args.embedding)
    started = time.perf_counter()
    pipeline = module.Diarization3Dspeaker(
        device=args.device,
        speaker_num=args.speaker_num,
        model_cache_dir=str(args.cache_dir),
    )
    pipeline.batchsize = MODEL_CONFIGS[args.embedding]["batch_size"]
    if args.segmentation_json is not None:
        segmentation = json.loads(args.segmentation_json.read_text(encoding="utf-8"))
        clean_regions = [
            [float(turn["start"]), float(turn["end"])]
            for turn in segmentation["turns"]
            if float(turn["end"]) > float(turn["start"])
        ]

        def model_segments(_wav):
            return clean_regions

        pipeline.do_vad = model_segments
    # CommonClustering routes short recordings to AHC, whose implementation
    # ignores speaker_num.  Oracle experiments must use the spectral branch or
    # they do not actually test the requested speaker count.
    if args.speaker_num is not None:
        pipeline.cluster.cluster_line = 0
        pipeline.cluster.min_cluster_size = 0
        pipeline.cluster.mer_cos = None
    captured: dict[str, np.ndarray] = {}
    original_extract = pipeline.do_emb_extraction

    def capture_embeddings(chunks, wav):
        embeddings = original_extract(chunks, wav)
        captured["chunk_times_s"] = np.asarray(chunks, dtype=np.float32)
        captured["embeddings"] = embeddings.astype(np.float32, copy=False)
        return embeddings

    pipeline.do_emb_extraction = capture_embeddings
    loaded = time.perf_counter() - started
    started = time.perf_counter()
    raw_turns = pipeline(str(args.audio), speaker_num=args.speaker_num)
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    turns = [
        {"start": round(float(start), 3), "end": round(float(end), 3),
         "speaker": f"SPEAKER_{int(speaker):02d}"}
        for start, end, speaker in raw_turns
    ]
    duration = audio_seconds(args.audio)
    payload = {
        "model": f"modelscope/3D-Speaker + {args.embedding}",
        "embedding_model": MODEL_CONFIGS[args.embedding]["model_id"],
        "speaker_count_mode": "oracle" if args.speaker_num is not None else "automatic",
        "requested_speaker_count": args.speaker_num,
        "segmentation_source": str(args.segmentation_json) if args.segmentation_json else None,
        "audio_seconds": round(duration, 3),
        "load_seconds": round(loaded, 3),
        "infer_seconds": round(elapsed, 3),
        "rtf": round(elapsed / duration, 4),
        "speakers": sorted({turn["speaker"] for turn in turns}),
        "num_turns": len(turns),
        "turns": turns,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.debug_npz is not None:
        args.debug_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.debug_npz, **captured)
    print(json.dumps({key: value for key, value in payload.items() if key != "turns"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
