"""Production adapter for the pinned 3D-Speaker inference components.

The upstream CLI imports pyannote at module import time even when overlap
detection is disabled. MomentSeek does not expose overlap detection, so this
adapter loads only the VAD, CAM++ and clustering components that we use.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics.pairwise import cosine_similarity

from speakerlab.process.processor import FBank
from speakerlab.models.campplus.DTDNN import CAMPPlus
from modelscope.pipelines import pipeline as modelscope_pipeline
from modelscope.utils.constant import Tasks

from app.model_sources import resolve_modelscope_model_source


EMBEDDING_MODEL_ID = "iic/speech_campplus_sv_zh_en_16k-common_advanced"
EMBEDDING_REVISION = "v1.0.0"
EMBEDDING_CHECKPOINT = "campplus_cn_en_common.pt"
VAD_MODEL_ID = "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"
VAD_REVISION = "v2.0.4"


def load_audio(
    source: str | Path | np.ndarray | torch.Tensor,
    ori_fs: int | None = None,
    obj_fs: int | None = None,
) -> torch.Tensor:
    """Load mono float audio without torchaudio's optional TorchCodec runtime.

    Platform callers normalise uploads and video soundtracks to PCM WAV with
    ffmpeg before reaching this adapter. SoundFile therefore covers the
    production path without introducing a second FFmpeg ABI via TorchCodec.
    """
    if isinstance(source, (str, Path)):
        samples, sample_rate = sf.read(
            str(source), dtype="float32", always_2d=True
        )
        values = samples.mean(axis=1, dtype=np.float32)
    elif isinstance(source, (np.ndarray, torch.Tensor)):
        values = (
            source.detach().cpu().numpy()
            if isinstance(source, torch.Tensor)
            else np.asarray(source)
        )
        if np.issubdtype(values.dtype, np.integer):
            info = np.iinfo(values.dtype)
            scale = float(max(abs(info.min), info.max))
            values = values.astype(np.float32) / scale
        else:
            values = values.astype(np.float32, copy=False)
        if values.ndim > 2:
            raise ValueError("audio arrays must have at most two dimensions")
        if values.ndim == 2:
            axis = 0 if values.shape[0] <= values.shape[1] else 1
            values = values.mean(axis=axis, dtype=np.float32)
        sample_rate = ori_fs
    else:
        raise TypeError(f"unsupported audio source: {type(source).__name__}")

    if sample_rate and obj_fs and sample_rate != obj_fs:
        divisor = math.gcd(int(sample_rate), int(obj_fs))
        values = resample_poly(
            values, int(obj_fs) // divisor, int(sample_rate) // divisor
        ).astype(np.float32, copy=False)
    return torch.from_numpy(np.ascontiguousarray(values)).unsqueeze(0)


class _ProductionClustering:
    """Dependency-light equivalent of the upstream common clustering policy."""

    def __call__(self, embeddings):
        values = np.asarray(embeddings, dtype=np.float32)
        count = len(values)
        if count <= 1:
            return np.zeros(count, dtype=np.int32)
        if count < 40:
            return AgglomerativeClustering(
                n_clusters=None,
                metric="cosine",
                linkage="average",
                distance_threshold=0.6,
            ).fit_predict(values).astype(np.int32)

        affinity = cosine_similarity(values)
        np.fill_diagonal(affinity, 0)
        laplacian = np.diag(np.abs(affinity).sum(axis=1)) - affinity
        eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
        maximum = min(15, count - 1)
        gaps = np.diff(eigenvalues[: maximum + 1])
        speakers = max(1, int(np.argmax(gaps)) + 1)
        labels = KMeans(n_clusters=speakers, n_init=10, random_state=0).fit_predict(
            eigenvectors[:, :speakers]
        )
        return labels.astype(np.int32)


def _model_dir(cache_dir: str | Path, model_id: str) -> Path:
    return Path(
        resolve_modelscope_model_source(
            cache_dir, model_id, local_files_only=True
        )
    )


def _normalize_device(value: str | torch.device | None) -> torch.device:
    if value is None or str(value) == "auto":
        if torch.cuda.is_available():
            value = "cuda"
        else:
            try:
                import torch_npu  # noqa: F401
                value = "npu" if torch.npu.is_available() else "cpu"
            except ImportError:
                value = "cpu"
    if str(value).split(":", 1)[0] == "npu":
        # TORCH_DEVICE_BACKEND_AUTOLOAD is disabled in production to keep model
        # workers deterministic, so register the private-use backend explicitly.
        import torch_npu  # noqa: F401
    device = torch.device(value)
    return device


class Diarization3Dspeaker:
    """Subset of the upstream API consumed by MomentSeek."""

    def __init__(self, device=None, model_cache_dir=None, **_: object):
        if model_cache_dir is None:
            raise ValueError("model_cache_dir is required in offline production mode")
        self.device = _normalize_device(device)
        self.fs = 16000
        self.batchsize = 64

        model_dir = _model_dir(model_cache_dir, EMBEDDING_MODEL_ID)
        checkpoint = model_dir / EMBEDDING_CHECKPOINT
        if not checkpoint.is_file():
            raise FileNotFoundError(f"speaker checkpoint is missing: {checkpoint}")

        self.feature_extractor = FBank(
            n_mels=80, sample_rate=self.fs, mean_nor=True
        )
        self.embedding_model = CAMPPlus(feat_dim=80, embedding_size=192)
        state = torch.load(checkpoint, map_location="cpu")
        self.embedding_model.load_state_dict(state)
        self.embedding_model.eval().to(self.device)

        vad_dir = _model_dir(model_cache_dir, VAD_MODEL_ID)
        # ModelScope's VAD pipeline is kept on CPU. CAM++ is the expensive part
        # and runs on NPU; forcing this pipeline through a CUDA-shaped device
        # adapter is less reliable and brings no useful throughput gain.
        self.vad_model = modelscope_pipeline(
            task=Tasks.voice_activity_detection,
            model=str(vad_dir),
            device="cpu",
            disable_pbar=True,
            disable_update=True,
        )
        self.cluster = _ProductionClustering()

    def do_vad(self, wav):
        result = self.vad_model(wav[0])[0]
        return [[start / 1000, end / 1000] for start, end in result["value"]]

    @staticmethod
    def chunk(start: float, end: float, dur: float = 1.5, step: float = 0.75):
        chunks = []
        current = start
        while current + dur < end + step:
            chunks.append([current, min(current + dur, end)])
            current += step
        return chunks

    def do_emb_extraction(self, chunks, wav):
        from speakerlab.utils.utils import circle_pad

        samples = [wav[0, int(start * self.fs):int(end * self.fs)] for start, end in chunks]
        maximum = max(sample.shape[0] for sample in samples)
        batch = torch.stack([circle_pad(sample, maximum) for sample in samples]).unsqueeze(1)
        with torch.no_grad():
            # torchaudio's Kaldi FBank reaches a complex FFT abs kernel which
            # torch_npu 2.9 does not implement. Feature extraction is small and
            # deterministic on CPU; move only the CAM++ input to the accelerator.
            features = torch.vmap(self.feature_extractor)(batch)
            return self.embedding_model(features.to(self.device)).cpu().numpy()
