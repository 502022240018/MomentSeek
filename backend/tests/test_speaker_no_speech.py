"""测试 speaker 阶段在无人声场景下的行为"""
import tempfile
from pathlib import Path
import numpy as np
import pytest


def test_speaker_should_skip_when_asr_empty():
    """当 ASR 阶段返回空结果（仅背景音乐）时，speaker 应该优雅跳过而不写入文件"""
    from app.indexing.speaker import build_speaker_index
    
    # 模拟 ASR 空结果：0 个 chunk
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        asr_npz = tmp_path / "asr.npz"
        
        # 模拟 ASR no_audio 场景的输出
        np.savez_compressed(
            asr_npz,
            chunk_times_ms=np.empty((0, 2), dtype=np.int32),
            texts=np.array([], dtype=object),
            embeddings=np.empty((0, 0), dtype=np.float16),
            embedding_chunk_indices=np.empty((0,), dtype=np.int32),
        )
        
        speaker_npz = tmp_path / "speaker.npz"
        work_dir = tmp_path / "work"
        work_dir.mkdir()
        
        # 需要一个实际的视频文件，这里用占位，实际测试需提供
        video_path = tmp_path / "dummy.mp4"
        video_path.write_bytes(b"")  # 占位
        
        # speaker 应该检测到 eligible=0 并提前返回
        try:
            result = build_speaker_index(
                video_path=str(video_path),
                asr_path=str(asr_npz),
                output_path=str(speaker_npz),
                working_dir=str(work_dir),
                model_repo="dummy",
                model_cache_dir="dummy",
                device="cpu",
                milvus_ctx=None,
            )
            # eligible=0 时在 line 261 就应该提前返回
            assert result["utterances"] == 0
            assert not speaker_npz.exists(), "speaker.npz should not exist when no eligible ASR chunks"
        except RuntimeError as e:
            # 如果抛出 "音频中没有可用于说话人索引的有效语音"，说明没有提前返回，继续执行到 VAD
            pytest.fail(f"Speaker should skip early when eligible=0, but got: {e}")


if __name__ == "__main__":
    test_speaker_should_skip_when_asr_empty()
