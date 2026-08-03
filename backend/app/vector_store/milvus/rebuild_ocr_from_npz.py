"""Offline OCR NPZ-to-Milvus rebuild command for the hybrid OCR schema.

This module is intentionally an operational migration tool. Production search
never reads NPZ files; it only uses retained assets as a one-time source for
rebuilding ``ocr_embeddings`` after the BM25 schema upgrade.
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

from app.core.settings import get_settings


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OcrNpzAsset:
    video_id: str
    index_dir: Path
    npz_path: Path


def discover_ocr_npz_assets(index_root: Path) -> list[OcrNpzAsset]:
    """Return one OCR NPZ asset per immediate video index directory."""
    if not index_root.exists():
        raise FileNotFoundError(f"OCR index root does not exist: {index_root}")
    return [
        OcrNpzAsset(
            video_id=npz_path.parent.name,
            index_dir=npz_path.parent,
            npz_path=npz_path,
        )
        for npz_path in sorted(index_root.glob("*/ocr.npz"))
    ]


def rebuild_ocr_assets(index_root: Path, *, dry_run: bool = False) -> int:
    """Upsert retained OCR NPZ assets into the current Milvus OCR collection."""
    assets = discover_ocr_npz_assets(index_root)
    if dry_run:
        for asset in assets:
            print(f"{asset.video_id}\t{asset.npz_path}")
        return len(assets)

    # Discovery and dry runs should also work in a lightweight development
    # environment that does not install the optional Milvus client package.
    from app.indexing.manifest import publish_recovery_channel_version
    from .milvus_asset_version import next_asset_version, publish_asset_version
    from .milvus_client import get_milvus_client
    from .milvus_indexer import reindex_from_file
    from .milvus_schema import MODEL_VERSIONS

    client = get_milvus_client()
    for position, asset in enumerate(assets, start=1):
        asset_version = next_asset_version(asset.index_dir)
        logger.info(
            "Rebuilding OCR %d/%d video=%s asset_version=%s from %s",
            position,
            len(assets),
            asset.video_id,
            asset_version,
            asset.npz_path,
        )
        written_rows = reindex_from_file(
            client=client,
            modality="ocr",
            video_id=asset.video_id,
            asset_version=asset_version,
            model_version=MODEL_VERSIONS["ocr"],
            npz_path=str(asset.npz_path),
        )
        persisted_rows = client.count_video_modality_version(
            asset.video_id, "ocr", asset_version
        )
        if persisted_rows != written_rows:
            raise RuntimeError(
                f"OCR recovery verification failed for {asset.video_id}: "
                f"expected={written_rows} persisted={persisted_rows}"
            )
        # Readers switch only after the replacement rows are fully visible.
        publish_recovery_channel_version(
            asset.index_dir,
            channel="ocr",
            asset_version=asset_version,
            row_count=persisted_rows,
        )
        deleted = client.delete_video_modality_except_version(
            asset.video_id, "ocr", asset_version
        )
        if deleted < 0:
            logger.warning("OCR recovery published but old rows could not be cleaned: %s", asset.video_id)
        publish_asset_version(asset.index_dir, asset_version)
    return len(assets)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Rebuild Milvus OCR hybrid data from retained OCR NPZ assets."
    )
    parser.add_argument(
        "--index-root",
        type=Path,
        default=get_settings().index_dir,
        help="Directory containing <video-id>/ocr.npz assets.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List source assets without connecting to Milvus or writing data.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    count = rebuild_ocr_assets(args.index_root, dry_run=args.dry_run)
    logger.info("OCR rebuild complete: %d asset(s)", count)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
