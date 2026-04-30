#!/usr/bin/env python3
"""
Batch-export raw videos into ranked face segments using dataset_processing.

Output layout:
  output_dir/
    confident/
      sample.mp4
      sample.json
    medium/
      ...
    unconfident/
      ...
    reports/
      source_report.json
      source_frame_log.csv  # optional
    export_manifest.jsonl
    export_resume_state.json
    summary.json

The inner per-video processor is source-agnostic. The outer batch exporter keeps
the existing archive-level orchestration contract:
  - batch-local JSONL manifest
  - resume from the next unfinished source video
  - tiered output folders for downstream packing
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import shutil
import sys
import time
from dataclasses import fields
from pathlib import Path
from typing import Any

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from batch.config import (
    ConfigError,
    get_bool,
    get_mapping,
    get_str,
    load_yaml_config,
    resolve_repo_path,
)
from batch.transcode import (
    resolve_ffmpeg_bin,
    select_video_encoder,
)


QUALITY_TIERS = ("confident", "medium", "unconfident")


def timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %Z")


def log(message: str) -> None:
    print(f"{timestamp()} {message}", flush=True)


def append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def load_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return None


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def iter_videos(input_dir: Path):
    for path in sorted(input_dir.glob("*.mp4")):
        if path.is_file():
            yield path


def resolve_dataset_kind(requested: str, source_archive: str, input_dir: Path) -> str:
    if requested != "auto":
        return requested
    lower = (source_archive or "").lower()
    if "talkvid" in lower:
        return "talkvid"
    if "hdtf" in lower:
        return "hdtf"
    if any(path.with_suffix(".json").exists() for path in input_dir.glob("*.mp4")):
        return "talkvid"
    return "hdtf"


def load_resume_progress(manifest_path: Path, resume_state_path: Path) -> tuple[dict, int, int, set[int]]:
    counters = {
        "ok": 0,
        "skip": 0,
        "discard": 0,
        "fail": 0,
        "confident": 0,
        "medium": 0,
        "unconfident": 0,
    }
    total_segments = 0
    latest_manifest_index = 0
    processed_video_indexes: set[int] = set()

    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue

                status = payload.get("status")
                tier = payload.get("tier")
                if status in counters:
                    counters[status] += 1
                if status == "ok" and tier in QUALITY_TIERS:
                    counters[tier] += 1
                total_segments += 1
                try:
                    video_index = int(payload.get("index") or 0)
                    if video_index > 0:
                        processed_video_indexes.add(video_index)
                        latest_manifest_index = max(latest_manifest_index, video_index)
                except (TypeError, ValueError):
                    pass

    resume_state = load_json(resume_state_path)
    next_video_index = 0
    if isinstance(resume_state, dict):
        try:
            next_video_index = int(resume_state.get("next_video_index") or 0)
        except (TypeError, ValueError):
            next_video_index = 0

    if processed_video_indexes:
        missing_before_latest = [
            idx for idx in range(1, latest_manifest_index + 1)
            if idx not in processed_video_indexes
        ]
        if missing_before_latest:
            next_video_index = min(missing_before_latest) - 1
        else:
            next_video_index = max(next_video_index, latest_manifest_index)
    else:
        next_video_index = 0
    return counters, total_segments, next_video_index, processed_video_indexes


def _apply_mapping(target: object, mapping: dict[str, Any], section_name: str) -> None:
    allowed = {field.name for field in fields(target)}
    for key, value in mapping.items():
        if key not in allowed:
            raise ConfigError(f"unsupported key in {section_name}: {key}")
        setattr(target, key, value)


def build_dataset_processing_config(config_path: Path, config: dict[str, Any], output_dir: Path):
    del config_path
    from dataset_processing.config import apply_dataset_config
    from face_processing.config import PipelineConfig

    pipeline_cfg = PipelineConfig()
    dataset_mapping = dict(get_mapping(config, "dataset_processing"))

    model_path = resolve_repo_path(REPO_ROOT, get_str(config, "dataset_processing", "model_path"))
    assert model_path is not None
    if not model_path.is_file():
        raise ConfigError(
            f"missing MediaPipe face landmarker model: {model_path}. "
            "Place the .task file at the configured path before launching processing."
        )
    dataset_mapping["model_path"] = str(model_path)

    ffmpeg_bin = resolve_ffmpeg_bin(get_str(config, "runtime", "ffmpeg_bin", allow_empty=True) or None)
    ffmpeg_threads = int(get_mapping(config, "runtime").get("ffmpeg_threads", 0))
    ffmpeg_timeout = int(get_mapping(config, "runtime").get("ffmpeg_timeout", 180))

    normalization_codec = select_video_encoder(str(dataset_mapping.get("normalization_codec", "auto")), ffmpeg_bin)
    export_codec = select_video_encoder(str(dataset_mapping.get("export_codec", "auto")), ffmpeg_bin)
    dataset_mapping["normalization_codec"] = normalization_codec
    dataset_mapping["export_codec"] = export_codec
    dataset_mapping["ffmpeg_bin"] = ffmpeg_bin
    dataset_mapping["ffmpeg_threads"] = ffmpeg_threads
    dataset_mapping["ffmpeg_timeout"] = ffmpeg_timeout

    apply_dataset_config(pipeline_cfg, dataset_mapping)
    pipeline_cfg.output_dir = str(output_dir)
    return pipeline_cfg, model_path


def copy_report_artifacts(video_work_dir: Path, reports_dir: Path, source_name: str) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("_report.json", "_frame_log.csv"):
        src = video_work_dir / f"{source_name}{suffix}"
        if src.exists():
            os.replace(src, reports_dir / src.name)


def promote_exported_segments(
    *,
    video_path: Path,
    source_archive: str,
    dataset_kind: str,
    video_work_dir: Path,
    batch_output_dir: Path,
) -> list[dict]:
    source_name = video_path.stem
    promoted: list[dict] = []
    raw_sidecar = load_json(video_path.with_suffix(".json"))

    for meta_path in sorted(video_work_dir.glob(f"{source_name}_seg_*.json")):
        payload = load_json(meta_path)
        if not isinstance(payload, dict):
            raise RuntimeError(f"invalid_segment_json: {meta_path}")

        rank = str(payload.get("rank") or "")
        if rank not in QUALITY_TIERS:
            raise RuntimeError(f"missing_or_invalid_rank in {meta_path}: {rank!r}")

        mp4_path = meta_path.with_suffix(".mp4")
        if not mp4_path.exists():
            raise RuntimeError(f"missing_segment_video: {mp4_path}")

        payload["source_archive"] = source_archive
        payload["source_dataset"] = dataset_kind
        payload["raw_sidecar_present"] = raw_sidecar is not None

        tier_dir = batch_output_dir / rank
        tier_dir.mkdir(parents=True, exist_ok=True)

        final_mp4 = tier_dir / mp4_path.name
        final_json = tier_dir / meta_path.name
        tmp_meta = meta_path.with_suffix(".json.tmp")
        with tmp_meta.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        os.replace(tmp_meta, meta_path)
        os.replace(mp4_path, final_mp4)
        os.replace(meta_path, final_json)
        promoted.append(payload)

    return promoted


def build_segment_name(source_name: str, segment_id: int) -> str:
    return f"{source_name}_seg_{int(segment_id):03d}"


def result_to_manifest_entries(result, source_name: str, promoted_payloads: list[dict]) -> list[dict]:
    promoted_by_id = {
        int(payload["segment_id"]): payload
        for payload in promoted_payloads
        if "segment_id" in payload
    }
    entries: list[dict] = []
    for segment in result.segments:
        segment_name = build_segment_name(source_name, segment.segment_id)
        if segment.status == "exported":
            payload = promoted_by_id.get(int(segment.segment_id), {})
            entries.append(
                {
                    "name": segment_name,
                    "status": "ok",
                    "tier": segment.rank,
                    "message": (
                        f"{segment_name}: ok frames={segment.length} "
                        f"source_range={segment.start_frame}:{segment.end_frame - 1} "
                        f"size={segment.output_size} rank={segment.rank}"
                    ),
                    "source_segment_start_frame": int(segment.start_frame),
                    "source_segment_end_frame": int(segment.end_frame - 1),
                    "segment_index": int(segment.segment_id) + 1,
                    "segment_count": len(result.segments),
                    "segment_meta": payload,
                }
            )
        elif segment.status == "dropped":
            reason = segment.drop_reason or "dropped"
            entries.append(
                {
                    "name": segment_name,
                    "status": "discard",
                    "tier": None,
                    "message": (
                        f"{segment_name}: dropped reason={reason} "
                        f"source_range={segment.start_frame}:{segment.end_frame - 1}"
                    ),
                    "source_segment_start_frame": int(segment.start_frame),
                    "source_segment_end_frame": int(segment.end_frame - 1),
                    "segment_index": int(segment.segment_id) + 1,
                    "segment_count": len(result.segments),
                }
            )

    if not entries and result.status == "dropped":
        entries.append(
            {
                "name": source_name,
                "status": "discard",
                "tier": None,
                "message": f"{source_name}: video_dropped reason={result.drop_reason or 'unknown'}",
                "source_segment_start_frame": 0,
                "source_segment_end_frame": -1,
                "segment_index": 1,
                "segment_count": 1,
            }
        )
    return entries


def summarize_video_entries(
    *,
    source_video: str,
    total_videos: int,
    video_index: int,
    total_source_frames: int,
    segment_entries: list[dict],
) -> tuple[str, dict[str, int]]:
    exported_segments = 0
    dropped_segments = 0
    failed_segments = 0
    exported_source_frames = 0
    tier_counts = {tier: 0 for tier in QUALITY_TIERS}

    for entry in segment_entries:
        status = str(entry.get("status") or "")
        if status == "ok":
            exported_segments += 1
            start_frame = int(entry.get("source_segment_start_frame") or 0)
            end_frame = int(entry.get("source_segment_end_frame") or -1)
            if end_frame >= start_frame:
                exported_source_frames += end_frame - start_frame + 1
            tier = entry.get("tier")
            if tier in QUALITY_TIERS:
                tier_counts[str(tier)] += 1
        elif status == "discard":
            dropped_segments += 1
        elif status == "fail":
            failed_segments += 1

    summary = (
        f"[FaceclipExport] [{video_index}/{total_videos}] {source_video}: "
        f"exported_segments={exported_segments} "
        f"dropped_segments={dropped_segments} "
        f"failed_segments={failed_segments} "
        f"processed_source_frames={exported_source_frames}/{total_source_frames} "
        f"confident={tier_counts['confident']} "
        f"medium={tier_counts['medium']} "
        f"unconfident={tier_counts['unconfident']}"
    )
    return summary, {
        "exported_segments": exported_segments,
        "dropped_segments": dropped_segments,
        "failed_segments": failed_segments,
        "processed_source_frames": exported_source_frames,
        **tier_counts,
    }


def process_one_video(
    *,
    video_path: Path,
    dataset_kind: str,
    source_archive: str,
    batch_output_dir: Path,
    work_root: Path,
    pipeline_cfg,
):
    from dataset_processing.pipeline import process_video

    source_name = video_path.stem
    pipeline_cfg.output_dir = str(work_root)
    result = process_video(str(video_path), pipeline_cfg)
    video_work_dir = work_root / source_name
    reports_dir = batch_output_dir / "reports"
    promoted_payloads = promote_exported_segments(
        video_path=video_path,
        source_archive=source_archive,
        dataset_kind=dataset_kind,
        video_work_dir=video_work_dir,
        batch_output_dir=batch_output_dir,
    )
    copy_report_artifacts(video_work_dir, reports_dir, source_name)
    shutil.rmtree(video_work_dir, ignore_errors=True)
    return result, result_to_manifest_entries(result, source_name, promoted_payloads)


def cleanup_video_artifacts(*, video_path: Path, batch_output_dir: Path, work_root: Path) -> None:
    source_name = video_path.stem
    shutil.rmtree(work_root / source_name, ignore_errors=True)
    reports_dir = batch_output_dir / "reports"
    for path in reports_dir.glob(f"{source_name}_*"):
        try:
            path.unlink()
        except OSError:
            pass
    for tier in QUALITY_TIERS:
        tier_dir = batch_output_dir / tier
        for path in tier_dir.glob(f"{source_name}_seg_*"):
            try:
                path.unlink()
            except OSError:
                pass


def release_video_result_memory(result) -> None:
    seen_frame_ids: set[int] = set()
    frame_lists = [getattr(result, "frame_data", None)]
    for segment in getattr(result, "segments", []):
        frame_lists.append(getattr(segment, "frame_data", None))

    for frame_list in frame_lists:
        if not frame_list:
            continue
        for fd in frame_list:
            if fd is None:
                continue
            frame_id = id(fd)
            if frame_id in seen_frame_ids:
                continue
            seen_frame_ids.add(frame_id)
            fd.landmarks = None
            fd.transform_matrix = None
            fd.bad_reasons.clear()
        frame_list.clear()


def probe_video_stats(video_path: Path) -> tuple[int, float, float]:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source video for stats: {video_path}")
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
    finally:
        cap.release()

    duration_sec = 0.0
    if total_frames > 0 and fps > 0:
        duration_sec = float(total_frames) / float(fps)
    return total_frames, fps, duration_sec


def should_use_gpu_for_video(
    *,
    base_use_gpu: bool,
    duration_sec: float,
    gpu_processing_clip_max_length_sec: float,
) -> bool:
    if not base_use_gpu:
        return False
    if gpu_processing_clip_max_length_sec <= 0:
        return True
    if duration_sec <= 0:
        return True
    return duration_sec <= gpu_processing_clip_max_length_sec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to the process-stage YAML config")
    parser.add_argument("--input-dir", required=True, help="Directory with source .mp4 files")
    parser.add_argument("--output-dir", required=True, help="Directory for tiered exported segments")
    parser.add_argument(
        "--normalized-dir",
        required=True,
        help="Per-video temporary work root used by the inner dataset_processing pipeline",
    )
    parser.add_argument("--source-archive", default="", help="Source archive name for provenance")
    parser.add_argument("--dataset-kind", choices=["auto", "talkvid", "hdtf"], default="auto")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        config_path, config = load_yaml_config(args.config)
    except ConfigError as exc:
        log(f"[FaceclipExport] config_error={exc}")
        return 2

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    work_root = Path(args.normalized_dir)
    manifest_path = output_dir / "export_manifest.jsonl"
    resume_state_path = output_dir / "export_resume_state.json"
    summary_path = output_dir / "summary.json"

    output_dir.mkdir(parents=True, exist_ok=True)
    work_root.mkdir(parents=True, exist_ok=True)

    base_pipeline_cfg, model_path = build_dataset_processing_config(config_path, config, work_root)
    dataset_kind = resolve_dataset_kind(args.dataset_kind, args.source_archive, input_dir)
    gpu_processing_clip_max_length_sec = float(
        get_mapping(config, "process", default={}).get("gpu_processing_clip_max_length_sec", 30.0)
    )

    counters, total_segments, start_video_index, processed_video_indexes = load_resume_progress(
        manifest_path,
        resume_state_path,
    )
    total_processed_source_frames = 0
    videos = list(iter_videos(input_dir))
    start_video_index = max(0, min(start_video_index, len(videos)))

    log(f"[FaceclipExport] input_dir={input_dir}")
    log(f"[FaceclipExport] output_dir={output_dir}")
    log(f"[FaceclipExport] work_root={work_root}")
    log(f"[FaceclipExport] source_archive={args.source_archive or '<none>'}")
    log(f"[FaceclipExport] dataset_kind={dataset_kind}")
    log(f"[FaceclipExport] model_path={model_path}")
    log(
        "[FaceclipExport] normalization="
        f"fps={base_pipeline_cfg.normalization.fps} "
        f"bitrate={base_pipeline_cfg.normalization.bitrate} "
        f"codec={base_pipeline_cfg.normalization.codec}"
    )
    log(
        "[FaceclipExport] export="
        f"fps={base_pipeline_cfg.export.fps} "
        f"bitrate={base_pipeline_cfg.export.bitrate} "
        f"codec={base_pipeline_cfg.export.codec}"
    )
    log(
        "[FaceclipExport] detection="
        f"num_faces={base_pipeline_cfg.detection.num_faces} "
        f"min_detection_confidence={base_pipeline_cfg.detection.min_detection_confidence} "
        f"min_presence_confidence={base_pipeline_cfg.detection.min_presence_confidence} "
        f"use_gpu={base_pipeline_cfg.detection.use_gpu} "
        f"gpu_processing_clip_max_length_sec={gpu_processing_clip_max_length_sec:g}"
    )
    log(f"[FaceclipExport] videos={len(videos)}")

    if start_video_index > 0 and start_video_index < len(videos):
        log(
            f"[FaceclipExport] resume_from_video_index={start_video_index + 1} "
            f"source_video={videos[start_video_index].name}"
        )
    elif start_video_index >= len(videos) and len(videos) > 0:
        log("[FaceclipExport] resume_at_end=true; no source videos left to export")

    for idx, video_path in enumerate(videos[start_video_index:], start=start_video_index + 1):
        if idx in processed_video_indexes:
            continue

        total_source_frames = 0
        source_fps = 0.0
        duration_sec = 0.0
        try:
            total_source_frames, source_fps, duration_sec = probe_video_stats(video_path)
        except Exception as exc:
            log(
                f"[FaceclipExport] source_video_stats_failed source_video={video_path.name} "
                f"error={type(exc).__name__}: {exc}"
            )

        video_use_gpu = should_use_gpu_for_video(
            base_use_gpu=bool(base_pipeline_cfg.detection.use_gpu),
            duration_sec=duration_sec,
            gpu_processing_clip_max_length_sec=gpu_processing_clip_max_length_sec,
        )
        if base_pipeline_cfg.detection.use_gpu and not video_use_gpu:
            log(
                f"[FaceclipExport] gpu_gate source_video={video_path.name} "
                f"duration_sec={duration_sec:.2f} threshold_sec={gpu_processing_clip_max_length_sec:g} "
                "route=cpu"
            )

        cleanup_video_artifacts(video_path=video_path, batch_output_dir=output_dir, work_root=work_root)
        pipeline_cfg = copy.deepcopy(base_pipeline_cfg)
        pipeline_cfg.detection.use_gpu = video_use_gpu

        try:
            result, segment_entries = process_one_video(
                video_path=video_path,
                dataset_kind=dataset_kind,
                source_archive=args.source_archive,
                batch_output_dir=output_dir,
                work_root=work_root,
                pipeline_cfg=pipeline_cfg,
            )
            total_source_frames = int(result.total_frames)
            release_video_result_memory(result)
        except Exception as exc:
            segment_entries = [
                {
                    "name": video_path.stem,
                    "status": "fail",
                    "tier": None,
                    "message": f"{video_path.stem}: unexpected_fail ({type(exc).__name__}: {exc})",
                    "source_segment_start_frame": 0,
                    "source_segment_end_frame": -1,
                    "segment_index": 1,
                    "segment_count": 1,
                }
            ]
            cleanup_video_artifacts(
                video_path=video_path,
                batch_output_dir=output_dir,
                work_root=work_root,
            )
        finally:
            gc.collect()

        if total_source_frames <= 0 and source_fps > 0 and duration_sec > 0:
            total_source_frames = int(round(source_fps * duration_sec))

        total_segments += len(segment_entries)
        for entry in segment_entries:
            status = entry["status"]
            tier = entry.get("tier")
            counters[status] += 1
            if status == "ok" and tier in QUALITY_TIERS:
                counters[tier] += 1
            append_jsonl(
                manifest_path,
                {
                    "ts": timestamp(),
                    "index": idx,
                    "total": len(videos),
                    "segment_index": entry["segment_index"],
                    "segment_count": entry["segment_count"],
                    "name": entry["name"],
                    "status": status,
                    "tier": tier,
                    "message": entry["message"],
                    "source_video": video_path.name,
                    "source_segment_start_frame": entry["source_segment_start_frame"],
                    "source_segment_end_frame": entry["source_segment_end_frame"],
                },
            )

        video_summary, video_counts = summarize_video_entries(
            source_video=video_path.stem,
            total_videos=len(videos),
            video_index=idx,
            total_source_frames=total_source_frames,
            segment_entries=segment_entries,
        )
        total_processed_source_frames += video_counts["processed_source_frames"]
        log(video_summary)

        write_json(
            resume_state_path,
            {
                "ts": timestamp(),
                "next_video_index": idx,
                "last_completed_index": idx,
                "last_completed_source_video": video_path.name,
            },
        )
        processed_video_indexes.add(idx)

    summary = {
        "ts": timestamp(),
        "processor": "dataset_processing",
        "config_path": str(config_path),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "work_root": str(work_root),
        "source_archive": args.source_archive,
        "dataset_kind": dataset_kind,
        "model_path": str(model_path),
        "normalization_codec": base_pipeline_cfg.normalization.codec,
        "normalization_bitrate": base_pipeline_cfg.normalization.bitrate,
        "export_codec": base_pipeline_cfg.export.codec,
        "export_bitrate": base_pipeline_cfg.export.bitrate,
        "total_videos": len(videos),
        "total_segments": total_segments,
        "processed_source_frames": total_processed_source_frames,
        **counters,
    }
    write_json(summary_path, summary)
    log(f"[FaceclipExport] summary={summary}")
    try:
        resume_state_path.unlink()
    except OSError:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
