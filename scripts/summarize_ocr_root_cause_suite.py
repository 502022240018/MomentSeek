#!/usr/bin/env python3
"""Summarize the isolated OCR root-cause matrix without hiding failed cases."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


CASES = ("semantic_only", "ocr_only", "ocr_semantic", "face_ocr_semantic")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"success": False, "error": f"report unreadable: {type(exc).__name__}: {exc}"}


def _monitor_max(path: Path) -> dict[str, int | float]:
    result: dict[str, int | float] = {
        "processes": 0,
        "threads": 0,
        "rss_mb": 0.0,
        "forkserver_processes": 0,
    }
    if not path.is_file():
        return result
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            value = json.loads(line).get("tree", {})
        except json.JSONDecodeError:
            continue
        for key in ("processes", "threads", "rss_mb"):
            result[key] = max(result[key], value.get(key, 0) or 0)
        commands = value.get("commands", {}) or {}
        result["forkserver_processes"] = max(
            result["forkserver_processes"], commands.get("python:forkserver", 0) or 0
        )
    return result


def _host_monitor_max(path: Path) -> dict[str, int]:
    result = {"container_pids": 0, "probe_threads": 0}
    if not path.is_file():
        return result
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        for token in line.split():
            if token.startswith("pids="):
                try:
                    result["container_pids"] = max(result["container_pids"], int(token.split("=", 1)[1]))
                except ValueError:
                    pass
        if line.startswith("Threads:"):
            try:
                result["probe_threads"] = max(result["probe_threads"], int(line.split(":", 1)[1].strip()))
            except ValueError:
                pass
    return result


def _diagnosis(cases: dict[str, dict[str, Any]]) -> list[str]:
    passed = {name: bool(value.get("success")) for name, value in cases.items()}
    lines: list[str] = []
    if set(cases) != set(CASES):
        failed = [name for name, success in passed.items() if not success]
        if failed:
            return [f"Selected probe(s) failed: {', '.join(failed)}."]
        return ["All selected probes passed; run the full four-case matrix before assigning a cross-stage cause."]
    if not passed["semantic_only"]:
        lines.append("A semantic_only failed: pure CPU semantic/OpenBLAS is independently unhealthy.")
    if not passed["ocr_only"]:
        lines.append("B ocr_only failed: investigate RapidOCR ACL, frame loop, shapes, or ACL resource lifecycle first.")
    if passed["semantic_only"] and passed["ocr_only"] and not passed["ocr_semantic"]:
        lines.append("A and B passed but C failed: ACL OCR and CPU semantic inference conflict in one process.")
    if passed["ocr_semantic"] and not passed["face_ocr_semantic"]:
        lines.append("C passed but D failed: Face CANN/TBE state is the differentiating trigger.")
    if all(passed.values()):
        lines.append("All short probes passed: the failure needs a longer scale/soak reproduction; no root cause is proven yet.")
    if not lines:
        lines.append("Multiple probes failed; inspect the earliest failing case and its phase/monitor logs before assigning one cause.")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite-dir", type=Path, required=True)
    parser.add_argument("--cases", default=",".join(CASES))
    args = parser.parse_args()
    selected_cases = tuple(name.strip() for name in args.cases.split(",") if name.strip())
    unsupported = [name for name in selected_cases if name not in CASES]
    if not selected_cases or unsupported:
        parser.error(f"unsupported or empty --cases value: {args.cases}")

    case_reports: dict[str, dict[str, Any]] = {}
    rows = []
    for name in selected_cases:
        case_dir = args.suite_dir / "cases" / name
        report = _read_json(case_dir / "report.json")
        log = (case_dir / "case.log").read_text(encoding="utf-8", errors="replace") if (case_dir / "case.log").is_file() else ""
        status = (case_dir / "status.txt").read_text(encoding="utf-8", errors="replace") if (case_dir / "status.txt").is_file() else ""
        maximum = _monitor_max(case_dir / "process-monitor.jsonl")
        host_maximum = _host_monitor_max(case_dir / "host-monitor.log")
        value = {
            "success": bool(report.get("success")) and "exit_code=0" in status,
            "error": report.get("error"),
            "max_processes": maximum["processes"],
            "max_threads": maximum["threads"],
            "max_rss_mb": maximum["rss_mb"],
            "max_forkserver_processes": maximum["forkserver_processes"],
            "max_container_pids": host_maximum["container_pids"],
            "openblas_warning": "OpenBLAS warning" in log,
            "bad_memory_unallocation": "Bad memory unallocation" in log,
            "tbe_trace": "tbe/common/repository_manager" in log or "multiprocess_util.py" in log,
            "timeout": "timed_out=1" in status,
            "phases": report.get("phases", []),
        }
        case_reports[name] = value
        rows.append({"case": name, **value})

    diagnosis = _diagnosis(case_reports)
    summary = {
        "schema_version": 1,
        "completed": True,
        "cases": rows,
        "diagnosis": diagnosis,
    }
    (args.suite_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = ["MomentSeek OCR root-cause suite", ""]
    for row in rows:
        lines.append(
            f"[{'PASS' if row['success'] else 'FAIL'}] {row['case']} "
            f"processes={row['max_processes']} threads={row['max_threads']} "
            f"container_pids={row['max_container_pids']} forkservers={row['max_forkserver_processes']} "
            f"rss_mb={row['max_rss_mb']} openblas={int(row['openblas_warning'])} "
            f"bad_free={int(row['bad_memory_unallocation'])} tbe={int(row['tbe_trace'])} "
            f"timeout={int(row['timeout'])}"
        )
        if row.get("error"):
            lines.append(f"  error={row['error']}")
    lines.extend(["", "Diagnosis:"] + [f"- {value}" for value in diagnosis])
    text = "\n".join(lines) + "\n"
    (args.suite_dir / "summary.txt").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
