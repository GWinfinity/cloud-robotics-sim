#!/usr/bin/env python3
"""End-to-end object-library expansion workflow.

One command takes externally generated 3D meshes (TRELLIS.2, Objaverse, ...)
all the way to verified, grasp-ready RoboTwin library classes::

    <staging_dir>/
        bottle-gen/          # bare name -> new class, auto id
            style_a.glb
            style_b.glb
        001_bottle/          # NNN_name -> append styles to an existing class
            extra.glb
        loose.glb            # loose files: only with --class-name

Stages:

1. **import** — clean, normalize and convert every mesh into the RoboTwin
   GLB layout via ``tools/import_generated_objects.py``. Instance numbering
   appends after existing instances of the class.
2. **smoke** — Genesis spawn smoke test per class in a fresh subprocess
   (VRAM isolation): each instance must settle at a finite, plausible
   height. Skip with ``--skip-smoke``.
3. **grasp** — optional (``--grasp``): run ``grasp_all_objects.py`` on the
   new classes only, optionally recording demonstration episodes
   (``--record``). Chunked subprocess mode is reused for VRAM safety.
4. **report** — ``<out>/report.json`` + ``<out>/report.md`` summarizing
   every class across all stages.

Usage:
    # import + spawn verification
    python scripts/expand_object_library.py data/asset_staging

    # full pipeline including grasp validation with recorded episodes
    python scripts/expand_object_library.py data/asset_staging \
        --grasp --record --episodes-per-class 5 --jitter-xy 0.05 --chunk-size 12
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tools.import_generated_objects import (  # noqa: E402
    MESH_SUFFIXES,
    import_meshes,
)

logger = logging.getLogger("expand_object_library")

SMOKE_SCRIPT = REPO_ROOT / "scripts" / "smoke_imported_objects.py"
GRASP_SCRIPT = REPO_ROOT / "examples" / "grasp" / "grasp_all_objects.py"

_SMOKE_LINE_RE = re.compile(r"instance (\d+): z=(\S+) finite=(\w+) -> (OK|FAIL)")


@dataclass
class StagingClass:
    """One class worth of staged meshes awaiting import."""

    label: str  # staging subdir name (bare or NNN_name)
    files: list[Path] = field(default_factory=list)


def discover_staging(
    staging_dir: str | Path, class_name: str | None = None
) -> list[StagingClass]:
    """Map a staging directory to per-class mesh file lists.

    Each subdirectory becomes one class (its name is the class label).
    Loose mesh files at the top level are grouped under ``class_name``
    (required when loose files are present).
    """
    staging_dir = Path(staging_dir)
    if not staging_dir.is_dir():
        raise FileNotFoundError(f"staging dir not found: {staging_dir}")
    classes: list[StagingClass] = []
    loose: list[Path] = []
    for entry in sorted(staging_dir.iterdir()):
        if entry.is_dir():
            files = sorted(
                p for p in entry.iterdir() if p.suffix.lower() in MESH_SUFFIXES
            )
            if files:
                classes.append(StagingClass(label=entry.name, files=files))
            else:
                logger.warning("staging subdir has no meshes, skipped: %s", entry)
        elif entry.suffix.lower() in MESH_SUFFIXES:
            loose.append(entry)
    if loose:
        if not class_name:
            raise ValueError(
                f"{len(loose)} loose mesh file(s) in {staging_dir}; "
                "pass --class-name to group them into a class"
            )
        classes.append(StagingClass(label=class_name, files=sorted(loose)))
    if not classes:
        raise ValueError(f"no mesh files found under {staging_dir}")
    return classes


def parse_smoke_output(text: str) -> dict[int, dict[str, object]]:
    """Parse ``smoke_imported_objects.py`` stdout into per-instance results."""
    results: dict[int, dict[str, object]] = {}
    for m in _SMOKE_LINE_RE.finditer(text):
        try:
            z = float(m.group(2))
        except ValueError:
            z = float("nan")
        results[int(m.group(1))] = {
            "z": z,
            "finite": m.group(3) == "True",
            "ok": m.group(4) == "OK",
        }
    return results


def run_import_stage(
    staging: list[StagingClass],
    objects_dir: str | Path,
    target_size: float = 0.15,
    max_faces: int = 20000,
    z_up: bool = False,
) -> dict[str, dict[str, object]]:
    """Import every staged class; returns ``{label: import-info}``.

    Import failures are recorded per class (``status: "error"``) and do not
    abort the remaining classes.
    """
    report: dict[str, dict[str, object]] = {}
    for staged in staging:
        try:
            results = import_meshes(
                [str(f) for f in staged.files],
                class_name=staged.label,
                objects_dir=objects_dir,
                target_size=target_size,
                max_faces=max_faces,
                z_up=z_up,
            )
        except (FileNotFoundError, ValueError) as exc:
            logger.error("import failed for %s: %s", staged.label, exc)
            report[staged.label] = {"status": "error", "message": str(exc)}
            continue
        report[staged.label] = {
            "status": "ok",
            "class_name": results[0].class_name,
            "instances": [r.index for r in results],
            "faces": [{"in": r.faces_in, "out": r.faces_out} for r in results],
            "extents": [list(r.extents) for r in results],
            "warnings": [w for r in results for w in r.warnings],
        }
        logger.info(
            "imported %s -> %s (%d instances)",
            staged.label,
            results[0].class_name,
            len(results),
        )
    return report


def run_smoke_stage(
    class_names: list[str],
    objects_dir: str | Path,
    out_dir: Path,
    timeout: float = 3600.0,
    steps: int = 300,
) -> dict[str, dict[str, object]]:
    """Genesis spawn smoke per class in a fresh subprocess (VRAM isolation)."""
    log_dir = out_dir / "smoke_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, dict[str, object]] = {}
    for name in class_names:
        cmd = [
            sys.executable,
            str(SMOKE_SCRIPT),
            "--objects-dir",
            str(objects_dir),
            "--class-name",
            name,
            "--steps",
            str(steps),
        ]
        logger.info("smoke: %s ...", name)
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout, check=False
            )
        except subprocess.TimeoutExpired:
            report[name] = {"status": "timeout"}
            logger.error("smoke: %s timed out after %.0f s", name, timeout)
            continue
        (log_dir / f"{name}.log").write_text(
            proc.stdout + proc.stderr, encoding="utf-8"
        )
        instances = parse_smoke_output(proc.stdout)
        ok = (
            proc.returncode == 0
            and bool(instances)
            and all(v["ok"] for v in instances.values())
        )
        report[name] = {
            "status": "ok" if ok else "fail",
            "exit_code": proc.returncode,
            "instances": instances,
        }
        logger.info("smoke: %s -> %s", name, "OK" if ok else "FAIL")
    return report


def run_grasp_stage(
    class_names: list[str],
    objects_dir: str | Path,
    out_dir: Path,
    planner: str = "auto",
    record: bool = False,
    episodes_per_class: int = 1,
    jitter_xy: float = 0.0,
    chunk_size: int = 12,
    seed: int = 0,
) -> dict[str, dict[str, object]]:
    """Grasp validation via ``grasp_all_objects.py`` (subprocess)."""
    grasp_out = out_dir / "grasp"
    grasp_out.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(GRASP_SCRIPT),
        "--objects-dir",
        str(objects_dir),
        "--out",
        str(grasp_out),
        "--planner",
        planner,
        "--chunk-size",
        str(chunk_size),
        "--seed",
        str(seed),
        "--classes",
        *class_names,
    ]
    if record:
        cmd += [
            "--record",
            "--episodes-per-class",
            str(episodes_per_class),
            "--jitter-xy",
            str(jitter_xy),
        ]
    logger.info("grasp: %d classes -> %s", len(class_names), grasp_out)
    ret = subprocess.call(cmd)
    report: dict[str, dict[str, object]] = {}
    report_path = grasp_out / "report.json"
    if report_path.is_file():
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        for raw in payload.get("records", []):
            report[raw["class_name"]] = {
                "status": raw.get("status", "unknown"),
                "planner": raw.get("planner"),
                "message": raw.get("message", ""),
            }
    for name in class_names:
        report.setdefault(
            name,
            {"status": "no_report", "planner": None, "message": f"driver exit={ret}"},
        )
    return report


def write_report(report: dict[str, object], out_dir: Path) -> None:
    """Write ``report.json`` and a human-readable ``report.md``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    classes: list[dict[str, object]] = report["classes"]  # type: ignore[assignment]
    summary: dict[str, object] = report["summary"]  # type: ignore[assignment]
    lines = [
        "# 资产扩充工作流报告",
        "",
        f"- 时间: {report['created_at']}",
        f"- staging: `{report['staging_dir']}`",
        f"- 资产库: `{report['objects_dir']}`",
        f"- 类数: {summary['classes']}, 导入实例: {summary['imported_instances']}, "
        f"冒烟通过: {summary['smoke_ok']}, 抓取成功: {summary['grasp_success']}",
        "",
        "| staging 类 | 库类名 | 实例数 | 冒烟 | 抓取 |",
        "|---|---|---|---|---|",
    ]
    for c in classes:
        imp: dict[str, object] = c["import"]  # type: ignore[assignment]
        smoke: dict[str, object] = c.get("smoke", {})  # type: ignore[assignment]
        grasp: dict[str, object] = c.get("grasp", {})  # type: ignore[assignment]
        n_inst = len(imp.get("instances", [])) if imp.get("status") == "ok" else 0
        lines.append(
            f"| {c['label']} | {imp.get('class_name', '-')} | {n_inst} "
            f"| {smoke.get('status', '-')} | {grasp.get('status', '-')} |"
        )
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_workflow(args: argparse.Namespace) -> dict[str, object]:
    """Run all enabled stages and write the combined report."""
    staging = discover_staging(args.staging_dir, class_name=args.class_name)
    import_report = run_import_stage(
        staging,
        args.objects_dir,
        target_size=args.target_size,
        max_faces=args.max_faces,
        z_up=args.z_up,
    )
    class_names = [
        str(info["class_name"])
        for info in import_report.values()
        if info.get("status") == "ok"
    ]

    smoke_report: dict[str, dict[str, object]] = {}
    if class_names and not args.skip_smoke:
        smoke_report = run_smoke_stage(
            class_names, args.objects_dir, args.out, timeout=args.smoke_timeout
        )

    grasp_report: dict[str, dict[str, object]] = {}
    grasp_classes = [
        n
        for n in class_names
        if not smoke_report or smoke_report.get(n, {}).get("status") == "ok"
    ]
    if args.grasp and grasp_classes:
        grasp_report = run_grasp_stage(
            grasp_classes,
            args.objects_dir,
            args.out,
            planner=args.planner,
            record=args.record,
            episodes_per_class=args.episodes_per_class,
            jitter_xy=args.jitter_xy,
            chunk_size=args.chunk_size,
            seed=args.seed,
        )

    classes = []
    for staged in staging:
        imp = import_report[staged.label]
        name = str(imp.get("class_name", ""))
        classes.append(
            {
                "label": staged.label,
                "import": imp,
                "smoke": smoke_report.get(name, {"status": "skipped"}),
                "grasp": grasp_report.get(name, {"status": "skipped"}),
            }
        )
    summary = {
        "classes": len(classes),
        "imported_instances": sum(
            len(c["import"].get("instances", []))  # type: ignore[union-attr]
            for c in classes
            if c["import"].get("status") == "ok"  # type: ignore[union-attr]
        ),
        "smoke_ok": sum(1 for c in classes if c["smoke"].get("status") == "ok"),  # type: ignore[union-attr]
        "grasp_success": sum(
            1 for c in classes if c["grasp"].get("status") == "success"  # type: ignore[union-attr]
        ),
    }
    report: dict[str, object] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "staging_dir": str(args.staging_dir),
        "objects_dir": str(args.objects_dir),
        "classes": classes,
        "summary": summary,
    }
    write_report(report, args.out)
    logger.info(
        "done: %d classes, %d instances, smoke_ok=%d, grasp_success=%d -> %s",
        summary["classes"],
        summary["imported_instances"],
        summary["smoke_ok"],
        summary["grasp_success"],
        args.out,
    )
    return report


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; returns 0 on success, 1 on staging/input errors."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "staging_dir", type=Path, help="staged meshes (see module docstring)"
    )
    parser.add_argument(
        "--objects-dir",
        type=Path,
        default=REPO_ROOT / "assets" / "robotwin" / "objects" / "objects",
        help="target RoboTwin object library (default: %(default)s)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "outputs" / "asset_expansion",
        help="report/log output dir (default: %(default)s)",
    )
    parser.add_argument(
        "--class-name",
        default=None,
        help="class label for loose mesh files in the staging dir",
    )
    parser.add_argument("--target-size", type=float, default=0.15)
    parser.add_argument("--max-faces", type=int, default=20000)
    parser.add_argument("--z-up", action="store_true", help="inputs are Z-up")
    parser.add_argument(
        "--skip-smoke", action="store_true", help="skip Genesis spawn smoke"
    )
    parser.add_argument(
        "--smoke-timeout",
        type=float,
        default=3600.0,
        help="per-class smoke timeout (s)",
    )
    parser.add_argument(
        "--grasp", action="store_true", help="run grasp validation stage"
    )
    parser.add_argument("--planner", choices=["auto", "curobo", "ompl"], default="auto")
    parser.add_argument("--record", action="store_true", help="record grasp episodes")
    parser.add_argument("--episodes-per-class", type=int, default=1)
    parser.add_argument("--jitter-xy", type=float, default=0.0)
    parser.add_argument("--chunk-size", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    try:
        run_workflow(args)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
