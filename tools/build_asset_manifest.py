#!/usr/bin/env python3
"""Build the home-5S asset manifest: per-asset provenance and license audit.

Walks the RoboTwin-OD object library (``assets/robotwin/objects/objects/``)
and emits ``asset_manifest.json`` recording, per class and per instance:

- layout kind (``glb`` collision meshes / ``urdf`` PartNet-Mobility);
- instance count and source metadata (``source`` / ``generator`` /
  ``imported_at`` / ``license`` / ``source_url`` / ``author``) as written by
  ``tools/import_generated_objects.py``;
- the resolved license: upstream-shipped instances (no ``generator`` field)
  fall back to the RoboTwin2.0 MIT license; imported instances missing a
  ``license`` field are flagged ``UNREGISTERED``.

This is the machine-readable counterpart of the per-asset IP registration
promised in the Track-B 移植方案设计书 §2 (逐资产登记来源与许可证).

Usage::

    uv run python tools/build_asset_manifest.py                 # write manifest
    uv run python tools/build_asset_manifest.py --check         # CI gate: fail on UNREGISTERED
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

_CLASS_DIR_RE = re.compile(r"^(\d{3})_(.+)$")
_MODEL_DATA_RE = re.compile(r"model_data(\d+)\.json$")

#: License applied to instances shipped with the upstream RoboTwin2.0 asset
#: repo (``assets/robotwin/objects/README.md`` states MIT License).
UPSTREAM_LICENSE = "MIT (upstream RoboTwin2.0)"

#: License ids accepted by ``--check`` for imported instances, mirroring the
#: compliance statement of the Track-B design doc §2 (Apache-2.0 / CC0 /
#: CC-BY / MIT / BSD / official authorization only; no GPL contamination).
ALLOWED_LICENSES = (
    "Apache-2.0",
    "MIT",
    "BSD-2-Clause",
    "BSD-3-Clause",
    "CC0-1.0",
    "CC-BY-3.0",  # Google Poly archive models on poly.pizza
    "CC-BY-4.0",
    "CC-BY-SA-4.0",
)

UNREGISTERED = "UNREGISTERED"


def class_kind(class_dir: Path) -> str:
    """``glb`` when ``model_data*.json`` files exist, else ``urdf``."""
    if any(_MODEL_DATA_RE.search(p.name) for p in class_dir.glob("model_data*.json")):
        return "glb"
    return "urdf"


def scan_glb_class(class_dir: Path) -> list[dict]:
    """Collect per-instance provenance from ``model_data<N>.json`` files."""
    instances: list[dict] = []
    for path in sorted(class_dir.glob("model_data*.json")):
        m = _MODEL_DATA_RE.search(path.name)
        if not m:
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("unreadable metadata %s: %s", path, exc)
            data = {}
        imported = "generator" in data
        license_ = data.get("license") or (
            UNREGISTERED if imported else UPSTREAM_LICENSE
        )
        instances.append(
            {
                "index": int(m.group(1)),
                "imported": imported,
                "source": data.get("source"),
                "generator": data.get("generator"),
                "imported_at": data.get("imported_at"),
                "license": license_,
                "source_url": data.get("source_url"),
                "author": data.get("author"),
            }
        )
    return instances


def scan_urdf_class(class_dir: Path) -> list[dict]:
    """One entry per PartNet-Mobility instance dir.

    Reads the instance's ``model_data.json`` when present: instances with a
    ``generator`` field (e.g. self-built furniture from
    ``tools/build_p1_assets.py``) are treated as imported and audited for
    license; others fall back to the upstream RoboTwin2.0 MIT license.
    """
    instances = []
    for sub in sorted(class_dir.iterdir()):
        if sub.is_dir() and sub.name.isdigit() and (sub / "mobility.urdf").is_file():
            meta_path = sub / "model_data.json"
            data: dict = {}
            if meta_path.is_file():
                try:
                    data = json.loads(meta_path.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError) as exc:
                    logger.warning("unreadable metadata %s: %s", meta_path, exc)
            imported = "generator" in data
            license_ = data.get("license") or (
                UNREGISTERED if imported else UPSTREAM_LICENSE
            )
            instances.append(
                {
                    "index": int(sub.name),
                    "imported": imported,
                    "source": data.get("source", sub.name),
                    "generator": data.get("generator"),
                    "imported_at": data.get("imported_at"),
                    "license": license_,
                    "source_url": data.get("source_url"),
                    "author": data.get("author"),
                }
            )
    return instances


def build_manifest(objects_dir: str | Path) -> dict:
    """Scan ``objects_dir`` and return the manifest dict."""
    objects_dir = Path(objects_dir)
    classes: list[dict] = []
    warnings: list[str] = []
    for class_dir in sorted(p for p in objects_dir.iterdir() if p.is_dir()):
        if not _CLASS_DIR_RE.match(class_dir.name):
            continue
        kind = class_kind(class_dir)
        instances = (
            scan_glb_class(class_dir) if kind == "glb" else scan_urdf_class(class_dir)
        )
        unregistered = [i for i in instances if i["license"] == UNREGISTERED]
        if unregistered:
            warnings.append(
                f"{class_dir.name}: {len(unregistered)} imported instance(s) "
                f"missing license ({[i['index'] for i in unregistered]})"
            )
        bad = [
            i
            for i in instances
            if i["imported"]
            and i["license"] != UNREGISTERED
            and i["license"] not in ALLOWED_LICENSES
        ]
        if bad:
            warnings.append(
                f"{class_dir.name}: license not in allowlist "
                f"({sorted({i['license'] for i in bad})})"
            )
        classes.append(
            {
                "class": class_dir.name,
                "kind": kind,
                "instances": len(instances),
                "imported_instances": sum(1 for i in instances if i["imported"]),
                "licenses": sorted({i["license"] for i in instances}),
                "instance_records": instances,
            }
        )
    return {
        "version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "objects_dir": str(objects_dir),
        "upstream_license": UPSTREAM_LICENSE,
        "allowed_licenses": list(ALLOWED_LICENSES),
        "totals": {
            "classes": len(classes),
            "instances": sum(c["instances"] for c in classes),
            "imported_instances": sum(c["imported_instances"] for c in classes),
            "unregistered_instances": sum(
                1
                for c in classes
                for i in c["instance_records"]
                if i["license"] == UNREGISTERED
            ),
            "disallowed_instances": sum(
                1
                for c in classes
                for i in c["instance_records"]
                if i["imported"]
                and i["license"] != UNREGISTERED
                and i["license"] not in ALLOWED_LICENSES
            ),
        },
        "warnings": warnings,
        "classes": classes,
    }


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; ``--check`` returns 1 when licenses are missing."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--objects-dir",
        default="assets/robotwin/objects/objects",
        help="RoboTwin object library root (default: %(default)s)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="manifest output path (default: <objects-dir>/../asset_manifest.json)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit 1 if any imported instance is UNREGISTERED (CI license gate)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    manifest = build_manifest(args.objects_dir)
    out = (
        Path(args.out)
        if args.out
        else Path(args.objects_dir).parent / "asset_manifest.json"
    )
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    totals = manifest["totals"]
    print(
        f"manifest -> {out} | classes={totals['classes']} "
        f"instances={totals['instances']} imported={totals['imported_instances']} "
        f"unregistered={totals['unregistered_instances']}"
    )
    for warning in manifest["warnings"]:
        logger.warning("%s", warning)
    if args.check and totals["unregistered_instances"]:
        logger.error("license gate failed: unregistered imported assets present")
        return 1
    if args.check and totals["disallowed_instances"]:
        logger.error(
            "license gate failed: imported assets with non-allowlist licenses present"
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
