#!/usr/bin/env python3
r"""Search and import models from the Objaverse-XL GitHub subset.

The ``github/github.parquet`` catalog of Objaverse-XL (mirrored on ModelScope
at ``allenai/objaverse-xl``, since huggingface.co is unreachable here) indexes
~5.2M 3D files hosted in GitHub repos. Each row carries:

* ``fileIdentifier`` — a ``https://github.com/<owner>/<repo>/blob/<sha>/<path>``
  URL, convertible to a raw.githubusercontent.com download URL (GitHub is
  reachable from this network, no token required);
* ``license`` — the *repository* license (may be ``None`` = all rights
  reserved, excluded); GPL / NonCommercial licenses are rejected, the rest is
  mapped to SPDX ids accepted by the asset-manifest whitelist;
* ``fileType`` — glb / gltf / obj / fbx / stl / ...

There are no semantic names in the catalog — search matches a regex against
file paths, so prefer distinctive nouns (``hairdryer``, ``wardrobe``) and eyeball
candidates before importing.

Subcommands:

* ``search <regex> [--name-only]`` — list license-clean candidates (path match).
* ``fetch <blob-url> --class-name NNN_class`` — download one model and import
  it as an additional class instance with full provenance
  (``license`` / ``source_url`` / ``source_site`` = ``objaverse-xl/github``).

Usage:
    python tools/fetch_objaverse_xl_github.py search "makeup.?organizer"
    python tools/fetch_objaverse_xl_github.py fetch \\
        "https://github.com/o/r/blob/<sha>/models/makeup_organizer.glb" \\
        --class-name 130_cosmetic_organizer
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("fetch_objaverse_xl_github")

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

DEFAULT_PARQUET = "outputs/asset_staging/objaverse_xl/github.parquet"
DEFAULT_CACHE_DIR = "outputs/asset_staging/objaverse_xl/github_glb"
USER_AGENT = "Mozilla/5.0 (compatible; genesis-cloud-sim asset importer)"

IMPORTABLE_TYPES = {"glb", "gltf", "obj", "stl", "ply"}

#: repo license string -> SPDX id accepted by tools/build_asset_manifest.py
_LICENSE_MAP = {
    "MIT License": "MIT",
    "Apache License 2.0": "Apache-2.0",
    'BSD 2-Clause "Simplified" License': "BSD-2-Clause",
    'BSD 3-Clause "New" or "Revised" License': "BSD-3-Clause",
    "Creative Commons Zero v1.0 Universal": "CC0-1.0",
    "Creative Commons - Attribution": "CC-BY-4.0",
    "Creative Commons Attribution 4.0 International": "CC-BY-4.0",
    "Creative Commons Attribution 3.0 Unported": "CC-BY-3.0",
}
_LICENSE_REJECT_RE = re.compile(r"(GPL|Affero|LGPL|Non-?Commercial|NoDerivs)", re.I)

_BLOB_URL_RE = re.compile(r"^https://github\.com/([^/]+)/([^/]+)/blob/([^/]+)/(.+)$")


@dataclass
class CatalogEntry:
    """One license-clean row of the github.parquet catalog."""

    blob_url: str
    license_spdx: str
    file_type: str

    @property
    def raw_url(self) -> str:
        return blob_to_raw_url(self.blob_url)


def map_repo_license(license_str: str | None) -> str | None:
    """Map a GitHub repo license string to an SPDX id, or ``None`` to reject."""
    if license_str is None:
        return None
    if license_str in _LICENSE_MAP:
        return _LICENSE_MAP[license_str]
    if _LICENSE_REJECT_RE.search(license_str):
        return None
    logger.debug("unmapped license (rejected): %s", license_str)
    return None


def blob_to_raw_url(blob_url: str) -> str:
    """Convert a github.com/.../blob/... URL to a raw.githubusercontent URL."""
    m = _BLOB_URL_RE.match(blob_url)
    if not m:
        raise ValueError(f"not a GitHub blob URL: {blob_url}")
    owner, repo, sha, path = m.groups()
    return f"https://raw.githubusercontent.com/{owner}/{repo}/{sha}/{path}"


def search_catalog(
    parquet_path: str | Path, pattern: str, name_only: bool = True
) -> list[CatalogEntry]:
    """Regex-search the catalog; returns license-clean importable entries."""
    import pyarrow.parquet as pq

    rx = re.compile(pattern, re.I)
    table = pq.read_table(
        str(parquet_path), columns=["fileIdentifier", "license", "fileType"]
    )
    cols = table.to_pydict()
    out: list[CatalogEntry] = []
    for url, lic, ftype in zip(
        cols["fileIdentifier"], cols["license"], cols["fileType"]
    ):
        if ftype not in IMPORTABLE_TYPES:
            continue
        spdx = map_repo_license(lic)
        if spdx is None:
            continue
        haystack = url.rsplit("/", 1)[-1] if name_only else url
        if rx.search(haystack):
            out.append(CatalogEntry(blob_url=url, license_spdx=spdx, file_type=ftype))
    return out


def fetch_url(url: str, timeout: float = 60.0) -> bytes:
    """GET ``url`` with a browser-ish user agent."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def cmd_search(args: argparse.Namespace) -> int:
    """List license-clean candidates matching a regex."""
    hits = search_catalog(args.parquet, args.pattern, name_only=args.name_only)
    if not hits:
        print(f"no license-clean candidates for /{args.pattern}/")
        return 0
    print(f"{len(hits)} candidate(s):")
    for h in hits[: args.limit]:
        print(f"  {h.license_spdx:<12} {h.file_type:<5} {h.blob_url}")
    if len(hits) > args.limit:
        print(f"  ... and {len(hits) - args.limit} more (--limit to show)")
    return 0


def cmd_fetch(args: argparse.Namespace) -> int:
    """Download one catalog model and import it as an additional instance."""
    from import_generated_objects import import_meshes

    if not _BLOB_URL_RE.match(args.blob_url):
        logger.error("not a GitHub blob URL: %s", args.blob_url)
        return 1
    spdx = map_repo_license(args.license_)
    if spdx is None:
        logger.error(
            "license %r is not importable (whitelist: %s)",
            args.license_,
            sorted(set(_LICENSE_MAP.values())),
        )
        return 1

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(args.blob_url.split("?")[0]).suffix or ".glb"
    m = _BLOB_URL_RE.match(args.blob_url)
    assert m
    cache_path = cache_dir / f"{m.group(1)}_{m.group(2)}_{m.group(3)[:8]}{suffix}"
    if not cache_path.is_file():
        cache_path.write_bytes(fetch_url(blob_to_raw_url(args.blob_url)))
    print(f"mesh: {cache_path} ({cache_path.stat().st_size} bytes)")

    results = import_meshes(
        [str(cache_path)],
        class_name=args.class_name,
        objects_dir=args.objects_dir,
        target_size=args.target_size,
        max_faces=args.max_faces,
        min_thickness=args.min_thickness,
        provenance={
            "license": spdx,
            "source_url": args.blob_url,
            "author": f"{m.group(1)}/{m.group(2)} (GitHub repo)",
            "source_site": "objaverse-xl/github",
        },
    )
    for r in results:
        warn = f" | warnings: {'; '.join(r.warnings)}" if r.warnings else ""
        print(
            f"[{r.class_name}] instance {r.index}: {r.glb_path} "
            f"({r.faces_in} -> {r.faces_out} faces, extents "
            f"{r.extents[0]:.3f} x {r.extents[1]:.3f} x {r.extents[2]:.3f} m){warn}"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; returns 0 on success, 1 on license/input errors."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_search = sub.add_parser("search", help="regex-search the catalog")
    p_search.add_argument("pattern", help="case-insensitive regex, matched on the file name by default")
    p_search.add_argument("--parquet", default=DEFAULT_PARQUET, help="catalog parquet (default: %(default)s)")
    p_search.add_argument("--limit", type=int, default=20, help="max rows to print (default: %(default)s)")
    p_search.add_argument(
        "--full-path",
        dest="name_only",
        action="store_false",
        help="match the regex against the full URL, not just the file name",
    )
    p_search.set_defaults(func=cmd_search)

    p_fetch = sub.add_parser("fetch", help="download + import one catalog model")
    p_fetch.add_argument("blob_url", help="github blob URL from the search output")
    p_fetch.add_argument(
        "--license",
        dest="license_",
        required=True,
        help="repo license string exactly as printed by search (mapped to SPDX)",
    )
    p_fetch.add_argument("--class-name", required=True, help="target class (NNN_name to append instances)")
    p_fetch.add_argument(
        "--objects-dir",
        default="assets/robotwin/objects/objects",
        help="RoboTwin object library root (default: %(default)s)",
    )
    p_fetch.add_argument(
        "--target-size",
        type=float,
        default=0.15,
        help="largest bounding-box edge after normalization, meters (default: %(default)s)",
    )
    p_fetch.add_argument(
        "--max-faces", type=int, default=20000, help="decimation cap (default: %(default)s)"
    )
    p_fetch.add_argument(
        "--min-thickness",
        type=float,
        default=0.012,
        help="pad any bounding-box axis thinner than this, meters (default: %(default)s)",
    )
    p_fetch.add_argument(
        "--cache-dir",
        default=DEFAULT_CACHE_DIR,
        help="where downloaded meshes are cached (default: %(default)s)",
    )
    p_fetch.set_defaults(func=cmd_fetch)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        return args.func(args)
    except (ValueError, OSError) as exc:
        logger.error("%s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
