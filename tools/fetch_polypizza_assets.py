#!/usr/bin/env python3
"""Search and import CC0 / CC-BY models from Poly Pizza into the object library.

Poly Pizza (https://poly.pizza) hosts low-poly 3D models, each labelled with
a license on its model page (Public Domain CC0, CC-BY, CC-BY-NC, ...). This
tool is the U1 rigid-asset upgrade route (replacing the unreachable
Objaverse/HuggingFace pipeline) and provides two subcommands:

* ``search <query>`` — list candidate models with title / triangle count /
  license / author, so a human can pick suitable ones.
* ``fetch <model-url> --class-name NNN_class`` — download the model's GLB,
  verify its license is importable (CC0-1.0 or CC-BY-4.0 only), and import it
  as an *additional instance* of an existing class via
  ``tools/import_generated_objects.py`` with full provenance
  (``license`` / ``source_url`` / ``author``) recorded in ``model_data<N>.json``.

License mapping follows the creativecommons.org link on each model page:

* ``publicdomain/zero/1.0``  -> ``CC0-1.0``     (importable)
* ``licenses/by/4.0`` / ``by/3.0`` -> ``CC-BY-4.0`` / ``CC-BY-3.0`` (importable, author recorded)
* ``licenses/by-sa/...``     -> share-alike     (rejected)
* ``licenses/by-nc*/...``    -> non-commercial  (rejected)

Only licenses on the asset-manifest whitelist (CC0-1.0 / CC-BY-4.0) are
importable; anything else aborts before download.

Usage:
    python tools/fetch_polypizza_assets.py search "hair dryer"
    python tools/fetch_polypizza_assets.py fetch https://poly.pizza/m/B5nWfdzHzO \
        --class-name 148_hairdryer_set
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("fetch_polypizza_assets")

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

SEARCH_URL = "https://poly.pizza/search/{query}"
STATIC_GLB_TEMPLATE = "https://static.poly.pizza/{uuid}.glb"
USER_AGENT = "Mozilla/5.0 (compatible; genesis-cloud-sim asset importer)"
DEFAULT_CACHE_DIR = "outputs/asset_staging/polypizza"

# creativecommons.org path -> SPDX id
_LICENSE_PATH_TO_SPDX = {
    "publicdomain/zero/1.0": "CC0-1.0",
    "licenses/by/4.0": "CC-BY-4.0",
    "licenses/by/3.0": "CC-BY-3.0",  # Google Poly archive models
    "licenses/by-sa/4.0": "CC-BY-SA-4.0",
    "licenses/by-nc/4.0": "CC-BY-NC-4.0",
    "licenses/by-nc/3.0": "CC-BY-NC-3.0",
    "licenses/by-nc-sa/4.0": "CC-BY-NC-SA-4.0",
}
# licenses allowed by tools/build_asset_manifest.py whitelist
IMPORTABLE_LICENSES = {"CC0-1.0", "CC-BY-4.0", "CC-BY-3.0"}

_SEARCH_CARD_RE = re.compile(
    r'href="/m/([A-Za-z0-9]+)"><div class="MuiCardMedia-root" title="([^"]+)"'
)
_LICENSE_URL_RE = re.compile(
    r"creativecommons\.org/(publicdomain/zero/1\.0|licenses/[a-z-]+/[34]\.0)"
)
_GLB_UUID_RE = re.compile(
    r"static\.poly\.pizza/([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})\.glb"
)
_MODEL_URL_RE = re.compile(r"^(?:https://poly\.pizza)?/m/([A-Za-z0-9]+)/?$")


@dataclass
class ModelInfo:
    """Metadata scraped from one Poly Pizza model page."""

    public_id: str
    page_url: str
    title: str
    author: str
    tris: int
    license_spdx: str
    license_url: str
    glb_url: str


def parse_search_results(html: str) -> list[tuple[str, str]]:
    """Extract ``(public_id, title)`` candidates from a search page, deduped."""
    seen: set[str] = set()
    out: list[tuple[str, str]] = []
    for public_id, title in _SEARCH_CARD_RE.findall(html):
        if public_id not in seen:
            seen.add(public_id)
            out.append((public_id, title))
    return out


def map_license_url(cc_path: str) -> str:
    """Map a creativecommons.org path to an SPDX id (raises on unknown)."""
    if cc_path not in _LICENSE_PATH_TO_SPDX:
        raise ValueError(f"unrecognized creativecommons path: {cc_path}")
    return _LICENSE_PATH_TO_SPDX[cc_path]


def parse_model_page(html: str, page_url: str) -> ModelInfo:
    """Extract model metadata from a Poly Pizza model page's HTML.

    The page embeds a plain JSON blob with ``PublicID`` / ``Title`` / ``Tris``
    / ``Creator.Username``, plus a creativecommons.org license link and a
    ``static.poly.pizza/<uuid>.glb`` download URL.
    """
    m = _MODEL_URL_RE.search(page_url)
    if not m:
        raise ValueError(f"not a Poly Pizza model URL: {page_url}")
    public_id = m.group(1)

    anchor = html.find(f'"PublicID":"{public_id}"')
    if anchor < 0:
        raise ValueError(f"model JSON blob for {public_id} not found in page")
    # fields of the same JSON object live around the PublicID anchor
    window = html[max(0, anchor - 2000) : anchor + 4000]

    mt = re.search(r'"Title":"((?:[^"\\]|\\.)*)"', window)
    ma = re.search(r'"Creator":\{"Username":"((?:[^"\\]|\\.)*)"', window)
    mtr = re.search(r'"Tris":(\d+)', window)
    ml = _LICENSE_URL_RE.search(html)
    mg = _GLB_UUID_RE.search(html)
    missing = [
        name
        for name, mm in [
            ("Title", mt),
            ("Creator", ma),
            ("license", ml),
            ("glb url", mg),
        ]
        if mm is None
    ]
    if missing:
        raise ValueError(f"model page missing fields: {', '.join(missing)}")
    assert mt and ma and ml and mg  # for type checkers

    return ModelInfo(
        public_id=public_id,
        page_url=f"https://poly.pizza/m/{public_id}",
        title=mt.group(1),
        author=ma.group(1),
        tris=int(mtr.group(1)) if mtr else -1,
        license_spdx=map_license_url(ml.group(1)),
        license_url=f"https://creativecommons.org/{ml.group(1)}",
        glb_url=STATIC_GLB_TEMPLATE.format(uuid=mg.group(1)),
    )


def fetch_url(url: str, timeout: float = 30.0) -> bytes:
    """GET ``url`` with a browser-ish user agent."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def cmd_search(args: argparse.Namespace) -> int:
    """List candidate models for a query with license / author info."""
    from urllib.parse import quote

    html = fetch_url(SEARCH_URL.format(query=quote(args.query))).decode(
        "utf-8", "replace"
    )
    candidates = parse_search_results(html)[: args.limit]
    if not candidates:
        print(f"no results for query: {args.query!r}")
        return 0
    print(
        f"{'public_id':<14} {'tris':>7}  {'license':<12} {'ok':<3} {'title':<30} author"
    )
    for public_id, title in candidates:
        page_url = f"https://poly.pizza/m/{public_id}"
        try:
            info = parse_model_page(
                fetch_url(page_url).decode("utf-8", "replace"), page_url
            )
            ok = "✓" if info.license_spdx in IMPORTABLE_LICENSES else "✗"
            print(
                f"{info.public_id:<14} {info.tris:>7}  {info.license_spdx:<12} {ok:<3} "
                f"{info.title[:30]:<30} {info.author}"
            )
        except (ValueError, OSError) as exc:
            print(f"{public_id:<14} {'?':>7}  {'?':<12} ?  {title[:30]:<30} ({exc})")
    return 0


def cmd_fetch(args: argparse.Namespace) -> int:
    """Download one model and import it as an additional class instance."""
    from import_generated_objects import import_meshes

    page_url = args.model_url
    if not page_url.startswith("http"):
        page_url = f"https://poly.pizza/m/{page_url}"
    info = parse_model_page(fetch_url(page_url).decode("utf-8", "replace"), page_url)
    print(
        f"model: {info.title!r} by {info.author} | {info.tris} tris | "
        f"license {info.license_spdx} ({info.license_url})"
    )
    if info.license_spdx not in IMPORTABLE_LICENSES:
        logger.error(
            "license %s is not importable (whitelist: %s); skipping",
            info.license_spdx,
            sorted(IMPORTABLE_LICENSES),
        )
        return 1

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    glb_path = cache_dir / f"{info.public_id}.glb"
    if not glb_path.is_file():
        payload = fetch_url(info.glb_url)
        if not payload.startswith(b"glTF"):
            logger.error("downloaded file is not a GLB: %s", info.glb_url)
            return 1
        glb_path.write_bytes(payload)
    print(f"glb: {glb_path} ({glb_path.stat().st_size} bytes)")

    results = import_meshes(
        [str(glb_path)],
        class_name=args.class_name,
        objects_dir=args.objects_dir,
        target_size=args.target_size,
        max_faces=args.max_faces,
        min_thickness=args.min_thickness,
        provenance={
            "license": info.license_spdx,
            "source_url": info.page_url,
            "author": info.author,
            "source_site": "poly.pizza",
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

    p_search = sub.add_parser("search", help="list candidate models for a query")
    p_search.add_argument("query", help="search terms, e.g. 'hair dryer'")
    p_search.add_argument(
        "--limit", type=int, default=10, help="max candidates (default: %(default)s)"
    )
    p_search.set_defaults(func=cmd_search)

    p_fetch = sub.add_parser("fetch", help="download + import one model")
    p_fetch.add_argument(
        "model_url",
        help="model page URL or public id, e.g. https://poly.pizza/m/B5nWfdzHzO",
    )
    p_fetch.add_argument(
        "--class-name",
        required=True,
        help="target class (NNN_name to append instances)",
    )
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
        "--max-faces",
        type=int,
        default=20000,
        help="decimation cap (default: %(default)s)",
    )
    p_fetch.add_argument(
        "--min-thickness",
        type=float,
        default=0.012,
        help="pad any bounding-box axis thinner than this, meters; thin low-poly "
        "models tunnel through the floor otherwise (default: %(default)s)",
    )
    p_fetch.add_argument(
        "--cache-dir",
        default=DEFAULT_CACHE_DIR,
        help="where downloaded GLBs are cached (default: %(default)s)",
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
