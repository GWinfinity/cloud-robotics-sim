"""CLI for the 3D asset generation API clients.

Usage::

    python -m cloud_robotics_sim.asset_gen providers
    python -m cloud_robotics_sim.asset_gen text "a wooden dining chair" \
        --provider tripo --out outputs/asset_staging/gen3d
    python -m cloud_robotics_sim.asset_gen image chair.png --provider meshy \
        --no-texture --topology quad

API keys are read from the environment (``python-dotenv`` is loaded when
installed, so a project ``.env`` file works too).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def _load_dotenv() -> None:
    """Load a local .env file when python-dotenv is installed."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m cloud_robotics_sim.asset_gen",
        description="Generate 3D assets via Tripo / Meshy / Hunyuan3D / Rodin APIs.",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable debug logging"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("providers", help="List providers and credential status")

    for name, help_text in (
        ("text", "Text-to-3D generation"),
        ("image", "Image-to-3D generation"),
    ):
        p = sub.add_parser(name, help=help_text)
        p.add_argument(
            "input", help='Text prompt (for "text") or image path (for "image")'
        )
        p.add_argument(
            "--provider",
            default=None,
            help="Provider name (default: first configured)",
        )
        p.add_argument(
            "--out",
            default="outputs/asset_staging/gen3d",
            help="Output directory for the .glb + provenance .json",
        )
        p.add_argument(
            "--no-texture",
            action="store_true",
            help="Generate geometry only (no PBR textures)",
        )
        p.add_argument(
            "--topology",
            choices=["quad", "triangle"],
            default=None,
            help="Preferred mesh topology (provider support varies)",
        )
        p.add_argument(
            "--negative-prompt", default=None, help="Things to avoid in the output"
        )
        p.add_argument(
            "--timeout",
            type=float,
            default=600.0,
            help="Total polling timeout in seconds (default: 600)",
        )
        p.add_argument(
            "--interval",
            type=float,
            default=5.0,
            help="Initial polling interval in seconds (default: 5)",
        )
    return parser


def _cmd_providers() -> int:
    from cloud_robotics_sim.asset_gen import (  # noqa: PLC0415
        get_provider,
        list_providers,
    )

    names = list_providers()
    if not names:
        print("No providers registered.")
        return 0
    for name in names:
        provider = get_provider(name)
        status = "configured" if provider.configured() else "missing credentials"
        envs = ", ".join(provider.env_vars)
        print(f"{name:12s} {status:20s} ({envs})")
    return 0


def _cmd_generate(args: argparse.Namespace) -> int:
    from cloud_robotics_sim.asset_gen import AssetGenClient  # noqa: PLC0415

    client = AssetGenClient(
        provider=args.provider,
        poll_timeout=args.timeout,
        poll_interval=args.interval,
    )
    kwargs = {
        "texture": not args.no_texture,
        "topology": args.topology,
        "negative_prompt": args.negative_prompt,
    }
    if args.command == "text":
        result = client.generate_text(args.input, out_dir=args.out, **kwargs)
    else:
        result = client.generate_image(args.input, out_dir=args.out, **kwargs)
    print(f"model:     {result.model_file}")
    print(f"provenance: {Path(result.model_file).with_suffix('.json')}")
    print(f"task_id:   {result.task_id} ({result.provider})")
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    _load_dotenv()
    parser = _build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO, format="%(message)s"
    )
    if args.command == "providers":
        return _cmd_providers()
    return _cmd_generate(args)


if __name__ == "__main__":
    sys.exit(main())
