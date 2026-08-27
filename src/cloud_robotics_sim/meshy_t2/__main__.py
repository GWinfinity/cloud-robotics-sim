"""CLI for the Meshy T2 reproduction.

Usage:
    uv run python -m cloud_robotics_sim.meshy_t2 train mesh-vae --tiny ...
    uv run python -m cloud_robotics_sim.meshy_t2 generate image.png \
        --checkpoints outputs/meshy_t2 --num-faces 4000 --out out.obj
    uv run python -m cloud_robotics_sim.meshy_t2 retopo dense.obj --out compact.obj
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
import trimesh
from PIL import Image

from .models.image_encoder import TinyViTImageEncoder
from .models.mesh_flow import MeshFlow, MeshFlowConfig
from .models.mesh_vae import MeshVAE, MeshVAEConfig
from .models.voxel_flow import VoxelFlow, VoxelFlowConfig
from .models.voxel_vae import VoxelVAE, VoxelVAEConfig
from .pipeline import MeshyT2Pipeline
from .train import main as train_main

logger = logging.getLogger(__name__)


def _build_pipeline(args: argparse.Namespace) -> MeshyT2Pipeline:
    device = args.device
    if args.tiny:
        mcfg, vcfg = MeshVAEConfig.tiny(), VoxelVAEConfig.tiny()
        fcfg, gcfg = VoxelFlowConfig.tiny(), MeshFlowConfig.tiny()
    else:
        mcfg, vcfg = MeshVAEConfig(), VoxelVAEConfig()
        fcfg, gcfg = VoxelFlowConfig(), MeshFlowConfig()
    fcfg.latent_channels, fcfg.latent_res = vcfg.latent_channels, vcfg.latent_res
    gcfg.latent_channels = mcfg.latent_channels
    gcfg.voxel_channels, gcfg.voxel_res = vcfg.latent_channels, vcfg.latent_res
    image_encoder = TinyViTImageEncoder(
        out_dim=fcfg.image_dim, image_size=args.image_size
    )
    pipeline = MeshyT2Pipeline.from_checkpoints(
        args.checkpoints,
        MeshVAE(mcfg),
        VoxelVAE(vcfg),
        VoxelFlow(fcfg),
        MeshFlow(gcfg),
        image_encoder,
        device,
    )
    return pipeline


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(prog="meshy_t2", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    tr = sub.add_parser("train", help="train a stage (forwards to train.py flags)")
    tr.add_argument("train_args", nargs=argparse.REMAINDER)

    gen = sub.add_parser("generate", help="image-to-mesh generation")
    gen.add_argument("image", type=Path)
    gen.add_argument("--checkpoints", type=Path, default=Path("outputs/meshy_t2"))
    gen.add_argument("--num-faces", type=int, default=4000)
    gen.add_argument("--num-vertices", type=int, default=None)
    gen.add_argument("--steps", type=int, default=25)
    gen.add_argument("--guidance", type=float, default=3.0)
    gen.add_argument("--out", type=Path, default=Path("outputs/meshy_t2/generated.obj"))

    ret = sub.add_parser("retopo", help="retopologize a dense mesh")
    ret.add_argument("mesh", type=Path)
    ret.add_argument("--checkpoints", type=Path, default=Path("outputs/meshy_t2"))
    ret.add_argument("--num-vertices", type=int, default=None)
    ret.add_argument("--steps", type=int, default=25)
    ret.add_argument("--out", type=Path, default=Path("outputs/meshy_t2/retopo.obj"))

    for p in (gen, ret):
        p.add_argument("--tiny", action="store_true", help="use tiny configs")
        p.add_argument(
            "--device", default="cuda" if torch.cuda.is_available() else "cpu"
        )
        p.add_argument("--image-size", type=int, default=768)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    if args.command == "train":
        return train_main(args.train_args)

    pipeline = _build_pipeline(args)
    if args.command == "generate":
        mesh = pipeline.generate(
            Image.open(args.image),
            num_vertices=args.num_vertices,
            num_faces=args.num_faces,
            steps=args.steps,
            guidance=args.guidance,
        )
    else:
        dense = trimesh.load(str(args.mesh), force="mesh")
        if not isinstance(dense, trimesh.Trimesh):
            raise ValueError(f"could not load a triangle mesh from {args.mesh}")
        mesh = pipeline.retopologize(
            dense, num_vertices=args.num_vertices, steps=args.steps
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(args.out))
    logger.info(
        "wrote %s (%d vertices, %d faces)",
        args.out,
        len(mesh.vertices),
        len(mesh.faces),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
