"""Training entry points for the four Meshy T2 stages.

Stages (paper Sec. 2):

* ``voxel-vae``   — dense 3D conv VAE over occupancy grids.
* ``mesh-vae``    — vertex-set mesh VAE (frozen in later stages).
* ``voxel-flow``  — Stage I image-conditioned scaffold flow.
* ``mesh-flow``   — Stage II image/voxel/count-conditioned latent flow.

All stages train on either a folder of meshes or the procedural
``SyntheticMeshDataset`` so smoke runs need no external assets. Image
conditioning is enabled when ``--images-root`` provides renders named
like the meshes; otherwise the image condition is always dropped (the
models then learn voxel/count-conditional generation only).
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from .data import MeshFolderDataset, MeshSample, SyntheticMeshDataset
from .flow import interpolate, sample_t
from .losses import mesh_vae_loss
from .models.image_encoder import TinyViTImageEncoder
from .models.mesh_flow import (
    MeshFlow,
    MeshFlowConfig,
    assign_latent_coords,
    image_token_coords,
    voxel_token_coords,
)
from .models.mesh_vae import MeshVAE, MeshVAEConfig
from .models.voxel_flow import VoxelFlow, VoxelFlowConfig
from .models.voxel_vae import VoxelVAE, VoxelVAEConfig

logger = logging.getLogger(__name__)


def _build_dataset(args: argparse.Namespace):
    if args.data == "synthetic":
        return SyntheticMeshDataset(
            size=args.dataset_size,
            num_samples=args.num_samples,
            voxel_res=args.voxel_res,
        )
    return MeshFolderDataset(
        args.data, num_samples=args.num_samples, voxel_res=args.voxel_res
    )


def _collate(batch: list[MeshSample]) -> list[MeshSample]:
    return batch  # variable vertex counts: keep a list


def _maybe_image_tokens(
    encoder: TinyViTImageEncoder | None,
    images_root: Path | None,
    idx: int,
    device: torch.device | str,
) -> torch.Tensor | None:
    if encoder is None or images_root is None:
        return None
    for suffix in (".png", ".jpg", ".jpeg"):
        path = images_root / f"{idx:06d}{suffix}"
        if path.exists():
            img = Image.open(path).convert("RGB")
            arr = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0)
            return encoder(arr.permute(2, 0, 1)[None].to(device))
    return None


def train_mesh_vae(args: argparse.Namespace) -> Path:
    """Train the vertex-set mesh VAE (Sec. 2.1)."""
    device = args.device
    cfg = MeshVAEConfig.tiny() if args.tiny else MeshVAEConfig()
    model = MeshVAE(cfg).to(device)
    dataset = _build_dataset(args)
    loader = DataLoader(
        dataset, batch_size=1, shuffle=True, collate_fn=_collate, num_workers=0
    )
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    step = 0
    for epoch in range(args.epochs):
        for batch in loader:
            sample = batch[0]
            v = sample.vertices.to(device)
            e = sample.edges.to(device)
            p = sample.points.to(device)
            n = sample.normals.to(device)
            positions, edge_emb, face_emb, _ = model(v, e, p, n)
            loss, parts = mesh_vae_loss(
                positions,
                edge_emb,
                face_emb,
                model.decoder.null_prev,
                model.decoder.null_next,
                v,
                e,
                [t.to(device) for t in sample.neighbors],
                [t.to(device) for t in sample.successors],
                cfg,
            )
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            if step % args.log_every == 0:
                logger.info("step %d loss %.4f %s", step, float(loss), parts)
            if step >= args.max_steps:
                torch.save(model.state_dict(), out_dir / "mesh_vae.pt")
                logger.info("saved %s", out_dir / "mesh_vae.pt")
                return out_dir / "mesh_vae.pt"
    torch.save(model.state_dict(), out_dir / "mesh_vae.pt")
    return out_dir / "mesh_vae.pt"


def train_voxel_vae(args: argparse.Namespace) -> Path:
    """Train the Voxel VAE over occupancy grids (Sec. 2.2)."""
    device = args.device
    cfg = VoxelVAEConfig.tiny() if args.tiny else VoxelVAEConfig(res=args.voxel_res)
    model = VoxelVAE(cfg).to(device)
    dataset = _build_dataset(args)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, collate_fn=_collate
    )
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    step = 0
    for _ in range(args.epochs):
        for batch in loader:
            occ = (
                torch.stack([s.occupancy for s in batch])
                .float()
                .unsqueeze(1)
                .to(device)
            )
            loss, parts = model.loss(occ)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            if step % args.log_every == 0:
                logger.info("step %d loss %.4f %s", step, float(loss), parts)
            if step >= args.max_steps:
                torch.save(model.state_dict(), out_dir / "voxel_vae.pt")
                return out_dir / "voxel_vae.pt"
    torch.save(model.state_dict(), out_dir / "voxel_vae.pt")
    return out_dir / "voxel_vae.pt"


def train_voxel_flow(args: argparse.Namespace) -> Path:
    """Train the Stage-I scaffold flow with the Voxel VAE frozen."""
    device = args.device
    vcfg = VoxelVAEConfig.tiny() if args.tiny else VoxelVAEConfig(res=args.voxel_res)
    voxel_vae = VoxelVAE(vcfg).to(device)
    ckpt = Path(args.out) / "voxel_vae.pt"
    if ckpt.exists():
        voxel_vae.load_state_dict(
            torch.load(ckpt, map_location=device, weights_only=True)
        )
    voxel_vae.eval()
    for p in voxel_vae.parameters():
        p.requires_grad_(False)

    fcfg = VoxelFlowConfig.tiny() if args.tiny else VoxelFlowConfig()
    fcfg.latent_channels = vcfg.latent_channels
    fcfg.latent_res = vcfg.latent_res
    flow = VoxelFlow(fcfg).to(device)
    encoder = TinyViTImageEncoder(out_dim=fcfg.image_dim) if args.images_root else None
    if encoder is not None:
        encoder.to(device)
        encoder.requires_grad_(False)
    params = list(flow.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    dataset = _build_dataset(args)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=_collate)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    step = 0
    for _ in range(args.epochs):
        for batch in loader:
            sample = batch[0]
            occ = sample.occupancy.float()[None, None].to(device)
            with torch.no_grad():
                z = voxel_vae.encode_deterministic(occ)
            x0 = z.flatten(2).transpose(1, 2)  # (1, res^3, C)
            x1 = torch.randn_like(x0)
            t = sample_t(1, device)
            x_t, target = interpolate(x0, x1, t)
            img = _maybe_image_tokens(encoder, _images_root(args), 0, device)
            pred = flow(x_t, t, img)
            loss = torch.nn.functional.mse_loss(pred, target)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            step += 1
            if step % args.log_every == 0:
                logger.info("step %d loss %.4f", step, float(loss))
            if step >= args.max_steps:
                torch.save(flow.state_dict(), out_dir / "voxel_flow.pt")
                return out_dir / "voxel_flow.pt"
    torch.save(flow.state_dict(), out_dir / "voxel_flow.pt")
    return out_dir / "voxel_flow.pt"


def train_mesh_flow(args: argparse.Namespace) -> Path:
    """Train the Stage-II mesh flow (Sobol OT, count condition, CFG drops)."""
    device = args.device
    mcfg = MeshVAEConfig.tiny() if args.tiny else MeshVAEConfig()
    mesh_vae = MeshVAE(mcfg).to(device)
    ckpt = Path(args.out) / "mesh_vae.pt"
    if ckpt.exists():
        mesh_vae.load_state_dict(
            torch.load(ckpt, map_location=device, weights_only=True)
        )
    mesh_vae.eval()
    for p in mesh_vae.parameters():
        p.requires_grad_(False)

    vcfg = VoxelVAEConfig.tiny() if args.tiny else VoxelVAEConfig(res=args.voxel_res)
    voxel_vae = VoxelVAE(vcfg).to(device)
    vckpt = Path(args.out) / "voxel_vae.pt"
    if vckpt.exists():
        voxel_vae.load_state_dict(
            torch.load(vckpt, map_location=device, weights_only=True)
        )

    fcfg = MeshFlowConfig.tiny() if args.tiny else MeshFlowConfig()
    fcfg.latent_channels = mcfg.latent_channels
    fcfg.voxel_channels = vcfg.latent_channels
    fcfg.voxel_res = vcfg.latent_res
    flow = MeshFlow(fcfg).to(device)

    encoder = TinyViTImageEncoder(out_dim=fcfg.image_dim) if args.images_root else None
    if encoder is not None:
        encoder.to(device)
        encoder.requires_grad_(False)

    # The voxel encoder is fine-tuned at 0.25x learning rate (Sec. 2.3).
    opt = torch.optim.AdamW(
        [
            {"params": flow.parameters(), "lr": args.lr},
            {"params": voxel_vae.parameters(), "lr": args.lr * 0.25},
        ],
        weight_decay=0.01,
    )
    dataset = _build_dataset(args)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=_collate)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    step = 0
    for _ in range(args.epochs):
        for batch in loader:
            sample = batch[0]
            t0 = time.time()
            del t0
            with torch.no_grad():
                z = mesh_vae.encode(
                    sample.vertices.to(device),
                    sample.edges.to(device),
                    sample.points.to(device),
                    sample.normals.to(device),
                )
            v = z.shape[0]
            num_pads = int(rng.uniform(0, fcfg.pad_ratio) * v)
            num_slots = v + num_pads
            pads = torch.cat(
                [
                    torch.zeros(num_pads, fcfg.latent_channels, device=device),
                    -torch.ones(num_pads, 1, device=device),
                ],
                dim=-1,
            )
            x0 = torch.cat([torch.cat([z, torch.ones(v, 1, device=device)], -1), pads])[
                None
            ]

            coords = torch.tensor(
                assign_latent_coords(sample.vertices.numpy(), num_slots, seed=step),
                dtype=torch.float32,
                device=device,
            )
            t = sample_t(1, device)
            x1 = torch.randn_like(x0)
            x_t, target = interpolate(x0, x1, t)

            # Independent condition dropout (Sec. 2.3).
            drop_all = rng.random() < fcfg.drop_all
            use_img = (
                not drop_all and encoder is not None and rng.random() >= fcfg.drop_image
            )
            use_vox = not drop_all and rng.random() >= fcfg.drop_voxel
            use_cnt = not drop_all and rng.random() >= fcfg.drop_count

            img = (
                _maybe_image_tokens(encoder, _images_root(args), 0, device)
                if use_img
                else None
            )
            img_coords = (
                image_token_coords(img.shape[1]).to(device) if img is not None else None
            )
            vox_tokens = vox_coords = None
            if use_vox:
                occ = sample.occupancy.float()[None, None].to(device)
                z_v = voxel_vae.encode_deterministic(occ)
                vox_tokens = z_v.flatten(2).transpose(1, 2)
                vox_coords = voxel_token_coords(z_v.shape[-1]).to(device)
            count = torch.tensor([float(num_slots)], device=device) if use_cnt else None

            pred = flow(
                x_t,
                t,
                latent_coords=coords,
                image_tokens=img,
                image_coords=img_coords,
                voxel_tokens=vox_tokens,
                voxel_coords=vox_coords,
                count=count,
            )
            loss = torch.nn.functional.mse_loss(pred, target)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(flow.parameters(), 1.0)
            opt.step()
            step += 1
            if step % args.log_every == 0:
                logger.info("step %d loss %.4f slots %d", step, float(loss), num_slots)
            if step >= args.max_steps:
                torch.save(flow.state_dict(), out_dir / "mesh_flow.pt")
                return out_dir / "mesh_flow.pt"
    torch.save(flow.state_dict(), out_dir / "mesh_flow.pt")
    return out_dir / "mesh_flow.pt"


def _images_root(args: argparse.Namespace) -> Path | None:
    return Path(args.images_root) if args.images_root else None


STAGES = {
    "mesh-vae": train_mesh_vae,
    "voxel-vae": train_voxel_vae,
    "voxel-flow": train_voxel_flow,
    "mesh-flow": train_mesh_flow,
}


def build_parser() -> argparse.ArgumentParser:
    """Build the training argument parser."""
    parser = argparse.ArgumentParser(description="Meshy T2 reproduction trainer")
    parser.add_argument("stage", choices=list(STAGES))
    parser.add_argument(
        "--data", default="synthetic", help="'synthetic' or a mesh folder"
    )
    parser.add_argument(
        "--images-root", default=None, help="optional renders for conditioning"
    )
    parser.add_argument("--out", default="outputs/meshy_t2")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--tiny", action="store_true", help="use tiny configs (CPU smoke runs)"
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--num-samples", type=int, default=4096)
    parser.add_argument("--voxel-res", type=int, default=64)
    parser.add_argument("--dataset-size", type=int, default=128)
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    path = STAGES[args.stage](args)
    logger.info("checkpoint written to %s", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
