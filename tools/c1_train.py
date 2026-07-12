#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
C1-a 残差头训练：冻结 RF（数据已烘焙），只优化 ResidualIndirectHead。

用法:
  python tools/c1_train.py --data_dir data/c1/bootstrap --epochs 30 --batch_size 2
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from renderformer.c1.dataset import C1ResidualDataset, c1_collate
from renderformer.c1.losses import ResidualIndirectLoss
from renderformer.c1.residual_head import ResidualIndirectHead


def _load_config(path: str | None) -> dict:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Train C1 residual indirect head")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--config", type=str, default="c1_profiles/default.json")
    parser.add_argument("--output_dir", type=str, default="checkpoints/c1")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--neural_dropout",
        type=float,
        default=None,
        help="训练时以该概率将 hdr_neural 置零，减轻「只抄 RF-Direct」",
    )
    args = parser.parse_args()

    cfg = _load_config(args.config if Path(args.config).is_file() else None)
    head_cfg = cfg.get("head", {})
    train_cfg = cfg.get("train", {})

    epochs = args.epochs or int(train_cfg.get("epochs", 30))
    batch_size = args.batch_size or int(train_cfg.get("batch_size", 2))
    lr = args.lr or float(train_cfg.get("lr", 1e-3))
    w_i = float(train_cfg.get("w_indirect", 1.0))
    w_c = float(train_cfg.get("w_compose", 0.5))
    neural_dropout = (
        args.neural_dropout
        if args.neural_dropout is not None
        else float(train_cfg.get("neural_dropout", 0.0))
    )

    device = torch.device(
        args.device
        or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_set = C1ResidualDataset(args.data_dir, split="train")
    try:
        val_set = C1ResidualDataset(args.data_dir, split="val")
        if len(val_set) == 0:
            val_set = None
    except Exception:
        val_set = None

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=c1_collate,
    )
    val_loader = None
    if val_set is not None:
        val_loader = DataLoader(
            val_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=c1_collate,
        )

    head = ResidualIndirectHead(
        use_neural=bool(head_cfg.get("use_neural", True)),
        use_depth=bool(head_cfg.get("use_depth", True)),
        base_channels=int(head_cfg.get("base_channels", 32)),
        num_blocks=int(head_cfg.get("num_blocks", 3)),
    ).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = ResidualIndirectLoss(w_indirect=w_i, w_compose=w_c)

    print(f"device={device} train={len(train_set)} val={len(val_set) if val_set else 0}")
    print(f"params={sum(p.numel() for p in head.parameters())} neural_dropout={neural_dropout}")

    best_val = float("inf")
    history = []

    for epoch in range(1, epochs + 1):
        head.train()
        t0 = time.perf_counter()
        running = 0.0
        n_batches = 0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            neural_in = batch["hdr_neural"]
            if neural_dropout > 0 and head.use_neural:
                if torch.rand(1).item() < neural_dropout:
                    neural_in = torch.zeros_like(neural_in)
            i_pred = head(
                batch["hdr_direct"],
                neural_in,
                batch["depth"],
            )
            loss, stats = loss_fn(
                i_pred,
                batch["indirect_target"],
                hdr_direct=batch["hdr_direct"],
                hdr_gt=batch["hdr_gt"],
                alpha=1.0,
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()
            running += stats["loss_total"]
            n_batches += 1
        train_loss = running / max(n_batches, 1)

        val_loss = None
        if val_loader is not None:
            head.eval()
            v_sum = 0.0
            v_n = 0
            with torch.no_grad():
                for batch in val_loader:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    i_pred = head(
                        batch["hdr_direct"],
                        batch["hdr_neural"],
                        batch["depth"],
                    )
                    loss, stats = loss_fn(
                        i_pred,
                        batch["indirect_target"],
                        hdr_direct=batch["hdr_direct"],
                        hdr_gt=batch["hdr_gt"],
                    )
                    v_sum += stats["loss_total"]
                    v_n += 1
            val_loss = v_sum / max(v_n, 1)

        dt = time.perf_counter() - t0
        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "sec": dt}
        history.append(row)
        msg = f"epoch {epoch:03d} train={train_loss:.5f}"
        if val_loss is not None:
            msg += f" val={val_loss:.5f}"
        msg += f" ({dt:.1f}s)"
        print(msg)

        ckpt = {
            "epoch": epoch,
            "model": head.state_dict(),
            "head_cfg": head_cfg,
            "train_cfg": {"lr": lr, "w_indirect": w_i, "w_compose": w_c},
        }
        torch.save(ckpt, out_dir / "last.pt")
        metric = val_loss if val_loss is not None else train_loss
        if metric < best_val:
            best_val = metric
            torch.save(ckpt, out_dir / "best.pt")

    with open(out_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"Saved checkpoints to {out_dir} (best={best_val:.5f})")


if __name__ == "__main__":
    main()
