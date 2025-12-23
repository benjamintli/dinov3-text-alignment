import argparse
import json
import os

import torch

from src.dinov3_text_align import NanoCLIP


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--output-dir", default="outputs_export")
    parser.add_argument("--text-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--img-model", default="facebook/dinov3-vitb16-pretrain-lvd1689m")
    parser.add_argument("--embed-size", type=int, default=64)
    parser.add_argument("--unfreeze-n-blocks", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=0)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--milestones", default="5,10,15")
    parser.add_argument("--lr-mult", type=float, default=0.1)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    model = NanoCLIP(
        txt_model=args.text_model,
        img_model=args.img_model,
        embed_size=args.embed_size,
        unfreeze_n_blocks=args.unfreeze_n_blocks,
        lr=args.lr,
        warmup_epochs=args.warmup_epochs,
        weight_decay=args.weight_decay,
        milestones=[int(m) for m in args.milestones.split(",") if m],
        lr_mult=args.lr_mult,
    )

    ckpt = torch.load(args.ckpt, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)

    img_path = os.path.join(args.output_dir, "img_encoder.pt")
    txt_path = os.path.join(args.output_dir, "txt_encoder.pt")
    torch.save(model.img_encoder.state_dict(), img_path)
    torch.save(model.txt_encoder.state_dict(), txt_path)

    config = {
        "text_model": args.text_model,
        "img_model": args.img_model,
        "embed_size": args.embed_size,
        "unfreeze_n_blocks": args.unfreeze_n_blocks,
        "lr": args.lr,
        "warmup_epochs": args.warmup_epochs,
        "weight_decay": args.weight_decay,
        "milestones": [int(m) for m in args.milestones.split(",") if m],
        "lr_mult": args.lr_mult,
    }
    config_path = os.path.join(args.output_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print(f"Saved image encoder: {img_path}")
    print(f"Saved text encoder: {txt_path}")
    print(f"Saved config: {config_path}")


if __name__ == "__main__":
    main()
