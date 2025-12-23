import argparse
import csv
import os
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor, AutoTokenizer

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)

from src.dinov3_text_align import DinoV3TextAlignment  # noqa: E402

try:
    from PIL import Image
except ImportError as exc:
    raise SystemExit("Missing deps. Install with: pip install pillow") from exc


class ImageTextDataset(Dataset):
    def __init__(
        self,
        csv_path,
        image_root,
        tokenizer,
        image_processor,
        max_length,
        image_col="image",
        caption_col="caption",
    ):
        self.items = []
        self.image_root = image_root
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_length = max_length
        self.image_col = image_col
        self.caption_col = caption_col

        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                image_path = row[self.image_col]
                caption = row[self.caption_col]
                self.items.append((image_path, caption))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        image_path, caption = self.items[idx]
        if self.image_root:
            image_path = os.path.join(self.image_root, image_path)

        with Image.open(image_path) as img:
            img = img.convert("RGB")
            image = self.image_processor(images=img, return_tensors="pt")[
                "pixel_values"
            ].squeeze(0)

        tokenized = self.tokenizer(
            caption,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )
        input_ids = tokenized["input_ids"].squeeze(0)
        attention_mask = tokenized["attention_mask"].squeeze(0)
        return image, input_ids, attention_mask


class HFDataset(Dataset):
    def __init__(
        self,
        hf_dataset,
        tokenizer,
        image_processor,
        max_length,
        image_col="image",
        caption_col="caption",
        caption_index=0,
    ):
        self.hf_dataset = hf_dataset
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_length = max_length
        self.image_col = image_col
        self.caption_col = caption_col
        self.caption_index = caption_index

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        row = self.hf_dataset[idx]
        image = row[self.image_col]
        caption = row[self.caption_col]
        if isinstance(caption, list):
            caption = caption[self.caption_index]

        if isinstance(image, str):
            with Image.open(image) as img:
                img = img.convert("RGB")
                pixel_values = self.image_processor(images=img, return_tensors="pt")[
                    "pixel_values"
                ].squeeze(0)
        else:
            pixel_values = self.image_processor(images=image, return_tensors="pt")[
                "pixel_values"
            ].squeeze(0)

        tokenized = self.tokenizer(
            caption,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )
        input_ids = tokenized["input_ids"].squeeze(0)
        attention_mask = tokenized["attention_mask"].squeeze(0)
        return pixel_values, input_ids, attention_mask


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_optimizer_and_scheduler(model):
    optimizer_params = [
        {
            "params": model.img_encoder.parameters(),
            "lr": model.lr,
            "weight_decay": model.weight_decay,
        },
        {
            "params": model.txt_encoder.parameters(),
            "lr": model.lr,
            "weight_decay": model.weight_decay,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_params)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=model.milestones, gamma=model.lr_mult
    )
    return optimizer, scheduler


def apply_warmup(optimizer, step, total_warmup_steps, base_lr):
    if total_warmup_steps <= 0:
        return
    lr_scale = min(1.0, (step + 1) / total_warmup_steps)
    for pg in optimizer.param_groups:
        pg["lr"] = lr_scale * base_lr


def evaluate(model, val_loader, device):
    model.eval()
    val_img = []
    val_txt = []
    with torch.no_grad():
        for images, captions, masks in val_loader:
            images = images.to(device, non_blocking=True)
            captions = captions.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            img_desc, txt_desc = model(images, captions, masks)
            val_img.append(img_desc.detach().cpu().numpy())
            val_txt.append(txt_desc.detach().cpu().numpy())

    img_desc = np.concatenate(val_img, axis=0)
    txt_desc = np.concatenate(val_txt, axis=0)
    labels = np.arange(img_desc.shape[0])
    recall_1, recall_5, recall_10 = model._calculate_recall(
        img_desc, txt_desc, labels, k_values=[1, 5, 10]
    )
    return recall_1, recall_5, recall_10


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv")
    parser.add_argument("--val-csv")
    parser.add_argument("--image-root", default="")
    parser.add_argument("--image-col", default="image")
    parser.add_argument("--caption-col", default="caption")
    parser.add_argument("--hf-dataset")
    parser.add_argument("--hf-split", default="train")
    parser.add_argument("--hf-val-split", default="validation")
    parser.add_argument("--hf-image-col", default="image")
    parser.add_argument("--hf-caption-col", default="caption")
    parser.add_argument("--hf-split-col", default="split")
    parser.add_argument("--caption-index", type=int, default=0)
    parser.add_argument(
        "--text-model", default="sentence-transformers/all-MiniLM-L6-v2"
    )
    parser.add_argument(
        "--img-model", default="facebook/dinov3-vitb16-pretrain-lvd1689m"
    )
    parser.add_argument("--embed-size", type=int, default=64)
    parser.add_argument("--unfreeze-n-blocks", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--milestones", default="5,10,15")
    parser.add_argument("--lr-mult", type=float, default=0.1)
    parser.add_argument("--warmup-epochs", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hf-upload-repo")
    parser.add_argument("--hf-upload-branch", default="main")
    parser.add_argument("--hf-upload-token", default="")
    args = parser.parse_args()

    try:
        from huggingface_hub import HfApi, login
    except ImportError as exc:
        raise SystemExit(
            "Missing deps. Install with: pip install huggingface_hub"
        ) from exc
    
    login(token=args.hf_upload_token)

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.text_model)
    image_processor = AutoImageProcessor.from_pretrained(args.img_model)

    if args.hf_dataset:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise SystemExit(
                "Missing deps. Install with: pip install datasets"
            ) from exc

        train_hf = load_dataset(args.hf_dataset, split=args.hf_split)
        val_hf = load_dataset(args.hf_dataset, split=args.hf_val_split)
        if args.hf_split_col in train_hf.column_names:
            train_hf = train_hf.filter(lambda x: x[args.hf_split_col] == args.hf_split)
        if args.hf_split_col in val_hf.column_names:
            val_hf = val_hf.filter(lambda x: x[args.hf_split_col] == args.hf_val_split)
        train_ds = HFDataset(
            train_hf,
            tokenizer,
            image_processor,
            args.max_length,
            image_col=args.hf_image_col,
            caption_col=args.hf_caption_col,
            caption_index=args.caption_index,
        )
        val_ds = HFDataset(
            val_hf,
            tokenizer,
            image_processor,
            args.max_length,
            image_col=args.hf_image_col,
            caption_col=args.hf_caption_col,
            caption_index=args.caption_index,
        )
    else:
        if not args.train_csv or not args.val_csv:
            raise SystemExit("Provide --train-csv/--val-csv or set --hf-dataset.")
        train_ds = ImageTextDataset(
            args.train_csv,
            args.image_root,
            tokenizer,
            image_processor,
            args.max_length,
            image_col=args.image_col,
            caption_col=args.caption_col,
        )
        val_ds = ImageTextDataset(
            args.val_csv,
            args.image_root,
            tokenizer,
            image_processor,
            args.max_length,
            image_col=args.image_col,
            caption_col=args.caption_col,
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = DinoV3TextAlignment(
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer, scheduler = build_optimizer_and_scheduler(model)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    total_warmup_steps = args.warmup_epochs * len(train_loader)
    global_step = 0
    best_recall = -1.0

    for epoch in range(args.epochs):
        model.train()
        for images, captions, masks in train_loader:
            images = images.to(device, non_blocking=True)
            captions = captions.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            apply_warmup(optimizer, global_step, total_warmup_steps, args.lr)

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                img_desc, txt_desc = model(images, captions, masks)
                loss, batch_acc = model.loss_fn(img_desc, txt_desc)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            global_step += 1

        scheduler.step()

        recall_1, recall_5, recall_10 = evaluate(model, val_loader, device)
        print(
            f"epoch {epoch} "
            f"loss={loss.item():.4f} "
            f"recall@1={recall_1:.4f} "
            f"recall@5={recall_5:.4f} "
            f"recall@10={recall_10:.4f}"
        )

        if recall_1 > best_recall:
            best_recall = recall_1
            ckpt_path = os.path.join(args.output_dir, "best.pt")
            torch.save({"model": model.state_dict(), "epoch": epoch}, ckpt_path)

        ckpt_path = os.path.join(args.output_dir, f"epoch_{epoch}.pt")
        torch.save({"model": model.state_dict(), "epoch": epoch}, ckpt_path)

        if args.hf_upload_repo:
            token = args.hf_upload_token or os.environ.get("HF_TOKEN")
            if not token:
                raise SystemExit("Set --hf-upload-token or HF_TOKEN to upload.")
            api = HfApi(token=token)
            api.upload_file(
                path_or_fileobj=ckpt_path,
                path_in_repo=os.path.basename(ckpt_path),
                repo_id=args.hf_upload_repo,
                repo_type="model",
                revision=args.hf_upload_branch,
                commit_message=f"Add checkpoint epoch {epoch}",
            )


if __name__ == "__main__":
    main()
