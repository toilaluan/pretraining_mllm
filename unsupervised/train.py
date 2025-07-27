"""
train.py
Unsupervised VLM pre-training:
- text → text causal loss (full sequence)
- (text_prefix + image + text_suffix) → text_suffix loss
- SigLIP-2 is frozen
"""

import os
import math
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
)
from datasets import load_dataset
from PIL import Image
import wandb
from tokenizing import setup_tokenizer, mixed_image_tokenize


# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
class CFG:
    # model
    llm_name = "meta-llama/Llama-3.1-8B-Instruct"
    siglip_name = "google/siglip-base-patch16-224"
    start_img = "<image>"
    end_img = "</image>"

    # training
    batch_size = 4
    grad_acc_steps = 8
    max_length = 1024
    lr = 5e-5
    num_epochs = 1
    warmup_steps = 1_000
    save_every = 1_000

    # dataset
    dataset_name = "HuggingFaceFW/fineweb"
    dataset_split = "train"
    streaming = True
    max_samples = 2_000_000

    device = "cuda"
    mixed_precision = True


cfg = CFG()

# ------------------------------------------------------------------
# TOKENIZER  (with the exact number of patch tokens)
# ------------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(cfg.llm_name, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# get number of patches from SigLIP config
from transformers import AutoConfig

siglip_cfg = AutoConfig.from_pretrained(cfg.siglip_name)
n_patches = (siglip_cfg.image_size // siglip_cfg.patch_size) ** 2  # 196
patch_tokens = [f"<patch_{i}>" for i in range(n_patches)]
tokenizer = setup_tokenizer(tokenizer, cfg.start_img, cfg.end_img, patch_tokens)
vocab_size = len(tokenizer)

# ------------------------------------------------------------------
# MODELS  (LLM trainable, SigLIP frozen)
# ------------------------------------------------------------------
llm = AutoModelForCausalLM.from_pretrained(
    cfg.llm_name,
    torch_dtype=torch.bfloat16,
    device_map={"": cfg.device},
    attn_implementation="flash_attention_2",
)
llm.resize_token_embeddings(vocab_size)

from transformers import AutoModel, AutoImageProcessor

siglip = AutoModel.from_pretrained(cfg.siglip_name).vision_model.to(cfg.device).eval()
for p in siglip.parameters():
    p.requires_grad = False

img_processor = AutoImageProcessor.from_pretrained(cfg.siglip_name)

# image-to-text projection (learned)
d_clip = siglip.config.hidden_size
d_llm = llm.config.hidden_size
image_proj = torch.nn.Linear(d_clip, d_llm, bias=False).to(cfg.device)


# ------------------------------------------------------------------
# DATASET
# ------------------------------------------------------------------
class MixedDataset(Dataset):
    def __init__(self, tokenizer, patch_tokens, max_len, stream):
        self.tokenizer = tokenizer
        self.patch_tokens = patch_tokens
        self.max_len = max_len
        self.stream = iter(stream)

    def __len__(self):
        return cfg.max_samples

    def __getitem__(self, _):
        sample = next(self.stream)
        text = sample["text"]
        return mixed_image_tokenize(
            self.tokenizer,
            text,
            self.patch_tokens,
            start_image_token=cfg.start_img,
            end_image_token=cfg.end_img,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
        )


def collate(batch):
    images = [b["mixed_image"]["image"].convert("RGB") for b in batch]
    pixel_values = img_processor(images, return_tensors="pt").pixel_values

    full_ids = torch.stack([b["full_text"]["input_ids"] for b in batch])
    full_mask = torch.stack([b["full_text"]["attention_mask"] for b in batch])

    mixed_ids = torch.stack([b["mixed_image"]["input_ids"] for b in batch])
    mixed_mask = torch.stack([b["mixed_image"]["attention_mask"] for b in batch])

    # locate post-image text for loss masking
    end_tok_id = tokenizer.convert_tokens_to_ids(cfg.end_img)
    post_start = []
    for seq in mixed_ids:
        pos = (seq == end_tok_id).nonzero(as_tuple=True)[0]
        post_start.append(pos.item() + 1 if len(pos) > 0 else len(seq))
    post_start = torch.tensor(post_start, dtype=torch.long)

    return {
        "full_ids": full_ids,
        "full_mask": full_mask,
        "mixed_ids": mixed_ids,
        "mixed_mask": mixed_mask,
        "pixel_values": pixel_values,
        "post_start": post_start,
    }


raw_ds = load_dataset(
    cfg.dataset_name,
    split=cfg.dataset_split,
    streaming=cfg.streaming,
    trust_remote_code=True,
)
train_ds = MixedDataset(tokenizer, patch_tokens, cfg.max_length, raw_ds)
train_loader = DataLoader(
    train_ds,
    batch_size=cfg.batch_size,
    collate_fn=collate,
    num_workers=0,
)

# ------------------------------------------------------------------
# OPTIMIZER / SCHEDULER
# ------------------------------------------------------------------
optimizer = torch.optim.AdamW(
    list(llm.parameters()) + list(image_proj.parameters()),
    lr=cfg.lr,
)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=cfg.warmup_steps,
    num_training_steps=(cfg.max_samples // (cfg.batch_size * cfg.grad_acc_steps)),
)
scaler = torch.cuda.amp.GradScaler(enabled=cfg.mixed_precision)

# ------------------------------------------------------------------
# TRAIN LOOP
# ------------------------------------------------------------------
wandb.init(project="unsupervised-vlm", config=cfg.__dict__)
step = 0
llm.train()
image_proj.train()

for epoch in range(cfg.num_epochs):
    for batch in train_loader:
        full_ids = batch["full_ids"].to(cfg.device)
        full_mask = batch["full_mask"].to(cfg.device)

        mixed_ids = batch["mixed_ids"].to(cfg.device)
        mixed_mask = batch["mixed_mask"].to(cfg.device)
        pixel_values = batch["pixel_values"].to(cfg.device)
        post_start = batch["post_start"].to(cfg.device)

        # -------------------------------------------------------------
        # 1) TEXT-ONLY LOSS (full causal LM)
        # -------------------------------------------------------------
        with torch.cuda.amp.autocast(enabled=cfg.mixed_precision):
            text_out = llm(
                input_ids=full_ids,
                attention_mask=full_mask,
                labels=full_ids,
            )
            text_loss = text_out.loss

        # -------------------------------------------------------------
        # 2) IMAGE-MIXED LOSS (only on tokens after </image>)
        # -------------------------------------------------------------
        # 2a) encode image
        with torch.no_grad():
            img_enc = siglip(pixel_values).last_hidden_state  # (B, 197, d_clip)
            img_enc = img_enc[:, 1:]  # drop cls token
            img_emb = image_proj(img_enc)  # (B, 196, d_llm)

        # 2b) build input embeddings
        tok_emb = llm.model.embed_tokens(mixed_ids)  # (B, L, d_llm)

        # locate patch placeholder positions
        patch_id = tokenizer.convert_tokens_to_ids(patch_tokens[0])
        mask = mixed_ids == patch_id  # (B, L) bool
        b_idx, pos_idx = torch.where(mask)
        for b in range(img_emb.size(0)):
            idx = pos_idx[b_idx == b]
            tok_emb[b, idx] = img_emb[b]  # in-place replace

        # 2c) create labels: keep only tokens after </image>
        labels = mixed_ids.clone()
        for b, start in enumerate(post_start):
            labels[b, :start] = -100
        labels[labels == tokenizer.pad_token_id] = -100

        with torch.cuda.amp.autocast(enabled=cfg.mixed_precision):
            mixed_out = llm(
                inputs_embeds=tok_emb,
                attention_mask=mixed_mask,
                labels=labels,
            )
            mixed_loss = mixed_out.loss

        # -------------------------------------------------------------
        # 3) COMBINED LOSS
        # -------------------------------------------------------------
        loss = 0.5 * text_loss + 0.5 * mixed_loss

        scaler.scale(loss).backward()
        if (step + 1) % cfg.grad_acc_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

        # -------------------------------------------------------------
        # LOGGING
        # -------------------------------------------------------------
        if step % 50 == 0:
            wandb.log(
                {
                    "loss": loss.item(),
                    "text_loss": text_loss.item(),
                    "mixed_loss": mixed_loss.item(),
                    "lr": scheduler.get_last_lr()[0],
                }
            )
            print(
                f"step={step:6d} | loss={loss.item():.4f} "
                f"(text={text_loss.item():.3f}, mixed={mixed_loss.item():.3f})"
            )

        if step > 0 and step % cfg.save_every == 0:
            ckpt_dir = f"ckpt/step_{step}"
            os.makedirs(ckpt_dir, exist_ok=True)
            llm.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)
            torch.save(image_proj.state_dict(), f"{ckpt_dir}/image_proj.pt")

        step += 1
        if step >= cfg.max_samples:
            break

# ------------------------------------------------------------------
# FINAL SAVE
# ------------------------------------------------------------------
llm.save_pretrained("ckpt/final")
tokenizer.save_pretrained("ckpt/final")
torch.save(image_proj.state_dict(), "ckpt/final/image_proj.pt")
print("Training finished.")
