"""
train.py – refactored
- AMP mixed precision handled with torch.cuda.amp
- Sample generation every cfg.generate_every steps
- Cleaner helper functions
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

from .tokenizing import setup_tokenizer, mixed_image_tokenize

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
class CFG:
    llm_name = "Qwen/Qwen2.5-0.5B"
    siglip_name = "google/siglip-base-patch16-224"
    start_img = "<image>"
    end_img = "</image>"

    # training
    batch_size = 1
    grad_acc_steps = 8
    max_length = 1024
    lr = 5e-5
    num_epochs = 1
    warmup_steps = 1_000
    save_every = 1_000
    generate_every = 500          # NEW

    # dataset
    dataset_name = "HuggingFaceFW/fineweb"
    dataset_split = "train"
    streaming = True
    max_samples = 2_000_000

    device = "cuda"
    mixed_precision = True        # AMP flag
    amp_dtype = torch.bfloat16    # bfloat16 is usually safe on A100+/H100

cfg = CFG()

# ------------------------------------------------------------------
# TOKENIZER
# ------------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(cfg.llm_name, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

from transformers import AutoConfig
siglip_cfg = AutoConfig.from_pretrained(cfg.siglip_name)
n_patches = (siglip_cfg.vision_config.image_size // siglip_cfg.vision_config.patch_size) ** 2
patch_tokens = [f"<patch_{i}>" for i in range(n_patches)]
tokenizer = setup_tokenizer(tokenizer, cfg.start_img, cfg.end_img, patch_tokens)
vocab_size = len(tokenizer)

# ------------------------------------------------------------------
# MODELS
# ------------------------------------------------------------------
llm_config = AutoConfig.from_pretrained(cfg.llm_name)
# llm = AutoModelForCausalLM.from_pretrained(cfg.llm_name).to(cfg.device)
llm = AutoModelForCausalLM.from_config(llm_config).to(cfg.device)
llm.resize_token_embeddings(vocab_size)

from transformers import AutoModel, AutoImageProcessor
siglip = AutoModel.from_pretrained(cfg.siglip_name).vision_model.to(cfg.device).eval()
for p in siglip.parameters():
    p.requires_grad = False
img_processor = AutoImageProcessor.from_pretrained(cfg.siglip_name)

d_clip = siglip.config.hidden_size
d_llm = llm.config.hidden_size
image_proj = torch.nn.Linear(d_clip, d_llm, bias=False).to(cfg.device)

patch_ids = torch.tensor(tokenizer.convert_tokens_to_ids(patch_tokens), device=cfg.device)

# ------------------------------------------------------------------
# DATASET / DATALOADER
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

    end_tok_id = tokenizer.convert_tokens_to_ids(cfg.end_img)
    post_start = [(seq == end_tok_id).nonzero(as_tuple=True)[0][0].item() + 1
                  if (seq == end_tok_id).any() else len(seq)
                  for seq in mixed_ids]
    post_start = torch.tensor(post_start, dtype=torch.long)

    return dict(
        full_ids=full_ids,
        full_mask=full_mask,
        mixed_ids=mixed_ids,
        mixed_mask=mixed_mask,
        pixel_values=pixel_values,
        post_start=post_start,
    )


raw_ds = load_dataset(cfg.dataset_name,
                      split=cfg.dataset_split,
                      streaming=cfg.streaming,
                      trust_remote_code=True)
train_ds = MixedDataset(tokenizer, patch_tokens, cfg.max_length, raw_ds)
train_loader = DataLoader(train_ds,
                          batch_size=cfg.batch_size,
                          collate_fn=collate,
                          num_workers=0)

# ------------------------------------------------------------------
# OPTIMIZER / SCHEDULER / SCALER
# ------------------------------------------------------------------
optimizer = torch.optim.AdamW(
    list(llm.parameters()) + list(image_proj.parameters()),
    lr=cfg.lr,
)
total_steps = cfg.max_samples // (cfg.batch_size * cfg.grad_acc_steps)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=cfg.warmup_steps,
    num_training_steps=total_steps,
)

# ------------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------------
def make_input_embeds(mixed_ids, pixel_values):
    tok_emb = llm.model.embed_tokens(mixed_ids)  # float32
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=cfg.mixed_precision,
                                     dtype=cfg.amp_dtype):
            img_enc = siglip(pixel_values).last_hidden_state  # (B, 197, d_clip)
            img_emb = image_proj(img_enc)                      # (B, 197, d_llm)

    img_emb = img_emb.to(tok_emb.dtype)  # float32
    mask = torch.isin(mixed_ids, patch_ids)
    b_idx, pos_idx = torch.where(mask)
    for b in range(img_emb.size(0)):
        idx = pos_idx[b_idx == b]
        tok_emb[b, idx] = img_emb[b]  # drop CLS token
    return tok_emb


def generate_sample(step):
    """
    Run a quick greedy generation using one random image + prefix from the dataloader.
    """
    llm.eval()
    # Grab a single sample
    batch = next(iter(train_loader))
    pixel_values = batch["pixel_values"][:1].to(cfg.device)
    mixed_ids = batch["mixed_ids"][:1].to(cfg.device)

    with torch.no_grad():
        input_embeds = make_input_embeds(mixed_ids, pixel_values)
        generated = llm.generate(
            inputs_embeds=input_embeds,
            max_new_tokens=64,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    prompt = tokenizer.decode(mixed_ids[0], skip_special_tokens=False)
    decoded = tokenizer.decode(generated[0], skip_special_tokens=False)
    llm.train()
    wandb.log({"generated": wandb.Html(f"<pre>{prompt}\n\n---\n\n{decoded}</pre>")},
              step=step)


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

        # AMP context
        with torch.cuda.amp.autocast(enabled=cfg.mixed_precision,
                                     dtype=cfg.amp_dtype):
            # 1) Text-only loss
            text_out = llm(input_ids=full_ids,
                           attention_mask=full_mask,
                           labels=full_ids)
            text_loss = text_out.loss

            # 2) Image-mixed loss
            tok_emb = make_input_embeds(mixed_ids, pixel_values)
            labels = mixed_ids.clone()
            for b, start in enumerate(post_start):
                labels[b, :start] = -100
            labels[labels == tokenizer.pad_token_id] = -100

            mixed_out = llm(inputs_embeds=tok_emb,
                            attention_mask=mixed_mask,
                            labels=labels)
            mixed_loss = mixed_out.loss

            loss = 0.5 * text_loss + 0.5 * mixed_loss

        # Backward
        loss.backward()
        if (step + 1) % cfg.grad_acc_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        # Logging
        if step % 50 == 0:
            wandb.log({
                "loss": loss.item(),
                "text_loss": text_loss.item(),
                "mixed_loss": mixed_loss.item(),
                "lr": scheduler.get_last_lr()[0],
            })
            print(f"step={step:6d} | "
                  f"loss={loss.item():.4f} "
                  f"(text={text_loss.item():.3f}, mixed={mixed_loss.item():.3f})")

        # Checkpoint
        if step > 0 and step % cfg.save_every == 0:
            ckpt_dir = f"ckpt/step_{step}"
            os.makedirs(ckpt_dir, exist_ok=True)
            llm.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)
            torch.save(image_proj.state_dict(), f"{ckpt_dir}/image_proj.pt")

        # Generate sample
        if step > 0 and step % cfg.generate_every == 0:
            generate_sample(step)

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