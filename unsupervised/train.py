"""
train.py – Distributed-safe, Accelerate-correct, with text-only & image-conditioned sampling.

- Single Accelerator instance (no globals), rank-0 logging/ckpt only
- Proper gradient accumulation via accelerate.accumulate
- Mixed precision via Accelerator(mixed_precision=...)
- Robust unwrap_model() and get_input_embeddings()
- Text-only sample generation added
- Safer label masking; attention_mask used for generate()
- Checkpoint rotation on rank 0 only
"""

import os
import math
import shutil
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    get_constant_schedule_with_warmup,
)
from transformers.models.perception_lm.modeling_perception_lm import (
    PerceptionLMMultiModalProjector,
)
from datasets import load_dataset
from PIL import Image
import wandb
import timm
from accelerate import Accelerator


# --------------------------- Configs ---------------------------


@dataclass
class TrainingConfig:
    # Models
    llm_name: str = "Qwen/Qwen3-0.6B"
    use_llm_pretrained: bool = False
    image_enc_name: str = "timm/vit_pe_core_large_patch14_336.fb"
    start_img: str = "<image>"
    end_img: str = "</image>"

    # Train
    batch_size: int = 8
    grad_acc_steps: int = 1
    max_length: int = 1024
    lr: float = 5e-4
    num_epochs: int = 1
    warmup_steps: int = 1_000
    max_steps: int = 1_000_000_000  # global steps cap

    # Logging / ckpts
    save_every: int = 1000
    generate_every: int = 500
    log_every: int = 25
    max_checkpoints: int = 1

    # Data
    dataset_name: str = "HuggingFaceFW/fineweb"
    dataset_split: str = "train"
    streaming: bool = True

    # HW / precision
    mixed_precision: str = "bf16"  # "bf16" | "fp16" | "no"
    num_workers: int = 8

    # Paths
    checkpoint_dir: str = "checkpoints"
    final_model_dir: str = "final_model"


# --------------------------- Utilities ---------------------------


class CheckpointManager:
    def __init__(
        self,
        base_dir: str,
        max_checkpoints: int = 3,
        accelerator: Optional[Accelerator] = None,
    ):
        self.base_dir = Path(base_dir)
        self.max_checkpoints = max_checkpoints
        self.base_dir.mkdir(exist_ok=True, parents=True)
        self.accelerator = accelerator

    def save_checkpoint(self, step: int, llm, tokenizer, image_proj) -> str:
        if not self.accelerator.is_main_process:
            return ""
        self.accelerator.wait_for_everyone()

        ckpt_dir = self.base_dir / f"step_{step}"
        ckpt_dir.mkdir(exist_ok=True)

        base = self.accelerator.unwrap_model(llm)
        base.save_pretrained(
            ckpt_dir,
            is_main_process=True,
            save_function=self.accelerator.save,
        )
        # image projector is small; save with accelerator.save to avoid N copies
        self.accelerator.save(image_proj.state_dict(), ckpt_dir / "image_proj.pt")

        tokenizer.save_pretrained(ckpt_dir)
        self._cleanup_old_checkpoints()
        return str(ckpt_dir)

    def _cleanup_old_checkpoints(self):
        if not self.accelerator.is_main_process:
            return
        checkpoints = []
        for path in self.base_dir.iterdir():
            if path.is_dir() and path.name.startswith("step_"):
                try:
                    step = int(path.name.split("_")[1])
                    checkpoints.append((step, path))
                except Exception:
                    continue
        checkpoints.sort(key=lambda x: x[0])
        while len(checkpoints) > self.max_checkpoints:
            _, path = checkpoints.pop(0)
            shutil.rmtree(path)
            self.accelerator.print(f"Removed old checkpoint: {path}")


# --------------------------- Dataset ---------------------------

# You already have these in your repo:
from .tokenizing import setup_tokenizer, mixed_image_tokenize  # noqa: E402


class MixedDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        patch_tokens: List[str],
        max_len: int,
        stream,
        config: TrainingConfig,
    ):
        self.tokenizer = tokenizer
        self.patch_tokens = patch_tokens
        self.max_len = max_len
        self.stream = iter(stream)
        self.config = config

    def __len__(self):
        # Acts like an infinite stream capped by max_steps in trainer
        return 10**12  # large sentinel; trainer caps by max_steps

    def __getitem__(self, _):
        try:
            sample = next(self.stream)
            text = sample.get("text", "")
            return mixed_image_tokenize(
                self.tokenizer,
                text,
                self.patch_tokens,
                start_image_token=self.config.start_img,
                end_image_token=self.config.end_img,
                max_length=self.max_len,
                padding="max_length",
                truncation=True,
            )
        except StopIteration:
            # re-create stream
            self.stream = iter(
                load_dataset(
                    self.config.dataset_name,
                    split=self.config.dataset_split,
                    streaming=self.config.streaming,
                    trust_remote_code=True,
                ).shuffle(seed=42)
            )
            return self.__getitem__(_)


# --------------------------- Trainer ---------------------------


class VisionLanguageTrainer:
    def __init__(self, config: TrainingConfig, accelerator: Accelerator):
        self.config = config
        self.accelerator = accelerator
        self.checkpoint_manager = CheckpointManager(
            config.checkpoint_dir, config.max_checkpoints, accelerator
        )
        self.device = accelerator.device

        self._setup_models()
        self._setup_data()
        self._setup_training()

        # Prepare trainables and dataloader
        (
            self.llm,
            self.image_proj,
            self.optimizer,
            self.scheduler,
            self.train_loader,
        ) = self.accelerator.prepare(
            self.llm,
            self.image_proj,
            self.optimizer,
            self.scheduler,
            self.train_loader,
        )

        # Keep frozen encoder on device
        self.image_enc.to(self.device).eval()
        for p in self.image_enc.parameters():
            p.requires_grad = False

    # ---- models

    def _setup_models(self):
        self.accelerator.print("Setting up models...")

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.llm_name, use_fast=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Vision encoder + preprocess
        self.image_enc = timm.create_model(self.config.image_enc_name, pretrained=True)
        data_config = timm.data.resolve_model_data_config(self.image_enc)
        self.img_transform = timm.data.create_transform(
            **data_config, is_training=False
        )

        # Infer number of patch tokens from a dummy forward (after dropping cls)
        with torch.no_grad():
            in_size = data_config.get("input_size", (3, 336, 336))
            dummy = torch.zeros(1, *in_size)
            feats = self.image_enc.forward_features(dummy)  # (1, 1+N, C) for ViT
            n = feats.shape[1] - 1
        # Match your original projector pooling ratio of 2 -> assume /4 tokens
        n_patch_tokens = max(1, n // 4)

        patch_tokens = [f"<patch_{i}>" for i in range(n_patch_tokens)]
        self.tokenizer = setup_tokenizer(
            self.tokenizer, self.config.start_img, self.config.end_img, patch_tokens
        )
        self.patch_tokens = patch_tokens
        self.patch_ids = torch.tensor(
            self.tokenizer.convert_tokens_to_ids(patch_tokens)
        )  # move to device later
        self.start_img_id = self.tokenizer.convert_tokens_to_ids(self.config.start_img)
        self.end_img_id = self.tokenizer.convert_tokens_to_ids(self.config.end_img)

        # LLM
        if self.config.use_llm_pretrained:
            self.accelerator.print(f"Loading pretrained LLM: {self.config.llm_name}")
            self.llm = AutoModelForCausalLM.from_pretrained(self.config.llm_name)
        else:
            self.accelerator.print(
                f"Initializing LLM from config: {self.config.llm_name}"
            )
            llm_config = AutoConfig.from_pretrained(self.config.llm_name)
            self.llm = AutoModelForCausalLM.from_config(llm_config)

        # Resize embeddings after adding tokens
        self.llm.resize_token_embeddings(len(self.tokenizer))

        # Projector
        self.image_proj = self._create_projection_layer()

        self.accelerator.print(
            f"Models ready. Vocab size: {len(self.tokenizer)} | #patch_tokens: {n_patch_tokens}"
        )

    def _create_projection_layer(self):
        @dataclass
        class TextConfig:
            hidden_size = self.llm.config.hidden_size

        @dataclass
        class VisionConfig:
            model_args = {
                "embed_dim": getattr(
                    self.image_enc, "embed_dim", self.llm.config.hidden_size
                )
            }

        @dataclass
        class PerceptionLMConfig:
            vision_config = VisionConfig()
            text_config = TextConfig()
            projector_pooling_ratio = 2

        return PerceptionLMMultiModalProjector(PerceptionLMConfig())

    # ---- data

    def _setup_data(self):
        self.accelerator.print("Setting up dataset stream...")
        raw_ds = load_dataset(
            self.config.dataset_name,
            split=self.config.dataset_split,
            streaming=self.config.streaming,
            trust_remote_code=True,
        ).shuffle(seed=42)

        train_ds = MixedDataset(
            self.tokenizer,
            self.patch_tokens,
            self.config.max_length,
            raw_ds,
            self.config,
        )

        self.train_loader = DataLoader(
            train_ds,
            batch_size=self.config.batch_size,
            collate_fn=self._collate_fn,
            num_workers=self.config.num_workers,
            pin_memory=True,
            prefetch_factor=8,
        )

    def _collate_fn(self, batch):
        images = [b["mixed_image"]["image"].convert("RGB") for b in batch]
        pixel_values = torch.stack(
            [self.img_transform(img) for img in images]
        )  # CPU tensor

        full_ids = torch.stack([b["full_text"]["input_ids"] for b in batch])
        full_mask = torch.stack([b["full_text"]["attention_mask"] for b in batch])
        mixed_ids = torch.stack([b["mixed_image"]["input_ids"] for b in batch])
        mixed_mask = torch.stack([b["mixed_image"]["attention_mask"] for b in batch])

        end_tok_id = self.tokenizer.convert_tokens_to_ids(self.config.end_img)
        post_start = [
            (seq == end_tok_id).nonzero(as_tuple=True)[0][0].item() + 1
            if (seq == end_tok_id).any()
            else int(seq.ne(self.tokenizer.pad_token_id).sum().item())
            for seq in mixed_ids
        ]
        post_start = torch.tensor(post_start, dtype=torch.long)

        return {
            "full_ids": full_ids,
            "full_mask": full_mask,
            "mixed_ids": mixed_ids,
            "mixed_mask": mixed_mask,
            "pixel_values": pixel_values,
            "post_start": post_start,
        }

    # ---- optim

    def _setup_training(self):
        params = []
        params += list(self.image_proj.parameters())
        params += list(self.llm.parameters())

        self.optimizer = torch.optim.AdamW(params, lr=self.config.lr, weight_decay=5e-5)
        self.scheduler = get_constant_schedule_with_warmup(
            self.optimizer, num_warmup_steps=self.config.warmup_steps
        )

    # ---- helpers

    def _make_input_embeds(
        self, mixed_ids: torch.Tensor, pixel_values: torch.Tensor
    ) -> torch.Tensor:
        base = self.accelerator.unwrap_model(self.llm)
        tok_emb = base.get_input_embeddings()(mixed_ids)

        with torch.no_grad():
            img_enc = self.image_enc.forward_features(pixel_values)[
                :, 1:, :
            ]  # drop cls
        img_emb = self.image_proj(img_enc).to(tok_emb.dtype)

        patch_ids = self.patch_ids.to(mixed_ids.device)
        mask = torch.isin(mixed_ids, patch_ids)
        b_idx, pos_idx = torch.where(mask)

        # assign in-order per batch
        for b in range(img_emb.size(0)):
            idx = pos_idx[b_idx == b]
            if idx.numel() > 0:
                tok_emb[b, idx] = img_emb[b, : idx.numel()]

        return tok_emb

    def _decode_image_for_wandb(self, img_tensor: torch.Tensor) -> Image.Image:
        # img_tensor: (3,H,W) normalized to ImageNet mean/std by timm transform
        mean = torch.tensor([0.485, 0.456, 0.406], device=img_tensor.device).view(
            3, 1, 1
        )
        std = torch.tensor([0.229, 0.224, 0.225], device=img_tensor.device).view(
            3, 1, 1
        )
        x = torch.clamp(img_tensor * std + mean, 0, 1)
        arr = (x.permute(1, 2, 0).detach().cpu().numpy() * 255).astype("uint8")
        return Image.fromarray(arr)

    # ---- sampling

    @torch.no_grad()
    def _generate_image_conditioned_sample(
        self, step: int, batch: Dict[str, torch.Tensor]
    ):
        base = self.accelerator.unwrap_model(self.llm)

        pixel_values = batch["pixel_values"][:1].to(self.device)
        mixed_ids = batch["mixed_ids"][:1].to(self.device)
        post_start = batch["post_start"][:1]

        prompt_ids = mixed_ids[:, : post_start[0].item()]
        attn = prompt_ids.ne(self.tokenizer.pad_token_id).long()

        input_embeds = self._make_input_embeds(prompt_ids, pixel_values)
        gen = base.generate(
            inputs_embeds=input_embeds,
            attention_mask=attn,
            max_new_tokens=64,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        post_ids = mixed_ids[:, post_start[0].item() :]
        post_text = self.tokenizer.decode(post_ids[0], skip_special_tokens=True)
        decoded = self.tokenizer.decode(gen[0], skip_special_tokens=False)
        pil_image = self._decode_image_for_wandb(pixel_values[0])

        if self.accelerator.is_main_process:
            self.accelerator.log(
                {
                    "samples/img_conditioned": wandb.Table(
                        data=[
                            [
                                post_text,
                                wandb.Image(pil_image, caption=f"step {step}"),
                                decoded,
                            ]
                        ],
                        columns=["target_text", "input_image", "generated_text"],
                    ),
                    "step": step,
                }
            )

    @torch.no_grad()
    def _generate_text_only_sample(self, step: int, batch: Dict[str, torch.Tensor]):
        base = self.accelerator.unwrap_model(self.llm)

        full_ids = batch["full_ids"][:1].to(self.device)
        full_mask = batch["full_mask"][:1].to(self.device)
        L = full_mask.sum(dim=1).item()
        cut = max(8, int(0.6 * L))  # 60% prefix

        prompt_ids = full_ids[:, :cut]
        attn = prompt_ids.ne(self.tokenizer.pad_token_id).long()
        gen = base.generate(
            input_ids=prompt_ids,
            attention_mask=attn,
            max_new_tokens=64,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        prompt_text = self.tokenizer.decode(prompt_ids[0], skip_special_tokens=False)
        out_text = self.tokenizer.decode(gen[0], skip_special_tokens=False)

        if self.accelerator.is_main_process:
            self.accelerator.log(
                {
                    "samples/text_only": wandb.Table(
                        data=[[prompt_text, out_text]],
                        columns=["prompt", "generated_text"],
                    ),
                    "step": step,
                }
            )

    # ---- train

    def train(self):
        if self.accelerator.is_main_process:
            self.accelerator.init_trackers(
                "unsupervised-vlm", config=asdict(self.config)
            )

        self.llm.train()
        self.image_proj.train()
        global_step = 0

        self.accelerator.print("Starting training...")
        for epoch in range(self.config.num_epochs):
            for batch in self.train_loader:
                # to device
                batch = {
                    k: (
                        v.to(self.device, non_blocking=True)
                        if torch.is_tensor(v)
                        else v
                    )
                    for k, v in batch.items()
                }

                with self.accelerator.accumulate(self.llm):
                    with self.accelerator.autocast():
                        # Text-only LM loss (mask pads)
                        text_labels = batch["full_ids"].masked_fill(
                            batch["full_ids"] == self.tokenizer.pad_token_id, -100
                        )
                        text_out = self.llm(
                            input_ids=batch["full_ids"],
                            attention_mask=batch["full_mask"],
                            labels=text_labels,
                        )
                        text_loss = text_out.loss

                        # Image-conditioned loss
                        tok_emb = self._make_input_embeds(
                            batch["mixed_ids"], batch["pixel_values"]
                        )
                        labels = batch["mixed_ids"].clone()
                        labels[labels == self.tokenizer.pad_token_id] = -100
                        patch_ids = self.patch_ids.to(labels.device)
                        labels[torch.isin(labels, patch_ids)] = -100
                        labels[labels == self.start_img_id] = -100
                        labels[labels == self.end_img_id] = -100

                        out = self.llm(
                            inputs_embeds=tok_emb,
                            attention_mask=batch["mixed_mask"],
                            labels=labels,
                        )
                        img_loss = out.loss

                        loss = 0.2 * text_loss + 0.8 * img_loss

                    self.accelerator.backward(loss)

                    if self.accelerator.sync_gradients:
                        self.optimizer.step()
                        self.optimizer.zero_grad(set_to_none=True)
                        self.scheduler.step()

                # Logging
                if (global_step % self.config.log_every) == 0:
                    if self.accelerator.is_main_process:
                        self.accelerator.log(
                            {
                                "loss": loss.item(),
                                "lr": self.scheduler.get_last_lr()[0],
                                "step": global_step,
                                "text_loss": text_loss.item(),
                                "img_loss": img_loss.item(),
                            },
                            step=global_step,
                        )
                    self.accelerator.print(
                        f"step {global_step:6d} | loss {loss.item():.4f} | text {text_loss.item():.4f} | img {img_loss.item():.4f}"
                    )

                # Checkpointing + samples
                if self.accelerator.is_main_process and global_step > 0:
                    if (global_step % self.config.save_every) == 0:
                        ckpt_path = self.checkpoint_manager.save_checkpoint(
                            global_step, self.llm, self.tokenizer, self.image_proj
                        )
                        self.accelerator.print(f"Saved checkpoint: {ckpt_path}")

                    if (global_step % self.config.generate_every) == 0:
                        # pull a small batch for sampling from the same loader
                        # (safe since called sparsely)
                        try:
                            sample_batch = batch  # reuse latest batch
                            self._generate_image_conditioned_sample(
                                global_step, sample_batch
                            )
                            self._generate_text_only_sample(global_step, sample_batch)
                        except Exception as e:
                            self.accelerator.print(
                                f"Sampling error at step {global_step}: {e}"
                            )

                global_step += 1
                if global_step >= self.config.max_steps:
                    break

            if global_step >= self.config.max_steps:
                break

        self._save_final_model()
        self.accelerator.print("Training completed!")

    def _save_final_model(self):
        if not self.accelerator.is_main_process:
            return
        final_dir = Path(self.config.final_model_dir)
        final_dir.mkdir(exist_ok=True, parents=True)
        base = self.accelerator.unwrap_model(self.llm)
        base.save_pretrained(final_dir)
        self.tokenizer.save_pretrained(final_dir)
        self.accelerator.save(self.image_proj.state_dict(), final_dir / "image_proj.pt")
        self.accelerator.print(f"Final model saved to: {final_dir}")


# --------------------------- Main ---------------------------


def main():
    cfg = TrainingConfig()

    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.grad_acc_steps,
        mixed_precision=cfg.mixed_precision,  # "bf16" | "fp16" | "no"
        log_with="wandb",
    )

    trainer = VisionLanguageTrainer(cfg, accelerator)
    trainer.train()


if __name__ == "__main__":
    main()
