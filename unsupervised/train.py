"""
train.py – Refactored and Cleaned
- AMP mixed precision with torch.cuda.amp
- Sample generation every cfg.generate_every steps
- Keep only last k checkpoints to save disk space
- Cleaner code organization and error handling
- Better logging and configuration management
"""

import os
import math
import shutil
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModel,
    AutoImageProcessor,
    AutoConfig,
    get_linear_schedule_with_warmup,
)
from transformers.models.perception_lm.modeling_perception_lm import PerceptionLMMultiModalProjector
from datasets import load_dataset
from PIL import Image
import wandb
import timm

from .tokenizing import setup_tokenizer, mixed_image_tokenize


@dataclass
class TrainingConfig:
    """Training configuration with all hyperparameters."""
    # Model configs
    llm_name: str = "Qwen/Qwen3-0.6B"
    use_llm_pretrained: bool = False
    image_enc_name: str = "timm/vit_pe_core_large_patch14_336.fb"
    start_img: str = "<image>"
    end_img: str = "</image>"
    
    # Training hyperparameters
    batch_size: int = 2
    grad_acc_steps: int = 1
    max_length: int = 1024
    lr: float = 5e-4
    num_epochs: int = 1
    warmup_steps: int = 1000
    max_samples: int = 2_000_000
    
    # Checkpointing and logging
    save_every: int = 10000
    generate_every: int = 500
    log_every: int = 25
    max_checkpoints: int = 1  # Only keep last k checkpoints
    
    # Dataset
    dataset_name: str = "HuggingFaceFW/fineweb"
    dataset_split: str = "train"
    streaming: bool = True
    
    # Hardware and precision
    device: str = "cuda"
    mixed_precision: bool = True
    amp_dtype: torch.dtype = torch.bfloat16
    num_workers: int = 0
    
    # Paths
    checkpoint_dir: str = "checkpoints"
    final_model_dir: str = "final_model"


class CheckpointManager:
    """Manages checkpoints and keeps only the last k versions."""
    
    def __init__(self, base_dir: str, max_checkpoints: int = 3):
        self.base_dir = Path(base_dir)
        self.max_checkpoints = max_checkpoints
        self.base_dir.mkdir(exist_ok=True)
    
    def save_checkpoint(self, step: int, llm, tokenizer, image_proj) -> str:
        """Save checkpoint and clean up old ones."""
        ckpt_dir = self.base_dir / f"step_{step}"
        ckpt_dir.mkdir(exist_ok=True)
        
        # Save models
        llm.save_pretrained(ckpt_dir)
        tokenizer.save_pretrained(ckpt_dir)
        torch.save(image_proj.state_dict(), ckpt_dir / "image_proj.pt")
        
        # Clean up old checkpoints
        self._cleanup_old_checkpoints()
        
        return str(ckpt_dir)
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints, keeping only the last k."""
        checkpoints = []
        for path in self.base_dir.iterdir():
            if path.is_dir() and path.name.startswith("step_"):
                try:
                    step = int(path.name.split("_")[1])
                    checkpoints.append((step, path))
                except (ValueError, IndexError):
                    continue
        
        # Sort by step number and remove old ones
        checkpoints.sort(key=lambda x: x[0])
        while len(checkpoints) > self.max_checkpoints:
            step, path = checkpoints.pop(0)
            shutil.rmtree(path)
            print(f"Removed old checkpoint: {path}")


class MixedDataset(Dataset):
    """Dataset that combines text and images for vision-language training."""
    
    def __init__(self, tokenizer, patch_tokens: List[str], max_len: int, stream, config: TrainingConfig):
        self.tokenizer = tokenizer
        self.patch_tokens = patch_tokens
        self.max_len = max_len
        self.stream = iter(stream)
        self.config = config

    def __len__(self):
        return self.config.max_samples

    def __getitem__(self, _):
        try:
            sample = next(self.stream)
            text = sample["text"]
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
            # Reset iterator if stream ends
            self.stream = iter(load_dataset(
                self.config.dataset_name,
                split=self.config.dataset_split,
                streaming=self.config.streaming,
                trust_remote_code=True
            ))
            return self.__getitem__(_)


class VisionLanguageTrainer:
    """Main trainer class for vision-language model."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.checkpoint_manager = CheckpointManager(config.checkpoint_dir, config.max_checkpoints)
        self._setup_models()
        self._setup_data()
        self._setup_training()
        
    def _setup_models(self):
        """Initialize all models and tokenizer."""
        print("Setting up models...")
        
        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.llm_name, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Setup vision patch tokens
        self.image_enc = timm.create_model(self.config.image_enc_name, pretrained=True)
        for p in self.image_enc.parameters():
            p.requires_grad = False
        data_config = timm.data.resolve_model_data_config(self.image_enc)
        self.img_transform = timm.data.create_transform(**data_config, is_training=False)
        self.image_enc.eval()
        self.image_enc.to(self.config.device)
        
        n_patches = self.image_enc.pos_embed.shape[1]
        if n_patches % 2 == 1:
            n_patches -= 1
        # n_patches = n_patches // (2 * 2)
        patch_tokens = [f"<patch_{i}>" for i in range(n_patches)]
        self.tokenizer = setup_tokenizer(
            self.tokenizer, self.config.start_img, self.config.end_img, patch_tokens
        )
        self.patch_tokens = patch_tokens
        vocab_size = len(self.tokenizer)
        
        # Language model
        if self.config.use_llm_pretrained:
            print(f"Loading pretrained LLM from {self.config.llm_name}")
            self.llm = AutoModelForCausalLM.from_pretrained(self.config.llm_name).to(self.config.device)
        else:
            print(f"Loading LLM from scratch from {self.config.llm_name}")
            llm_config = AutoConfig.from_pretrained(self.config.llm_name)
            self.llm = AutoModelForCausalLM.from_config(llm_config).to(self.config.device)
        self.llm.resize_token_embeddings(vocab_size)

        # for p in self.llm.parameters():
        #     p.requires_grad = False
        
        # Vision model
        
        # Vision-language projection
        self.image_proj = self._create_projection_layer()
        self.patch_ids = torch.tensor(
            self.tokenizer.convert_tokens_to_ids(patch_tokens), 
            device=self.config.device
        )
        self.start_img_id = self.tokenizer.convert_tokens_to_ids(self.config.start_img)
        self.end_img_id = self.tokenizer.convert_tokens_to_ids(self.config.end_img)
        
        print(f"Models loaded. Vocab size: {vocab_size}")
    
    def _create_projection_layer(self):
        """Create the vision-to-language projection layer."""
        @dataclass
        class TextConfig:
            hidden_size = self.llm.config.hidden_size

        @dataclass  
        class VisionConfig:
            model_args = {"embed_dim": self.image_enc.embed_dim}

        @dataclass
        class PerceptionLMConfig:
            vision_config = VisionConfig()
            text_config = TextConfig()
            projector_pooling_ratio = 1

        model = PerceptionLMMultiModalProjector(PerceptionLMConfig()).to(self.config.device)
        return model
    
    def _setup_data(self):
        """Setup dataset and dataloader."""
        print("Setting up dataset...")
        
        raw_ds = load_dataset(
            self.config.dataset_name,
            split=self.config.dataset_split,
            streaming=self.config.streaming,
            trust_remote_code=True
        )
        
        train_ds = MixedDataset(
            self.tokenizer, self.patch_tokens, self.config.max_length, raw_ds, self.config
        )
        
        self.train_loader = DataLoader(
            train_ds,
            batch_size=self.config.batch_size,
            collate_fn=self._collate_fn,
            num_workers=self.config.num_workers
        )
    

    def _collate_fn(self, batch):
        """Collate function for batching."""
        images = [b["mixed_image"]["image"].convert("RGB") for b in batch]
        pixel_values = torch.stack([self.img_transform(img) for img in images])

        full_ids = torch.stack([b["full_text"]["input_ids"] for b in batch])
        full_mask = torch.stack([b["full_text"]["attention_mask"] for b in batch])
        mixed_ids = torch.stack([b["mixed_image"]["input_ids"] for b in batch])
        mixed_mask = torch.stack([b["mixed_image"]["attention_mask"] for b in batch])

        end_tok_id = self.tokenizer.convert_tokens_to_ids(self.config.end_img)
        post_start = [
            (seq == end_tok_id).nonzero(as_tuple=True)[0][0].item() + 1
            if (seq == end_tok_id).any() else len(seq)
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
    def _setup_training(self):
        """Setup optimizer, scheduler, and scaler."""
        print("Setting up training components...")
        
        # Collect trainable parameters
        trainable_params = []
        trainable_params.extend(self.image_proj.parameters())
        trainable_params.extend(self.llm.parameters())
        
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.config.lr,
            weight_decay=5e-5,
        )
        
        total_steps = self.config.max_samples // (self.config.batch_size * self.config.grad_acc_steps)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=total_steps,
        )
        
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.config.mixed_precision)
    
    def _make_input_embeds(self, mixed_ids: torch.Tensor, pixel_values: torch.Tensor, max_length: int) -> torch.Tensor:
        """Create input embeddings by combining text and image tokens."""
        tok_emb = self.llm.model.embed_tokens(mixed_ids)
        
        with torch.no_grad():
            with torch.amp.autocast(enabled=self.config.mixed_precision, dtype=self.config.amp_dtype, device_type=self.config.device):
                img_enc = self.image_enc.forward_features(pixel_values)[:, 1:, :]
        img_emb = self.image_proj(img_enc)
        
        img_emb = img_emb.to(tok_emb.dtype)
        mask = torch.isin(mixed_ids, self.patch_ids)
        b_idx, pos_idx = torch.where(mask)
        
        for b in range(img_emb.size(0)):
            idx = pos_idx[b_idx == b]
            if len(idx) > 0:
                tok_emb[b, idx] = img_emb[b, :len(idx)]

        
        
        return tok_emb
    
    def _generate_sample(self, step: int):
        """Generate a sample for logging purposes."""
        self.llm.eval()
        
        try:
            batch = next(iter(self.train_loader))
            pixel_values = batch["pixel_values"][:1].to(self.config.device)
            mixed_ids = batch["mixed_ids"][:1].to(self.config.device)
            post_start = batch["post_start"][:1].to(self.config.device)
            
            # Truncate to post_start
            prompt_ids = mixed_ids[:, :post_start[0]]
            post_ids = mixed_ids[:, post_start[0]:]
            post_text = self.tokenizer.decode(post_ids[0], skip_special_tokens=True)
            prompt = self.tokenizer.decode(prompt_ids[0], skip_special_tokens=False)
            
            with torch.no_grad():
                input_embeds = self._make_input_embeds(prompt_ids, pixel_values, self.config.max_length)
                generated = self.llm.generate(
                    inputs_embeds=input_embeds,
                    max_new_tokens=64,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            
            decoded = self.tokenizer.decode(generated[0], skip_special_tokens=False)
            
            # Convert tensor to PIL Image for logging
            img_tensor = pixel_values[0]
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(img_tensor.device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(img_tensor.device)
            
            img_tensor = torch.clamp(img_tensor * std + mean, 0, 1)
            img_np = (img_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')
            pil_image = Image.fromarray(img_np)
            
            # Log to wandb
            wandb.log({
                "samples": wandb.Table(
                    data=[[post_text, wandb.Image(pil_image, caption=f"Step {step}"), decoded, prompt]],
                    columns=["target_text", "input_image", "generated_text", "full_prompt"]
                ),
                "step": step
            })
            
        except Exception as e:
            print(f"Error during sample generation: {e}")
        finally:
            self.llm.train()
    
    def train(self):
        """Main training loop."""
        wandb.init(project="unsupervised-vlm", config=self.config.__dict__)
        
        step = 0
        self.llm.train()
        self.image_proj.train()
        
        print("Starting training...")
        
        for epoch in range(self.config.num_epochs):
            for batch in self.train_loader:
                # Move batch to device
                batch = {k: v.to(self.config.device) if torch.is_tensor(v) else v 
                        for k, v in batch.items()}
                
                with torch.cuda.amp.autocast(enabled=self.config.mixed_precision, dtype=self.config.amp_dtype):
                    # Compute text loss
                    text_loss = self.llm(
                        input_ids=batch["full_ids"],
                        attention_mask=batch["full_mask"],
                        labels=batch["full_ids"]
                    )
                    text_loss = text_loss.loss
                    
                    # Compute loss
                    tok_emb = self._make_input_embeds(batch["mixed_ids"], batch["pixel_values"], self.config.max_length)
                    
                    # Prepare labels (mask everything before post_start)
                    labels = batch["mixed_ids"].clone()
                    
                    labels[labels == self.tokenizer.pad_token_id] = -100
                    labels[torch.isin(labels, self.patch_ids)] = -100
                    labels[labels == self.start_img_id] = -100
                    labels[labels == self.end_img_id] = -100
                    
                    # Forward pass
                    outputs = self.llm(
                        inputs_embeds=tok_emb,
                        attention_mask=batch["mixed_mask"],
                        labels=labels
                    )

                    img_loss = outputs.loss

                    loss = (0.2*text_loss + 0.8*img_loss) / self.config.grad_acc_steps
                
                # Backward pass
                # self.scaler.scale(loss).backward()
                loss.backward()
                
                # Optimizer step
                if (step + 1) % self.config.grad_acc_steps == 0:
                    # self.scaler.step(self.optimizer)
                    # self.scaler.update()
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.scheduler.step()
                
                # Logging
                if step % self.config.log_every == 0:
                    wandb.log({
                        "loss": loss.item() * self.config.grad_acc_steps,
                        "lr": self.scheduler.get_last_lr()[0],
                        "step": step,
                        "text_loss": text_loss.item(),
                        "img_loss": img_loss.item(),
                    })
                    print(f"Step {step:6d} | Loss: {loss.item() * self.config.grad_acc_steps:.4f} | Text Loss: {text_loss.item():.4f} | Img Loss: {img_loss.item():.4f}")
                
                # Checkpointing
                if step > 0 and step % self.config.save_every == 0:
                    ckpt_path = self.checkpoint_manager.save_checkpoint(
                        step, self.llm, self.tokenizer, self.image_proj
                    )
                    print(f"Saved checkpoint: {ckpt_path}")
                
                # Sample generation
                if step > 0 and step % self.config.generate_every == 0:
                    self._generate_sample(step)
                
                step += 1
                if step >= self.config.max_samples:
                    break
            
            if step >= self.config.max_samples:
                break
        
        # Final save
        self._save_final_model()
        print("Training completed!")
    
    def _save_final_model(self):
        """Save the final trained model."""
        final_dir = Path(self.config.final_model_dir)
        final_dir.mkdir(exist_ok=True)
        
        self.llm.save_pretrained(final_dir)
        self.tokenizer.save_pretrained(final_dir)
        torch.save(self.image_proj.state_dict(), final_dir / "image_proj.pt")
        
        print(f"Final model saved to: {final_dir}")


def main():
    """Main training function."""
    config = TrainingConfig()
    trainer = VisionLanguageTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()