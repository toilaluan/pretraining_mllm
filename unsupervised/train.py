#!/usr/bin/env python
"""
train.py - Unsupervised Vision-Language Model Pretraining

This script implements unsupervised pretraining for a Vision-Language Model (VLM)
using the FineWeb dataset. The training approach uses a mixed loss function that
helps the LLM understand image embeddings from SigLIP without requiring image-text pairs.

Loss function: loss = text_only_loss * 0.5 + image_mixed_loss * 0.5

Key features:
- Uses SigLIP2 as the image encoder
- Processes FineWeb dataset for unsupervised training
- Implements batched training with mixed text/image representations
- Integrates with the provided tokenizing and image utilities
"""

import os
import time
import logging
import random
from typing import Dict, List, Tuple, Optional, Union

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    AutoModelForCausalLM,
    AutoConfig,
    get_scheduler,
    default_data_collator,
    set_seed,
)
from datasets import load_dataset
from tqdm import tqdm

# Import our custom tokenization and image utilities
from tokenizing import setup_tokenizer, mixed_image_tokenize, create_note_image
from image_utils import create_note_image as create_note_image_util

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# Configuration parameters
class TrainingConfig:
    # Model configuration
    base_model_name = "meta-llama/Llama-3.1-8B-Instruct"
    image_encoder_name = "google/siglip-so400m-patch14-384"

    # Token configuration
    start_image_token = "<image>"
    end_image_token = "</image>"
    num_patch_tokens = 256  # SigLIP typically uses 256 patch tokens

    # Dataset configuration
    dataset_name = "HuggingFaceFW/fineweb"
    dataset_subset = "sample-10BT"
    max_train_samples = None  # Set to integer to limit samples
    max_eval_samples = None

    # Training configuration
    per_device_train_batch_size = 2
    per_device_eval_batch_size = 1
    learning_rate = 2e-5
    weight_decay = 0.01
    num_train_epochs = 3
    max_seq_length = 2048
    gradient_accumulation_steps = 4
    max_grad_norm = 1.0
    warmup_ratio = 0.03
    logging_steps = 100
    eval_steps = 500
    save_steps = 1000
    output_dir = "./vlm_pretrained"
    seed = 42

    # Mixed training configuration
    text_only_ratio = 0.5  # Ratio of text-only samples in each batch
    mid_token_range = (0.1, 0.9)  # Range for selecting middle section for images

    def __init__(self):
        # Generate patch tokens
        self.patch_tokens = [f"<patch_{i}>" for i in range(self.num_patch_tokens)]

        # Calculate effective batch size
        self.train_batch_size = (
            self.per_device_train_batch_size * torch.cuda.device_count()
            if torch.cuda.is_available()
            else self.per_device_train_batch_size
        )
        self.eval_batch_size = (
            self.per_device_eval_batch_size * torch.cuda.device_count()
            if torch.cuda.is_available()
            else self.per_device_eval_batch_size
        )


class FineWebDataset(Dataset):
    """Dataset class for FineWeb that handles mixed text/image representations."""

    def __init__(
        self, dataset, tokenizer, config: TrainingConfig, split: str = "train"
    ):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.config = config
        self.split = split
        self.is_train = split == "train"

        # Setup special tokens in tokenizer if not already done
        if not hasattr(tokenizer, "_image_tokens_setup"):
            self.tokenizer = setup_tokenizer(
                tokenizer,
                config.start_image_token,
                config.end_image_token,
                config.patch_tokens,
            )
            tokenizer._image_tokens_setup = True

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        text = example["text"]

        # For training, randomly decide whether to use text-only or mixed representation
        if self.is_train and random.random() > self.config.text_only_ratio:
            # Mixed image-text representation
            try:
                return mixed_image_tokenize(
                    self.tokenizer,
                    text,
                    self.config.patch_tokens,
                    start_image_token=self.config.start_image_token,
                    end_image_token=self.config.end_image_token,
                    mid_token_range=self.config.mid_token_range,
                    max_length=self.config.max_seq_length,
                )["mixed_image"]
            except Exception as e:
                logger.warning(
                    f"Error processing mixed image: {e}. Falling back to text-only."
                )

        # Text-only representation
        return self.tokenizer(
            text,
            add_special_tokens=True,
            padding="max_length",
            truncation=True,
            max_length=self.config.max_seq_length,
            return_tensors="pt",
        )


class SigLIPImageEncoder(nn.Module):
    """Wrapper for SigLIP image encoder to extract image features."""

    def __init__(self, model_name: str = "google/siglip-so400m-patch14-384"):
        super().__init__()
        from transformers import SiglipVisionModel, SiglipVisionConfig

        logger.info(f"Loading SigLIP image encoder: {model_name}")
        self.config = SiglipVisionConfig.from_pretrained(model_name)
        self.model = SiglipVisionModel.from_pretrained(model_name)
        self.patch_size = self.config.patch_size

        # Freeze the image encoder parameters
        for param in self.model.parameters():
            param.requires_grad = False

        logger.info(f"SigLIP image encoder loaded with patch size: {self.patch_size}")

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Extract image features from pixel values."""
        with torch.no_grad():
            outputs = self.model(pixel_values=pixel_values)
            return outputs.last_hidden_state  # [batch_size, num_patches, hidden_size]


class VLMMixedModel(nn.Module):
    """Wrapper for the LLM model that handles mixed text/image inputs."""

    def __init__(self, llm_model, image_encoder, config: TrainingConfig):
        super().__init__()
        self.llm = llm_model
        self.image_encoder = image_encoder
        self.config = config

        # Get token IDs for special tokens
        self.start_image_token_id = config.start_image_token_id
        self.end_image_token_id = config.end_image_token_id
        self.patch_token_ids = config.patch_token_ids

        # Create a learnable projection from image features to LLM hidden size
        image_hidden_size = image_encoder.config.hidden_size
        llm_hidden_size = llm_model.config.hidden_size

        self.image_projection = nn.Linear(
            image_hidden_size, llm_hidden_size, bias=False
        )

        logger.info(
            f"Created image projection: {image_hidden_size} -> {llm_hidden_size}"
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for mixed text/image inputs.

        Args:
            input_ids: Token IDs, with image regions marked by special tokens
            attention_mask: Attention mask
            labels: Optional labels for computing loss
            pixel_values: Optional batch of images corresponding to the image regions

        Returns:
            Dictionary containing loss and logits
        """
        batch_size = input_ids.shape[0]
        seq_length = input_ids.shape[1]

        # First, get the LLM embeddings for all tokens
        inputs_embeds = self.llm.get_input_embeddings()(input_ids)

        # If we have image regions to process
        if pixel_values is not None and hasattr(self, "image_projection"):
            # Process images through SigLIP
            image_features = self.image_encoder(
                pixel_values
            )  # [batch_size, num_patches, hidden_size]
            projected_features = self.image_projection(
                image_features
            )  # [batch_size, num_patches, llm_hidden_size]

            # Replace patch token embeddings with projected image features
            for i in range(batch_size):
                # Find patch token positions in this sequence
                patch_positions = (input_ids[i] == self.patch_token_ids[0]).nonzero(
                    as_tuple=True
                )[0]

                if len(patch_positions) > 0:
                    # Replace patch token embeddings with image features
                    # Only take as many features as we have patch tokens
                    num_patches_to_use = min(
                        len(projected_features[i]), len(patch_positions)
                    )
                    inputs_embeds[i, patch_positions[:num_patches_to_use]] = (
                        projected_features[i, :num_patches_to_use]
                    )

        # Forward through the LLM
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
        )

        return outputs


def prepare_patch_token_ids(tokenizer, config: TrainingConfig) -> None:
    """Prepare token IDs for special image tokens."""
    # Get token IDs
    config.start_image_token_id = tokenizer.convert_tokens_to_ids(
        config.start_image_token
    )
    config.end_image_token_id = tokenizer.convert_tokens_to_ids(config.end_image_token)

    # Get patch token IDs
    config.patch_token_ids = [
        tokenizer.convert_tokens_to_ids(token) for token in config.patch_tokens
    ]

    logger.info(
        f"Special token IDs: start={config.start_image_token_id}, "
        f"end={config.end_image_token_id}, "
        f"patch tokens count={len(config.patch_token_ids)}"
    )


def create_dataloaders(config: TrainingConfig, tokenizer):
    """Create training and evaluation dataloaders."""
    # Load the dataset
    logger.info(
        f"Loading FineWeb dataset: {config.dataset_name} - {config.dataset_subset}"
    )
    dataset = load_dataset(config.dataset_name, config.dataset_subset, split="train")

    # Split into train and eval
    train_test_split = dataset.train_test_split(test_size=0.05, seed=config.seed)
    train_dataset = train_test_split["train"]
    eval_dataset = train_test_split["test"]

    # Limit samples if needed
    if config.max_train_samples is not None:
        train_dataset = train_dataset.select(range(config.max_train_samples))
    if config.max_eval_samples is not None:
        eval_dataset = eval_dataset.select(range(config.max_eval_samples))

    logger.info(f"Training set size: {len(train_dataset)}")
    logger.info(f"Evaluation set size: {len(eval_dataset)}")

    # Create dataset objects
    train_dataset = FineWebDataset(train_dataset, tokenizer, config, split="train")
    eval_dataset = FineWebDataset(eval_dataset, tokenizer, config, split="eval")

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=default_data_collator,
        batch_size=config.per_device_train_batch_size,
        num_workers=4,
    )

    eval_dataloader = DataLoader(
        eval_dataset,
        collate_fn=default_data_collator,
        batch_size=config.per_device_eval_batch_size,
        num_workers=2,
    )

    return train_dataloader, eval_dataloader


def train():
    """Main training function."""
    # Parse arguments and set up configuration
    config = TrainingConfig()

    # Set seed for reproducibility
    set_seed(config.seed)

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load tokenizer
    logger.info(f"Loading tokenizer: {config.base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model_name, use_fast=True, padding_side="right"
    )

    # Set pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Set pad token to eos_token")

    # Prepare special token IDs
    prepare_patch_token_ids(tokenizer, config)

    # Create dataloaders
    train_dataloader, eval_dataloader = create_dataloaders(config, tokenizer)

    # Load base LLM model
    logger.info(f"Loading base LLM model: {config.base_model_name}")
    llm_model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    # Load image encoder
    image_encoder = SigLIPImageEncoder(config.image_encoder_name)
    image_encoder.to(device)

    # Create VLM model
    model = VLMMixedModel(llm_model, image_encoder, config)

    # Move model to device
    model.to(device)

    # Prepare optimizer and scheduler
    optimizer = AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    # Calculate total training steps
    num_update_steps_per_epoch = (
        len(train_dataloader) // config.gradient_accumulation_steps
    )
    max_train_steps = num_update_steps_per_epoch * config.num_train_epochs

    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=int(max_train_steps * config.warmup_ratio),
        num_training_steps=max_train_steps,
    )

    # Training loop
    logger.info("***** Starting training *****")
    logger.info(f"  Num examples = {len(train_dataloader.dataset)}")
    logger.info(f"  Num Epochs = {config.num_train_epochs}")
    logger.info(
        f"  Instantaneous batch size per device = {config.per_device_train_batch_size}"
    )
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {config.train_batch_size * config.gradient_accumulation_steps}"
    )
    logger.info(f"  Gradient Accumulation steps = {config.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0

    # For tracking metrics
    start_time = time.time()

    for epoch in range(config.num_train_epochs):
        model.train()
        epoch_loss = 0.0
        steps_in_epoch = 0

        progress_bar = tqdm(
            total=num_update_steps_per_epoch,
            desc=f"Epoch {epoch + 1}/{config.num_train_epochs}",
        )

        for step, batch in enumerate(train_dataloader):
            # Prepare batch
            batch = {
                k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)
            }

            # Determine if this is a text-only or image-mixed batch
            has_images = "image" in batch and batch["image"] is not None

            # For image-mixed batches, we need to process the images
            pixel_values = None
            if has_images:
                # Convert PIL images to tensors
                images = batch.pop("image", None)
                if images is not None and len(images) > 0:
                    # Process each image in the batch
                    processed_images = []
                    for img in images:
                        if isinstance(img, Image.Image):
                            # Convert PIL image to tensor
                            img_tensor = torch.tensor(
                                torchvision.transforms.functional.to_tensor(img)
                            ).unsqueeze(0)
                            processed_images.append(img_tensor)

                    if processed_images:
                        pixel_values = torch.cat(processed_images, dim=0).to(device)

            # Forward pass
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["input_ids"].clone(),
                pixel_values=pixel_values,
            )

            loss = outputs.loss
            loss = loss / config.gradient_accumulation_steps
            tr_loss += loss.item()
            epoch_loss += loss.item()
            steps_in_epoch += 1

            # Backward pass
            loss.backward()

            # Update weights every gradient_accumulation_steps
            if (step + 1) % config.gradient_accumulation_steps == 0 or (
                step + 1
            ) == len(train_dataloader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # Log metrics
                if global_step % config.logging_steps == 0:
                    avg_loss = (tr_loss - logging_loss) / config.logging_steps
                    elapsed_time = time.time() - start_time
                    logger.info(
                        f"Step {global_step}, Loss: {avg_loss:.4f}, Time: {elapsed_time:.2f}s"
                    )
                    logging_loss = tr_loss

                progress_bar.update(1)
                progress_bar.set_postfix({"loss": loss.item()})

        # End of epoch
        avg_epoch_loss = epoch_loss / steps_in_epoch
        logger.info(f"Epoch {epoch + 1} completed. Average loss: {avg_epoch_loss:.4f}")

        # Evaluation
        if config.eval_steps > 0 and (epoch + 1) % 1 == 0:  # Evaluate every epoch
            model.eval()
            eval_loss = 0.0
            eval_steps = 0

            with torch.no_grad():
                for eval_step, eval_batch in enumerate(
                    tqdm(eval_dataloader, desc="Evaluating")
                ):
                    eval_batch = {
                        k: v.to(device)
                        for k, v in eval_batch.items()
                        if isinstance(v, torch.Tensor)
                    }

                    # Similar processing for eval batch as in training
                    has_images = (
                        "image" in eval_batch and eval_batch["image"] is not None
                    )
                    pixel_values = None

                    if has_images:
                        images = eval_batch.pop("image", None)
                        if images is not None and len(images) > 0:
                            processed_images = []
                            for img in images:
                                if isinstance(img, Image.Image):
                                    img_tensor = torch.tensor(
                                        torchvision.transforms.functional.to_tensor(img)
                                    ).unsqueeze(0)
                                    processed_images.append(img_tensor)

                            if processed_images:
                                pixel_values = torch.cat(processed_images, dim=0).to(
                                    device
                                )

                    outputs = model(
                        input_ids=eval_batch["input_ids"],
                        attention_mask=eval_batch["attention_mask"],
                        labels=eval_batch["input_ids"].clone(),
                        pixel_values=pixel_values,
                    )

                    eval_loss += outputs.loss.item()
                    eval_steps += 1

            avg_eval_loss = eval_loss / eval_steps
            logger.info(f"Epoch {epoch + 1} evaluation loss: {avg_eval_loss:.4f}")

        # Save checkpoint
        if config.output_dir is not None:
            output_dir = os.path.join(config.output_dir, f"epoch_{epoch + 1}")
            os.makedirs(output_dir, exist_ok=True)

            # Save model
            model.llm.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)

            logger.info(f"Checkpoint saved to {output_dir}")

    # Final save
    if config.output_dir is not None:
        final_dir = os.path.join(config.output_dir, "final_model")
        os.makedirs(final_dir, exist_ok=True)
        model.llm.save_pretrained(final_dir)
        tokenizer.save_pretrained(final_dir)
        logger.info(f"Final model saved to {final_dir}")


if __name__ == "__main__":
    try:
        # Make sure we can import the custom modules
        from tokenizing import create_note_image
        from image_utils import create_note_image as create_note_image_util
        import torchvision

        train()
    except ImportError as e:
        logger.error(f"Missing required module: {e.name}")
        logger.error("Please ensure all dependencies are installed:")
        logger.error("pip install transformers datasets torch torchvision tqdm Pillow")
        raise
