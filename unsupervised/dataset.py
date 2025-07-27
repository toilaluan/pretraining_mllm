from datasets import load_dataset
from torch.utils.data import Dataset
from .tokenizing import mixed_image_tokenize
from transformers import AutoTokenizer


class MixedImageDataset(Dataset):
    def __init__(
        self,
        dataset_name: str,
        tokenizer: AutoTokenizer,
        image_mixed_rate: float = 0.5,
        image_text_tokens: tuple[int, int] = (100, 500),
        start_image_token: str = "<image>",
        end_image_token: str = "</image>",
        tokenize_args: dict = {
            "add_special_tokens": False,
            "padding": "max_length",
            "truncation": True,
            "max_length": 2048,
        },
    ):
        self.dataset = load_dataset(dataset_name, split="train", num_proc=16)
        self.tokenizer = tokenizer
        self.image_mixed_rate = image_mixed_rate
        self.image_text_tokens = image_text_tokens
        self.start_image_token = start_image_token
        self.end_image_token = end_image_token
        self.tokenize_args = tokenize_args

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        text = item["text"]
        char_count = len(text)
        if char_count < self.image_text_tokens[1]:
            output = self.tokenizer(text, **self.tokenize_args)
            input_ids = output.input_ids
            labels = input_ids.copy()
            attention_mask = output.attention_mask
            text_as_image = None
        else:
            output = mixed_image_tokenize(
                self.tokenizer,
                text,
                self.image_text_tokens,
                self.start_image_token,
                self.end_image_token,
                self.tokenize_args,
            )
            input_ids = output["input_ids"]
            labels = output["labels"]
            attention_mask = output["attention_mask"]
            text_as_image = output["image"]

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "text_as_image": text_as_image,
        }
