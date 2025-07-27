from transformers import AutoTokenizer
import random
from PIL import Image, ImageDraw, ImageFont
import textwrap
from typing import Dict, List, Union, Tuple, Any
import torch


def create_note_image(text: str, size: Tuple[int, int] = (512, 512)) -> Image.Image:
    """
    Create a note-style image from text with proper formatting and readability.
    """
    # Create image with white background
    img = Image.new("RGB", size, color="white")
    draw = ImageDraw.Draw(img)

    # Try to use a bold sans-serif font, fallback to default
    try:
        # Try different font paths for cross-platform compatibility
        font_paths = [
            "/System/Library/Fonts/Arial Bold.ttf",  # macOS
            "C:\\Windows\\Fonts\\arialbd.ttf",  # Windows
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",  # Linux
        ]
        font = None
        for path in font_paths:
            try:
                font = ImageFont.truetype(path, 24)
                break
            except (OSError, IOError):
                continue

        if font is None:
            font = ImageFont.load_default()
    except:
        font = ImageFont.load_default()

    # Calculate margins
    margin = 40
    max_width = size[0] - 2 * margin
    max_height = size[1] - 2 * margin

    # Clean and prepare text
    text = text.strip()
    if not text:
        text = "Empty text"

    # Wrap text to fit image width
    wrapped_lines = []
    for paragraph in text.split("\n"):
        if paragraph.strip():
            # Estimate characters per line based on font size
            avg_char_width = font.getbbox("A")[2] - font.getbbox("A")[0]
            chars_per_line = max(1, max_width // avg_char_width)
            lines = textwrap.wrap(paragraph, width=chars_per_line)
            wrapped_lines.extend(lines)
        else:
            wrapped_lines.append("")  # Preserve empty lines

    # Calculate line height
    line_height = font.getbbox("Ag")[3] - font.getbbox("Ag")[1] + 6

    # Limit lines to fit in image
    max_lines = max_height // line_height
    if len(wrapped_lines) > max_lines:
        wrapped_lines = wrapped_lines[: max_lines - 1] + ["..."]

    # Draw text
    y = margin
    for line in wrapped_lines:
        if y + line_height > size[1] - margin:
            break
        draw.text((margin, y), line, fill="black", font=font)
        y += line_height

    return img


def setup_tokenizer(
    tokenizer: AutoTokenizer,
    start_image_token: str,
    end_image_token: str,
    patch_tokens: List[str],
) -> AutoTokenizer:
    """
    Setup tokenizer with special tokens for image processing.
    """
    # Add image boundary tokens
    special_tokens = [start_image_token, end_image_token]

    # Add patch tokens
    special_tokens.extend(patch_tokens)

    # Add all special tokens at once
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

    return tokenizer


def mixed_image_tokenize(
    tokenizer: AutoTokenizer,
    text: str,
    patch_tokens: List[str],
    start_image_token: str = "<image>",
    end_image_token: str = "</image>",
    mid_token_range: Tuple[float, float] = (0.1, 0.9),
    padding: bool = True,
    truncation: bool = True,
    max_length: int = 2048,
    return_tensors: str = "pt",
) -> Dict[str, Union[Dict[str, List[int]], Image.Image, str]]:
    """
    Tokenize text and replace a middle section with an image representation.

    Args:
        tokenizer: The tokenizer to use
        text: Input text to process
        tokenize_args: Arguments for tokenization
        start_image_token: Token to mark start of image
        end_image_token: Token to mark end of image
        patch_tokens: List of patch tokens to represent image content
        mid_token_range: Range for selecting middle section (min_ratio, max_ratio)

    Returns:
        Dictionary containing full text tokens and mixed image representation
    """
    # Tokenize the full text
    tokenizer_output = tokenizer(
        text,
        add_special_tokens=False,
        padding=padding,
        truncation=truncation,
        max_length=max_length,
        return_tensors=return_tensors,
    )

    full_input_ids = tokenizer_output.input_ids.squeeze()
    full_attention_mask = tokenizer_output.attention_mask.squeeze()

    output = {
        "full_text": {
            "input_ids": full_input_ids,
            "attention_mask": full_attention_mask,
        },
        "mixed_image": {},
    }

    valid_length = full_attention_mask.sum()

    if valid_length < 3:  # Need at least 3 tokens to create a meaningful middle section
        # Fallback: use the entire sequence as image
        mid_start = 0
        mid_end = valid_length
    else:
        mid_ratio = random.uniform(mid_token_range[0], mid_token_range[1])
        mid_length = max(1, int(mid_ratio * valid_length))
        max_start = max(0, valid_length - mid_length)
        mid_start = random.randint(0, max_start) if max_start > 0 else 0
        mid_end = min(mid_start + mid_length, valid_length)

    # Split the sequence
    pre_ids = full_input_ids[:mid_start]
    mid_ids = full_input_ids[mid_start:mid_end]
    post_ids = full_input_ids[mid_end:]

    # Convert middle section to text and create image
    mid_text = tokenizer.decode(
        mid_ids,
        skip_special_tokens=True,
    )
    image = create_note_image(mid_text)
    start_image_token_id = tokenizer.encode(start_image_token, add_special_tokens=False)
    end_image_token_id = tokenizer.encode(end_image_token, add_special_tokens=False)

    # Create image token sequence
    image_token_text = "".join(patch_tokens)
    image_token_ids = tokenizer.encode(
        image_token_text,
        add_special_tokens=False,
        return_tensors="pt",
        padding=False,
        truncation=False,
    ).squeeze()

    # Create new attention mask
    pre_attention = full_attention_mask[:mid_start]
    image_attention = torch.ones(len(image_token_ids))
    post_attention = full_attention_mask[mid_end:]

    # Store mixed image representation
    output["mixed_image"] = {
        "input_ids": torch.cat(
            [
                pre_ids,
                torch.tensor(start_image_token_id),
                image_token_ids,
                torch.tensor(end_image_token_id),
                post_ids,
            ]
        ),
        "attention_mask": torch.cat([pre_attention, image_attention, post_attention]),
        "image": image,
        "original_text": mid_text,
        "image_token_ids": image_token_ids,
    }

    return output


def test_mixed_image_tokenize():
    """
    Test function for mixed_image_tokenize
    """
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

    # Set pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Setup special tokens
    start_token = "<image>"
    end_token = "</image>"
    patch_tokens = [f"<patch_{i}>" for i in range(16)]

    print("Setting up tokenizer with special tokens...")
    tokenizer = setup_tokenizer(tokenizer, start_token, end_token, patch_tokens)

    # Test text
    text = (
        """
For best results, use bold sans-serif fonts (Arial Bold, Helvetica Bold)
System font detection works on Windows/macOS/Linux
Handles edge cases: single-word lines, long words, minimal text
Maintains 512x512 output size exactly
White space optimized for maximum readability
Uses proportional font metrics for accurate spacing
The function balances text density with readability, ensuring your text fills the available space while maintaining comfortable reading characteristics. The justification algorithm minimizes visual gaps while keeping letter spacing natural and readable.
"""
        * 2
    )

    print("Processing text...")
    result = mixed_image_tokenize(tokenizer, text, patch_tokens)

    # Print results
    print(f"\nOriginal text length: {len(text)}")
    print(f"Full text token count: {len(result['full_text']['input_ids'])}")
    print(f"Mixed image token count: {len(result['mixed_image']['input_ids'])}")
    print(f"Image represents text: '{result['mixed_image']['original_text'][:100]}...'")

    # Save the generated image
    if result["mixed_image"]["image"]:
        result["mixed_image"]["image"].save("test_note.png")
        print("Image saved as 'test_note.png'")

    # Verify token consistency
    original_non_pad = [
        t for t in result["full_text"]["input_ids"] if t != tokenizer.pad_token_id
    ]
    mixed_non_pad = [
        t for t in result["mixed_image"]["input_ids"] if t != tokenizer.pad_token_id
    ]

    print(f"Original non-pad tokens: {len(original_non_pad)}")
    print(f"Mixed non-pad tokens: {len(mixed_non_pad)}")
    print(f"Image token IDs: {result['mixed_image']['image_token_ids']}")

    # Test edge cases
    print("\nTesting edge cases...")

    # Short text
    short_result = mixed_image_tokenize(tokenizer, "Hello world!", patch_tokens)
    print(
        f"Short text processed successfully: {len(short_result['mixed_image']['input_ids'])} tokens"
    )

    # Empty text
    try:
        empty_result = mixed_image_tokenize(tokenizer, "")
        print("Empty text handled successfully")
    except Exception as e:
        print(f"Empty text error: {e}")

    # Very long text
    long_text = text * 10
    long_result = mixed_image_tokenize(tokenizer, long_text, patch_tokens)
    print(
        f"Long text processed successfully: {len(long_result['mixed_image']['input_ids'])} tokens"
    )

    return result


if __name__ == "__main__":
    result = test_mixed_image_tokenize()
    print("\nTest completed successfully!")
