from PIL import Image, ImageDraw, ImageFont
import os


def create_note_image(
    text,
    background_color=(255, 255, 255),
    text_color=(0, 0, 0),
    font_path=None,
    margin_lr=12,
    margin_tb=30,
):
    """
    Create a 512x512 note image with justified text.

    Args:
        text (str): Text content (4-512 tokens)
        background_color (tuple): RGB background color
        text_color (tuple): RGB text color
        font_path (str): Path to .ttf font file (optional)
        margin_lr (int): Left/right margin in pixels
        margin_tb (int): Top/bottom margin in pixels

    Returns:
        PIL.Image: Generated note image
    """
    # Constants
    IMG_SIZE = 512
    MIN_FONT_SIZE = 10
    MAX_FONT_SIZE = 40

    # Create base image
    img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), background_color)
    draw = ImageDraw.Draw(img)

    # Calculate available space
    available_width = IMG_SIZE - 2 * margin_lr
    available_height = IMG_SIZE - 2 * margin_tb

    # Handle empty text
    if not text.strip():
        return img

    # Find optimal font size
    font_size = MIN_FONT_SIZE
    best_lines = []

    for size in range(MAX_FONT_SIZE, MIN_FONT_SIZE - 1, -1):
        # Try to load font
        try:
            if font_path and os.path.exists(font_path):
                font = ImageFont.truetype(font_path, size)
            else:
                # Try system fonts in order of availability
                for fallback in [
                    "arialbd.ttf",
                    "arial.ttf",
                    "DejaVuSans-Bold.ttf",
                    "DejaVuSans.ttf",
                    "Helvetica",
                    "LiberationSans-Bold",
                ]:
                    try:
                        font = ImageFont.truetype(fallback, size)
                        break
                    except (OSError, ValueError):
                        continue
                else:
                    # Final fallback: default PIL font (fixed size)
                    font = ImageFont.load_default()
                    if size > 12:  # Default font doesn't scale well
                        continue
        except Exception:
            continue

        # Break text into lines
        words = text.split()
        lines = []
        current_line = []
        current_width = 0

        for word in words:
            word_width = font.getlength(word + " ")  # Include space width
            if current_width + word_width <= available_width:
                current_line.append(word)
                current_width += word_width
            else:
                lines.append(current_line)
                current_line = [word]
                current_width = word_width

        if current_line:
            lines.append(current_line)

        # Calculate total height
        line_spacing = size * 1.25
        total_height = (len(lines) * line_spacing) - (
            0.25 * size
        )  # Adjust for baseline

        # Check if fits in available height
        if total_height <= available_height:
            font_size = size
            best_lines = lines
            break

    # Final font setup
    try:
        if font_path and os.path.exists(font_path):
            font = ImageFont.truetype(font_path, font_size)
        else:
            for fallback in [
                "arialbd.ttf",
                "arial.ttf",
                "DejaVuSans-Bold.ttf",
                "DejaVuSans.ttf",
                "Helvetica",
                "LiberationSans-Bold",
            ]:
                try:
                    font = ImageFont.truetype(fallback, font_size)
                    break
                except (OSError, ValueError):
                    continue
            else:
                font = ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()

    # Draw text (justified for all lines except last)
    y = margin_tb
    line_spacing = font_size * 1.25

    for i, words in enumerate(best_lines):
        line_text = " ".join(words)
        line_width = font.getlength(line_text)

        # Last line: left-aligned
        if i == len(best_lines) - 1:
            draw.text((margin_lr, y), line_text, font=font, fill=text_color)
        # Justified lines
        else:
            # Calculate extra space to distribute
            extra_space = available_width - line_width
            num_spaces = max(1, len(words) - 1)
            space_width = font.getlength(" ")
            extra_per_space = extra_space / num_spaces

            x = margin_lr
            for j, word in enumerate(words):
                draw.text((x, y), word, font=font, fill=text_color)
                word_width = font.getlength(word)
                x += word_width + space_width

                # Distribute extra space
                if j < len(words) - 1:
                    x += extra_per_space

        y += line_spacing

    return img


if __name__ == "__main__":
    from PIL import Image

    # Create note with sample text
    # text = " ".join(["Hello"] * 100)  # ~100 tokens
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
    note = create_note_image(
        text,
        background_color=(240, 248, 255),  # AliceBlue
        text_color=(47, 79, 79),  # DarkSlateGray
        font_path="arialbd.ttf",  # Optional custom font
    )

    note.save("note.png")
