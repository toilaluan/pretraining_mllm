from PIL import Image, ImageDraw, ImageFont
import textwrap
import os


def create_note_image(
    text,
    background_image=None,
    width=800,
    height=600,
    output_path="note_image.png",
    font_path="ARIAL.TTF",
    text_color="black",
    padding=5,
):
    """
    Create a note image with dynamic font sizing that fills the available space.

    Args:
        text (str): The text content to render
        background_image (str, optional): Path to background image file
        width (int): Canvas width in pixels (default: 800)
        height (int): Canvas height in pixels (default: 600)
        output_path (str): Output file path (default: "note_image.png")
        font_path (str, optional): Path to custom font file (.ttf/.otf)
        text_color (str): Text color (default: "black")
        padding (int): Padding from edges in pixels (default: 40)

    Returns:
        PIL.Image: The generated image object
    """

    # Create base image
    if background_image and os.path.exists(background_image):
        # Load and resize background image
        bg_img = Image.open(background_image)
        bg_img = bg_img.resize((width, height), Image.Resampling.LANCZOS)
        img = bg_img.copy()

        # Add semi-transparent overlay for better text readability
        overlay = Image.new("RGBA", (width, height), (255, 255, 255, 25))
        img = Image.alpha_composite(img.convert("RGBA"), overlay)
        img = img.convert("RGB")

        # Adjust text color for background images
        if text_color == "black":
            text_color = "white"
    else:
        # Create white background
        img = Image.new("RGB", (width, height), "white")

    draw = ImageDraw.Draw(img)

    # Calculate available space
    max_width = width - (2 * padding)
    max_height = height - (2 * padding)

    # Find optimal font size
    font_size = find_optimal_font_size(draw, text, max_width, max_height, font_path)

    # Load font
    try:
        if font_path and os.path.exists(font_path):
            font = ImageFont.truetype(font_path, font_size)
        else:
            # Try to use system fonts
            try:
                font = ImageFont.truetype("arial.ttf", font_size)
            except OSError:
                try:
                    font = ImageFont.truetype(
                        "/System/Library/Fonts/Arial.ttf", font_size
                    )  # macOS
                except OSError:
                    try:
                        font = ImageFont.truetype(
                            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size
                        )  # Linux
                    except OSError:
                        font = ImageFont.load_default()
    except OSError:
        font = ImageFont.load_default()

    # Render text
    render_text_block(
        draw,
        text,
        padding,
        padding,
        max_width,
        font,
        text_color,
        background_image is not None,
    )

    # Save image
    img.save(output_path, "PNG", quality=95)

    return img


def find_optimal_font_size(draw, text, max_width, max_height, font_path=None):
    """
    Use binary search to find the optimal font size that fits the text in the given dimensions.
    """
    min_size = 12
    max_size = 200
    best_size = min_size

    while min_size <= max_size:
        current_size = (min_size + max_size) // 2

        # Load font with current size
        try:
            if font_path and os.path.exists(font_path):
                font = ImageFont.truetype(font_path, current_size)
            else:
                try:
                    font = ImageFont.truetype("arial.ttf", current_size)
                except OSError:
                    try:
                        font = ImageFont.truetype(
                            "/System/Library/Fonts/Arial.ttf", current_size
                        )
                    except OSError:
                        try:
                            font = ImageFont.truetype(
                                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                                current_size,
                            )
                        except OSError:
                            font = ImageFont.load_default()
        except OSError:
            font = ImageFont.load_default()

        # Measure text block
        text_width, text_height = measure_text_block(draw, text, max_width, font)

        if text_width <= max_width and text_height <= max_height:
            best_size = current_size
            min_size = current_size + 1
        else:
            max_size = current_size - 1

    return max(best_size, 12)  # Ensure minimum readable size


def measure_text_block(draw, text, max_width, font):
    """
    Measure the dimensions of a text block with word wrapping.
    """
    words = text.split()
    lines = []
    current_line = ""
    max_line_width = 0

    for word in words:
        test_line = current_line + (" " if current_line else "") + word
        bbox = draw.textbbox((0, 0), test_line, font=font)
        line_width = bbox[2] - bbox[0]

        if line_width <= max_width:
            current_line = test_line
            max_line_width = max(max_line_width, line_width)
        else:
            if current_line:
                lines.append(current_line)
                current_line = word
            else:
                # Single word too long, add it anyway
                lines.append(word)
                current_line = ""

    if current_line:
        lines.append(current_line)
        bbox = draw.textbbox((0, 0), current_line, font=font)
        max_line_width = max(max_line_width, bbox[2] - bbox[0])

    # Calculate total height
    if lines:
        bbox = draw.textbbox((0, 0), "Ay", font=font)  # Get line height
        line_height = bbox[3] - bbox[1]
        total_height = len(lines) * line_height * 1.4  # 1.4x spacing
    else:
        total_height = 0

    return max_line_width, total_height


def render_text_block(draw, text, x, y, max_width, font, color, has_background=False):
    """
    Render text with word wrapping, left-aligned, top to bottom.
    """
    words = text.split()
    lines = []
    current_line = ""

    # Create lines with word wrapping
    for word in words:
        test_line = current_line + (" " if current_line else "") + word
        bbox = draw.textbbox((0, 0), test_line, font=font)
        line_width = bbox[2] - bbox[0]

        if line_width <= max_width:
            current_line = test_line
        else:
            if current_line:
                lines.append(current_line)
                current_line = word
            else:
                lines.append(word)
                current_line = ""

    if current_line:
        lines.append(current_line)

    # Calculate line height
    bbox = draw.textbbox((0, 0), "Ay", font=font)
    line_height = int((bbox[3] - bbox[1]) * 1.4)  # 1.4x spacing

    # Draw each line
    current_y = y
    for line in lines:
        if has_background:
            # Add stroke/outline for better visibility on background images
            for adj in range(-2, 3):
                for adj2 in range(-2, 3):
                    if adj != 0 or adj2 != 0:
                        draw.text(
                            (x + adj, current_y + adj2), line, font=font, fill="black"
                        )

        draw.text((x, current_y), line, font=font, fill=color)
        current_y += line_height


# Example usage and test function
def example_usage():
    """
    Example usage of the create_note_image function.
    """

    # Example 1: Simple white background
    text1 = """Welcome to the Dynamic Note Image Generator!

This Python function creates beautiful text overlays with automatic font sizing that adapts to fill your canvas while maintaining readability.

Perfect for quotes, notes, social media posts, and more!"""

    img1 = create_note_image(
        text=text1,
        width=512,
        height=512,
        output_path="example_white_bg.png",
        padding=10,
    )
    print("Created example_white_bg.png")

    # Example 2: Custom dimensions and color
    text2 = "Short quote with custom styling"

    img2 = create_note_image(
        text=text2,
        width=512,
        height=512,
        text_color="navy",
        padding=60,
        output_path="example_custom.png",
    )
    print("Created example_custom.png")

    # Example 3: With background image (uncomment if you have a background image)
    # img3 = create_note_image(
    #     text="Text overlay on background image",
    #     background_image="background.jpg",
    #     width=800,
    #     height=600,
    #     output_path="example_with_bg.png"
    # )
    # print("Created example_with_bg.png")


if __name__ == "__main__":
    example_usage()
