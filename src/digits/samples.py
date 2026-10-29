"""Grid of training images as PNG (plan 9, D17).

train writes samples/augmented.png into the run folder: a batch exactly as the model gets it, for
the visual check on the workbench's Live-run page. Each tile has its label above it.
"""

import numpy as np
from PIL import Image, ImageDraw

ZOOM = 3
CAPTION_PX = 12
GAP_PX = 2
BACKGROUND = 200  # light grey between tiles, so the black image border stays visible


def image_grid(images: np.ndarray, labels, cols: int = 8) -> Image.Image:
    """images (N, H, W) in [0, 1] (clipped) -> greyscale PNG-ready image, row by row."""
    n, h, w = images.shape
    rows = -(-n // cols)
    tile_w, tile_h = w * ZOOM, h * ZOOM + CAPTION_PX
    grid = Image.new("L", (GAP_PX + cols * (tile_w + GAP_PX), GAP_PX + rows * (tile_h + GAP_PX)))
    grid.paste(BACKGROUND, (0, 0, *grid.size))
    draw = ImageDraw.Draw(grid)
    pixels = (np.clip(images, 0, 1) * 255).round().astype(np.uint8)
    for i, (image, label) in enumerate(zip(pixels, labels, strict=True)):
        row, col = divmod(i, cols)
        x, y = GAP_PX + col * (tile_w + GAP_PX), GAP_PX + row * (tile_h + GAP_PX)
        draw.text((x + 2, y), str(label), fill=0)
        tile = Image.fromarray(image).resize((tile_w, h * ZOOM), Image.Resampling.NEAREST)
        grid.paste(tile, (x, y + CAPTION_PX))
    return grid
