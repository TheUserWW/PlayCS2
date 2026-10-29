"""CS2 model configuration. All context lengths are observation time steps."""

from dataclasses import dataclass


@dataclass(frozen=True)
class CS2PolicyConfig:
    n_embed: int = 384
    n_head: int = 6
    n_layer: int = 8
    block_size: int = 128
    dropout: float = 0.1
    vision_dim: int = 384
    image_height: int = 128
    image_width: int = 224
    # Apply the same radar mask to demo and live inputs; normalized x0,y0,x1,y1.
    mask_rects: tuple = ((0.0, 0.0, 0.2, 0.36),)

    def __post_init__(self):
        dimensions = (self.n_embed, self.n_head, self.n_layer, self.block_size,
                      self.vision_dim, self.image_height, self.image_width)
        if any(type(v) is not int or v <= 0 for v in dimensions):
            raise ValueError("Model dimensions must be positive integers")
        if self.n_embed % self.n_head or not 0 <= self.dropout < 1:
            raise ValueError("n_embed must be divisible by n_head; dropout must be in [0, 1)")
        if min(self.image_height, self.image_width) < 32:
            raise ValueError("CNN input dimensions must be at least 32")
        if any(len(r) != 4 or not (0 <= r[0] < r[2] <= 1 and 0 <= r[1] < r[3] <= 1) for r in self.mask_rects):
            raise ValueError("mask_rects must contain normalized x0,y0,x1,y1 rectangles")
