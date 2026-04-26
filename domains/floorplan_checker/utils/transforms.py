import numpy as np
from PIL import Image
import mlx.core as mx

def load_and_transform_image(image_path: str, size: tuple[int, int] = (256, 256)) -> mx.array:
    with Image.open(image_path) as img:
        if img.mode == 'P' and 'transparency' in img.info:
            img = img.convert('RGBA')
        img = img.convert('RGB')
        img = img.resize(size, Image.Resampling.BILINEAR)
        img_np = np.array(img, dtype=np.float32) / 255.0
        return mx.array(img_np)
