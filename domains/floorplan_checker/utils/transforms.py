import numpy as np
from PIL import Image
import mlx.core as mx

def load_and_transform_image(image_path: str, size: tuple[int, int] = (256, 256)) -> mx.array:
    try:
        with Image.open(image_path) as img:
            if img.mode == 'P' and 'transparency' in img.info:
                img = img.convert('RGBA')
            img = img.convert('RGB')
            img = img.resize(size, Image.Resampling.BILINEAR)
            # Convert to numpy array, normalize to [0, 1]
            img_np = np.array(img, dtype=np.float32) / 255.0
            # MLX convolution layers expect NHWC format, so shape [H, W, C] is correct.
            return mx.array(img_np)
    except Exception as e:
        print(f"Error loading {image_path}: {e}")
        return mx.zeros((*size, 3), dtype=mx.float32)
