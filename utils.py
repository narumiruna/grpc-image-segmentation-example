import io

import numpy as np
from PIL import Image


def load_bytes_image(bytes_data, mode='RGB'):
    bytes_buffer = io.BytesIO(bytes_data)
    return Image.open(bytes_buffer).convert(mode)


def load_bytes(f):
    with open(f, 'rb') as fp:
        return fp.read()


def draw_mask(pred, num_classes=21, seed=1):
    np.random.seed(seed)

    h, w = pred.shape

    mask = np.zeros(shape=(h, w, 3), dtype=np.uint8)
    for i in range(1, num_classes):
        color = np.random.randint(0, 256, size=3, dtype=np.uint8)
        mask[pred == i, :] = color[None, :]

    return Image.fromarray(mask.astype(np.uint8))
