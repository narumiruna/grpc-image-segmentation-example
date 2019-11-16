from PIL import Image
import io


def load_bytes_image(bytes_data, mode='RGB'):
    bytes_buffer = io.BytesIO(bytes_data)
    return Image.open(bytes_buffer).convert(mode)


def load_bytes(f):
    with open(f, 'rb') as fp:
        return fp.read()