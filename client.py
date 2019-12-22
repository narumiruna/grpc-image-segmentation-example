import click
import grpc
import numpy as np
from PIL import Image

import segmentation_pb2
import segmentation_pb2_grpc
from server import ResizeCenterCrop
from utils import draw_mask, load_bytes


@click.command()
@click.argument('image-path')
@click.option('--host', default='127.0.0.1')
@click.option('--port', default='50051')
def main(image_path, host, port):
    size = 480
    num_classes = 21
    max_message_length = num_classes * size * size * 4 + 5

    address = '{}:{}'.format(host, port)
    options = [('grpc.max_receive_message_length', max_message_length)]
    channel = grpc.insecure_channel(address, options)

    bytes_data = load_bytes(image_path)
    stub = segmentation_pb2_grpc.SegmentationStub(channel)
    response = stub.Predict(segmentation_pb2.Request(image=bytes_data))

    pred = np.frombuffer(response.prediction, dtype=np.float32)
    pred = np.resize(pred, (num_classes, size, size)).argmax(axis=0)

    # draw result
    img = Image.open(image_path).convert('RGB')
    transform = ResizeCenterCrop(size)

    img = transform(img)
    mask = draw_mask(pred)
    blend = Image.blend(img, mask, alpha=0.5)

    width, height = img.size
    new_img_size = (width, height * 3)
    new_img = Image.new('RGB', new_img_size, (0, 0, 0))
    new_img.paste(img, (0, 0))
    new_img.paste(mask, (0, height))
    new_img.paste(blend, (0, height * 2))

    new_img.show()


if __name__ == '__main__':
    main()
