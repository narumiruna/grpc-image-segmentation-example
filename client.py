import click
import grpc
import numpy as np
from PIL import Image

import segmentation_pb2
import segmentation_pb2_grpc
from segmentator import ResizeCenterCrop
from utils import draw_mask, load_bytes

INPUT_SIZE = (480, 480)


@click.command()
@click.option('--host', default='127.0.0.1')
@click.option('--port', default='50051')
@click.option('--image-path')
def main(host, port, image_path):
    address = '{}:{}'.format(host, port)
    channel = grpc.insecure_channel(address)

    bytes_data = load_bytes(image_path)
    stub = segmentation_pb2_grpc.SegmentationStub(channel)
    response = stub.Predict(segmentation_pb2.Request(image=bytes_data))

    pred = np.frombuffer(response.prediction, dtype=np.int64)
    pred = np.resize(pred, INPUT_SIZE)

    # draw prediction result
    img = Image.open(image_path).convert('RGB')
    transform = ResizeCenterCrop(480)
    img = transform(img)

    mask = draw_mask(pred)

    Image.blend(img, mask, alpha=0.5).show()


if __name__ == '__main__':
    main()
