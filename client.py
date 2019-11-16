import click
import grpc
import numpy as np

import segmentation_pb2
import segmentation_pb2_grpc
from utils import load_bytes


@click.command()
@click.option('--host', default='127.0.0.1')
@click.option('--port', default='50051')
@click.option('--image-path')
def main(host, port, image_path):
    address = '{}:{}'.format(host, port)
    channel = grpc.insecure_channel(address)

    bytes_data = load_bytes(image_path)
    stub = segmentation_pb2_grpc.SegmentationStub(channel)
    pred = stub.Predict(segmentation_pb2.Request(image=bytes_data))

    pred = np.frombuffer(pred.prediction, dtype=np.int64).reshape((480, 480))
    print(pred.shape)


if __name__ == '__main__':
    main()
