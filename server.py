import time
from concurrent import futures

import click
import grpc

import segmentation_pb2
import segmentation_pb2_grpc
from segmentator import Segmentator
from utils import load_bytes_image

_ONE_DAY_IN_SECONDS = 60 * 60 * 24


class SegmentationServicer(segmentation_pb2_grpc.SegmentationServicer):
    def __init__(self):
        self.segmentator = Segmentator()

    def Predict(self, request, context):
        bytes_data = request.image
        img = load_bytes_image(bytes_data)
        out = self.segmentator(img)
        return segmentation_pb2.Response(prediction=out.tobytes())


@click.command()
@click.option('--host', default='127.0.0.1')
@click.option('--port', default='50051')
@click.option('--max-workers', default=1)
def main(host, port, max_workers):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    servicer = SegmentationServicer()
    segmentation_pb2_grpc.add_SegmentationServicer_to_server(servicer, server)

    address = '{}:{}'.format(host, port)
    server.add_insecure_port(address)
    server.start()
    print("Listening at {}".format(address))
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == '__main__':
    main()
