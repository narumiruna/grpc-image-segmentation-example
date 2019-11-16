import time
from concurrent import futures

import click
import grpc
import torch
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet101

import segmentation_pb2
import segmentation_pb2_grpc
from utils import load_bytes_image

_ONE_DAY_IN_SECONDS = 60 * 60 * 24


class ResizeCenterCrop(transforms.Compose):
    def __init__(self, size):
        super(ResizeCenterCrop, self).__init__([
            transforms.Resize(size),
            transforms.CenterCrop(size),
        ])


class Segmentator(object):
    def __init__(self, use_cuda=True):
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() and use_cuda else 'cpu')
        self.model = deeplabv3_resnet101(pretrained=True)
        self.model.to(self.device)

        self.transform = transforms.Compose([
            ResizeCenterCrop(480),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def __call__(self, image):
        self.model.eval()

        x = self.transform(image).unsqueeze(dim=0).to(self.device)
        out = self.model(x)['out']
        return out.squeeze(dim=0).cpu().numpy()


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
