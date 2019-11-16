FROM pytorch/pytorch:1.3-cuda10.1-cudnn7-devel

ENV LANG C.UTF-8
ENV LANGUAGE C.UTF-8
ENV LC_ALL C.UTF-8

RUN pip install \
    click \
    grpcio \
    pillow-simd \
    && rm -rf ~/.cache/pip

RUN python -c "from torchvision.models.segmentation import deeplabv3_resnet101;deeplabv3_resnet101(pretrained=True)"

WORKDIR /workspace
COPY segmentation_pb2.py .
COPY segmentation_pb2_grpc.py .
COPY server.py .
COPY utils.py .

CMD python server.py

EXPOSE 50051
