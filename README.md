# gRPC Image Segmentation Example

## Generate code

```shell
$ python -m grpc_tools.protoc --proto_path=. --python_out=. --grpc_python_out=. segmentation.proto
```

## Start server

```shell
$ python server.py
```

or start server in docker
```shell
$ docker build -t grpcseg .
$ docker run -d -p 50051:50051 grpcseg
```
## Run client

```shell
$ python client.py img.jpg
```
