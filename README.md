# gRPC Image Segmentation Example

## Generate code

```shell
$ python -m grpc_tools.protoc --proto_path=. --python_out=. --grpc_python_out=. segmentation.proto
```

## Start server

```shell
$ python server.py
```

## Run client

```shell
$ python client.py img.jpg
```
