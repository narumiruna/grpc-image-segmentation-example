# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
import grpc

import segmentation_pb2 as segmentation__pb2


class SegmentationStub(object):
  # missing associated documentation comment in .proto file
  pass

  def __init__(self, channel):
    """Constructor.

    Args:
      channel: A grpc.Channel.
    """
    self.Predict = channel.unary_unary(
        '/Segmentation/Predict',
        request_serializer=segmentation__pb2.Request.SerializeToString,
        response_deserializer=segmentation__pb2.Response.FromString,
        )


class SegmentationServicer(object):
  # missing associated documentation comment in .proto file
  pass

  def Predict(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')


def add_SegmentationServicer_to_server(servicer, server):
  rpc_method_handlers = {
      'Predict': grpc.unary_unary_rpc_method_handler(
          servicer.Predict,
          request_deserializer=segmentation__pb2.Request.FromString,
          response_serializer=segmentation__pb2.Response.SerializeToString,
      ),
  }
  generic_handler = grpc.method_handlers_generic_handler(
      'Segmentation', rpc_method_handlers)
  server.add_generic_rpc_handlers((generic_handler,))