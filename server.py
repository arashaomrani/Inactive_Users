import grpc
from concurrent import futures
import pandas as pd
from churn_pb2 import *
from churn_pb2_grpc import ChurnServiceServicer, add_ChurnServiceServicer_to_server
from utils import data_preprocessing, model, train, predict

class ChurnServicer(ChurnServiceServicer):

    def GetChurned(self, request, context):
        if request.training == 'True':
            df, df_fill, df_fill_scale, X_train, X_test, y_train, y_test = data_preprocessing(request.input)
            Model = model(X_train)
            history = train(Model, X_train, X_test, y_train, y_test)
            return ChurnResponse()
        else:
            prediction, classes = predict('model.keras', request.input, request.output)
            return ChurnResponse(prediction=prediction, churn_class=classes)

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    add_ChurnServiceServicer_to_server(ChurnServicer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    serve()

