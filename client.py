import grpc
import churn_pb2
import churn_pb2_grpc
import argparse

def run(host, port, input_file, output_file, training):
    channel = grpc.insecure_channel(f'{host}:{port}')
    stub = churn_pb2_grpc.ChurnServiceStub(channel)
    response = stub.GetChurned(churn_pb2.ChurnRequest(input=input_file,output=output_file, training=training))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='localhost', help='The server host')
    parser.add_argument('--port', default=50051, help='The server port')
    parser.add_argument('--input', required=True, help='Input data file')
    parser.add_argument('--output', required=True, help='Output data file')
    parser.add_argument('--training', type=str, default='False', help='Specify if in training mode')
    args = parser.parse_args()
    run(args.host, args.port, args.input, args.output, args.training)

