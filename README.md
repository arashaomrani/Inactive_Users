# Inactive_Users

This project is an implementation of predicting inactive users using gRPC. The service gets a csv file as data that can be used for training, or prediction (if already there is a model trained for this problem).

## Project Structure

- This solution requires clean installation of Ubuntu 22.04 .
- `churn.proto`: The Protocol Buffers definition file for the service.
- `setup.sh`: Run this file (`./setup.sh`) in the Ubuntu environment to update the system, install python3 and pip3 and also install required python libararies. 
- `build.sh`: Run this file (`./build.sh`) in the Ubuntu environment to create `churn_pb2.py` and `churn_pb2_grpc.py` from `churn.proto`.
- `server.py`: The Python script for the churn service server.
- `client.py`: The Python script for the churn service client.

## Run Server:
### Run server locally:
To run server locally, can use following command: `python3 server.py --host localhost --port 50051`
### Run server remotely: 
To run server to be accessible from a remote machine, can use following command: `python3 server.py --host 0.0.0.0 --port 50051` 
The --host argument of the server script should be set to 0.0.0.0 to accept connections from any IP address.

## Run Client:
To run client, run the following command: `python3 client.py --input activity_data.csv --output output.csv --training True --host server_public_ip_or_hostname --port 50051`

Client can accept 3 arguements (`input csv file location`, `output csv file location`, `training`).
If ther `--training` arguement be `True` then it run a training on the input csv file and save the trained model and other preprocessing objects. If the `--training` be `False`, it runs prediction on the input csv using the trained model, and saves the results in the output csv file.

The format of input and ouput files should be csv.

