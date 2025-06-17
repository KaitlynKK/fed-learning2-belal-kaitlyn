import flwr as fl

# Define the strategy
strategy = fl.server.strategy.FedAvg(
    min_fit_clients=1,
    min_evaluate_clients=1,
    min_available_clients=1,
)

# Start the server with correct config
fl.server.start_server(
    server_address="localhost:8080",
    config=fl.server.ServerConfig(num_rounds=3),
    strategy=strategy
)
