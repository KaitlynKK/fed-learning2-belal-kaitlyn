import flwr as fl

def aggregate_fit(server_round, results, failures):
    return [], {}

strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,
    fraction_evaluate=0.0,
    min_fit_clients=1,
    min_available_clients=1,
)

def main():
    fl.server.start_server(config=fl.server.ServerConfig(num_rounds=1), strategy=strategy)

if __name__ == "__main__":
    main()