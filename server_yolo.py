# server_yolo.py
import flwr as fl

def aggregate_fit(server_round, results, failures):
    detections = []
    for _, _, metrics in results:
        detections.extend(metrics.get("detections", []))
    return [], {}

strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,
    fraction_evaluate=0.0,
    min_fit_clients=1,
    min_available_clients=1,
    on_fit_config_fn=lambda r: {},
    fit_metrics_aggregation_fn=aggregate_fit,
)

def main():
    fl.server.start_server(config=fl.server.ServerConfig(num_rounds=1), strategy=strategy)