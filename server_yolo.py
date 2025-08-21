import flwr as fl
from flwr.common import parameters_to_ndarrays
import torch
from ultralytics import YOLO
import os

# Custom strategy that stores the final global parameters after training
class FedAvgWithSave(fl.server.strategy.FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.final_parameters = None

    def aggregate_fit(self, server_round, results, failures):
        aggregated, metrics = super().aggregate_fit(server_round, results, failures)
        if aggregated:
            self.final_parameters = aggregated
        return aggregated, metrics

# Initialize the strategy
strategy = FedAvgWithSave(
    min_fit_clients=4,
    min_evaluate_clients=4,
    min_available_clients=4,
)

# Start the federated learning server
fl.server.start_server(
    server_address="localhost:8080",
    config=fl.server.ServerConfig(num_rounds=3),
    strategy=strategy
)

# After training, save the final aggregated model if it exists
if strategy.final_parameters:
    print("[SERVER] ✅ Aggregation complete, saving model...")

    # Load the base model used by clients
    base_model = YOLO("model/my_model.pt")

    # Convert FL parameters to a PyTorch-compatible format
    ndarrays = parameters_to_ndarrays(strategy.final_parameters)
    new_state_dict = {
        k: torch.tensor(v)
        for k, v in zip(base_model.model.state_dict().keys(), ndarrays)
    }

    # Apply weights to base model and save
    base_model.model.load_state_dict(new_state_dict, strict=False)
    os.makedirs("static/output", exist_ok=True)
    base_model.save("static/output/final_model.pt")
    print("[SERVER] ✅ Final model saved to: static/output/final_model.pt")
else:
    print("[SERVER] ⚠️ No final parameters were found to save.")

