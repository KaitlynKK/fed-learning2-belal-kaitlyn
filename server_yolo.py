import flwr as fl
from flwr.server.strategy import FedAvg
from flwr.server import ServerConfig

class CustomFedAvg(FedAvg):
    def aggregate_fit(self, server_round, results, failures):
        if not results:
            print("[SERVER] ⚠️ No client results returned. Returning empty model.")
            return None, {}
        print(f"[SERVER] ✅ Aggregating {len(results)} client results...")
        try:
            return super().aggregate_fit(server_round, results, failures)
        except ZeroDivisionError:
            print("[SERVER] ⚠️ Prevented ZeroDivisionError. Returning dummy update.")
            dummy_parameters = results[0][1].parameters  # reuse any result
            return dummy_parameters, {}

def main():
    strategy = CustomFedAvg(
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        min_fit_clients=1,
        min_available_clients=1,
    )

    # ✅ Modern call with explicit server address and config
    fl.server.start_server(
        server_address="localhost:8080",
        config=ServerConfig(num_rounds=1),
        strategy=strategy
    )

if __name__ == "__main__":
    main()
