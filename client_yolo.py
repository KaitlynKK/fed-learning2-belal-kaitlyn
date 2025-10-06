import os
import argparse
import torch
import flwr as fl
from flwr.common import parameters_to_ndarrays
from ultralytics import YOLO

# --- Strategy that remembers last aggregated params ---
class FedAvgWithSave(fl.server.strategy.FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.final_parameters = None

    def aggregate_fit(self, server_round, results, failures):
        aggregated, metrics = super().aggregate_fit(server_round, results, failures)
        if aggregated is not None:
            self.final_parameters = aggregated
        return aggregated, metrics


def save_final_model(params, base_ckpt="model/my_model.pt", out_path="static/output/final_model.pt"):
    print("[SERVER] Aggregation complete — saving model...")

    base_model = YOLO(base_ckpt)
    base_sd = base_model.model.state_dict()

    ndarrays = parameters_to_ndarrays(params)
    if len(ndarrays) != len(base_sd):
        raise RuntimeError("Mismatched param counts! Check client get_parameters order.")

    new_sd = {}
    for (k, v), arr in zip(base_sd.items(), ndarrays):
        t = torch.from_numpy(arr).to(v.device).to(v.dtype)
        if t.shape != v.shape:
            raise RuntimeError(f"Shape mismatch for {k}: got {t.shape}, expected {v.shape}")
        new_sd[k] = t

    base_model.model.load_state_dict(new_sd, strict=True)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    base_model.save(out_path)
    print(f"[SERVER] Final global model saved at: {out_path}")
    return out_path


# --- Test-only: evaluate final_model.pt on unseen videos ---
def test_final_model(model_path="static/output/final_model.pt", test_videos_dir="data/test_videos"):
    print("\n[SERVER] Testing final model on unseen videos (no training)...")
    model = YOLO(model_path)

    save_dir = "static/output/test_results"
    os.makedirs(save_dir, exist_ok=True)

    # Run inference on each test video
    for fname in os.listdir(test_videos_dir):
        if fname.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
            src = os.path.join(test_videos_dir, fname)
            print(f"[SERVER] Inference on: {fname}")
            model.predict(source=src, save=True, save_dir=save_dir, imgsz=640, conf=0.25, verbose=True)

    print(f"\n[SERVER] Inference complete! Results saved to {save_dir}\n")
    print("[SERVER] Evaluating accuracy (mAP, precision, recall)...")

    # Evaluate accuracy using labelled validation set (not test videos)
    metrics = model.val(data="data/labelfront2", imgsz=640, conf=0.25, split="val")
    print("\n[SERVER] === Performance Summary ===")
    print(metrics)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_rounds", type=int, default=4)
    ap.add_argument("--test_videos_dir", default="data/test_videos")
    ap.add_argument("--base_ckpt", default="model/my_model.pt")
    ap.add_argument("--final_out", default="static/output/final_model.pt")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()

    strategy = FedAvgWithSave(
        min_fit_clients=1,
        min_evaluate_clients=1,
        min_available_clients=1,
    )

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
    )

    if strategy.final_parameters is not None:
        model_path = save_final_model(strategy.final_parameters, base_ckpt=args.base_ckpt, out_path=args.final_out)
        test_final_model(model_path, args.test_videos_dir)
    else:
        print("[SERVER] No parameters found after training.")
