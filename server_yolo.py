import os
import argparse
import torch
import flwr as fl
from flwr.common import parameters_to_ndarrays
from ultralytics import YOLO
import shutil
from datetime import datetime


# --- Strategy that remembers the last aggregated parameters ---
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
    print("[SERVER] Aggregation complete, saving model...")

    base_model = YOLO(base_ckpt)
    base_sd = base_model.model.state_dict()

    ndarrays = parameters_to_ndarrays(params)
    if len(ndarrays) != len(base_sd):
        raise RuntimeError(
            f"Mismatched param counts: agg={len(ndarrays)} vs model={len(base_sd)}. "
            "Ensure client get_parameters() iterates state_dict in a stable order."
        )

    new_sd = {}
    for (k, v), arr in zip(base_sd.items(), ndarrays):
        t = torch.from_numpy(arr).to(v.device).to(v.dtype)
        if t.shape != v.shape:
            raise RuntimeError(f"Shape mismatch for {k}: got {t.shape}, expected {v.shape}")
        new_sd[k] = t

    base_model.model.load_state_dict(new_sd, strict=True)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    base_model.save(out_path)
    print(f"[SERVER] Final model saved to: {out_path}")
    return out_path


# --- Test the saved global model on unseen videos + evaluate performance ---
def test_final_model(model_path="static/output/final_model.pt",
                     test_videos_dir="data/test_videos",
                     save_dir="static/output/test_results_run1"):

    print("[SERVER] Testing final model on unseen videos...")
    model = YOLO(model_path)
    os.makedirs(save_dir, exist_ok=True)

    # 1️⃣ Run inference on all test videos
    for fname in os.listdir(test_videos_dir):
        if fname.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
            src = os.path.join(test_videos_dir, fname)
            print(f"[SERVER] Inference on: {fname}")
            model.predict(source=src, save=True, save_dir=save_dir, conf=0.25, verbose=True)

    # 2️⃣ Evaluate model performance (mAP, Precision, Recall)
    print("\n[SERVER] Evaluating model performance on test dataset...")
    metrics = model.val(split="val")  # Ultralytics built-in validation

    results_summary = {
        "Precision": round(metrics.box.p.mean().item(), 3),
        "Recall": round(metrics.box.r.mean().item(), 3),
        "mAP50": round(metrics.box.map50.mean().item(), 3),
        "mAP50-95": round(metrics.box.map.mean().item(), 3),
    }

    print("\n📊 [SERVER] TEST PERFORMANCE SUMMARY")
    for k, v in results_summary.items():
        print(f"  {k:<10}: {v}")

    # 3️⃣ Save metrics to TXT and CSV
    metrics_txt = os.path.join(save_dir, "metrics_summary.txt")
    metrics_csv = os.path.join(save_dir, "metrics_summary.csv")

    with open(metrics_txt, "w") as f:
        f.write("Test Performance Summary\n")
        f.write("=========================\n")
        for k, v in results_summary.items():
            f.write(f"{k}: {v}\n")

    with open(metrics_csv, "w") as f:
        f.write("Metric,Value\n")
        for k, v in results_summary.items():
            f.write(f"{k},{v}\n")

    print(f"\n[SERVER] ✅ Metrics saved to:")
    print(f"   - {metrics_txt}")
    print(f"   - {metrics_csv}")

    print(f"\n[SERVER] Testing complete. Visual results saved to: {save_dir}")
    return results_summary


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_rounds", type=int, default=4)
    ap.add_argument("--test_videos_dir", default="data/test_videos")
    ap.add_argument("--base_ckpt", default="model/my_model.pt")
    ap.add_argument("--final_out", default="static/output/final_model.pt")
    ap.add_argument("--test_only", action="store_true", help="Run testing only on final model without training")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # ✅ TEST-ONLY MODE
    if args.test_only:
        print("[SERVER] Running in TEST-ONLY mode.")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = f"static/output/test_results_{timestamp}"
        os.makedirs(save_dir, exist_ok=True)
        test_final_model(model_path=args.final_out, test_videos_dir=args.test_videos_dir, save_dir=save_dir)
        print(f"[SERVER] Test-only run completed. Results in: {save_dir}")
        exit(0)

    # ✅ NORMAL FEDERATED TRAINING MODE
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

    # Save model + test after aggregation
    if strategy.final_parameters is not None:
        model_path = save_final_model(strategy.final_parameters, base_ckpt=args.base_ckpt, out_path=args.final_out)
        test_model_path = "static/output/final_model_eval.pt"
        shutil.copy(model_path, test_model_path)
        print(f"[SERVER] Copied model to test-only file: {test_model_path}")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        test_results_dir = f"static/output/test_results_{timestamp}"
        os.makedirs(test_results_dir, exist_ok=True)

        test_final_model(model_path=test_model_path, test_videos_dir=args.test_videos_dir, save_dir=test_results_dir)
        print(f"[SERVER] Test results saved in: {test_results_dir}")
    else:
        print("[SERVER] No final parameters were found to save.")
