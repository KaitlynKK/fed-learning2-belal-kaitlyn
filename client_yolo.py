import os
import argparse
import torch
from pathlib import Path
import flwr as fl
from flwr.common import parameters_to_ndarrays
from ultralytics import YOLO
import random

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


# --- Test-only: run final model on unseen videos and export results ---
def test_final_model(model_path="static/output/final_model.pt",
                     test_videos_dir="data/test_videos",
                     save_dir="static/output/test_results_run1"):

    print("[SERVER] Testing final model on unseen videos...")
    model = YOLO(model_path)
    os.makedirs(save_dir, exist_ok=True)

    summary_path = os.path.join(save_dir, "test_summary.txt")
    summary_lines = ["=== TEST PERFORMANCE SUMMARY ===\n"]

    # --- Video inference ---
    for fname in os.listdir(test_videos_dir):
        if fname.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
            src = os.path.join(test_videos_dir, fname)
            print(f"[SERVER] Inference on: {fname}")
            try:
                model.predict(
                    source=src,
                    save=True,
                    save_dir=os.path.join(save_dir, os.path.splitext(fname)[0]),
                    conf=0.25,
                    imgsz=640,
                    stream=True,
                    verbose=False
                )
                summary_lines.append(f"{fname}: Success\n")
            except Exception as e:
                print(f"[ERROR] Could not process {fname}: {e}")
                summary_lines.append(f"{fname}: Failed ({e})\n")

            torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # --- Evaluate performance metrics on labelled dataset ---
    try:
        from pathlib import Path
        print("\n[SERVER] Evaluating model performance on labelled dataset (labelfront2)...")

        # Use absolute path so Ultralytics doesn’t prepend datasets_dir
        data_yaml = str(Path("data/labelfront2/data.yaml").resolve())

        metrics = model.val(data=data_yaml, imgsz=640, conf=0.25, split="val")

        results_summary = {
            "Precision": round(metrics.box.p.mean().item(), 3),
            "Recall": round(metrics.box.r.mean().item(), 3),
            "mAP50": round(metrics.box.map50.mean().item(), 3),
            "mAP50-95": round(metrics.box.map.mean().item(), 3),
        }

        summary_lines.append("\n--- Overall Metrics ---\n")
        for k, v in results_summary.items():
            summary_lines.append(f"{k:<10}: {v}\n")

    except Exception as e:
        summary_lines.append(f"\nMetrics evaluation skipped due to: {e}\n")

    # --- Save test summary ---
    with open(summary_path, "w", encoding="utf-8", errors="ignore") as f:
        f.writelines(summary_lines)

    print(f"\n📄 Test summary saved at: {summary_path}")
    print(f"[SERVER] Visual results saved to: {save_dir}")
    return summary_path


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
