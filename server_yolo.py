# server_yolo.py — Flower YOLOv8 server with test-only mode and proper evaluation

import os
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Dict

import torch
import flwr as fl
from flwr.common import parameters_to_ndarrays
from ultralytics import YOLO
import yaml


# ----------------------------
# Strategy that remembers last aggregated params
# ----------------------------
class FedAvgWithSave(fl.server.strategy.FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.final_parameters = None

    def aggregate_fit(self, server_round, results, failures):
        aggregated, metrics = super().aggregate_fit(server_round, results, failures)
        if aggregated is not None:
            self.final_parameters = aggregated
        return aggregated, metrics


# ----------------------------
# Save aggregated parameters into a YOLO checkpoint
# ----------------------------
def save_final_model(
    params,
    base_ckpt: str = "model/my_model.pt",
    out_path: str = "static/output/final_model.pt",
) -> str:
    print("[SERVER] Aggregation complete — saving model...")

    base_model = YOLO(base_ckpt)
    base_sd = base_model.model.state_dict()  # OrderedDict of tensors

    ndarrays = parameters_to_ndarrays(params)
    if len(ndarrays) != len(base_sd):
        raise RuntimeError(
            f"Mismatched param counts: agg={len(ndarrays)} vs model={len(base_sd)}. "
            "Ensure each client get_parameters()/set_parameters uses a stable key order."
        )

    new_sd: Dict[str, torch.Tensor] = {}
    for (k, v), arr in zip(base_sd.items(), ndarrays):
        t = torch.from_numpy(arr).to(v.device).to(v.dtype)
        if t.shape != v.shape:
            raise RuntimeError(f"Shape mismatch for '{k}': got {t.shape}, expected {v.shape}")
        new_sd[k] = t

    base_model.model.load_state_dict(new_sd, strict=True)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    # Save as a standard Ultralytics checkpoint path; YOLO().save() supports file path
    base_model.save(out_path)
    print(f"[SERVER] Final global model saved at: {out_path}")
    return out_path


# ----------------------------
# Test helpers
# ----------------------------
VIDEO_EXTS: List[str] = [".mp4", ".mov", ".avi", ".mkv"]


def ensure_test_yaml(
    test_root: Path,
    yaml_path: Path = Path("data/test_data.yaml"),
    class_names: List[str] | None = None,
) -> Path:
    """
    Build a YOLO data.yaml that points 'val' to a list of ALL images found under:
      test_root/**/images/*.jpg|png|jpeg
    Assumes matching labels live at test_root/**/labels/<same_stem>.txt

    On Windows, writes POSIX-style paths to avoid YAML escape issues.
    """
    test_root = test_root.resolve()
    yaml_path = yaml_path.resolve()

    if class_names is None or len(class_names) == 0:
        class_names = ["object"]  # default single-class

    # collect images recursively only from 'images' folders
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    imgs: list[str] = []
    for root, _, files in os.walk(test_root):
        if os.path.basename(root).lower() == "images":
            for fn in files:
                if Path(fn).suffix.lower() in exts:
                    p = Path(root, fn).resolve()
                    imgs.append(p.as_posix())

    if not imgs:
        raise FileNotFoundError(f"No images found under {test_root}/**/images")

    # write list file next to yaml
    list_file = yaml_path.with_name("_auto_test_list.txt")
    list_file.parent.mkdir(parents=True, exist_ok=True)
    with open(list_file, "w", encoding="utf-8") as f:
        f.write("\n".join(imgs))

    # write YAML with only 'val' populated
    content = {
        "path": test_root.as_posix(),
        "train": [],
        "val": list_file.as_posix(),
        "nc": len(class_names),
        "names": class_names,
    }
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(content, f, sort_keys=False)
    return yaml_path


def test_final_model(
    model_path: str = "static/output/final_model.pt",
    test_videos_dir: str = "data/test_videos",
    save_dir: str | None = None,
    test_yaml_path: str = "data/test_data.yaml",
    visuals_conf: float = 0.25,  # only for saving pretty videos; NOT used for metrics
) -> str:
    """
    - Runs inference on each video file in test_videos_dir (optional visuals with stream=True).
    - Evaluates metrics on labelled frames under test_videos/**/images + labels using a generated data.yaml.
    - IMPORTANT: We do NOT clamp 'conf' during model.val(); we use defaults so recall/mAP are computed properly.
    """
    model_path = str(Path(model_path).resolve())
    test_videos_dir = str(Path(test_videos_dir).resolve())
    if save_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = str(Path(f"static/output/test_results_{timestamp}").resolve())
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    print("\n[SERVER] Testing final model on unseen videos...")
    model = YOLO(model_path)

    # --- Video inference visuals (optional, uses a typical runtime conf like 0.25)
    for root, _, files in os.walk(test_videos_dir):
        for fname in files:
            if Path(fname).suffix.lower() in VIDEO_EXTS:
                src = str(Path(root) / fname)
                rel = Path(root).relative_to(test_videos_dir)
                out_dir = str(Path(save_dir) / rel / Path(fname).stem)
                print(f"[SERVER] Inference on: {Path(rel) / fname}")
                try:
                    preds = model.predict(
                        source=src,
                        save=True,
                        save_dir=out_dir,
                        imgsz=640,
                        conf=visuals_conf,
                        stream=True,     # don't keep frames in RAM
                        verbose=False,
                        vid_stride=1,    # adjust if you want to skip frames
                    )
                    # consume generator
                    for _ in preds:
                        pass
                except Exception as e:
                    print(f"[SERVER][WARN] Could not process {fname}: {e}")
                finally:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

    # --- Metrics summary
    summary_path = str(Path(save_dir) / "test_summary.txt")
    lines = ["=== TEST PERFORMANCE SUMMARY ===\n"]
    for root, _, files in os.walk(test_videos_dir):
        for fname in files:
            if Path(fname).suffix.lower() in VIDEO_EXTS:
                rel = Path(root).relative_to(test_videos_dir)
                lines.append(f"{Path(rel) / fname}: processed\n")

    try:
        print("\n[SERVER] Evaluating model performance on labelled test_videos...")
        yaml_abs = ensure_test_yaml(test_root=Path(test_videos_dir), yaml_path=Path(test_yaml_path))
        # ⚠️ DO NOT pass a high 'conf' here — let Ultralytics sweep properly
        metrics = model.val(data=str(yaml_abs), imgsz=640)

        # Extract key metrics safely
        def _to_float(x):
            try:
                if hasattr(x, "item"):
                    return float(x.item())
                if hasattr(x, "mean"):
                    m = x.mean()
                    return float(m.item()) if hasattr(m, "item") else float(m)
                if isinstance(x, (list, tuple)) and len(x) > 0:
                    return float(sum(map(_to_float, x)) / len(x))
                return float(x)
            except Exception:
                return float("nan")

        p = _to_float(getattr(metrics.box, "p", float("nan")))
        r = _to_float(getattr(metrics.box, "r", float("nan")))
        map50 = _to_float(getattr(metrics.box, "map50", float("nan")))
        map5095 = _to_float(getattr(metrics.box, "map", float("nan")))

        lines.append("\n--- Overall Metrics (labelled test_videos) ---\n")
        lines.append(f"Precision : {p:.3f}\n")
        lines.append(f"Recall    : {r:.3f}\n")
        lines.append(f"mAP50     : {map50:.3f}\n")
        lines.append(f"mAP50-95  : {map5095:.3f}\n")

    except Exception as e:
        lines.append(f"\nMetrics evaluation skipped due to: {e}\n")

    with open(summary_path, "w", encoding="utf-8") as f:
        f.writelines(lines)

    print(f"\n📄 Test summary saved at: {summary_path}")
    print(f"[SERVER] Visual results saved to: {save_dir}\n")
    return summary_path


# ----------------------------
# CLI
# ----------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_rounds", type=int, default=4)
    ap.add_argument("--base_ckpt", default="model/my_model.pt")
    ap.add_argument("--final_out", default="static/output/final_model.pt")
    ap.add_argument("--test_videos_dir", default="data/test_videos")
    ap.add_argument("--test_yaml", default="data/test_data.yaml")
    ap.add_argument("--test_only", action="store_true", help="Run only evaluation on final_model.pt")
    return ap.parse_args()


# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    args = parse_args()

    if args.test_only:
        print("[SERVER] Running in TEST-ONLY mode.")
        if not Path(args.final_out).exists():
            print(f"[SERVER][WARN] {args.final_out} not found. "
                  f"Make sure you saved a model there, or pass --final_out to where your model is.")
        test_final_model(
            model_path=args.final_out,
            test_videos_dir=args.test_videos_dir,
            test_yaml_path=args.test_yaml,
        )
        print("[SERVER] Test-only run completed.")
        raise SystemExit(0)

    # --- Normal federated training mode ---
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

    if strategy.final_parameters is None:
        print("[SERVER] No final parameters were found to save.")
        raise SystemExit(0)

    # Save aggregated global model and then evaluate it
    model_path = save_final_model(
        params=strategy.final_parameters,
        base_ckpt=args.base_ckpt,
        out_path=args.final_out,
    )

    test_final_model(
        model_path=str(model_path),
        test_videos_dir=args.test_videos_dir,
        test_yaml_path=args.test_yaml,
    )
