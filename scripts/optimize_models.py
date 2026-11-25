"""ONNX Runtime model optimization (INT8 dynamic quantization) and latency benchmarking."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _dummy_inputs_for(stem: str, models_dir: Path) -> dict[str, np.ndarray]:
    if "word" in stem:
        cfg = _load_json(models_dir / "word_model_config.json")
        seq_len = int(cfg.get("seq_len", 48))
        feat_dim = int(cfg.get("feature_dim", 688))
        return {"sequence": np.zeros((1, seq_len, feat_dim), dtype=np.float32)}
    cfg = _load_json(models_dir / "letter_model_config.json")
    crop = int(cfg.get("crop_size", 224))
    geo_dim = int(cfg.get("geo_dim", 98))
    return {
        "image": np.zeros((1, 3, crop, crop), dtype=np.float32),
        "geo": np.zeros((1, geo_dim), dtype=np.float32),
    }


def benchmark_onnx_model(model_path: Path, models_dir: Path, warmup: int = 5, runs: int = 30) -> dict[str, Any]:
    """Measure file size and ORT_ENABLE_ALL CPU inference latency for one ONNX model."""
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(str(model_path), sess_options=opts, providers=["CPUExecutionProvider"])
    feeds = _dummy_inputs_for(model_path.stem.lower(), models_dir)
    out_names = [o.name for o in sess.get_outputs()]

    for _ in range(warmup):
        sess.run(out_names, feeds)

    latencies_ms: list[float] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        sess.run(out_names, feeds)
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    arr = np.asarray(latencies_ms, dtype=np.float64)
    return {
        "file": model_path.name,
        "precision": "int8" if model_path.stem.endswith("_int8") else "fp32",
        "size_mb": round(model_path.stat().st_size / (1024.0 * 1024.0), 3),
        "mean_ms": round(float(arr.mean()), 3),
        "p95_ms": round(float(np.percentile(arr, 95)), 3),
        "runs": runs,
    }


def quantize_to_int8(fp32_path: Path, int8_path: Path) -> tuple[bool, str]:
    """Apply ORT dynamic INT8 quantization when the `onnx` Python dependency is installed."""
    try:
        import onnx  # noqa: F401
        from onnxruntime.quantization import QuantType, quantize_dynamic
    except Exception as exc:
        return False, f"Quantization skipped (install `onnx` to quantize locally or export INT8 in Colab): {exc}"

    quantize_dynamic(
        model_input=str(fp32_path),
        model_output=str(int8_path),
        weight_type=QuantType.QInt8,
    )
    return True, f"Quantized {fp32_path.name} -> {int8_path.name}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Optimize & benchmark ASL ONNX models (FP32 / INT8).")
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--quantize", action="store_true", help="Attempt INT8 dynamic quantization.")
    parser.add_argument("--benchmark-only", action="store_true", help="Only benchmark existing ONNX files.")
    parser.add_argument("--runs", type=int, default=30)
    args = parser.parse_args()

    models_dir: Path = args.models_dir
    if not models_dir.exists():
        print(f"Models directory does not exist: {models_dir}")
        return 1

    onnx_files = sorted(models_dir.glob("*.onnx"))
    if not onnx_files:
        print(f"No .onnx files found in {models_dir}")
        return 1

    if args.quantize and not args.benchmark_only:
        for p in list(onnx_files):
            if p.stem.endswith("_int8"):
                continue
            target = p.with_name(f"{p.stem}_int8.onnx")
            ok, msg = quantize_to_int8(p, target)
            print(msg)
            if ok and target not in onnx_files:
                onnx_files.append(target)

    results = [benchmark_onnx_model(p, models_dir, runs=args.runs) for p in sorted(onnx_files)]
    report_path = models_dir / "optimization_report.json"
    report_path.write_text(
        json.dumps({"onnxruntime_version": ort.__version__, "benchmarks": results}, indent=2),
        encoding="utf-8",
    )
    for row in results:
        print(
            f"{row['file']:32s} | {row['precision']:4s} | "
            f"{row['size_mb']:7.3f} MB | mean {row['mean_ms']:7.3f} ms | p95 {row['p95_ms']:7.3f} ms"
        )
    print(f"Saved benchmark report -> {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
