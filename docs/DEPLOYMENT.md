# Deployment & Model Optimization Guide

This application supports both **Cloud Container Deployment** (Hugging Face Spaces Docker SDK / Render / Railway / Fly.io) and **Local Edge Deployment** with **ONNX Runtime graph optimizations (`ORT_ENABLE_ALL`)** and **INT8 dynamic quantization**.

---

## 1. Local Deployment (Windows / macOS / Linux)

```bash
python web_app.py
```

By default the local server runs at `http://127.0.0.1:8000` (`DEPLOYMENT_ENV=local`).  
To default to INT8 quantized ONNX models when available:

```bash
# PowerShell
$env:ASL_PREFER_INT8="1"; python web_app.py
```

You can also switch between **FP32** and **INT8** live from the top navigation bar in the web UI.

---

## 2. Free-Tier Cloud Deployment (Hugging Face Spaces — Docker SDK)

Hugging Face Spaces free tier provides **2 vCPU, 16 GB RAM** with Docker support on port **`7860`**, ideal for both WebSockets (`/ws/live`) and MediaPipe + ONNX Runtime CPU inference.

1. Go to [huggingface.co/new-space](https://huggingface.co/new-space)
2. Enter a Space name (e.g., `asl-live-translator`)
3. Select **Space SDK: Docker** (Blank template) and **CPU basic (free)**
4. Push this repository (`Dockerfile`, `requirements.txt`, `web_app.py`, `src/`, `web/`, `models/`, `demo_images/`) to the Hugging Face Space git remote:
   ```bash
   git remote add hf https://huggingface.co/spaces/<YOUR_USERNAME>/<SPACE_NAME>
   git push hf main
   ```
5. Once built, Hugging Face serves the live HTTPS URL over WebSockets (`wss://.../ws/live`), which is required by browsers to grant webcam permissions in production.
6. Verify deployment health at `https://<your-space>.hf.space/health`.

### Local Docker Verification

```bash
docker compose up --build
# Open http://127.0.0.1:7860
```

---

## 3. Vercel production and GitHub branches

- `main` is the GitHub default branch and the Vercel production branch. Successful Vercel builds from `main` update [ASL Live](https://asl-live-aai3001.vercel.app/).
- `thawzinhtun` is a personal development branch. Open a pull request into `main` to release changes from it.
- Hosting CI runs on pushes and pull requests targeting `main`. It builds `Dockerfile`, runs the Python suite in that image, and checks the running container on `PORT=80` for model readiness, image upload, and live WebSocket responses in Words and Letters modes.

## 4. Model Optimization & Quantization Benchmark

* **Training Export (v2 Colab Notebooks):** Cell 10 of [01_Train_ASL_Words_Conformer_2026-11-19.ipynb](../colab/01_Train_ASL_Words_Conformer_2026-11-19.ipynb) and Cell 9 of [02_Train_ASL_Letters_Hybrid_2026-11-19.ipynb](../colab/02_Train_ASL_Letters_Hybrid_2026-11-19.ipynb) automatically export:
  * `asl_words_conformer.onnx` (FP32) + `asl_words_conformer_int8.onnx` (INT8 Dynamic Quantized) + `optimization_benchmark_words.csv`
  * `asl_letters_hybrid.onnx` (FP32) + `asl_letters_hybrid_int8.onnx` (INT8 Dynamic Quantized) + `optimization_benchmark_letters.csv`
* **Local Optimization & Latency Benchmark Script:**
  ```bash
  # Benchmark existing ONNX models (uses ORT_ENABLE_ALL graph fusion)
  python scripts/optimize_models.py --benchmark-only

  # Quantize + benchmark (if `onnx` is installed in the active environment)
  python scripts/optimize_models.py --quantize
  ```
  Outputs are written to `models/optimization_report.json`.
