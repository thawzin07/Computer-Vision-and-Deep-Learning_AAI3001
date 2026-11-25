# ASL Live — Real-Time ASL to English Web Translator

An academic Deep Learning & Computer Vision system for real-time **American Sign Language (ASL)** recognition across both **15 dynamic words** and **26 fingerspelled letters (`A–Z`)**, served through a minimalist browser interface powered by **FastAPI**, **WebSockets**, **ONNX Runtime**, and **MediaPipe Tasks**.

---

## Key Capabilities & Model Benchmarks

| Track | Active Model Architecture | Size (FP32 → INT8) | Local CPU Latency (FP32 / INT8) | Held-Out Test Performance |
| :--- | :--- | :--- | :--- | :--- |
| **Words (15)** | **v2 Dual Architecture + SupCon** (`asl_words_conformer.onnx` + `_int8.onnx`) | `9.185 MB → 5.650 MB` (`-38.5%`) | `3.48 ms` / `4.58 ms` | **100.00% Top-1 · 100.00% Macro-F1** (5-Fold CV: **97.68% ± 2.17%**) |
| **Letters (`A–Z`)** | **HybridHandNet v2 (ModDrop + Aux + SupCon)** (`asl_letters_hybrid.onnx` + `_int8.onnx`) | `17.931 MB → 4.888 MB` (`-72.7%`) | `4.37 ms` / `94.24 ms` | **99.84% Top-1 · 99.84% Macro-F1** (INT8: `95.21%`) |

*(Original v1 baseline weights — `90.78%` Words, `99.90%` Letters — are preserved in [`models/packs/v1_legacy/`](models/packs/v1_legacy/pack_manifest.json) and can be switched live in the header **Pack** selector.)*

**15 Supported ASL Words:**
`eat`, `family`, `finish`, `friend`, `hello`, `help`, `love`, `more`, `no`, `please`, `stop`, `thank you`, `want`, `water`, `yes`

---

## Project Structure

```text
.
├── colab/
│   ├── 01_Train_ASL_Words_Conformer.ipynb              # Executed v1 Word Conformer notebook
│   ├── 01_Train_ASL_Words_Conformer_2026-11-19.ipynb   # v2 balanced Conformer + BiGRU ensemble & 5-fold CV
│   ├── 02_Train_ASL_Letters_Hybrid.ipynb               # Executed v1 HybridHandNet notebook
│   └── 02_Train_ASL_Letters_Hybrid_2026-11-19.ipynb    # v2 Modality Dropout + Synthetic->Real OOD benchmark
├── demo_images/                             # Reference illustrations for all 15 ASL words
├── models/                                  # Active v2 FP32+INT8 ONNX weights, configs & MediaPipe .task bundles
│   ├── asl_words_conformer.onnx
│   ├── asl_words_conformer_int8.onnx
│   ├── word_model_config.json
│   ├── asl_letters_hybrid.onnx
│   ├── asl_letters_hybrid_int8.onnx
│   ├── letter_model_config.json
│   ├── optimization_report.json
│   ├── packs/                               # Pluggable packs (includes preserved `v1_legacy/` baseline)
│   └── mediapipe/
│       ├── hand_landmarker.task
│       └── pose_landmarker_lite.task
├── docs/
│   ├── DEPLOYMENT.md                        # Local & Hugging Face Spaces Docker deployment + INT8 guide
│   └── ARCHITECTURE_AND_SCALABILITY.md      # Pipeline diagram, pack spec & annotation schema
├── scripts/                                 # Reproducible notebook generators, validator & ORT optimizer
│   ├── build_notebooks.py
│   ├── nb_common.py
│   ├── nb_letters.py
│   ├── nb_letters_v2.py
│   ├── nb_words.py
│   ├── nb_words_v2.py
│   └── optimize_models.py
├── Dockerfile                               # Cloud container deployment (port 7860, non-headless OpenCV libs)
├── docker-compose.yml
├── src/asl_live/                            # Shared feature extraction, landmarks, ONNX, calibration & stabilizers
├── tests/                                   # Automated unit & API tests
├── web/static/                              # Minimalist white-theme frontend (HTML / CSS / JS)
│   ├── index.html
│   ├── app.css
│   └── app.js
├── web_app.py                               # FastAPI + WebSocket server entrypoint
├── check_environment.py                     # Local Python, MediaPipe, ONNX & webcam diagnostic
├── requirements.txt
└── README.md
```

---

## Quick Start (For Teammates)

All trained ONNX models, JSON configs, and MediaPipe `.task` bundles are included in `models/`, so you can run the translator immediately after cloning.

### 1. Install Dependencies
Tested on Windows with Python 3.10–3.14:

```powershell
pip install -r requirements.txt
```

### 2. Verify Local Environment
```powershell
python check_environment.py
```

### 3. Launch the Web App
```powershell
python web_app.py
```

This starts the local server at **`http://127.0.0.1:8000`** and opens your browser automatically *(optional flags: `--host`, `--port`, `--models-dir`, `--no-browser`)*.

---

## Web App Features

- **Words (15) Mode:**
  - Live browser webcam inference streamed over WebSocket (`/ws/live`) with camera switcher dropdown, crisp `<canvas>` skeleton overlay, active-signing border pulse, **Mirror**, and **Skeleton** toggles.
  - Switch between **Auto Motion Gate** (sign then pause briefly) and **Manual Capture** (**Start Sign** → **Stop & Predict**).
  - Drag-and-drop or **Upload Video** (`.mp4`, `.mov`, `.avi`, `.webm`, `.mkv`) for full-clip word evaluation via `POST /api/predict/video`.
  - Live **Top-3 Probabilities**, **Hand Motion Energy** gauge, and auto-highlighting 15-word grid.
- **Letters (`A–Z`) Mode:**
  - Live webcam fingerspelling with hold-to-confirm stabilization and isolated per-session dynamic `J`/`Z` motion stroke detection.
  - Drag-and-drop or **Upload Image** for single-frame hand crop classification via `POST /api/predict/image`.
- **Live Tuning & Team Dataset Annotation Studio:**
  - Click **Tune** to adjust word confidence, word motion gate, letter confidence, and letter hold duration live over WebSocket.
  - Click **Dataset & Annotate** to record 2s trimmed word clips (`POST /api/calibration/word`) or snapshots (`POST /api/calibration/letter`) with `signer_id`, `lighting`, `background`, `handedness`, and custom new sign labels, manage recent recordings (`DELETE /api/calibration/sample/{id}`), and download `calibration_data.zip` containing `annotations.csv` + `annotations.jsonl`.
- **Cloud Deployment, Precision Switch & Model Packs:**
  - Switch between **FP32** and **INT8** quantized ONNX sessions (`POST /api/precision`) and swap pluggable domain/sign-language packs (`models/packs/<pack_id>/`, `POST /api/packs/select`) directly from the header bar.
  - Deployable locally or in cloud containers (`Dockerfile` / Hugging Face Spaces port `7860`, `/health`, CORS).
- **Demo Signs Guide (`Demo Signs` Tab):**
  - Switch between **Words (15)** (illustrated guides from `demo_images/`) and **Letters (A–Z)** (handshape descriptions + dynamic `J`/`Z` motion tags) with keyword filtering and one-click **Practice** jumps.
- **Transcript Builder & Model Hot-Reload:**
  - Built-in **Auto-append** transcript with optional **Polish English** formatting, **Space**, **Undo**, **Clear**, **Copy**, and browser **Speak** (`speechSynthesis`).
  - Click **Reload Models** (`POST /api/reload`) anytime updated ONNX files are placed in `models/`.

---

## AAI3001 Rubric Alignment Summary

| Criterion | Implemented Evidence | Details & Guides |
| :--- | :--- | :--- |
| **1. Dataset & Annotation** | Curated multi-source ASL (`ASL Citizen` + `WLASL`, `Lexset` + `Grassknoted` + OOD real hands) + browser **Dataset & Annotate Studio** (`signer_id`, `lighting`, `background`, `handedness`, clip trimming, custom labels, `annotations.csv`/`.jsonl`) ingested into Colab v2 signer-disjoint splits | [docs/ARCHITECTURE_AND_SCALABILITY.md](docs/ARCHITECTURE_AND_SCALABILITY.md) |
| **2. Model Development** | v2 ablation architectures: `Conformer v2 + BiGRU Ensemble + SupervisedContrastiveLoss (SupCon)` (Words) and `HybridHandNet v2` with **Modality Dropout + Auxiliary Supervision + SupCon** + **Synthetic→Real OOD benchmark** (Letters) | [colab/01_Train_ASL_Words_Conformer_2026-11-19.ipynb](colab/01_Train_ASL_Words_Conformer_2026-11-19.ipynb) · [colab/02_Train_ASL_Letters_Hybrid_2026-11-19.ipynb](colab/02_Train_ASL_Letters_Hybrid_2026-11-19.ipynb) |
| **3. Deployment & Optimization** | **Cloud** (`Dockerfile`, Hugging Face Spaces port `7860`, `/health`, CORS) + **Local** (`ORT_ENABLE_ALL` graph fusion, FP32 vs INT8 dynamic quantization, `scripts/optimize_models.py`, live FP32/INT8 switcher) | [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) |
| **4. User Interface** | Minimalist white React-free SPA with live webcam/upload inference, dismissible onboarding tips, **Demo Signs** interactive guide, live tuning sliders, and team annotation manager | [web/static/index.html](web/static/index.html) |
| **5. Scalability** | Pluggable `models/packs/<pack_id>/` registry adapting dynamically to arbitrary class counts/domains (`General ASL`, `Medical`, `SgSL`) plus custom-label annotation studio | [docs/ARCHITECTURE_AND_SCALABILITY.md](docs/ARCHITECTURE_AND_SCALABILITY.md) |

---

## Retraining on Google Colab

1. Open [Google Colab](https://colab.research.google.com/) and upload the dated v2 notebooks:
   - [01_Train_ASL_Words_Conformer_2026-11-19.ipynb](colab/01_Train_ASL_Words_Conformer_2026-11-19.ipynb)
   - [02_Train_ASL_Letters_Hybrid_2026-11-19.ipynb](colab/02_Train_ASL_Letters_Hybrid_2026-11-19.ipynb)
2. Select **Runtime → Change runtime type → GPU (`A100` / `L4`) + High-RAM**.
3. Run all cells — both v2 notebooks automatically reuse your existing v1 Google Drive feature caches if present, ingest any uploaded team `annotations.csv` calibration data, run ablations (including `SupervisedContrastiveLoss`) / 5-fold signer-independent CV / Synthetic→Real OOD evaluation, generate report figures, and download updated FP32 + INT8 export bundles (`asl_words_export_v2_2026-11-25.zip` / `asl_letters_export_v2_2026-11-25.zip`).
4. Replace `asl_words_conformer.onnx` (+ `_int8.onnx`), `word_model_config.json`, `asl_letters_hybrid.onnx` (+ `_int8.onnx`), and `letter_model_config.json` in `models/` and click **Reload Models** in the web app.

*(To regenerate the dated v2 `.ipynb` sources programmatically after modifying `scripts/nb_*_v2.py`, run `python scripts/build_notebooks.py`.)*

---

## Running Tests

```powershell
pytest -v
```
