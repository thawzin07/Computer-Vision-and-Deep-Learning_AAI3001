# ASL Live — Real-Time ASL to English Web Translator

An academic Deep Learning & Computer Vision system for real-time **American Sign Language (ASL)** recognition across both **15 dynamic words** and **26 fingerspelled letters (`A–Z`)**, served through a minimalist browser interface powered by **FastAPI**, **WebSockets**, **ONNX Runtime**, and **MediaPipe Tasks**.

---

## Key Capabilities & Model Benchmarks

| Track | Model Architecture | Input Representation | Held-Out Test Performance |
| :--- | :--- | :--- | :--- |
| **Words (15)** | **Kinematic Conformer** (`asl_words_conformer.onnx`) | 48-frame × 688-dim normalized upper-body + dual-hand pose, joint angles, velocity & acceleration | **90.78% Top-1 Accuracy** · **91.04% Macro-F1** |
| **Letters (`A–Z`)** | **HybridHandNet** (`asl_letters_hybrid.onnx`) | Dual-branch `224×224` canonical hand crop (`EfficientNet-B0`) + 98-dim 3D hand geometry + dynamic `J`/`Z` trajectory | **99.90% Top-1 Accuracy** · **99.89% Macro-F1** |

**15 Supported ASL Words:**
`eat`, `family`, `finish`, `friend`, `hello`, `help`, `love`, `more`, `no`, `please`, `stop`, `thank you`, `want`, `water`, `yes`

---

## Project Structure

```text
.
├── colab/
│   ├── 01_Train_ASL_Words_Conformer.ipynb   # Word Conformer training, baselines & ONNX export
│   └── 02_Train_ASL_Letters_Hybrid.ipynb    # Letter HybridHandNet training, Grad-CAM & ONNX export
├── demo_images/                             # Reference illustrations for all 15 ASL words
├── models/                                  # Pre-exported ONNX weights, configs & MediaPipe .task bundles
│   ├── asl_words_conformer.onnx
│   ├── word_model_config.json
│   ├── asl_letters_hybrid.onnx
│   ├── letter_model_config.json
│   └── mediapipe/
│       ├── hand_landmarker.task
│       └── pose_landmarker_lite.task
├── scripts/                                 # Reproducible notebook generators & schema validator
│   ├── build_notebooks.py
│   ├── nb_common.py
│   ├── nb_letters.py
│   └── nb_words.py
├── src/asl_live/                            # Shared feature extraction, landmarks, ONNX & stabilizers
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
  - Live browser webcam inference streamed over WebSocket (`/ws/live`) with crisp `<canvas>` skeleton overlay, **Mirror**, and **Skeleton** toggles.
  - Switch between **Auto Motion Gate** (sign then pause briefly) and **Manual Capture** (**Start Sign** → **Stop & Predict**).
  - Drag-and-drop or **Upload Video** (`.mp4`, `.mov`, `.avi`, `.webm`, `.mkv`) for full-clip word evaluation via `POST /api/predict/video`.
  - Live **Top-3 Probabilities**, **Hand Motion Energy** gauge, and auto-highlighting 15-word grid.
- **Letters (`A–Z`) Mode:**
  - Live webcam fingerspelling with hold-to-confirm stabilization and dynamic `J`/`Z` motion stroke detection.
  - Drag-and-drop or **Upload Image** for single-frame hand crop classification via `POST /api/predict/image`.
- **Demo Signs Guide (`Demo Signs` Tab):**
  - Displays all 15 reference illustrations from `demo_images/` with beginner step-by-step instructions, keyword filtering, and one-click **Practice** jump to Words mode.
- **Transcript Builder & Model Hot-Reload:**
  - Built-in **Auto-append** transcript with **Space**, **Undo**, **Clear**, **Copy**, and browser **Speak** (`speechSynthesis`).
  - Click **Reload Models** (`POST /api/reload`) anytime updated ONNX files are placed in `models/`.

---

## Retraining on Google Colab

1. Open [Google Colab](https://colab.research.google.com/) and upload [01_Train_ASL_Words_Conformer.ipynb](colab/01_Train_ASL_Words_Conformer.ipynb) or [02_Train_ASL_Letters_Hybrid.ipynb](colab/02_Train_ASL_Letters_Hybrid.ipynb).
2. Select **Runtime → Change runtime type → GPU (`L4` / `A100` / `T4`) + High-RAM**.
3. Run all cells to train, benchmark 3 model baselines, generate report figures, and export `.onnx` + `.json` bundles (`asl_words_export.zip` / `asl_letters_export.zip`).
4. Replace the exported files in `models/` and click **Reload Models** in the web app.

*(To regenerate the `.ipynb` sources programmatically after modifying `scripts/nb_*.py`, run `python scripts/build_notebooks.py`.)*

---

## Running Tests

```powershell
pytest -v
```
