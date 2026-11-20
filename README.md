# Twizzle — Figure Skating Element Classification

Twizzle is an end-to-end deep learning web application for classifying **11 figure skating elements** (jumps, spins, and step sequences) from short video clips using **X3D** and **R(2+1)D** models exported to **ONNX**.

## Project Team
- **Anabelle Gabriella Imandjaja** — Data & Preprocessing Lead (Binus University — Computer Science, Data Science)
- **Kenzie Bellinda** — Deep Learning & Model Engineer (Binus University — Computer Science, Data Science)
- **William Jovaski Trikesumah** — ONNX Inference & Frontend Engineer (Binus University — Computer Science, Data Science)

## Repository Structure

```text
Twizzle/
├── app.py                  # FastAPI + Gradio application entry point
├── pipeline.py             # ONNX Runtime inference pipeline (X3D & Fused X3D + R(2+1)D)
├── preprocess.py           # Video probing and 16-frame uniform sampling
├── contoh.mp4              # Sample video clip for quick testing
├── requirements.txt        # Python dependencies for inference & deployment
├── vercel.json             # Vercel static frontend deployment config
├── export/                 # Exported ONNX models and calibration metadata
│   ├── x3d/                # Fast mode (X3D ONNX + deployment_meta.json)
│   └── r2plus1d/           # Accurate Fused mode (R(2+1)D ONNX + deployment_meta.json)
├── notebook/               # Training & evaluation notebooks + training library
│   ├── model_x3d.ipynb
│   ├── model_r2plus1d.ipynb
│   ├── model_compare.ipynb
│   └── twizzle_train.py
├── predictions/            # Saved validation & test prediction probabilities (.npz)
├── scripts/                # Model testing, benchmarking & calibration analysis scripts
│   ├── bench.py
│   ├── compare.py
│   ├── fused_margin.py
│   ├── test_model.py
│   └── model_comparison.csv
└── static/                 # Frontend web application (HTML, CSS, JS, JSON, media assets)
    ├── index.html
    ├── styles.css
    ├── script.js
    ├── elements.json
    ├── skaters.json
    ├── team.json
    ├── elements/           # Sample clips for the 11 skating elements
    ├── skaters/            # Skater spotlight portraits
    └── teams/              # Project team portraits
```

## Running Locally

```bash
pip install -r requirements.txt gradio fastapi uvicorn
python app.py
```

Then open `http://127.0.0.1:7860` in your browser.
