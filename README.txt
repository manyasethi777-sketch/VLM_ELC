# ============================================================
#  ELC 2025-26 | Vision-Language Model Project
#  README — Run Instructions
# ============================================================

## Project Title
Vision-Language Models for Image Understanding:
Image Captioning & Visual Question Answering using BLIP

## Student Info
[Your Name] | [Roll Number] | [Section]

---

## What This Project Does
This project implements a Vision-Language Model (VLM) pipeline
capable of:
1. **Image Captioning** — Automatically describing any image in
   natural language using the BLIP model.
2. **Visual Question Answering (VQA)** — Answering free-text
   questions about uploaded images.

A browser-based interactive demo is provided via Gradio.

---

## Requirements
- Python 3.8 or higher
- Internet connection (first run downloads ~1 GB model weights)
- No GPU needed — runs on any CPU laptop

---

## Setup Instructions

### Step 1: Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run the interactive demo
```bash
python code/vlm_app.py
```
Then open your browser at: http://127.0.0.1:7860

### Step 3 (Optional): Run the evaluation script
```bash
python code/evaluate.py
```
This computes BLEU-4 scores on sample MS-COCO images.

---

## Dataset
- **Pre-training data used by BLIP**: MS-COCO Captions, Visual Genome
- **VQA evaluation**: VQA v2 benchmark
- No local dataset download needed — model weights are pre-trained
  and pulled automatically from HuggingFace Hub.

---

## Model Details
| Component | Details |
|-----------|---------|
| Model Name | BLIP (Salesforce) |
| Vision Encoder | ViT-B/16 (Vision Transformer) |
| Text Decoder | BERT-based causal LM |
| Captioning weights | blip-image-captioning-base |
| VQA weights | blip-vqa-base |
| Source | HuggingFace Transformers |

---

## Evaluation Metrics
| Metric | Task | Score (paper) |
|--------|------|--------------|
| BLEU-4 | Captioning | 36.7 |
| CIDEr | Captioning | 121.7 |
| METEOR | Captioning | 29.0 |
| Accuracy | VQA | 78.25% |

---

## Folder Structure
```
VLM_ELC/
├── code/
│   ├── vlm_app.py       ← Main Gradio demo app
│   └── evaluate.py      ← Evaluation / metrics script
├── requirements.txt     ← Python dependencies
├── README.txt           ← This file
├── Report.pdf           ← Project write-up
└── results/             ← Screenshots / sample outputs
```

---

## References
1. Li et al., "BLIP: Bootstrapping Language-Image Pre-training
   for Unified Vision-Language Understanding and Generation",
   ICML 2022. https://arxiv.org/abs/2201.12086
2. HuggingFace BLIP: https://huggingface.co/Salesforce/blip-image-captioning-base
3. VQA v2 Dataset: https://visualqa.org/
4. MS-COCO: https://cocodataset.org/
