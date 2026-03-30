# Vietnamese-English Code-Switching ASR

> **Paper:** "A Vietnamese-English Code-Switching Speech Dataset and Benchmark Evaluation for Automatic Speech Recognition"  
> Submitted to *Procedia Computer Science* — KES 2026

---

## Overview

This repository contains the dataset, code, and model weights accompanying the paper. We present:

- The **first labeled speech corpus** for Vietnamese-English code-switching (~12 hours, 12,672 utterances across 5 real-world domains)
- A **benchmark evaluation** of 4 ASR systems: Whisper large-v3, PhoWhisper-large, wav2vec2-base-vietnamese-250h, and Qwen3-ASR-1.7B
- An **ablation study** showing that LoRA fine-tuning + speed perturbation + SpecAugment reduces WER-N from 20.95% → 4.56% on the holdout set

---

## Results

### Ablation study (validation set)

| Model | Condition | WER | WER-N | CER-N | SER-N |
|---|---|---|---|---|---|
| Whisper large-v3 | Baseline | 21.01 | 13.62 | 7.37 | 61.16 |
| Whisper large-v3 | + Aug | 5.86 | 5.15 | 2.67 | 34.45 |
| Whisper large-v3 | + LoRA | 6.48 | 5.72 | 2.93 | 37.50 |
| Whisper large-v3 | **+ Aug + LoRA** | **5.86** | **5.15** | **2.69** | **34.26** |
| PhoWhisper-large | Baseline | 29.76 | 20.95 | 11.44 | 77.29 |
| PhoWhisper-large | + Aug | 6.02 | 5.18 | 2.77 | 32.92 |
| PhoWhisper-large | + LoRA | 6.35 | 5.53 | 2.87 | 35.21 |
| PhoWhisper-large | **+ Aug + LoRA** ✓ | **6.00** | **5.12** | **2.71** | **33.11** |
| wav2vec2-VI-250h | Baseline (zero-shot) | 64.68 | 51.30 | 28.10 | 100.00 |
| Qwen3-ASR-1.7B | Baseline (zero-shot) | 23.83 | 14.43 | 7.74 | 62.88 |

### Holdout evaluation — PhoWhisper-large + Aug + LoRA

| Metric | Raw | Normalized |
|---|---|---|
| WER | 5.76 | **4.56** |
| CER | 2.76 | 2.44 |
| SER | 37.26 | 31.48 |

### Cross-domain: VIVOS monolingual Vietnamese

| Model | Condition | WER-N |
|---|---|---|
| wav2vec2-VI-250h | Untuned | 10.72 |
| Qwen3-ASR-1.7B | Untuned | 7.17 |
| Whisper large-v3 | Untuned | 12.26 |
| PhoWhisper-large | Untuned | 4.61 |
| PhoWhisper-large | Fine-tuned | 7.20 |

Fine-tuning on code-switching data increases VIVOS WER-N by only 2.59pp — no catastrophic forgetting.

---

## Dataset

The corpus comprises **12,672 utterances / 12.2 hours** spanning 5 domains: classrooms, supermarkets, airports, offices, and casual conversation.

| Split | Utterances | Duration |
|---|---|---|
| Train | 10,188 | 9.8 h |
| Dev | 1,048 | 1.0 h |
| Test | 1,436 | 1.5 h |

- 99.8% of utterances contain intra-sentential code-switching
- English token ratio: 18.6% (train)
- Vocabulary: 6,685 unique word types (4,599 EN / 1,858 VI)
- Audio: 16 kHz, mono, `.wav`

📦 **Dataset on HuggingFace Hub:** `[link coming soon]`

---

## Models

📦 **Fine-tuned model weights on HuggingFace Hub:** [rinhoooo/phowhisper-large-vien-cs-asr](https://huggingface.co/rinhoooo/phowhisper-large-vien-cs-asr)

| Model | HuggingFace ID |
|---|---|
| PhoWhisper-large + Aug + LoRA (best) | `rinhoooo/phowhisper-large-vien-cs-asr` |
| Whisper large-v3 + Aug + LoRA | `[coming soon]` |

---

## Repository structure

```
vien-cs-asr/
├── notebooks/
│   ├── 01_eda.ipynb              # Dataset statistics & visualizations
│   ├── 02_train.ipynb            # LoRA fine-tuning with augmentation
│   ├── 03_eval.ipynb             # WER / CER / SER evaluation
│   └── 04_error_analysis.ipynb   # Hallucination & number normalization analysis
├── src/
│   ├── dataset.py                # HuggingFace Dataset loader
│   ├── augment.py                # Speed perturbation + SpecAugment
│   └── metrics.py                # WER / CER / SER + text normalization
├── configs/
│   ├── train.yaml                # Training hyperparameters
│   └── lora.yaml                 # LoRA configuration
├── data/
│   ├── raw/                      # Original audio (gitignored)
│   ├── processed/                # 16 kHz normalized wav files (gitignored)
│   └── manifests/                # train.json, dev.json, test.json
├── requirements.txt
├── .gitignore
├── DATASHEET.md
└── LICENSE                       # CC BY 4.0
```

---

## Setup

Tested on Python 3.10, CUDA 12.1, single NVIDIA A100 80 GB.

```bash
git clone https://github.com/[org]/vien-cs-asr.git
cd vien-cs-asr
```

Install PyTorch with the correct CUDA variant first:

```bash
pip install torch==2.3.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
```

Then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

> **Note:** If you are running on a different CUDA version, replace `cu121` with the appropriate tag (e.g. `cu118` for CUDA 11.8). See [pytorch.org/get-started](https://pytorch.org/get-started/locally/) for the full matrix.

---

## Training

Open `notebooks/02_train.ipynb` and set paths in the first cell:

```python
DATA_DIR   = "data/manifests"
OUTPUT_DIR = "models/checkpoints"
```

Key hyperparameters (see `configs/train.yaml` and `configs/lora.yaml`):

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 5e-5 |
| LR schedule | Cosine annealing (2 restarts) |
| Effective batch size | 32 |
| Label smoothing | 0.1 |
| Gradient clipping | 1.0 |
| Early stopping patience | 3 |
| Max epochs | 5 |
| LoRA rank *r* | 32 |
| LoRA alpha *α* | 64 |
| LoRA dropout | 0.1 |
| LoRA target modules | q_proj, k_proj, v_proj, out_proj, fc1, fc2 |
| Trainable params | 57.7M (3.6% of total) |

---

## Evaluation

```bash
# Run full evaluation on test split
# (see notebooks/03_eval.ipynb for detailed breakdown)
```

Metrics reported: WER, CER, SER on both raw and normalized transcriptions.  
**Primary metric: WER-N** (punctuation removed, lowercased, language tags stripped).

---

## Citation

```bibtex
@inproceedings{[author]2026viencsasr,
  title     = {A Vietnamese-English Code-Switching Speech Dataset and Benchmark
               Evaluation for Automatic Speech Recognition},
  author    = {[Author] and [Author] and [Author]},
  booktitle = {Procedia Computer Science (KES 2026)},
  year      = {2026}
}
```

---

## License

- **Code:** MIT License  
- **Dataset:** Creative Commons Attribution 4.0 International (CC BY 4.0)

---

## Acknowledgements

[Acknowledgements here.]
