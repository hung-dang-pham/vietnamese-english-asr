# 🎙️ Vietnamese-English Code-Switching ASR

> Fine-tuned Whisper large-v3 for Vietnamese-English code-switching automatic speech recognition.

[![Paper](https://img.shields.io/badge/Paper-ArXiv-red)](https://arxiv.org/abs/XXXX.XXXXX)
[![Model](https://img.shields.io/badge/Model-HuggingFace-yellow)](https://huggingface.co/yourname/whisper-vi-en-codeswitching)
[![License](https://img.shields.io/badge/License-MIT-blue)](LICENSE)

---

## 📊 Results

| Test Set | Source | Duration | WER (norm) | CER (norm) |
|---|---|---|---|---|
| Short utterances (2–3s) | Scripted | ~1h | **0.25%** | 0.17% |
| Long utterances (8–20s) | YouTube | ~20min | **3.50%** | 1.51% |

Baseline (Whisper large-v3, no fine-tuning): `[X]%` WER on short, `[X]%` on long.

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/yourname/vietnamese-english-asr
cd vietnamese-english-asr
pip install -r requirements.txt
```

### Inference

```python
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import numpy as np
from pydub import AudioSegment

# Load model
processor = WhisperProcessor.from_pretrained("yourname/whisper-vi-en-codeswitching")
model = WhisperForConditionalGeneration.from_pretrained("yourname/whisper-vi-en-codeswitching")

# Load audio
audio = AudioSegment.from_wav("your_audio.wav").set_frame_rate(16000).set_channels(1)
audio_array = np.array(audio.get_array_of_samples()).astype(np.float32) / 32768.0

# Transcribe
inputs = processor(audio_array, sampling_rate=16000, return_tensors="pt")
predicted_ids = model.generate(**inputs, language="vi", task="transcribe")
transcript = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
print(transcript)
```

Or use the inference script directly:

```bash
python src/inference.py --audio your_audio.wav
```

---

## 📁 Repository Structure

```
vietnamese-english-asr/
├── README.md
├── paper/
│   └── paper.pdf                  # Full paper
├── data/
│   ├── README.md                  # Dataset description & download links
│   └── sample/                    # Sample audio files for testing
│       ├── sample_01.wav
│       └── sample_transcript.json
├── src/
│   ├── inference.py               # Run inference on audio files
│   ├── preprocess.py              # VAD trimming & audio chunking
│   └── evaluate.py                # Compute WER/CER metrics
├── notebooks/
│   ├── 01_data_pipeline.ipynb     # Data collection from YouTube
│   ├── 02_fine_tuning.ipynb       # Fine-tune Whisper + LoRA
│   └── 03_evaluation.ipynb        # Reproduce paper results
├── configs/
│   └── train_config.yaml          # Training hyperparameters
├── requirements.txt
└── LICENSE
```

---

## 🗂️ Dataset

Our dataset consists of two sources:

- **Scripted speech**: Read by individual speakers, one sentence per utterance (2–3s)
- **Spontaneous speech**: Extracted from YouTube educational and tech videos featuring natural Vietnamese-English code-switching (8–20s)

Dataset is available at: [HuggingFace Datasets](https://huggingface.co/datasets/yourname/vi-en-codeswitching) *(coming soon)*

### Sample

```json
{
  "000001.wav": {
    "text": "Hiện tại Dũng đang làm giảng viên trường Đại học Swinburne ở bên Úc, có dạy về trí tuệ nhân tạo, hệ thống thông minh, và Dũng có bằng tiến sĩ về xây dựng hệ thống AI."
  },
  "000002.wav": {
    "text": "Thứ hai là AI Training, là những cái người đào tạo người khác cách sử dụng AI, có thể là đào tạo ở mức rất là cơ bản, cách sử dụng ChatGPT cho công việc hằng ngày."
  }
}
```

---

## 🔧 Training

### Requirements

- GPU: NVIDIA A100 40GB+ recommended for Whisper large-v3
- CUDA 11.8+

### Fine-tuning

```bash
# Install dependencies
pip install -r requirements.txt

# Run fine-tuning
python src/train.py --config configs/train_config.yaml
```

Key hyperparameters (`configs/train_config.yaml`):

```yaml
model: openai/whisper-large-v3
lora_rank: 8
lora_target_modules: [q_proj, v_proj, out_proj, fc1, fc2]
learning_rate: 1e-5
warmup_steps: 100
max_steps: 2000
batch_size: 8
gradient_accumulation_steps: 2
```

See `notebooks/02_fine_tuning.ipynb` for the full fine-tuning walkthrough.

---

## 📈 Reproduce Results

```bash
# Download test set
# (link coming soon)

# Run evaluation
python src/evaluate.py \
  --model yourname/whisper-vi-en-codeswitching \
  --test_data data/test_short.json \
  --output results/test_short.csv

python src/evaluate.py \
  --model yourname/whisper-vi-en-codeswitching \
  --test_data data/test_long.json \
  --output results/test_long.csv
```

Or follow `notebooks/03_evaluation.ipynb` for the full evaluation walkthrough.

---

## 🔍 Key Findings

**1. Chunking quality matters**
Sentence boundary accuracy significantly impacts WER. Utterances cut mid-sentence show substantially higher error rates than complete sentences. We recommend using VAD + forced alignment (WhisperX) for optimal segmentation.

**2. Hallucination prevention**
Whisper hallucinates when audio begins with music or non-speech noise. We apply Silero VAD to trim audio, retaining only 0.5s of silence before the first speech segment. This substantially reduces repetitive hallucination artifacts.

**3. Utterance length domain mismatch**
Training on short utterances (2–3s) leads to degraded performance on long utterances (8–20s). Balanced training data across utterance lengths is important for robust performance.

---

## 📄 Citation

If you use this work, please cite:

```bibtex
@article{yourname2025vienasr,
  title     = {Fine-tuning Whisper for Vietnamese-English Code-Switching Automatic Speech Recognition},
  author    = {[Author Names]},
  journal   = {[Venue]},
  year      = {2025},
  url       = {https://arxiv.org/abs/XXXX.XXXXX}
}
```

---

## 📬 Contact

Questions or feedback? Open an issue or contact `[your email]`.
