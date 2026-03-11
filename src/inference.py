"""
inference.py — Vietnamese-English Code-Switching ASR
Usage:
    # Single file
    python inference.py --audio path/to/audio.wav --adapter path/to/adapter

    # Folder of wav files
    python inference.py --audio_dir path/to/folder --adapter path/to/adapter

    # Save results to JSON
    python inference.py --audio_dir path/to/folder --adapter path/to/adapter --output results.json
"""

import argparse
import json
import os
import re

import librosa
import numpy as np
import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor

# ── Constants ──────────────────────────────────────────────────────────────────
BASE_MODEL_ID = "openai/whisper-large-v3"
SAMPLING_RATE = 16000
MAX_DURATION_S = 30
LANGUAGE = "vi"
BATCH_SIZE = 8


# ── Audio ──────────────────────────────────────────────────────────────────────
def load_audio(path: str, target_sr: int = SAMPLING_RATE) -> np.ndarray:
    audio, _ = librosa.load(path, sr=target_sr, mono=True)
    max_samples = MAX_DURATION_S * target_sr
    return audio[:max_samples] if len(audio) > max_samples else audio


# ── Model ──────────────────────────────────────────────────────────────────────
def load_model(adapter_dir: str):
    print(f"Loading base model: {BASE_MODEL_ID}")
    processor = WhisperProcessor.from_pretrained(BASE_MODEL_ID)
    base_model = WhisperForConditionalGeneration.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    print(f"Loading LoRA adapter: {adapter_dir}")
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    model = model.merge_and_unload()
    model.eval()
    print("Model ready.")
    return model, processor


# ── Inference ──────────────────────────────────────────────────────────────────
def transcribe_batch(model, processor, audio_paths: list[str], batch_size: int = BATCH_SIZE) -> list[str]:
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    forced_decoder_ids = processor.get_decoder_prompt_ids(language=LANGUAGE, task="transcribe")

    results = []
    for i in tqdm(range(0, len(audio_paths), batch_size), desc="Transcribing"):
        batch_paths = audio_paths[i : i + batch_size]
        audios = [load_audio(p) for p in batch_paths]

        inputs = processor.feature_extractor(
            audios,
            sampling_rate=SAMPLING_RATE,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        ).to(device)
        inputs["input_features"] = inputs["input_features"].to(dtype)

        with torch.no_grad():
            ids = model.generate(
                **inputs,
                forced_decoder_ids=forced_decoder_ids,
                max_new_tokens=225,
            )

        preds = processor.tokenizer.batch_decode(ids, skip_special_tokens=True)
        results.extend(preds)

    return results


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Vietnamese-English ASR Inference")
    parser.add_argument("--audio", type=str, help="Path to a single .wav file")
    parser.add_argument("--audio_dir", type=str, help="Path to folder of .wav files")
    parser.add_argument("--adapter", type=str, required=True, help="Path to LoRA adapter directory")
    parser.add_argument("--output", type=str, default=None, help="Save results to JSON file")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    if not args.audio and not args.audio_dir:
        parser.error("Provide --audio or --audio_dir")

    # Collect audio files
    if args.audio:
        audio_paths = [args.audio]
    else:
        audio_paths = sorted([
            os.path.join(args.audio_dir, f)
            for f in os.listdir(args.audio_dir)
            if f.endswith(".wav")
        ])
    print(f"Found {len(audio_paths)} file(s)")

    # Load model
    model, processor = load_model(args.adapter)

    # Run inference
    transcripts = transcribe_batch(model, processor, audio_paths, batch_size=args.batch_size)

    # Print results
    results = {}
    for path, text in zip(audio_paths, transcripts):
        filename = os.path.basename(path)
        results[filename] = {"text": text}
        print(f"{filename}: {text}")

    # Save if requested
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
