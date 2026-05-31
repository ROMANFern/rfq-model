"""
Download the T5 model from Google Drive before the server starts.
Run this once: python download_model.py
Render build command: python download_model.py && uvicorn main:app ...
"""

import os
import gdown

MODEL_DIR   = os.getenv("MODEL_DIR", "./model")
DRIVE_URL   = os.getenv("MODEL_DRIVE_URL", "")  # set in Render env vars

FILES = [
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "tokenizer_config.json",
    "tokenizer.json",
    "special_tokens_map.json",
    "spiece.model",
]

def download_model():
    if not DRIVE_URL:
        print("MODEL_DRIVE_URL not set — skipping model download (assume already present)")
        return

    os.makedirs(MODEL_DIR, exist_ok=True)

    # Check if already downloaded
    if os.path.exists(os.path.join(MODEL_DIR, "model.safetensors")):
        size = os.path.getsize(os.path.join(MODEL_DIR, "model.safetensors"))
        if size > 100_000_000:  # > 100MB means it's real
            print(f"Model already present ({size/1e6:.0f}MB) — skipping download")
            return

    print(f"Downloading model from Google Drive to {MODEL_DIR}/")
    # Download entire folder
    gdown.download_folder(DRIVE_URL, output=MODEL_DIR, quiet=False, use_cookies=False)
    print("Model download complete.")

if __name__ == "__main__":
    download_model()
