"""
Download the T5 model weights from Google Drive at build time.
The tokenizer is loaded directly from HuggingFace (google-t5/t5-small)
so we only need model.safetensors + config files from Drive.

Run:  python download_model.py
Render build command: pip install -r requirements.txt && python download_model.py
"""

import os
import gdown

MODEL_DIR       = os.getenv("MODEL_DIR", "./model")
# Set this to the shareable link of the entire Drive FOLDER
DRIVE_FOLDER_URL = os.getenv("MODEL_DRIVE_URL", "")

# Individual file IDs - more reliable than folder download
# Set these as env vars in Render, OR extract from the folder URL
# Format: https://drive.google.com/file/d/FILE_ID/view
MODEL_FILE_ID        = os.getenv("MODEL_FILE_ID", "")         # model.safetensors
CONFIG_FILE_ID       = os.getenv("CONFIG_FILE_ID", "")        # config.json
GEN_CONFIG_FILE_ID   = os.getenv("GEN_CONFIG_FILE_ID", "")    # generation_config.json


def download_file(file_id: str, dest_path: str, label: str):
    if not file_id:
        print(f"  Skipping {label} — no file ID set")
        return False
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1000:
        print(f"  {label} already present — skipping")
        return True
    print(f"  Downloading {label}...")
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, dest_path, quiet=False, fuzzy=True)
    if os.path.exists(dest_path):
        size_mb = os.path.getsize(dest_path) / 1e6
        print(f"  {label} downloaded ({size_mb:.1f} MB)")
        return True
    print(f"  ERROR: {label} download failed")
    return False


def download_via_folder(folder_url: str):
    """Fallback: try gdown folder download"""
    print(f"Attempting folder download from {folder_url}")
    try:
        gdown.download_folder(folder_url, output=MODEL_DIR, quiet=False, use_cookies=False)
        return True
    except Exception as e:
        print(f"Folder download failed: {e}")
        return False


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    weights_path = os.path.join(MODEL_DIR, "model.safetensors")

    # Check if already fully downloaded
    if os.path.exists(weights_path) and os.path.getsize(weights_path) > 100_000_000:
        print(f"Model weights already present ({os.path.getsize(weights_path)/1e6:.0f}MB) — skipping")
        return

    print("Downloading model files...")

    if MODEL_FILE_ID:
        # Individual file download - most reliable
        download_file(MODEL_FILE_ID,      weights_path,                              "model.safetensors")
        download_file(CONFIG_FILE_ID,     os.path.join(MODEL_DIR, "config.json"),    "config.json")
        download_file(GEN_CONFIG_FILE_ID, os.path.join(MODEL_DIR, "generation_config.json"), "generation_config.json")
    elif DRIVE_FOLDER_URL:
        # Folder download fallback
        download_via_folder(DRIVE_FOLDER_URL)
    else:
        print("WARNING: No MODEL_FILE_ID or MODEL_DRIVE_URL set.")
        print("Model must be present in MODEL_DIR before starting.")
        return

    # Verify
    if os.path.exists(weights_path) and os.path.getsize(weights_path) > 100_000_000:
        print(f"Model download complete. ({os.path.getsize(weights_path)/1e6:.0f}MB)")
    else:
        print("ERROR: model.safetensors missing or too small after download.")
        raise RuntimeError("Model download failed — check Drive file IDs and share permissions.")


if __name__ == "__main__":
    main()
