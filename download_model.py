"""
download_model.py
Run this once on Render startup to pull model weights.
Add to Render Build Command:
  pip install -r requirements.txt && python download_model.py
"""

import os
import urllib.request

# ─────────────────────────────────────────────────────────
# PASTE YOUR GOOGLE DRIVE DIRECT DOWNLOAD LINK BELOW
# How to get it:
#   1. Upload best_crack.pt to Google Drive
#   2. Right-click → Share → Anyone with the link
#   3. Copy the file ID from the share URL:
#      https://drive.google.com/file/d/FILE_ID_HERE/view
#   4. Replace FILE_ID_HERE below
# ─────────────────────────────────────────────────────────
MODEL_FILE_ID = "15EjnkQEZ-15EjnkQEZ-HGxj2PLKOKPLwpphuNsojmr"
MODEL_NAME    = "wall_crack.pt"

def download_from_gdrive(file_id, dest):
    url = f"https://drive.google.com/uc?export=download&id=15EjnkQEZ-17Xhz_Jygy99Fg5PO2dNfOQhfoKkZFB7M"
    print(f"⬇️  Downloading {dest} from Google Drive...")
    try:
        urllib.request.urlretrieve(url, dest)
        size_mb = os.path.getsize(dest) / 1024 / 1024
        print(f"✅ Downloaded {dest} ({size_mb:.1f} MB)")
    except Exception as e:
        print(f"❌ Download failed: {e}")

if __name__ == "__main__":
    os.makedirs("uploads", exist_ok=True)

    if not os.path.exists(MODEL_NAME):
        if MODEL_FILE_ID == "FILE_ID_HERE":
            print("⚠️  No Google Drive file ID set — skipping model download.")
            print("   App will use yolov8n.pt (auto-downloaded by ultralytics)")
        else:
            download_from_gdrive(MODEL_FILE_ID, MODEL_NAME)
    else:
        print(f"✅ Model already exists: {MODEL_NAME}")