import os
import gdown

MODEL_FILE_ID = "17Xhz_Jygy99Fg5PO2dNfOQhfoKkZFB7M"
MODEL_NAME    = "wall_crack.pt"

if __name__ == "__main__":
    os.makedirs("uploads", exist_ok=True)
    if not os.path.exists(MODEL_NAME):
        print(f"⬇️  Downloading {MODEL_NAME} from Google Drive...")
        gdown.download(id=MODEL_FILE_ID, output=MODEL_NAME, quiet=False)
        if os.path.exists(MODEL_NAME):
            size_mb = os.path.getsize(MODEL_NAME) / 1024 / 1024
            print(f"✅ Downloaded {MODEL_NAME} ({size_mb:.1f} MB)")
        else:
            print("❌ Download failed — app will use yolov8n.pt fallback")
    else:
        print(f"✅ Model already exists: {MODEL_NAME}")
```
Save.

**Step 3 — Push:**
```
git add .
git commit -m "fix model download using gdown"
git push
```

---

**But for now your app is already live and working with yolov8n.pt!** Open it and test:
```
https://crackscansystem.onrender.com