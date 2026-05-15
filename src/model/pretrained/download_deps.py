"""
Run this once in src/model/pretrained/ to download the missing MotionBERT dependency.

    cd src/model/pretrained
    python3 download_deps.py
"""
from huggingface_hub import hf_hub_download

files = [
    "lib/model/drop.py",
    "lib/utils/tools.py",
]

for f in files:
    path = hf_hub_download(repo_id="walterzhu/MotionBERT", filename=f, local_dir=".")
    print(f"downloaded: {path}")

print("Done.")
