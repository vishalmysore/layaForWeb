#!/usr/bin/env python3
"""Upload a built model/ folder to a Hugging Face model repo, so the website does not have to carry it itself.

    set HF_TOKEN=hf_xxx              (a token with write access; macOS/Linux: export HF_TOKEN=hf_xxx)
    python scripts/upload_to_hf.py yourname/laya-en-web
    python scripts/upload_to_hf.py yourname/laya-typed-decisions-web --folder build-typed/model

Then build the site pointing at it:
    node scripts/prepare_site.mjs --external-model-base https://huggingface.co/yourname/laya-en-web/resolve/main/
    node scripts/prepare_site.mjs --external-model-base-typed https://huggingface.co/yourname/laya-typed-decisions-web/resolve/main/
"""
import argparse, os, sys
from pathlib import Path
from huggingface_hub import HfApi

ap = argparse.ArgumentParser()
ap.add_argument("repo", nargs="?", default=os.environ.get("HF_MODEL_REPO"), help="<user>/<repo> (or set HF_MODEL_REPO)")
ap.add_argument("--folder", default=None, help="the built model/ folder to upload (default: build/model)")
a = ap.parse_args()
if not a.repo:
    sys.exit("usage: python scripts/upload_to_hf.py <user>/<repo> [--folder build-typed/model]   (or set HF_MODEL_REPO)")
folder = Path(a.folder) if a.folder else Path(__file__).resolve().parent.parent / "build" / "model"
if not (folder / "manifest.json").exists():
    sys.exit(f"{folder} not found or has no manifest.json; run scripts/build_model.py first")
api = HfApi(token=os.environ.get("HF_TOKEN"))
api.create_repo(a.repo, repo_type="model", exist_ok=True)
api.upload_folder(repo_id=a.repo, repo_type="model", folder_path=str(folder), commit_message=f"Upload converted model from {folder} for ONNX Runtime Web")
print(f"uploaded. Model base URL: https://huggingface.co/{a.repo}/resolve/main/")
