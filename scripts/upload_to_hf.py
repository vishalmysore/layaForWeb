#!/usr/bin/env python3
"""Upload build/model to a Hugging Face model repo, so the website does not have to carry the 700+ MB itself.

    set HF_TOKEN=hf_xxx              (a token with write access; macOS/Linux: export HF_TOKEN=hf_xxx)
    python scripts/upload_to_hf.py yourname/laya-en-web

Then build the site pointing at it:
    node scripts/prepare_site.mjs --external-model-base https://huggingface.co/yourname/laya-en-web/resolve/main/
"""
import os, sys
from pathlib import Path
from huggingface_hub import HfApi

repo = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("HF_MODEL_REPO")
if not repo:
    sys.exit("usage: python scripts/upload_to_hf.py <user>/<repo>   (or set HF_MODEL_REPO)")
folder = Path(__file__).resolve().parent.parent / "build" / "model"
if not (folder / "manifest.json").exists():
    sys.exit("build/model not found; run scripts/build_model.py first")
api = HfApi(token=os.environ.get("HF_TOKEN"))
api.create_repo(repo, repo_type="model", exist_ok=True)
api.upload_folder(repo_id=repo, repo_type="model", folder_path=str(folder), commit_message="Upload converted Laya (English) for ONNX Runtime Web")
print(f"uploaded. Model base URL: https://huggingface.co/{repo}/resolve/main/")
