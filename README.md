# layaForWeb

Run the English [Laya](https://huggingface.co/convaiinnovations/laya) decision model inside a browser tab. You give it some text and a few typed questions (a labeled choice, an ordered score, a yes/no), and it answers with probabilities, entirely on the visitor's machine. No server, no API key, nothing leaves the page.

Laya is not a chat model, so WebLLM cannot run it. It is an encoder (ModernBERT-large plus small decision heads) that answers every question in one forward pass, which is exactly what ONNX Runtime Web is built for. This repository converts the published checkpoint to ONNX, shrinks it, checks it against the original PyTorch model, and publishes a static demo page with GitHub Pages.

## Deploy with GitHub Actions

Push this folder as the root of a new GitHub repository, then open the repository's Settings, go to Pages, and set Source to GitHub Actions. Pushing to `main` (or running the workflow by hand from the Actions tab) builds and deploys the site. The first run takes a while, since it installs PyTorch, downloads the 843 MB checkpoint from Hugging Face, converts it and runs the tests. Later runs reuse the cached model unless `scripts/build_model.py` or `requirements-build.txt` changes.

The workflow in `.github/workflows/deploy.yml` does this, in order: converts and quantizes the model, compares every build with the original PyTorch model on 48 questions and fails if one drifts further than allowed, installs the web dependencies, checks that the JavaScript port builds token sequences identical to Python's, assembles the site into `dist/`, loads the finished site in headless Chromium and checks its answers against saved PyTorch outputs, and finally uploads and deploys it.

By default the workflow publishes two builds, `q8e8` and `q4e8`, and the Pages site comes to about 760 MB. GitHub Pages sites are limited to 1 GB, so there is not much room to add more. Two ways to make it lighter: run the workflow by hand and set the `variants` input to `q8e8`, or keep the model on Hugging Face instead (next section), which leaves the Pages site at roughly 30 MB. Weights are split into 24 MiB parts, so no single file comes near GitHub's file size limits, and the page stitches the parts back together in memory.

## Keeping the model on Hugging Face

The converted files can live in a Hugging Face model repo, and the page can load them from there. I checked that Hugging Face's `resolve` URLs answer cross-origin requests (they echo the requesting origin, and the CDN they redirect to allows any origin), which is what a page on `github.io` needs. To do it, create a repository variable named `HF_MODEL_REPO` (for example `yourname/laya-en-web`) and a repository secret named `HF_TOKEN` with write access. The workflow then uploads `build/model` to that repo and builds the Pages site with a `site-config.json` that points at it. You can do the same by hand:

```
set HF_TOKEN=hf_xxx
python scripts/upload_to_hf.py yourname/laya-en-web
node scripts/prepare_site.mjs --external-model-base https://huggingface.co/yourname/laya-en-web/resolve/main/
```

I could not test the upload step itself, because that needs your token. It uses the standard `HfApi.upload_folder` call. The model card that goes with the files declares Apache-2.0 and names Laya as the base model, which is what its license asks for.

GitHub Pages cannot send the `Cross-Origin-Opener-Policy` and `Cross-Origin-Embedder-Policy` headers that WebAssembly needs for multithreading. The page registers a small service worker (`web/coi-sw.js`) that adds them and reloads once, so the CPU fallback can use several threads even on Pages. The smoke test serves the site without those headers on purpose, to prove this works.

## Run it on your own machine

You need Python 3.11, Node 20 or newer, and about 8 GB of free disk for the conversion. In a terminal in this folder:

```
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements-build.txt
python scripts/build_model.py
python scripts/verify_model.py
npm ci
node scripts/prepare_site.mjs
python serve.py
```

Then open http://localhost:8000 in Chrome or Edge, pick a build, and press Load model. (`python serve.py` serves the `dist/` folder; the model files are inside it, so this works without any hosting.) The conversion takes a few minutes on a laptop. `python scripts/build_model.py --variants q8e8` builds only the int8 version.

## What gets built

The whole network is exported as one ONNX graph with dynamic batch and sequence length. The graph takes token ids, an attention mask, the positions of the option markers, and the question type, and returns one logit per option. Everything around it stays in JavaScript and mirrors the Python package line by line: building the token sequence for each question (`buildSequence` in `web/laya-core.js`), applying the calibration temperature, taking the softmax, and turning it into a choice, a score or a yes/no probability. `tests/seq_parity.mjs` checks that the JavaScript builds exactly the same token ids as Python for 24 cases, including emoji, right-to-left text, special tokens typed into the state, JSON states and very long input.

Weight-only quantized builds are produced. Activations stay in floating point, which is what keeps the probabilities close to the original, since Laya's confidence numbers are the whole point of the model.

| Build | Size on disk | How it is quantized |
|---|---|---|
| `q8e8` (default) | about 440 MB | int8 block-128 weights using the `MatMulNBits` operator for MatMuls, int8 embeddings |
| `q4e8` | about 290 MB | int4 block-32 weights (`MatMulNBits`) for MatMuls, int8 embeddings |
| `qdq8` (optional) | about 430 MB | int8 per-channel weights stored as int8 plus a scale and dequantized by the runtime; add it with `--variants qdq8,q8e8,q4e8` |

`q8e8` and `qdq8` are about equally faithful, but `q8e8` uses an operator that ONNX Runtime has dedicated CPU and WebGPU kernels for, so it is the safer default.

I also tried the common dynamic int8 quantization (activations quantized on the fly). It was much worse, with probability differences up to 0.35 on a single question, so it is not used.

## How close is it to the original?

`scripts/verify_model.py` runs 12 short texts against 4 typed questions each, 48 questions in all, through the original PyTorch model and through each ONNX file, and compares the resulting probabilities. These are the numbers from the build in this repository:

| Build | Same top answer | Mean of the largest probability difference per question | Worst single difference |
|---|---|---|---|
| fp32 ONNX (unquantized) | 100% | 0.0000 | 0.0000 |
| `q8e8` | 97.9% | 0.013 | 0.081 |
| `qdq8` | 97.9% | 0.014 | 0.059 |
| `q4e8` | 97.9% | 0.063 | 0.319 |

The in-browser check on the page (and in the CI smoke test) runs six saved cases through the WASM backend. For `q8e8` it agreed with PyTorch on the top answer for 11 of 12 questions, with a largest probability difference of 0.059 (`qdq8` gave 11 of 12 and 0.069), and for `qdq8` the one disagreement was a near tie (37% against 35% in the original). This measures how faithfully the conversion reproduces Laya, not how accurate Laya is. Laya's temperature scaling was fitted on the full-precision model, so treat the confidence values from either quantized build as approximate and re-check them on your own labeled data before letting them trigger anything automatically.

## Speed and backends

The page defaults to the WASM backend, which runs on the CPU and is the one I could check against PyTorch. On a two-core cloud machine a three-question call took roughly 2 to 5 seconds, so expect it to be quicker on a modern laptop but far from Laya's published 33 ms, which is a GPU number for the PyTorch model.

WebGPU is offered as experimental. I had no real GPU to test on, only a software WebGPU adapter (SwiftShader), and on that adapter the per-channel int8 build (`qdq8`) returned wrong probabilities, for example 0 for a yes/no question the original answers with 18%. I could not tell whether that is a bug in that quantization format's WebGPU path or a limitation of the software adapter. Because of that I made WASM the default, and made the default build `q8e8`, whose operator has dedicated WebGPU kernels (still untested on a real GPU). On your own GPU, run the "Run comparison" button on the page: it takes a minute and tells you whether WebGPU agrees with PyTorch before you trust it. If you find WebGPU results that are off, please treat WASM as the reference.

Loading is the slow part for a visitor, since the page has to download the whole weight file (430 MB or 290 MB). It keeps the parts in the browser's Cache Storage, keyed by the file's hash, so a repeat visit skips the download (in my test the second load fetched all 12 parts from cache in about a second). The cache lives in the visitor's browser and counts against its storage quota, and clearing site data removes it.

## What Laya is and is not good at

Laya is a small model, and the demo page includes an example that shows why you should test it on your own data. Asked whether an agent's unreviewed `DELETE` on a production table is safe to run without approval, the original PyTorch model answered "yes" with 83% probability, and the converted models give similar answers (84% and 89%). Asked instead whether the command is destructive, it said only 63%. That is one hand-written case, not a benchmark, but it is a useful reminder that wording matters and that a confident number is not a correct one. Its own documentation states that it supports about 20 options per choice question, that options must fit in 192 tokens, and that the text plus the question are cut at 512 tokens. This build covers the English checkpoint only.

## Hosting the model somewhere else

The page reads `model/manifest.json` and the files it lists from `./model/` next to it. Add `?modelBase=https://example.com/laya/model/` to the URL to load them from another host instead. That host has to send CORS headers that allow your site's origin. I have not tested this with a specific host; anything that serves static files with CORS enabled, including Hugging Face repositories and object storage buckets, is a candidate. Upload the contents of `build/model/`.

## Layout

| Path | What it is |
|---|---|
| `scripts/build_model.py` | download, export to ONNX, quantize, split into parts, write the manifest |
| `scripts/verify_model.py` | compare each build with PyTorch on 48 questions, fail if it drifts too far |
| `scripts/gen_fixtures.py` | regenerate the small test fixtures from the PyTorch model |
| `scripts/prepare_site.mjs` | assemble `dist/` from `web/`, the npm packages and `build/model/` |
| `web/` | the page: `index.html`, `app.js`, `laya-core.js` (the inference code), the service worker |
| `tests/` | token sequence parity test and the headless browser smoke test |
| `serve.py` | local static server; `--no-coi` mimics GitHub Pages |
| `.github/workflows/deploy.yml` | the build, test and deploy workflow |

## Licenses

Laya, its tokenizer and weights are Apache-2.0 from ConvAI Innovations. ONNX Runtime Web is MIT and Tokenizers.js is Apache-2.0. See `NOTICE.md`. This project is an unofficial port and is not affiliated with ConvAI Innovations. Add your own license file for the code in this repository.
