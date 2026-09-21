# layaForWeb

**A decision model that runs in your browser.** Give it some text and a few typed questions. It answers each one with probabilities and a confidence score, in a single forward pass, entirely on the visitor's device. No server, no API key, no data leaves the page.

**Live demo:** https://vishalmysore.github.io/layaForWeb/

Built on the Laya English checkpoint ([convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya), Apache-2.0), converted to ONNX and quantized for ONNX Runtime Web.

## What it does

You provide a piece of text (or a JSON object) and up to about 20 options per question. The model returns:

- **Choice**: a probability for each labeled option, plus the top pick
- **Score**: a probability for each step of an ordered scale, plus a continuous score
- **Yes/No**: the probability that a statement is true

Every answer carries a **confidence** (1 minus normalized entropy) and the model's temperature is calibrated per question type, so the numbers can be used to decide when to act automatically and when to hand off to a person.

Unlike a chat model, it does not generate text. It reads the input once and scores the options, which makes it deterministic and small enough to ship to a browser.

### Example

Input:

```js
const state = {
  ticket: {
    subject: "App crashes on launch",
    text: "Since the last update, the app closes as soon as I open it. I have a demo in one hour!"
  }
};

const questions = {
  team:    { type: "choice", instructions: "Which team should handle this?",
             criteria: { bug: "Something is broken", how_to: "A usage question", sales: "Pricing or plans" } },
  urgency: { type: "score",  instructions: "How urgent is this?",
             criteria: ["Can wait", "This week", "Today", "Right now"] },
  angry:   { type: "noul",   instructions: "The customer sounds angry" }
};

const result = await laya.systemOne(state, questions);
```

Output (abridged):

```json
{
  "team":    { "choice": "bug", "probabilities": { "bug": 0.957, "how_to": 0.026, "sales": 0.017 }, "confidence": 0.813 },
  "urgency": { "score": 1.62, "probabilities": { "0": 0.065, "1": 0.438, "2": 0.303, "3": 0.194 }, "confidence": 0.120 },
  "angry":   { "noul": 0.185, "confidence": 0.815 }
}
```

The team is clear (96% bug, high confidence). Urgency is spread across "This week", "Today" and "Right now", and the low confidence says so.

## The model

| | |
|---|---|
| Architecture | ModernBERT-large encoder plus a 2-layer transformer head (421M parameters in total) |
| Decoding | Non-autoregressive: all options scored in one pass, typed heads for choice, score and yes/no |
| Input | Up to 512 tokens of text plus question, options within 192 tokens |
| Language | English |
| Runtime | ONNX Runtime Web (WASM on CPU by default, WebGPU experimental) |

### Builds

Weight-only quantization; activations stay in floating point, which keeps the probabilities close to the original.

| Build | Size | Quantization |
|---|---|---|
| `q8e8` (default) | ~440 MB | int8 block-128 weights (`MatMulNBits`), int8 embeddings |
| `q4e8` | ~290 MB | int4 block-32 weights (`MatMulNBits`), int8 embeddings |
| `qdq8` (optional) | ~430 MB | int8 per-channel weights with runtime dequantization |

The weights are split into 24 MiB parts with SHA-256 hashes in `manifest.json`. The page caches them in the browser (Cache Storage, keyed by hash), so repeat visits skip the download.

### Fidelity to the PyTorch model

`scripts/verify_model.py` runs 48 questions (12 texts × 4 typed questions) through PyTorch and each ONNX build.

| Build | Same top answer | Mean max probability difference | Worst difference |
|---|---|---|---|
| fp32 ONNX | 100% | 0.0000 | 0.0000 |
| `q8e8` | 97.9% | 0.013 | 0.081 |
| `qdq8` | 97.9% | 0.014 | 0.059 |
| `q4e8` | 97.9% | 0.063 | 0.319 |

The CI browser test loads the site in headless Chromium and compares six saved cases against PyTorch outputs. For `q8e8`, 11 of 12 questions matched the top answer, with a worst probability difference of 0.059.

The calibration temperature was fitted on the full-precision model, so confidence values from the quantized builds are approximate.

## Try it locally

Requires Python 3.11, Node 20+ and about 8 GB of free disk for the conversion.

```
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements-build.txt
python scripts/build_model.py   # download, export to ONNX, quantize, split into parts
python scripts/verify_model.py  # compare every build with PyTorch
npm ci
node scripts/prepare_site.mjs   # assemble dist/
python serve.py                 # http://localhost:8000
```

Open http://localhost:8000 in Chrome or Edge, choose a build and press **Load model**. The demo has presets for support tickets, lead scoring, patient message routing, delivery exceptions, product reviews and agent guardrails. `python scripts/build_model.py --variants q8e8` builds only the default build.

## Deploy with GitHub Actions

1. Push this repo to GitHub.
2. Settings → Pages → Source: **GitHub Actions**.
3. Add a repository **secret** `HF_TOKEN` (a Hugging Face token with write access) and a **variable** `HF_MODEL_REPO` (for example `yourname/laya-en-web`).
4. Push to `main`, or run the workflow from the Actions tab.

The workflow (`.github/workflows/deploy.yml`) converts and quantizes the model, verifies each build against PyTorch, checks that the JavaScript port produces the same token sequences as Python, runs the site in headless Chromium, uploads the model files to your Hugging Face repo, and deploys a small Pages site (about 30 MB) that loads the model from there. The first run installs PyTorch and downloads the 843 MB checkpoint. Later runs reuse the cached model unless `scripts/build_model.py` or `requirements-build.txt` change.

Without `HF_MODEL_REPO`, the model files are placed inside the Pages site (about 760 MB with two builds; Pages sites are limited to 1 GB). Run the workflow manually with the `variants` input set to `q8e8` to shrink it.

To upload by hand:

```
set HF_TOKEN=hf_xxx
python scripts/upload_to_hf.py yourname/laya-en-web
node scripts/prepare_site.mjs --external-model-base https://huggingface.co/yourname/laya-en-web/resolve/main/
```

Any static host with CORS enabled works for the model files: open the page with `?modelBase=https://example.com/laya/model/`.

## Use it in your own page

`web/laya-core.js` exports the `Laya` class and the pieces it is built from. It mirrors the Python `laya` package: it builds the token sequence for each question, runs the ONNX graph, applies the calibration temperature and softmax, and formats the choice, score or yes/no result. `tests/seq_parity.mjs` checks that it builds identical token ids to Python on 24 cases, including emoji, right-to-left text, special tokens in the input, JSON state and very long input.

`loadModel()` in `web/app.js` shows the full loading sequence: fetch the tokenizer and config, download and reassemble the weight parts, create the ONNX Runtime session, then construct `new Laya(ort, session, tokenizer, cfg)`. After that, `laya.systemOne(state, questions)` returns the answers shown above.

## Backends

- **WASM (default)**: runs on the CPU and is the verified reference. A three-question call took about 2 to 5 seconds on a 2-core machine; modern laptops are faster. The page enables multithreading on GitHub Pages through a small service worker (`web/coi-sw.js`) that adds the required COOP/COEP headers.
- **WebGPU (experimental)**: the page includes a **Run comparison** button that checks WebGPU output against the PyTorch reference on your GPU. It has not been verified on real GPU hardware; on a software adapter the `qdq8` build gave incorrect probabilities.

## Repository layout

| Path | What it is |
|---|---|
| `web/` | the page: `index.html`, `app.js`, `laya-core.js`, service worker |
| `scripts/build_model.py` | download, export to ONNX, quantize, split into parts, write manifest |
| `scripts/verify_model.py` | compare each build with PyTorch on 48 questions, fail if it drifts too far |
| `scripts/upload_to_hf.py` | upload `build/model` to a Hugging Face model repo |
| `scripts/gen_fixtures.py` | regenerate test fixtures from the PyTorch model |
| `scripts/prepare_site.mjs` | assemble `dist/` from `web/`, npm packages and the model |
| `tests/` | token sequence parity test and headless browser smoke test |
| `serve.py` | local static server (`--no-coi` mimics GitHub Pages) |
| `.github/workflows/deploy.yml` | build, test and deploy workflow |
| `LICENSE`, `NOTICE.md`, `licenses/` | license text, notices and third-party license texts |

## License

The code in this repository (build scripts, demo page, JavaScript inference code) is licensed under the Apache License, Version 2.0; see `LICENSE`.

The model is a modified derivative of [Laya](https://huggingface.co/convaiinnovations/laya) by ConvAI Innovations (Apache-2.0), which is built on ModernBERT-large by Answer.AI and LightOn (Apache-2.0). The demo page ships ONNX Runtime Web (MIT) and Tokenizers.js (Apache-2.0). Copyright notices, the list of changes made to Laya, and the third-party license texts are in `NOTICE.md` and the `licenses/` folder, and they are included in the deployed site and in the model files on Hugging Face.

This is an unofficial browser port and is not affiliated with ConvAI Innovations.
