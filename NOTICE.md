# Notice

layaForWeb is an unofficial browser port of the English Laya checkpoint. It is not affiliated with or endorsed by ConvAI Innovations, Microsoft, Hugging Face, Answer.AI or LightOn.

## This project

The build scripts, the demo page and the JavaScript inference code in this repository are Copyright 2026 vishalmysore and licensed under the Apache License, Version 2.0 (see `LICENSE`).

## Laya (model weights, tokenizer, reference code)

- Source: https://huggingface.co/convaiinnovations/laya and https://github.com/NandhaKishorM/laya (PyPI: `laya`)
- Copyright: ConvAI Innovations
- License: Apache License, Version 2.0 (see `LICENSE`)

**Changes made in this project** (Apache-2.0 section 4b):

- The English checkpoint was exported to ONNX.
- The weights were quantized (weight-only int8 or int4, int8 embeddings) and split into 24 MiB parts.
- `web/laya-core.js` is a JavaScript port of the Python `laya/common.py` (`build_sequence`) and `laya/agent.py` (`system_one`) inference code.

The quantized files distributed with this project are modified derivatives of Laya and are not the original release. Their outputs differ slightly from the original, as documented in the README.

## ModernBERT

The encoder architecture and initial weights of Laya come from ModernBERT-large by Answer.AI and LightOn, released under the Apache License, Version 2.0. https://huggingface.co/answerdotai/ModernBERT-large

## Third-party software shipped with the demo page

- **ONNX Runtime Web** (`onnxruntime-web` 1.30.0, files in `vendor/`): Copyright (c) Microsoft Corporation, MIT License. The license text is in `licenses/onnxruntime-LICENSE.txt`. The WebAssembly binary includes further third-party components whose notices are in `licenses/onnxruntime-ThirdPartyNotices.txt`.
- **Tokenizers.js** (`@huggingface/tokenizers` 0.2.0, `vendor/tokenizers.min.mjs`): Hugging Face, Apache License, Version 2.0. The license text is in `licenses/tokenizers.js-LICENSE.txt`.

## Build-time tools (not distributed)

PyTorch, Transformers, safetensors, huggingface_hub, NumPy, ONNX, ONNX Script, ONNX Runtime and the `laya` Python package are used only to convert the model and are not part of the published site or the model files. Each is available under its own open-source license.
