#!/usr/bin/env python3
"""Convert the English Laya checkpoint into browser-sized ONNX models.

    python scripts/build_model.py                      # builds q8e8 and q4e8
    python scripts/build_model.py --variants qdq8      # only the closest-to-original build

What it does
  1. Downloads the English checkpoint (repo root of convaiinnovations/laya on Hugging Face).
  2. Exports the whole network (ModernBERT-large encoder + typed decision heads) to one ONNX
     graph with dynamic batch and sequence length. Softmax, temperature and the
     choice/score/noul post-processing stay in JavaScript, exactly like laya/agent.py.
  3. Quantizes the weights (activations stay float, which is what keeps the probabilities close):
        qdq8  int8 per-channel weights for every MatMul and the embedding table (~430 MB)
        q8e8  int8 block-128 weights (MatMulNBits operator) + int8 embeddings    (~440 MB)
        q4e8  int4 block-32 weights (MatMulNBits operator) + int8 embeddings     (~290 MB)
  4. Splits each weight file into <= 24 MiB parts and writes model/manifest.json, so the site
     can be hosted on services with per-file size limits (GitHub Pages, git repos, CDNs).

Outputs (all under --out, default build/):
  build/en/        downloaded checkpoint
  build/onnx/      full, unsplit ONNX files (used by verify_model.py)
  build/model/     what the website serves: graph files, weight parts, tokenizer, config, manifest
"""
import argparse, gc, hashlib, json, os, shutil, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPO = "convaiinnovations/laya"
CHUNK = 24 * 1024 * 1024
LABELS = {
    "qdq8": "int8 weights, per-channel",
    "q8e8": "int8 weights, block-wise",
    "q4e8": "int4 weights (smaller, drifts more)",
}


MODEL_CARD = """---
license: apache-2.0
base_model: convaiinnovations/laya
library_name: onnx
tags: [onnx, onnxruntime-web, browser, decision-model, quantized]
---
# Laya (English) converted for the browser

Quantized ONNX conversion of the English checkpoint of [convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya)
(Apache-2.0), made for ONNX Runtime Web. Unofficial; not affiliated with ConvAI Innovations.

The weight files are split into parts (`*.onnx.data.partNNN`) and listed in `manifest.json`; the demo page reassembles them.
Build scripts: the `layaForWeb` repository that produced this folder.
"""


def log(msg):
    print(f"[build] {msg}", flush=True)


# --------------------------------------------------------------------------- stage: download
def stage_download(out):
    from huggingface_hub import snapshot_download
    dst = out / "en"
    patterns = ["model.safetensors", "rl_agent_config.json", "encoder/*", "tokenizer/*"]
    if (dst / "model.safetensors").exists() and (dst / "rl_agent_config.json").exists():
        log(f"checkpoint already in {dst}")
        return
    snapshot_download(REPO, local_dir=str(dst), allow_patterns=patterns)
    log(f"downloaded checkpoint to {dst}")


# --------------------------------------------------------------------------- stage: export
def stage_export(out):
    import torch, onnx
    import laya
    from laya.common import build_sequence, collate_items, QTYPES
    from torch.export import Dim

    torch.set_num_threads(max(1, os.cpu_count() or 1))
    torch.backends.mha.set_fastpath_enabled(False)  # keep nn.TransformerEncoderLayer exportable
    onnx_dir = out / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)

    agent = laya.load(str(out / "en"), device="cpu")
    model = agent.model.eval().float()

    class Wrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, input_ids, attention_mask, marker_pos, marker_mask, qtype):
            logits, act = self.m(input_ids, attention_mask, marker_pos, marker_mask, qtype)
            return logits, act

    state = {"ticket": {"subject": "App crashes on launch", "text": "Since the last update, the app closes as soon as I open it."}}
    qs = [
        {"t": "choice", "ins": "Which team should handle this?", "crit": {"bug": "Something is broken", "how_to": "A usage question", "sales": "Pricing or plans"}},
        {"t": "score", "ins": "How urgent is this?", "crit": ["Can wait", "This week", "Today", "Right now"]},
        {"t": "noul", "ins": "The customer sounds angry", "crit": None},
    ]
    items = []
    for q in qs:
        seq, mk = build_sequence(agent.tok, state, q, 512, 192)
        items.append({"ids": seq, "markers": mk, "qtype": QTYPES[q["t"]]})
    b = collate_items([items], agent.tok.pad_token_id)
    args = (b["input_ids"], b["attention_mask"], b["marker_pos"], b["marker_mask"], b["qtype"])

    # one shared Dim object per axis name, so the exporter treats them as the same dimension
    batch, seq, kk = Dim("batch", min=1, max=64), Dim("seq", min=8, max=512), Dim("k", min=2, max=255)
    dyn = {"input_ids": {0: batch, 1: seq}, "attention_mask": {0: batch, 1: seq},
           "marker_pos": {0: batch, 1: kk}, "marker_mask": {0: batch, 1: kk}, "qtype": {0: batch}}
    log("exporting to ONNX (torch.export path)...")
    prog = torch.onnx.export(
        Wrapper(model).eval(), args, dynamo=True, dynamic_shapes=dyn,
        input_names=["input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"],
        output_names=["logits", "act_logits"], opset_version=18, optimize=True)
    raw = onnx_dir / "laya_fp32_raw.onnx"
    prog.save(str(raw), external_data=True)
    del prog, model, agent
    gc.collect()

    # the exporter leaves stale shape annotations that trip onnxruntime's quantizer; drop them
    m = onnx.load(str(raw), load_external_data=False)
    del m.graph.value_info[:]
    onnx.save(m, str(onnx_dir / "laya_fp32.onnx"))
    # laya_fp32.onnx points at laya_fp32_raw.onnx.data
    log(f"exported: {(onnx_dir / 'laya_fp32_raw.onnx.data').stat().st_size / 1e6:.0f} MB of fp32 weights")


# --------------------------------------------------------------------------- stage: quantize
def _embedding_to_int8(m):
    import numpy as np
    from onnx import numpy_helper, helper, TensorProto
    g = m.graph
    inits = {i.name: i for i in g.initializer}
    new_nodes, drop, added = [], set(), []
    for node in g.node:
        if node.op_type == "Gather" and node.input[0] in inits:
            t = inits[node.input[0]]
            if t.data_type == TensorProto.FLOAT and len(t.dims) == 2 and t.dims[0] > 1000:
                w = numpy_helper.to_array(t).astype(np.float32)
                scale = np.maximum(np.abs(w).max(axis=1, keepdims=True) / 127.0, 1e-8).astype(np.float32)
                q = np.clip(np.round(w / scale), -127, 127).astype(np.int8)
                n = t.name
                added += [numpy_helper.from_array(q, n + "_q8"), numpy_helper.from_array(scale.reshape(-1), n + "_scale")]
                new_nodes.append(helper.make_node("DequantizeLinear", [n + "_q8", n + "_scale"], [n + "_dq"], axis=0, name=n + "_DQ"))
                drop.add(n)
                node.input[0] = n + "_dq"
    keep = [i for i in g.initializer if i.name not in drop]
    del g.initializer[:]
    g.initializer.extend(keep)
    g.initializer.extend(added)
    nodes = list(g.node)
    del g.node[:]
    g.node.extend(new_nodes + nodes)


def _save(m, path):
    import onnx
    name = Path(path).name
    onnx.save(m, str(path), save_as_external_data=True, all_tensors_to_one_file=True,
              location=name + ".data", size_threshold=1024)


def stage_qdq8(out):
    """Weight-only per-channel symmetric int8: initializer -> DequantizeLinear -> MatMul/Gather."""
    import numpy as np, onnx
    from onnx import numpy_helper, helper, TensorProto
    onnx_dir = out / "onnx"
    m = onnx.load(str(onnx_dir / "laya_fp32.onnx"), load_external_data=True)
    g = m.graph
    inits = {i.name: i for i in g.initializer}
    new_nodes, drop, added, n_q = [], set(), [], 0
    for node in g.node:
        target = None
        if node.op_type == "MatMul" and node.input[1] in inits:
            target, axis, idx = node.input[1], 1, 1  # B is [K, N]: one scale per output channel N
        elif node.op_type == "Gather" and node.input[0] in inits:
            t = inits[node.input[0]]
            if len(t.dims) == 2 and t.dims[0] > 1000:
                target, axis, idx = node.input[0], 0, 0  # embedding table: one scale per row
        if target is not None and target not in drop and inits[target].data_type == TensorProto.FLOAT:
            w = numpy_helper.to_array(inits[target]).astype(np.float32)
            red = 0 if axis == 1 else 1
            scale = np.maximum(np.abs(w).max(axis=red, keepdims=True) / 127.0, 1e-8).astype(np.float32)
            q = np.clip(np.round(w / scale), -127, 127).astype(np.int8)
            qn, sn, dn = target + "_q8", target + "_scale", target + "_dq"
            added += [numpy_helper.from_array(q, qn), numpy_helper.from_array(scale.reshape(-1), sn)]
            new_nodes.append(helper.make_node("DequantizeLinear", [qn, sn], [dn], axis=axis, name=target + "_DQ"))
            drop.add(target)
            n_q += 1
            node.input[idx] = dn
        elif target is not None and target in drop:
            node.input[idx] = target + "_dq"  # weight shared by several nodes
    keep = [i for i in g.initializer if i.name not in drop]
    del g.initializer[:]
    g.initializer.extend(keep)
    g.initializer.extend(added)
    nodes = list(g.node)
    del g.node[:]
    g.node.extend(new_nodes + nodes)
    _save(m, onnx_dir / "laya_qdq8.onnx")
    log(f"qdq8: quantized {n_q} tensors -> {(onnx_dir / 'laya_qdq8.onnx.data').stat().st_size / 1e6:.0f} MB")


def stage_nbits(out, name, bits, block):
    """Block-wise weight-only quantization with the MatMulNBits operator, plus int8 embeddings."""
    import onnx
    from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer, DefaultWeightOnlyQuantConfig
    onnx_dir = out / "onnx"
    m = onnx.load(str(onnx_dir / "laya_fp32.onnx"), load_external_data=True)
    cfg = DefaultWeightOnlyQuantConfig(block_size=block, is_symmetric=True, bits=bits)
    q = MatMulNBitsQuantizer(m, algo_config=cfg)
    q.process()
    tmp = f"laya_{name}_tmp.onnx"
    q.model.save_model_to_file(str(onnx_dir / tmp), use_external_data_format=True)
    del q, m
    gc.collect()
    mq = onnx.load(str(onnx_dir / tmp), load_external_data=True)
    _embedding_to_int8(mq)
    _save(mq, onnx_dir / f"laya_{name}.onnx")
    for f in (tmp, tmp + ".data"):
        (onnx_dir / f).unlink(missing_ok=True)
    log(f"{name} -> {(onnx_dir / f'laya_{name}.onnx.data').stat().st_size / 1e6:.0f} MB")


def stage_q4e8(out):
    stage_nbits(out, "q4e8", bits=4, block=32)


def stage_q8e8(out):
    stage_nbits(out, "q8e8", bits=8, block=128)


# --------------------------------------------------------------------------- stage: package
def stage_package(out, variants):
    onnx_dir, model_dir, en = out / "onnx", out / "model", out / "en"
    if model_dir.exists():
        shutil.rmtree(model_dir)
    model_dir.mkdir(parents=True)
    shutil.copy(en / "tokenizer" / "tokenizer.json", model_dir / "tokenizer.json")
    shutil.copy(en / "tokenizer" / "tokenizer_config.json", model_dir / "tokenizer_config.json")
    shutil.copy(en / "rl_agent_config.json", model_dir / "rl_agent_config.json")
    manifest = {"version": 1, "source": REPO, "chunk_bytes": CHUNK, "variants": {}}
    for v in variants:
        graph = onnx_dir / f"laya_{v}.onnx"
        data = onnx_dir / f"laya_{v}.onnx.data"
        shutil.copy(graph, model_dir / graph.name)
        parts, h, size = [], hashlib.sha256(), data.stat().st_size
        with open(data, "rb") as f:
            i = 0
            while True:
                buf = f.read(CHUNK)
                if not buf:
                    break
                h.update(buf)
                name = f"{data.name}.part{i:03d}"
                (model_dir / name).write_bytes(buf)
                parts.append(name)
                i += 1
        manifest["variants"][v] = {"label": LABELS[v], "onnx": graph.name,
                                   "data": {"name": data.name, "size": size, "sha256": h.hexdigest(), "parts": parts}}
        log(f"{v}: {size / 1e6:.0f} MB in {len(parts)} parts")
    (model_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (model_dir / "README.md").write_text(MODEL_CARD)
    total = sum(p.stat().st_size for p in model_dir.iterdir())
    log(f"model folder ready: {model_dir} ({total / 1e6:.0f} MB)")


# --------------------------------------------------------------------------- driver
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variants", default="q8e8,q4e8")
    ap.add_argument("--out", default=str(ROOT / "build"))
    ap.add_argument("--stage", choices=["download", "export", "qdq8", "q8e8", "q4e8", "package"], help=argparse.SUPPRESS)
    a = ap.parse_args()
    out = Path(a.out)
    variants = [v for v in a.variants.split(",") if v]
    for v in variants:
        if v not in LABELS:
            sys.exit(f"unknown variant {v!r}; choose from {', '.join(LABELS)}")
    if a.stage:  # child process: run exactly one stage
        if a.stage == "package":
            stage_package(out, variants)
        else:
            {"download": stage_download, "export": stage_export, "qdq8": stage_qdq8, "q8e8": stage_q8e8, "q4e8": stage_q4e8}[a.stage](out)
        return
    out.mkdir(parents=True, exist_ok=True)
    onnx_dir = out / "onnx"
    plan = ["download"]
    if not (onnx_dir / "laya_fp32.onnx").exists():
        plan.append("export")
    plan += [v for v in variants if not (onnx_dir / f"laya_{v}.onnx.data").exists()]
    plan.append("package")
    for stage in plan:  # one process per stage keeps peak memory low
        log(f"stage: {stage}")
        cmd = [sys.executable, __file__, "--stage", stage, "--out", str(out), "--variants", ",".join(variants)]
        r = subprocess.run(cmd)
        if r.returncode != 0 and stage == "download":
            # some networks block the Xet download backend; plain HTTPS works everywhere. The switch is read
            # when huggingface_hub is imported, so it has to be set for a fresh process.
            log("download failed; retrying without Xet")
            r = subprocess.run(cmd, env={**os.environ, "HF_HUB_DISABLE_XET": "1"})
        if r.returncode != 0:
            sys.exit(f"stage {stage} failed with exit code {r.returncode}")
    log("done")


if __name__ == "__main__":
    main()
