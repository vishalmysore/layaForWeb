#!/usr/bin/env python3
"""Compare every built ONNX variant with the original PyTorch model on 48 typed questions.

    python scripts/verify_model.py                # checks build/
    python scripts/verify_model.py --out build-typed

Works against any checkpoint build_model.py produced, using that checkpoint's own sequence-length
limits from <out>/en/rl_agent_config.json (see build_model.py for why this is read dynamically
rather than assumed).

The exported fp32 graph must match PyTorch almost exactly. Quantized builds are allowed to drift a little;
the thresholds below fail the build if they drift more than we measured when this was written.
"""
import argparse, json, os, sys
from pathlib import Path
import numpy as np, torch, onnxruntime as ort
import laya
from laya.common import build_sequence, collate_items, QTYPES, temp_bucket, clamp_temperature

ROOT = Path(__file__).resolve().parent.parent
# (min top-answer agreement, max mean of the largest per-question probability difference)
LIMITS = {"fp32": (1.00, 0.002), "qdq8": (0.93, 0.03), "q8e8": (0.93, 0.03), "q4e8": (0.90, 0.12)}
FILES = {"fp32": "laya_fp32.onnx", "qdq8": "laya_qdq8.onnx", "q8e8": "laya_q8e8.onnx", "q4e8": "laya_q4e8.onnx"}

STATES = [
    "Customer wrote: I was charged twice for the same subscription this month and support has not replied in five days. This is unacceptable.",
    "Claim report: rear-ended at a traffic light, minor bumper damage, no injuries, photos attached, other driver admitted fault at the scene.",
    "Flight 214 was cancelled due to a crew shortage. Passenger has a connecting flight and travels with two small children.",
    "Package scanned at depot. Address label partially unreadable. Recipient phone number missing. Delivery attempt scheduled tomorrow.",
    "Patient message: Can I get a refill on my usual allergy medication? I also wanted to ask about my appointment next Tuesday.",
    "Code review comment: this function silently swallows exceptions and returns None, callers assume it always returns a list.",
    "Email: Congratulations, you have won a free cruise! Click the link and confirm your details in the next 10 minutes to claim.",
    "Meeting notes: team agreed to move the launch to next quarter because the integration tests are still failing on two platforms.",
    "User review: Battery life is great and the screen is sharp, but the speaker crackles at high volume. Would still recommend.",
    "Agent plan: rename the temporary folder, then upload the exported report to the shared team drive and post a link in the channel.",
    "Support chat: Hi! How do I change the language of the app to French? I could not find it in settings.",
    "Incident: the login service latency doubled after the last deploy, error rate at 4%, on-call has been paged twice.",
]
QDEFS = [
    {"type": "choice", "instructions": "What is the main topic of this text?", "criteria": {"complaint": "A complaint or problem", "question": "A question or request for help", "update": "A status update", "spam": "Unwanted or fraudulent message"}},
    {"type": "score", "instructions": "How urgent is this?", "criteria": ["Not urgent", "Low", "Medium", "High", "Critical"]},
    {"type": "noul", "instructions": "A human should review this before any automatic action"},
    {"type": "noul", "instructions": "The author sounds frustrated or negative"},
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "build"), help="the --out directory build_model.py wrote to (default: %(default)s)")
    a = ap.parse_args()
    OUT = Path(a.out)

    torch.set_num_threads(os.cpu_count() or 1)
    agent = laya.load(str(OUT / "en"), device="cpu")
    cfg = json.loads((OUT / "en" / "rl_agent_config.json").read_text())
    max_len, head_max_len = int(cfg.get("max_len", 512)), int(cfg.get("head_max_len", 192))
    variants = [v for v in FILES if (OUT / "onnx" / FILES[v]).exists()]
    sessions = {v: ort.InferenceSession(str(OUT / "onnx" / FILES[v]), providers=["CPUExecutionProvider"]) for v in variants}
    stats = {v: {"same": [], "dp": [], "dscore": []} for v in variants}

    for state in STATES:
        items, iq = [], []
        for d in QDEFS:
            q = agent._to_internal(d)
            iq.append(q)
            seq, mk = build_sequence(agent.tok, state, q, max_len, head_max_len)
            items.append({"ids": seq, "markers": mk, "qtype": QTYPES[q["t"]]})
        b = collate_items([items], agent.tok.pad_token_id)
        with torch.no_grad():
            ref = agent.model(b["input_ids"], b["attention_mask"], b["marker_pos"], b["marker_mask"], b["qtype"])[0].numpy()
        feeds = {"input_ids": b["input_ids"].numpy().astype(np.int64), "attention_mask": b["attention_mask"].numpy().astype(np.int64),
                 "marker_pos": b["marker_pos"].numpy().astype(np.int64), "marker_mask": b["marker_mask"].numpy(), "qtype": b["qtype"].numpy().astype(np.int64)}
        for v in variants:
            lo = sessions[v].run(None, feeds)[0]
            for r, q in enumerate(iq):
                k = len(items[r]["markers"]); qt = QTYPES[q["t"]]
                ts = clamp_temperature(agent.temperature_by_options.get(temp_bucket(qt, k), agent.temperature[qt]))
                def P(l):
                    z = l[r, :k] / ts; e = np.exp(z - z.max()); return e / e.sum()
                pr, po = P(ref), P(lo)
                stats[v]["dp"].append(float(np.abs(pr - po).max()))
                stats[v]["same"].append(bool(pr.argmax() == po.argmax()))
                if q["t"] == "score":
                    stats[v]["dscore"].append(abs(float((np.arange(k) * pr).sum() - (np.arange(k) * po).sum())))

    report, failed = {}, False
    lines = ["| build | questions | same top answer | mean max prob diff | p95 | worst | mean score diff |", "|---|---|---|---|---|---|---|"]
    for v in variants:
        s = stats[v]
        agree, mean_dp = float(np.mean(s["same"])), float(np.mean(s["dp"]))
        report[v] = {"n": len(s["dp"]), "agree": agree, "mean_dp": mean_dp, "p95_dp": float(np.percentile(s["dp"], 95)),
                     "worst_dp": float(np.max(s["dp"])), "mean_score_diff": float(np.mean(s["dscore"]))}
        lim = LIMITS[v]
        ok = agree >= lim[0] and mean_dp <= lim[1]
        failed |= not ok
        lines.append(f"| {v}{'' if ok else ' (FAILED)'} | {report[v]['n']} | {agree * 100:.1f}% | {mean_dp:.4f} | {report[v]['p95_dp']:.4f} | {report[v]['worst_dp']:.4f} | {report[v]['mean_score_diff']:.3f} |")
    table = "\n".join(lines)
    print(table)
    (OUT / "verify_report.json").write_text(json.dumps(report, indent=2))
    (OUT / "verify_report.md").write_text(table + "\n")
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as f:
            f.write("### Model check against the original PyTorch model\n\n" + table + "\n")
    if failed:
        sys.exit("verification failed: a build drifted more than allowed")


if __name__ == "__main__":
    main()
