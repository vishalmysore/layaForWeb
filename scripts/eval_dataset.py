"""Run the labeled cases in tests/data through the original PyTorch Laya model and report accuracy.

    pip install laya
    python scripts/eval_dataset.py                      # downloads convaiinnovations/laya, runs on CPU
    python scripts/eval_dataset.py --model build/en     # use a local checkpoint folder
    python scripts/eval_dataset.py --out results.json   # also write every answer to a file

How answers are scored:
  choice  correct when the most probable option equals the label
  noul    correct when P(yes) >= 0.5 matches the label
  score   correct when the most probable level equals the label; "within one" allows one level off

Cases can list question names under "hard": those labels are judgement calls, so they are
reported separately ("clear" = every other label).
"""
import argparse, glob, json, os, sys, time


def top(probs):
    return max(probs, key=probs.get)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="convaiinnovations/laya")
    ap.add_argument("--data", default=os.path.join(os.path.dirname(__file__), "..", "tests", "data"))
    ap.add_argument("--domain", help="only run this domain (file name without .json)")
    ap.add_argument("--threshold", type=float, default=0.90, help="confidence needed to act without a person")
    ap.add_argument("--out", help="write all answers to this JSON file")
    args = ap.parse_args()

    import laya
    device = "cpu"
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        pass
    agent = laya.load(args.model, device=device)
    print(f"model: {args.model}  device: {device}\n")

    files = sorted(p for p in glob.glob(os.path.join(args.data, "*.json")) if not p.endswith("index.json"))
    if args.domain:
        files = [p for p in files if os.path.basename(p) == args.domain + ".json"]
    if not files:
        sys.exit("no data files found")

    rows, dump = [], []   # rows: (domain, qtype, correct, within1, hard, confidence)
    t0 = time.time()
    for path in files:
        doc = json.load(open(path, encoding="utf-8"))
        for case in doc["cases"]:
            res = agent.system_one(case["state"], doc["questions"])["answers"]
            hard = set(case.get("hard", []))
            for q, spec in doc["questions"].items():
                a = res[q]
                want = case["expected"][q]
                if spec["type"] == "choice":
                    got = top(a["probabilities"]); ok = got == want; w1 = ok
                elif spec["type"] == "noul":
                    got = a["noul"] >= 0.5; ok = got == want; w1 = ok
                else:
                    got = int(top(a["probabilities"])); ok = got == want; w1 = abs(got - want) <= 1
                rows.append((doc["domain"], spec["type"], ok, w1, q in hard, a["confidence"]))
                dump.append({"domain": doc["domain"], "id": case["id"], "question": q, "type": spec["type"],
                             "expected": want, "got": got, "correct": ok, "confidence": round(a["confidence"], 4),
                             "hard": q in hard, "answer": a})
    secs = time.time() - t0

    def pct(n, d):
        return f"{100 * n / d:5.1f}%" if d else "    - "

    def summarize(sel):
        n = len(sel)
        return n, sum(r[2] for r in sel), sum(r[3] for r in sel)

    print(f"{'domain':22} {'answers':>7} {'correct':>8} {'clear only':>11}")
    for dom in sorted({r[0] for r in rows}):
        sel = [r for r in rows if r[0] == dom]
        clear = [r for r in sel if not r[4]]
        n, c, _ = summarize(sel); n2, c2, _ = summarize(clear)
        print(f"{dom:22} {n:7d} {pct(c, n):>8} {pct(c2, n2):>11}")
    n, c, _ = summarize(rows); n2, c2, _ = summarize([r for r in rows if not r[4]])
    print(f"{'ALL':22} {n:7d} {pct(c, n):>8} {pct(c2, n2):>11}\n")

    print("by question type (all labels):")
    for t in ("choice", "score", "noul"):
        sel = [r for r in rows if r[1] == t]
        n, c, w = summarize(sel)
        extra = f"   within one level: {pct(w, n)}" if t == "score" else ""
        print(f"  {t:7} {n:4d} answers  correct {pct(c, n)}{extra}")

    hi = [r for r in rows if r[5] >= args.threshold]
    lo = [r for r in rows if r[5] < args.threshold]
    print(f"\nwith a confidence threshold of {args.threshold:.2f}:")
    print(f"  acts automatically on {len(hi)} of {len(rows)} answers, {pct(sum(r[2] for r in hi), len(hi))} of them correct")
    print(f"  hands {len(lo)} answers to a person; the model was right on {pct(sum(r[2] for r in lo), len(lo))} of those")
    print(f"\n{len(rows)} answers in {secs:.0f}s")

    if args.out:
        json.dump(dump, open(args.out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
