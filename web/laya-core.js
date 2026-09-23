// Laya (English) in the browser: sequence builder, ONNX Runtime Web inference, calibrated outputs.
// This is a line-by-line port of laya/common.py (build_sequence) and laya/agent.py (system_one).
//
// Derived from Laya by ConvAI Innovations (https://github.com/NandhaKishorM/laya), licensed under the
// Apache License, Version 2.0 (see LICENSE and NOTICE.md). Changed: ported from Python to JavaScript and
// adapted to run the exported ONNX graph with ONNX Runtime Web.

const MASK = "[MASK]";
const QTYPES = { choice: 0, score: 1, noul: 2 };
const QTYPE_NAMES = ["choice", "score", "noul"];

// Python's json.dumps(..., ensure_ascii=False) uses ", " and ": " separators; JS JSON.stringify uses none.
export function pyDumps(v) {
  if (v === null || v === undefined) return "null";
  if (typeof v === "boolean") return v ? "true" : "false";
  if (typeof v === "number") return Number.isInteger(v) ? String(v) : String(v);
  if (typeof v === "string") return JSON.stringify(v);
  if (Array.isArray(v)) return "[" + v.map(pyDumps).join(", ") + "]";
  return "{" + Object.entries(v).map(([k, x]) => JSON.stringify(k) + ": " + pyDumps(x)).join(", ") + "}";
}
const serializeState = (s) => (typeof s === "string" ? s : pyDumps(s));
const renderCriterion = (v) => (typeof v === "string" ? v : pyDumps(v));

export function toInternal(q) {
  let crit = q.criteria;
  if (q.type === "choice" && Array.isArray(crit)) crit = Object.fromEntries(crit.map((c) => [c, null]));
  const ins = typeof q.instructions === "string" ? q.instructions : pyDumps(q.instructions);
  return { t: q.type, ins, crit };
}

export function renderOptions(q) {
  if (q.t === "choice") {
    return Object.entries(q.crit).map(([k, v]) => (v === null || v === undefined || v === "" ? k : `${k}: ${renderCriterion(v)}`));
  }
  if (q.t === "score") return q.crit.map((c, i) => `level ${i}: ${renderCriterion(c)}`);
  const crit = q.crit || {};
  const f = crit.false, t = crit.true;
  return [
    "false: " + (f !== null && f !== undefined && f !== "" ? renderCriterion(f) : "no, the statement does not hold"),
    "true: " + (t !== null && t !== undefined && t !== "" ? renderCriterion(t) : "yes, the statement holds"),
  ];
}

export function specialIds(tok) {
  const g = (t) => tok.token_to_id(t);
  return { cls: g("[CLS]"), sep: g("[SEP]"), mask: g("[MASK]"), pad: g("[PAD]") };
}

export function buildSequence(tok, sp, state, q, maxLen = 512, headMaxLen = 192) {
  const enc = (t) => Array.from(tok.encode(t, { add_special_tokens: false }).ids);
  const opts = renderOptions(q);
  const ins = String(q.ins).split(MASK).join(" ");
  let headIds = enc(`${q.t} question: ${ins}`);
  let optIds = opts.map((o) => [sp.mask, ...enc(" " + o.split(MASK).join(" ")).slice(0, 48)]);
  const total = (arr) => arr.reduce((a, o) => a + o.length, 0);
  let optBudget = headMaxLen - total(optIds);
  if (optBudget < 16) {
    const per = Math.max(4, Math.floor((headMaxLen - 16) / Math.max(1, optIds.length)));
    optIds = optIds.map((o) => o.slice(0, per));
    optBudget = headMaxLen - total(optIds);
  }
  headIds = headIds.slice(0, Math.max(8, optBudget));
  let ids = [sp.cls, ...headIds, sp.sep];
  const markers = [];
  for (const o of optIds) { markers.push(ids.length); for (const t of o) ids.push(t); }
  ids.push(sp.sep);
  const room = Math.max(0, maxLen - ids.length - 1);
  const st = enc(serializeState(state).split(MASK).join(" ")).slice(0, room);
  ids = ids.concat(st, [sp.sep]).slice(0, maxLen);
  return { ids, markers: markers.filter((m) => m < maxLen) };
}

export function tempBucket(qt, k) {
  const size = k <= 2 ? "2" : k <= 5 ? "3-5" : k <= 10 ? "6-10" : "11+";
  return `${QTYPE_NAMES[qt]}:${size}`;
}

// Port of Python's clamp_temperature (laya/common.py). Both shipped checkpoints have
// choice:11+ = 0.10058, the one bucket outside [0.5, 5]; using it raw sharpens those logits ~10x
// and inflates confidence for choice questions with 11+ options. Non-numeric, NaN and +-inf fall
// back to 1.0, like Python's float() coercion; everything else is clamped into [TEMP_MIN, TEMP_MAX].
const TEMP_MIN = 0.5, TEMP_MAX = 5.0;
export function clampTemperature(t) {
  let v;
  if (typeof t === "number") v = t;
  else if (typeof t === "string" && t.trim() !== "") v = Number(t);
  else return 1.0;
  if (!Number.isFinite(v)) return 1.0;
  return Math.min(TEMP_MAX, Math.max(TEMP_MIN, v));
}
export function confidenceFromProbs(p, k) {
  if (k < 2) return 1.0;
  let ent = 0;
  for (let i = 0; i < k; i++) ent -= p[i] * Math.log(Math.min(Math.max(p[i], 1e-12), 1.0));
  return Math.min(Math.max(1.0 - ent / Math.log(k), 0.0), 1.0);
}
const r4 = (x) => Math.round(x * 1e4) / 1e4;

export function collate(items, padId) {
  const n = items.length;
  const L = Math.max(...items.map((it) => it.ids.length));
  const kmax = Math.max(...items.map((it) => it.markers.length));
  const ids = new BigInt64Array(n * L).fill(BigInt(padId));
  const att = new BigInt64Array(n * L);
  const mpos = new BigInt64Array(n * kmax);
  const mmask = new Uint8Array(n * kmax);
  const qtype = new BigInt64Array(n);
  items.forEach((it, i) => {
    it.ids.forEach((t, j) => { ids[i * L + j] = BigInt(t); att[i * L + j] = 1n; });
    it.markers.forEach((m, j) => { mpos[i * kmax + j] = BigInt(m); mmask[i * kmax + j] = 1; });
    qtype[i] = BigInt(it.qtype);
  });
  return { n, L, kmax, ids, att, mpos, mmask, qtype };
}

export class Laya {
  constructor(ort, session, tokenizer, cfg) {
    this.ort = ort; this.session = session; this.tok = tokenizer; this.cfg = cfg;
    this.sp = specialIds(tokenizer);
  }

  async systemOne(state, questions) {
    const cfg = this.cfg, ort = this.ort;
    const ids = Object.keys(questions);
    const items = [], qs = [];
    for (const qid of ids) {
      const q = toInternal(questions[qid]);
      const { ids: seq, markers } = buildSequence(this.tok, this.sp, state, q, cfg.max_len ?? 512, cfg.head_max_len ?? 192);
      if (markers.length !== renderOptions(q).length) throw new Error(`question "${qid}": options exceed head_max_len=${cfg.head_max_len ?? 192}`);
      items.push({ ids: seq, markers, qtype: QTYPES[q.t] });
      qs.push(q);
    }
    const b = collate(items, this.sp.pad);
    const feeds = {
      input_ids: new ort.Tensor("int64", b.ids, [b.n, b.L]),
      attention_mask: new ort.Tensor("int64", b.att, [b.n, b.L]),
      marker_pos: new ort.Tensor("int64", b.mpos, [b.n, b.kmax]),
      marker_mask: new ort.Tensor("bool", b.mmask, [b.n, b.kmax]),
      qtype: new ort.Tensor("int64", b.qtype, [b.n]),
    };
    const t0 = performance.now();
    const out = await this.session.run(feeds);
    const ms = performance.now() - t0;
    const logits = out.logits.data; // [n, kmax] float32
    const answers = {};
    ids.forEach((qid, r) => {
      const q = qs[r], k = items[r].markers.length, qt = items[r].qtype;
      const tScale = clampTemperature(cfg.temperature_by_options?.[tempBucket(qt, k)] ?? cfg.temperature[qt]);
      const z = Array.from({ length: k }, (_, i) => logits[r * b.kmax + i] / tScale);
      const zmax = Math.max(...z);
      const e = z.map((v) => Math.exp(v - zmax));
      const s = e.reduce((a, v) => a + v, 0);
      const p = e.map((v) => v / s);
      const conf = r4(confidenceFromProbs(p, k));
      if (q.t === "choice") {
        const keys = Object.keys(q.crit);
        const top = p.indexOf(Math.max(...p));
        answers[qid] = { type: "choice", choice: keys[top], probabilities: Object.fromEntries(keys.map((kk, i) => [kk, r4(p[i])])), confidence: conf };
      } else if (q.t === "score") {
        const score = p.reduce((a, v, i) => a + i * v, 0);
        answers[qid] = { type: "score", score: r4(score), legend: Object.fromEntries(q.crit.map((c, i) => [String(i), c])), probabilities: Object.fromEntries(p.map((v, i) => [String(i), r4(v)])), confidence: conf };
      } else {
        answers[qid] = { type: "noul", noul: r4(p[1]), confidence: r4(Math.max(p[1], 1 - p[1])) };
      }
    });
    return { model: "laya-en-onnx-web", answers, usage: { input_tokens: b.att.reduce((a, v) => a + Number(v), 0), output_tokens: 0 }, latency_ms: ms };
  }
}
