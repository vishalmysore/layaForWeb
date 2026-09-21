import * as ort from "./vendor/ort.min.mjs";
import { Tokenizer } from "./vendor/tokenizers.min.mjs";
import { Laya } from "./laya-core.js";

const $ = (id) => document.getElementById(id);
// Where the model files live. Override with ?modelBase=https://host/path/ to serve them from elsewhere (needs CORS).
// A deployment can also point at a Hugging Face repo through site-config.json ({"modelBase": "..."}).
let MODEL_DIR = "./model/";
let MANIFEST = null;

const PRESETS = {
  "Support ticket": {
    state: { ticket: { subject: "App crashes on launch", text: "Since the last update, the app closes as soon as I open it. I have a demo in one hour!" } },
    questions: {
      team: { type: "choice", instructions: "Which team should handle this?", criteria: { bug: "Something is broken", how_to: "A usage question", sales: "Pricing or plans" } },
      urgency: { type: "score", instructions: "How urgent is this?", criteria: ["Can wait", "This week", "Today", "Right now"] },
      angry: { type: "noul", instructions: "The customer sounds angry" },
    },
  },
  "Agent guardrail (two wordings)": {
    state: "Agent plan: run `DELETE FROM customers WHERE last_login < '2020-01-01'` on the production database. No backup has been taken and no human has reviewed this command.",
    questions: {
      safe_without_approval: { type: "noul", instructions: "Is this safe to run without a human approving it first?" },
      destructive: { type: "noul", instructions: "The command is destructive and cannot be undone" },
      needs_human: { type: "noul", instructions: "A human should approve this command before it runs" },
    },
  },
  "Sales lead scoring": {
    state: "Hi, I'm the VP of Engineering at a 400-person logistics company. We're evaluating vendors this quarter, have budget approved, and want a demo next week with our security team.",
    questions: {
      lead_quality: { type: "score", instructions: "How qualified is this sales lead?", criteria: ["Not a fit", "Weak interest", "Some interest", "Strong buying signals"] },
      next_step: { type: "choice", instructions: "What should sales do next?", criteria: { book_demo: "Schedule a demo", nurture: "Add to a nurture campaign", ignore: "No action needed" } },
    },
  },
  "Patient message routing": {
    state: "Patient message: I've had chest tightness and shortness of breath since this morning, and my left arm feels numb.",
    questions: {
      route: { type: "choice", instructions: "Where should this message go?", criteria: { emergency: "Needs immediate emergency attention", nurse: "Nurse triage within a day", admin: "Administrative request", refill: "Medication refill" } },
      urgent: { type: "noul", instructions: "The patient describes symptoms that could be life-threatening" },
    },
  },
  "Delivery exception": {
    state: "Package scanned at depot. Address label partially unreadable. Recipient phone number missing. The customer needs the parcel by Friday for an event.",
    questions: {
      action: { type: "choice", instructions: "What should support do?", criteria: { reship: "Send a replacement", notify: "Notify the customer of the delay", fix_address: "Contact the customer to confirm the address", wait: "Wait, no action needed" } },
      at_risk: { type: "noul", instructions: "The delivery is at risk of missing the customer's deadline" },
    },
  },
  "Product review": {
    state: "Battery life is great and the screen is sharp, but the speaker crackles at high volume. Would still recommend.",
    questions: {
      sentiment: { type: "score", instructions: "Overall sentiment of the review", criteria: ["Very negative", "Negative", "Mixed", "Positive", "Very positive"] },
      mentions_defect: { type: "noul", instructions: "The reviewer reports a hardware defect" },
    },
  },
};

let laya = null;
let backendUsed = "";
let lastResult = null;

function setStatus(msg, cls = "") { const s = $("status"); s.textContent = msg; s.className = "status " + cls; }
function setProgress(f) { $("progress").style.display = f == null ? "none" : "block"; if (f != null) $("progressBar").style.width = (f * 100).toFixed(1) + "%"; }
function badge(text, ok = false) { const b = document.createElement("span"); b.className = "badge" + (ok ? " ok" : ""); b.textContent = text; $("badges").appendChild(b); }

async function fetchBytes(url, onProgress) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}. Has the model been built? See the README.`);
  const total = Number(res.headers.get("content-length")) || 0;
  const reader = res.body.getReader();
  const chunks = []; let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value); got += value.length; onProgress?.(got, total);
  }
  const out = new Uint8Array(got); let o = 0;
  for (const c of chunks) { out.set(c, o); o += c.length; }
  return out;
}

// The weight file is stored as <= 24 MiB parts (hosting limits); stitch them back together here.
// Parts are kept in the browser's Cache Storage, keyed by the file's hash, so a repeat visit does not download them again.
async function fetchParts(entry, onProgress) {
  let cache = null;
  try { cache = await caches.open("laya-" + entry.sha256.slice(0, 16)); } catch { /* private mode etc.: just download */ }
  const out = new Uint8Array(entry.size); let off = 0, fromCache = 0;
  for (const part of entry.parts) {
    const url = MODEL_DIR + part;
    let bytes = null;
    try { const hit = cache && await cache.match(url); if (hit) { bytes = new Uint8Array(await hit.arrayBuffer()); fromCache++; } } catch {}
    if (!bytes) {
      bytes = await fetchBytes(url, (got) => onProgress(off + got, entry.size));
      try { if (cache) await cache.put(url, new Response(bytes)); } catch { /* quota exceeded: fine */ }
    }
    if (off + bytes.length > entry.size) throw new Error("weights are larger than the manifest says");
    out.set(bytes, off); off += bytes.length; onProgress(off, entry.size);
  }
  if (off !== entry.size) throw new Error(`weights incomplete: got ${off} of ${entry.size} bytes`);
  fetchParts.lastFromCache = fromCache;
  return out;
}

async function hasWebGPU() {
  try { return !!navigator.gpu && !!(await navigator.gpu.requestAdapter()); } catch { return false; }
}

async function loadModel() {
  $("loadBtn").disabled = true; $("runBtn").disabled = true; $("verifyBtn").disabled = true;
  $("badges").innerHTML = ""; laya = null;
  try {
    const v = MANIFEST.variants[$("variant").value];
    let want = $("backend").value;
    const gpu = await hasWebGPU();
    // ONNX Runtime's WebGPU MatMulNBits kernel only supports 2- and 4-bit weights, so the 8-bit build cannot create a WebGPU session.
    const key = $("variant").value;
    const noWebGPU = key === "q8e8";
    if (want === "auto") want = gpu && !noWebGPU ? "webgpu" : "wasm";
    if (want === "webgpu" && noWebGPU) throw new Error("The int8 block-wise build cannot run on WebGPU (its 8-bit operator has no WebGPU kernel; only 2- and 4-bit are supported). Choose the int4 build for WebGPU, or use WASM for this build.");
    if (want === "webgpu" && !gpu) throw new Error("WebGPU is not available in this browser. Choose WASM or use a recent Chrome/Edge.");

    ort.env.wasm.wasmPaths = new URL("./vendor/", import.meta.url).href;
    ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 2) : 1;

    setStatus("Loading tokenizer and config…"); setProgress(0);
    const [tj, tc, cfg] = await Promise.all(
      ["tokenizer.json", "tokenizer_config.json", "rl_agent_config.json"].map((f) => fetch(MODEL_DIR + f).then((r) => { if (!r.ok) throw new Error(f + ": HTTP " + r.status); return r.json(); })));
    const tokenizer = new Tokenizer(tj, tc);

    const t0 = performance.now();
    const progress = (name) => (got, total) => { setProgress(total ? got / total : null); setStatus(`Downloading ${name}: ${(got / 1048576).toFixed(0)}${total ? " / " + (total / 1048576).toFixed(0) : ""} MB`); };
    const graph = await fetchBytes(MODEL_DIR + v.onnx);
    const data = await fetchParts(v.data, progress(v.data.name));
    const dlMs = performance.now() - t0;

    setStatus("Creating inference session (this can take a while on first run)…"); setProgress(null);
    const t1 = performance.now();
    const session = await ort.InferenceSession.create(graph, {
      executionProviders: want === "webgpu" ? ["webgpu", "wasm"] : ["wasm"],
      graphOptimizationLevel: "all",
      externalData: [{ path: v.data.name, data }],
    });
    const initMs = performance.now() - t1;
    laya = new Laya(ort, session, tokenizer, cfg);
    backendUsed = want;
    setStatus("Warming up…");
    const w0 = performance.now();
    await laya.systemOne("warm up", { w: { type: "noul", instructions: "This is a warm-up call" } });
    const warmMs = performance.now() - w0;
    setStatus(`Ready. Download ${(dlMs / 1000).toFixed(1)} s${fetchParts.lastFromCache ? ` (${fetchParts.lastFromCache} of ${v.data.parts.length} parts from browser cache)` : ""}, session init ${(initMs / 1000).toFixed(1)} s, warm-up call ${warmMs.toFixed(0)} ms.`, "");
    badge(want === "webgpu" ? "WebGPU" : "WASM" + (ort.env.wasm.numThreads > 1 ? ` · ${ort.env.wasm.numThreads} threads` : " · 1 thread"), true);
    badge($("variant").selectedOptions[0].textContent.split(" (")[0]);
    badge("cross-origin isolated: " + (self.crossOriginIsolated ? "yes" : "no"));
    $("runBtn").disabled = false; $("verifyBtn").disabled = false;
  } catch (e) {
    console.error(e); setStatus("Could not load the model: " + (e?.message || e), "warn");
  } finally { setProgress(null); $("loadBtn").disabled = false; }
}

function fmtPct(x) { return (x * 100).toFixed(1) + "%"; }

function barRow(name, p, top) {
  const d = document.createElement("div"); d.className = "bar" + (top ? " top" : "");
  d.innerHTML = `<span class="name"></span><span class="track"><span class="fill" style="display:block;width:${(p * 100).toFixed(1)}%"></span></span><span class="val">${fmtPct(p)}</span>`;
  d.querySelector(".name").textContent = name; d.querySelector(".name").title = name;
  return d;
}

function render(res) {
  lastResult = res;
  $("resultCard").style.display = "block";
  $("sLatency").textContent = res.latency_ms.toFixed(0);
  $("sTokens").textContent = res.usage.input_tokens;
  $("sBackend").textContent = backendUsed === "webgpu" ? "WebGPU" : "WASM";
  $("raw").textContent = JSON.stringify(res, null, 2);
  const thr = Number($("thresh").value);
  const box = $("answers"); box.innerHTML = "";
  for (const [qid, a] of Object.entries(res.answers)) {
    const q = document.createElement("div"); q.className = "q";
    const head = document.createElement("div"); head.className = "qhead";
    const headline = a.type === "choice" ? a.choice : a.type === "score" ? `score ${a.score.toFixed(2)}` : `P(true) ${fmtPct(a.noul)}`;
    head.innerHTML = `<span><span class="qname"></span> <span class="qtype">${a.type}</span></span><span><b class="hl"></b></span>`;
    head.querySelector(".qname").textContent = qid; head.querySelector(".hl").textContent = headline;
    q.appendChild(head);
    if (a.type === "choice") {
      const entries = Object.entries(a.probabilities); const mx = Math.max(...entries.map((e) => e[1]));
      entries.forEach(([k, p]) => q.appendChild(barRow(k, p, p === mx)));
    } else if (a.type === "score") {
      const mx = Math.max(...Object.values(a.probabilities));
      Object.entries(a.probabilities).forEach(([i, p]) => q.appendChild(barRow(`${i}: ${a.legend[i]}`, p, p === mx)));
    } else {
      q.appendChild(barRow("yes", a.noul, a.noul >= 0.5)); q.appendChild(barRow("no", 1 - a.noul, a.noul < 0.5));
    }
    const m = document.createElement("div"); m.className = "meta"; m.textContent = `confidence ${a.confidence.toFixed(3)} (Laya defines this as 1 minus the normalized entropy of the distribution, so a spread-out score question reads low even when its top level is likely)`; q.appendChild(m);
    const dec = document.createElement("div"); dec.className = "decision";
    const auto = a.confidence >= thr;
    dec.innerHTML = auto ? `Your code would <b class="auto">act automatically</b> (confidence ≥ ${thr.toFixed(2)})` : `Your code would <b class="human">send to a person</b> (confidence below ${thr.toFixed(2)})`;
    q.appendChild(dec);
    box.appendChild(q);
  }
}

async function run() {
  if (!laya) return;
  let state, questions;
  const raw = $("state").value.trim();
  try { state = raw.startsWith("{") || raw.startsWith("[") ? JSON.parse(raw) : $("state").value; } catch { state = $("state").value; }
  try { questions = JSON.parse($("questions").value); } catch (e) { setStatus("Questions is not valid JSON: " + e.message, "warn"); return; }
  $("runBtn").disabled = true;
  try { render(await laya.systemOne(state, questions)); setStatus("Done."); }
  catch (e) { console.error(e); setStatus("Run failed: " + (e?.message || e), "warn"); }
  finally { $("runBtn").disabled = false; }
}

async function verify() {
  if (!laya) return;
  $("verifyBtn").disabled = true; $("verifyOut").textContent = "Running…";
  try {
    const ref = await fetch("./reference.json").then((r) => r.json());
    let rows = "", worst = 0, agree = 0, total = 0;
    for (const c of ref) {
      const res = await laya.systemOne(c.state, c.questions);
      for (const [qid, a] of Object.entries(res.answers)) {
        const r = c.answers[qid];
        let dp, headRef, headNow, same;
        if (a.type === "noul") { dp = Math.abs(a.noul - r.noul); headRef = r.noul.toFixed(3); headNow = a.noul.toFixed(3); same = (a.noul >= 0.5) === (r.noul >= 0.5); }
        else { const ks = Object.keys(a.probabilities); dp = Math.max(...ks.map((k) => Math.abs(a.probabilities[k] - r.probabilities[k])));
          const top = (o) => Object.entries(o.probabilities).sort((x, y) => y[1] - x[1])[0][0];
          headRef = a.type === "choice" ? top(r) : r.score.toFixed(2); headNow = a.type === "choice" ? top(a) : a.score.toFixed(2); same = top(a) === top(r); }
        worst = Math.max(worst, dp); total++; agree += same ? 1 : 0;
        rows += `<tr><td>${c.name}</td><td>${qid}</td><td>${headRef}</td><td>${headNow}</td><td class="num">${dp.toFixed(3)}</td><td>${same ? "same" : "<b class='warn'>differs</b>"}</td></tr>`;
      }
    }
    $("verifyOut").innerHTML = `<p><b>${agree}/${total}</b> answers pick the same top option as the original; largest probability difference <b>${worst.toFixed(3)}</b>.</p><table><thead><tr><th>Case</th><th>Question</th><th>Original</th><th>This build</th><th>Max |Δp|</th><th>Top answer</th></tr></thead><tbody>${rows}</tbody></table>`;
    window.__verify = { agree, total, worst };
  } catch (e) { $("verifyOut").textContent = "Comparison failed: " + (e?.message || e); }
  finally { $("verifyBtn").disabled = false; }
}

async function initManifest() {
  try {
    let base = new URLSearchParams(location.search).get("modelBase");
    if (!base) { try { const c = await fetch("./site-config.json"); if (c.ok) base = (await c.json()).modelBase; } catch { /* no config: model sits next to the page */ } }
    if (base) MODEL_DIR = base.replace(/\/?$/, "/");
    const r = await fetch(MODEL_DIR + "manifest.json");
    if (!r.ok) throw new Error("HTTP " + r.status);
    MANIFEST = await r.json();
    const sel = $("variant"); sel.innerHTML = "";
    for (const [k, v] of Object.entries(MANIFEST.variants)) sel.add(new Option(`${v.label} (~${Math.round(v.data.size / 1048576)} MB)`, k));
    $("loadBtn").disabled = false;
  } catch (e) {
    setStatus("No model found next to this page (" + (e?.message || e) + "). Build it first, see the README.", "warn");
  }
}
$("loadBtn").disabled = true;
initManifest();

// wiring
for (const k of Object.keys(PRESETS)) $("preset").add(new Option(k, k));
function applyPreset() {
  const p = PRESETS[$("preset").value];
  $("state").value = typeof p.state === "string" ? p.state : JSON.stringify(p.state, null, 2);
  $("questions").value = JSON.stringify(p.questions, null, 2);
}
$("preset").addEventListener("change", applyPreset); applyPreset();
$("thresh").addEventListener("input", () => { $("threshVal").textContent = Number($("thresh").value).toFixed(2); if (lastResult) render(lastResult); });
$("loadBtn").addEventListener("click", loadModel);
$("runBtn").addEventListener("click", run);
$("verifyBtn").addEventListener("click", verify);
document.addEventListener("keydown", (e) => { if ((e.ctrlKey || e.metaKey) && e.key === "Enter") run(); });
window.__laya = { loadModel, run, verify, get ready() { return !!laya; }, get instance() { return laya; } };
