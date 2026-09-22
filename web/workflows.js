// Nine small "apps" built on the layaForWeb model: type a real message, Laya answers a fixed set of
// typed questions about it, and a short rule turns those answers into an action. This page is
// independent of index.html/app.js — it does not read or change anything there — but it loads the
// same unmodified model files and the same unmodified Laya port (laya-core.js) that page uses.
//
// The "convert to Laya format" step is the WORKFLOWS table below: each entry pairs a domain with the
// fixed typed questions a real system would ask about that kind of message. Typing a message and
// pressing Run wraps it in that domain's state shape and sends state + questions to Laya, exactly
// like any other caller of laya.systemOne() would.
import * as ort from "./vendor/ort.min.mjs";
import { Tokenizer } from "./vendor/tokenizers.min.mjs";
import { Laya } from "./laya-core.js";

const $ = (id) => document.getElementById(id);
// Same two-checkpoint setup as app.js: a general base model, and an optional second one fine-tuned
// for typed-decision workflows. See app.js for the full rationale.
const BASES = {
  laya: { label: "Laya (general)", dir: "./model/", param: "modelBase", configKey: "modelBase" },
  typed: { label: "Laya (typed decisions)", dir: "./model-typed/", param: "modelBaseTyped", configKey: "modelBaseTyped" },
};
let MODEL_DIR = BASES.laya.dir;
let MANIFEST = null;

// ---------------------------------------------------------------------------------------------
// Nine workflows. `questions` is the fixed typed schema a real system would use for that job
// (unchanged by what the visitor types). `toState` builds the Laya `state` from the message (and
// subject, for the one domain that uses it). `decide` reads Laya's answers and returns the action.
// `examples` are two messages to try, taken from tests/data/<domain>.json in this repository.
const AUTO = "auto", HOLD = "hold", BLOCK = "block";
const chip = (t) => t;

const WORKFLOWS = {
  "support-tickets": {
    label: "Support ticket → route to a team",
    description: "Routes a ticket to a team queue and priority, or to a person, based on Laya's answers.",
    needsSubject: true,
    questions: {
      team: { type: "choice", instructions: "Which team should handle this?", criteria: { bug: "Something is broken", how_to: "A usage question", feature_request: "A request for a new capability", account_access: "Login, password or account access" } },
      urgency: { type: "score", instructions: "How urgent is this?", criteria: ["Can wait", "This week", "Today", "Right now"] },
      angry: { type: "noul", instructions: "The customer sounds angry" },
    },
    toState: (subject, text) => ({ ticket: { subject: subject || "(no subject)", text } }),
    decide(a, thr) {
      const team = a.team, urg = a.urgency, angry = a.angry;
      const level = urg.legend[String(Math.round(urg.score))] ?? urg.legend["0"];
      const chips = [chip(`team: ${team.choice}`), chip(`priority: ${level}`), chip(angry.noul >= 0.5 ? "customer sounds upset" : "customer sounds calm")];
      if (team.confidence >= thr && urg.confidence >= thr)
        return { status: AUTO, title: `Routed to ${team.choice}, priority "${level}"`, detail: `Both the team and the urgency call were confident (≥ ${thr.toFixed(2)}), so the ticket is filed automatically.`, chips };
      return { status: HOLD, title: `Suggested: ${team.choice}, priority "${level}" — needs a person to confirm`, detail: `Team confidence ${team.confidence.toFixed(2)} or urgency confidence ${urg.confidence.toFixed(2)} is below ${thr.toFixed(2)}, so this goes to a triage queue with the suggestion attached instead of being filed on its own.`, chips };
    },
    examples: [
      { subject: "App crashes on launch", text: "Since the last update, the app closes as soon as I open it. I have a demo in one hour!" },
      { subject: "Lost my work AGAIN", text: "This is the THIRD time this month your app has lost my work. I am sick of it. Fix your product or I'm leaving." },
    ],
  },

  "agent-guardrails": {
    label: "Agent action → approve or block",
    description: "Gates a plan an AI agent wants to run. Any sign of risk blocks it, whatever the confidence.",
    questions: {
      destructive: { type: "noul", instructions: "The action is destructive and cannot be undone" },
      needs_human: { type: "noul", instructions: "A human should approve this action before it runs" },
      safe_without_approval: { type: "noul", instructions: "It is safe to run this action without a human approving it first" },
      risk: { type: "score", instructions: "How risky is this action?", criteria: ["Low", "Medium", "High", "Critical"] },
    },
    toState: (subject, text) => text,
    decide(a, thr) {
      const { destructive, needs_human, safe_without_approval, risk } = a;
      const chips = [chip(`destructive: ${destructive.noul >= 0.5 ? "yes" : "no"} (${(destructive.noul * 100).toFixed(0)}%)`), chip(`needs human: ${needs_human.noul >= 0.5 ? "yes" : "no"} (${(needs_human.noul * 100).toFixed(0)}%)`), chip(`risk: ${risk.legend[String(Math.round(risk.score))]}`)];
      if (needs_human.noul >= 0.5 || destructive.noul >= 0.5 || Math.round(risk.score) >= 2)
        return { status: BLOCK, title: "Blocked — held for a human to approve", detail: "Any one of “needs a human”, “destructive” or “high/critical risk” being the more likely answer blocks the run, regardless of confidence. This mirrors how a real approval gate should fail closed.", chips };
      if (safe_without_approval.noul >= 0.5 && safe_without_approval.confidence >= thr)
        return { status: AUTO, title: "Approved — runs automatically", detail: `Every risk signal points to "safe", and the "safe without approval" call is confident (≥ ${thr.toFixed(2)}).`, chips };
      return { status: HOLD, title: "Held for review — no clear signal either way", detail: "Nothing screams risk, but the model is not confident enough that it's safe to skip a human.", chips };
    },
    examples: [
      { text: "Agent plan: run `DELETE FROM customers WHERE last_login < '2020-01-01'` on the production database. No backup has been taken and no human has reviewed this command." },
      { text: "Agent plan: create a new feature branch and open a draft pull request with the changes." },
    ],
  },

  "content-moderation": {
    label: "Comment → publish, hold or remove",
    description: "Gates a public comment. Only acts automatically when the model is confident; otherwise a moderator sees it.",
    questions: {
      verdict: { type: "choice", instructions: "What should happen to this comment?", criteria: { allow: "Fine to publish", review: "Needs a human moderator", remove: "Breaks the rules and should be removed" } },
      toxicity: { type: "score", instructions: "How toxic is the language?", criteria: ["None", "Mild", "Strong", "Severe"] },
      spam: { type: "noul", instructions: "The comment is spam or an advertisement" },
    },
    toState: (subject, text) => text,
    decide(a, thr) {
      const v = a.verdict, tox = a.toxicity, spam = a.spam;
      const chips = [chip(`verdict: ${v.choice} (${(v.probabilities[v.choice] * 100).toFixed(0)}%)`), chip(`toxicity: ${tox.legend[String(Math.round(tox.score))]}`), chip(spam.noul >= 0.5 ? "looks like spam" : "not spam")];
      if (v.confidence < thr) return { status: HOLD, title: "Held for a moderator", detail: `The verdict's confidence (${v.confidence.toFixed(2)}) is below ${thr.toFixed(2)}, so it is not applied automatically.`, chips };
      if (v.choice === "remove") return { status: BLOCK, title: "Removed automatically", detail: `Confidently rated "remove" (≥ ${thr.toFixed(2)}).`, chips };
      if (v.choice === "allow") return { status: AUTO, title: "Published automatically", detail: `Confidently rated "allow" (≥ ${thr.toFixed(2)}).`, chips };
      return { status: HOLD, title: "Held for a moderator", detail: "The model itself says this needs a human to look at it.", chips };
    },
    examples: [
      { text: "You are a complete idiot and everyone here knows it. Shut up." },
      { text: "Great article, thanks for sharing the details on the setup!" },
    ],
  },

  "it-incidents": {
    label: "Monitoring alert → page a team or log it",
    description: "Pages the owning team for major or critical incidents; otherwise just files a ticket.",
    questions: {
      severity: { type: "score", instructions: "How severe is this incident?", criteria: ["Minor", "Moderate", "Major", "Critical"] },
      owner: { type: "choice", instructions: "Which team should own it?", criteria: { network: "Networking and DNS", database: "Databases", application: "Application developers", security: "Security team" } },
      customer_impact: { type: "noul", instructions: "Customers are affected right now" },
    },
    toState: (subject, text) => text,
    decide(a, thr) {
      const sev = a.severity, owner = a.owner, impact = a.customer_impact;
      const level = sev.legend[String(Math.round(sev.score))];
      const chips = [chip(`severity: ${level}`), chip(`owner: ${owner.choice}`), chip(impact.noul >= 0.5 ? "customers affected now" : "no customer impact reported")];
      if (Math.round(sev.score) >= 2) return { status: BLOCK, title: `Paging the ${owner.choice} team now`, detail: "Major or critical severity pages the owning team immediately, even if the owner or severity call isn't highly confident — for incidents, a false page is cheaper than a missed one.", chips };
      if (owner.confidence >= thr) return { status: AUTO, title: `Logged for the ${owner.choice} team, no page`, detail: `Minor/moderate severity, and the owning team is a confident call (≥ ${thr.toFixed(2)}).`, chips };
      return { status: HOLD, title: "Logged, but the owning team is unclear", detail: `Owner confidence ${owner.confidence.toFixed(2)} is below ${thr.toFixed(2)}, so a person should triage which team it belongs to.`, chips };
    },
    examples: [
      { text: "Alert: primary database CPU at 100% for 20 minutes. Search and login return 500 errors for all users." },
      { text: "Disk usage on the log server at 71%. No services are affected." },
    ],
  },

  "email-triage": {
    label: "Email → file with an action and a due time",
    description: "Files an email into an action queue (reply now, reply later, archive, delegate) with a due time.",
    questions: {
      action: { type: "choice", instructions: "What should the recipient do with this email?", criteria: { reply_now: "Reply today", reply_later: "Reply within the week", archive: "No reply needed", delegate: "Forward it to someone else" } },
      needs_reply: { type: "noul", instructions: "The sender expects a reply" },
      urgency: { type: "score", instructions: "How urgent is this email?", criteria: ["Low", "Normal", "High", "Urgent"] },
    },
    toState: (subject, text) => text,
    decide(a, thr) {
      const act = a.action, reply = a.needs_reply, urg = a.urgency;
      const due = { reply_now: "due today", reply_later: "due this week", archive: "no due date", delegate: "forward now" }[act.choice];
      const chips = [chip(`action: ${act.choice.replace("_", " ")}`), chip(due), chip(reply.noul >= 0.5 ? "expects a reply" : "no reply expected")];
      if (act.confidence >= thr) return { status: AUTO, title: `Filed: ${act.choice.replace("_", " ")} (${due})`, detail: `Confidence ${act.confidence.toFixed(2)} ≥ ${thr.toFixed(2)}, filed automatically.`, chips };
      return { status: HOLD, title: `Suggested: ${act.choice.replace("_", " ")} (${due}) — low confidence`, detail: `Confidence ${act.confidence.toFixed(2)} is below ${thr.toFixed(2)}; left in the inbox with a suggested label instead of being auto-filed.`, chips };
    },
    examples: [
      { text: "URGENT: the server room is flooding, please call me right now." },
      { text: "Newsletter: Ten tips for better spreadsheets. You can unsubscribe at any time." },
    ],
  },

  "delivery-exceptions": {
    label: "Delivery problem → dispatch a support action",
    description: "Turns a delivery problem into a support action: reship, notify, fix the address, or wait.",
    questions: {
      action: { type: "choice", instructions: "What should support do?", criteria: { reship: "Send a replacement", notify: "Tell the customer about the delay or outcome", fix_address: "Contact the customer to fix the delivery details", wait: "Wait, no action needed" } },
      urgency: { type: "score", instructions: "How urgent is this?", criteria: ["Not urgent", "Soon", "Urgent", "Immediate"] },
      upset: { type: "noul", instructions: "The customer is upset" },
    },
    toState: (subject, text) => text,
    decide(a, thr) {
      const act = a.action, urg = a.urgency, upset = a.upset;
      const label = { reship: "Ship a replacement", notify: "Send a status update", fix_address: "Contact customer to confirm address", wait: "No action, keep watching" }[act.choice];
      const chips = [chip(label), chip(`urgency: ${urg.legend[String(Math.round(urg.score))]}`), chip(upset.noul >= 0.5 ? "customer is upset" : "customer is calm")];
      if (act.confidence >= thr) return { status: AUTO, title: label, detail: `Dispatched automatically (confidence ${act.confidence.toFixed(2)} ≥ ${thr.toFixed(2)}).`, chips };
      return { status: HOLD, title: `Suggested: ${label}`, detail: `Confidence ${act.confidence.toFixed(2)} is below ${thr.toFixed(2)}, so an agent should confirm before it goes out.`, chips };
    },
    examples: [
      { text: "The package has been lost for 10 days. Tracking has not updated since it left the warehouse, and the customer has emailed three times." },
      { text: "Parcel is in transit and on schedule. Estimated delivery is tomorrow." },
    ],
  },

  "sales-leads": {
    label: "Inbound lead → a CRM action",
    description: "Turns an inbound message into a CRM action: book a demo, add to nurture, or archive.",
    questions: {
      lead_quality: { type: "score", instructions: "How qualified is this sales lead?", criteria: ["Not a fit", "Weak interest", "Some interest", "Strong buying signals"] },
      next_step: { type: "choice", instructions: "What should sales do next?", criteria: { book_demo: "Schedule a demo", nurture: "Add to a nurture campaign", ignore: "No action needed" } },
    },
    toState: (subject, text) => text,
    decide(a, thr) {
      const step = a.next_step, q = a.lead_quality;
      const label = { book_demo: "Create a demo task for a rep", nurture: "Add to the nurture sequence", ignore: "Archive, no action" }[step.choice];
      const chips = [chip(label), chip(`lead quality: ${q.legend[String(Math.round(q.score))]}`)];
      if (step.confidence >= thr) return { status: AUTO, title: label, detail: `Confidence ${step.confidence.toFixed(2)} ≥ ${thr.toFixed(2)}, actioned in the CRM automatically.`, chips };
      return { status: HOLD, title: `Suggested: ${label}`, detail: `Confidence ${step.confidence.toFixed(2)} is below ${thr.toFixed(2)}; a rep should look at it before it's actioned.`, chips };
    },
    examples: [
      { text: "Hi, I'm the VP of Engineering at a 400-person logistics company. We're evaluating vendors this quarter, have budget approved, and want a demo next week with our security team." },
      { text: "I'm a student working on a class project about project management tools. Do you have a free version?" },
    ],
  },

  "patient-messages": {
    label: "Clinic message → route it (synthetic, routing only)",
    description: "Routes a portal message. A possibly life-threatening message escalates immediately, whatever the confidence. Synthetic text, for routing logic only — not medical advice.",
    questions: {
      route: { type: "choice", instructions: "Where should this message go?", criteria: { emergency: "Needs immediate emergency attention", nurse: "Nurse triage within a day", admin: "Administrative request", refill: "Medication refill" } },
      urgent: { type: "noul", instructions: "The patient describes symptoms that could be life-threatening" },
    },
    toState: (subject, text) => text,
    decide(a, thr) {
      const route = a.route, urgent = a.urgent;
      const chips = [chip(`route: ${route.choice}`), chip(urgent.noul >= 0.5 ? "possibly life-threatening" : "not flagged as life-threatening")];
      if (urgent.noul >= 0.5 || route.choice === "emergency")
        return { status: BLOCK, title: "Escalated immediately", detail: "Either signal alone is enough to escalate, regardless of confidence. A routing tool like this should fail toward caution, not toward its own certainty.", chips };
      if (route.confidence >= thr) return { status: AUTO, title: `Routed to ${route.choice}`, detail: `Confidence ${route.confidence.toFixed(2)} ≥ ${thr.toFixed(2)}.`, chips };
      return { status: HOLD, title: `Suggested: ${route.choice} — low confidence`, detail: `Confidence ${route.confidence.toFixed(2)} is below ${thr.toFixed(2)}; a person should route it.`, chips };
    },
    examples: [
      { text: "Patient message: I've had chest tightness and shortness of breath since this morning, and my left arm feels numb." },
      { text: "Patient message: Could you send my vaccination record to my new school?" },
    ],
  },

  "product-reviews": {
    label: "Product review → publish or escalate",
    description: "Escalates a clearly negative review or one that reports a defect; otherwise files it as feedback.",
    questions: {
      sentiment: { type: "score", instructions: "How does the reviewer feel about the product?", criteria: ["Very negative", "Negative", "Neutral", "Positive", "Very positive"] },
      topic: { type: "choice", instructions: "What is the review mostly about?", criteria: { quality: "How well the product is made or works", shipping: "Delivery and packaging", value: "Price compared with what you get", usability: "Setup and ease of use", support: "Customer service" } },
      recommends: { type: "noul", instructions: "The reviewer would recommend this product" },
    },
    toState: (subject, text) => text,
    decide(a, thr) {
      const sent = a.sentiment, topic = a.topic, rec = a.recommends;
      const level = sent.legend[String(Math.round(sent.score))];
      const chips = [chip(`sentiment: ${level}`), chip(`about: ${topic.choice}`), chip(rec.noul >= 0.5 ? "would recommend" : "would not recommend")];
      const negative = Math.round(sent.score) <= 1 || (rec.noul < 0.5 && rec.confidence >= thr);
      if (negative) return { status: HOLD, title: `Escalated to the product team (${topic.choice})`, detail: "Negative sentiment or a confident “would not recommend” sends the review to a person instead of just publishing it.", chips };
      return { status: AUTO, title: `Published, filed as feedback about ${topic.choice}`, detail: "Sentiment is neutral or better and there's no confident negative signal, so it's published and logged for product analytics.", chips };
    },
    examples: [
      { text: "Broke after two days. Cheap plastic, the hinge snapped. Do not buy." },
      { text: "Absolutely love this blender. Crushes ice in seconds and cleans up easily. Best purchase I've made all year." },
    ],
  },
};

const ORDER = ["support-tickets", "agent-guardrails", "content-moderation", "it-incidents", "email-triage", "delivery-exceptions", "sales-leads", "patient-messages", "product-reviews"];

// ---------------------------------------------------------------------------------------------
// Model loading. Same files, same manifest and the same load steps as the main demo (index.html /
// app.js), written fresh here so this page works on its own and app.js is not touched.
let laya = null;
let backendUsed = "";
let exampleIdx = {};

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
      try { if (cache) await cache.put(url, new Response(bytes)); } catch {}
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
  $("loadBtn").disabled = true; $("runBtn").disabled = true;
  $("badges").innerHTML = ""; laya = null;
  try {
    const v = MANIFEST.variants[$("variant").value];
    let want = $("backend").value;
    const gpu = await hasWebGPU();
    const key = $("variant").value;
    const noWebGPU = key === "q8e8";
    let backendNote = "";
    if (want === "auto" && gpu && noWebGPU) backendNote = "Auto chose WASM: this build cannot run on WebGPU. Pick the int4 build to use WebGPU.";
    if (want === "auto") want = gpu && !noWebGPU ? "webgpu" : "wasm";
    if (want === "webgpu" && noWebGPU) throw new Error("The int8 block-wise build cannot run on WebGPU (only 2- and 4-bit are supported). Choose the int4 build for WebGPU, or use WASM.");
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
    setStatus(`Ready. Download ${(dlMs / 1000).toFixed(1)} s${fetchParts.lastFromCache ? ` (${fetchParts.lastFromCache} of ${v.data.parts.length} parts from browser cache)` : ""}, session init ${(initMs / 1000).toFixed(1)} s, warm-up call ${warmMs.toFixed(0)} ms.`);
    badge(want === "webgpu" ? "WebGPU" : "WASM" + (ort.env.wasm.numThreads > 1 ? ` · ${ort.env.wasm.numThreads} threads` : " · 1 thread"), true);
    badge($("variant").selectedOptions[0].textContent.split(" (")[0]);
    if (backendNote) badge(backendNote);
    $("runBtn").disabled = false;
  } catch (e) {
    console.error(e); setStatus("Could not load the model: " + (e?.message || e), "warn");
  } finally { setProgress(null); $("loadBtn").disabled = false; }
}

async function resolveBaseDir(key) {
  const b = BASES[key];
  let dir = new URLSearchParams(location.search).get(b.param);
  if (!dir) { try { const c = await fetch("./site-config.json"); if (c.ok) dir = (await c.json())[b.configKey]; } catch {} }
  return (dir || b.dir).replace(/\/?$/, "/");
}

async function probeBase(key) {
  const dir = await resolveBaseDir(key);
  try {
    const r = await fetch(dir + "manifest.json");
    if (!r.ok) throw new Error("HTTP " + r.status);
    return { key, dir, manifest: await r.json() };
  } catch (e) {
    return { key, dir, manifest: null, error: e };
  }
}

let AVAILABLE = {};
function applyBase(key) {
  const a = AVAILABLE[key];
  MODEL_DIR = a.dir; MANIFEST = a.manifest;
  const vsel = $("variant"); vsel.innerHTML = "";
  for (const [k, v] of Object.entries(MANIFEST.variants)) vsel.add(new Option(`${v.label} (~${Math.round(v.data.size / 1048576)} MB)`, k));
  laya = null; $("runBtn").disabled = true; $("loadBtn").disabled = false; $("badges").innerHTML = "";
  setStatus(`${BASES[key].label} selected. Not loaded yet.`);
}

async function initManifests() {
  const probes = await Promise.all(Object.keys(BASES).map(probeBase));
  const sel = $("baseModel"); sel.innerHTML = "";
  for (const p of probes) if (p.manifest) { AVAILABLE[p.key] = p; sel.add(new Option(BASES[p.key].label, p.key)); }
  if (!Object.keys(AVAILABLE).length) {
    const first = probes[0];
    setStatus("No model found next to this page (" + (first.error?.message || first.error) + "). Build it first, see the README.", "warn");
    return;
  }
  $("baseModelRow").style.display = Object.keys(AVAILABLE).length > 1 ? "" : "none";
  applyBase(Object.keys(AVAILABLE)[0]);
}

// ---------------------------------------------------------------------------------------------
// Rendering
function fmtPct(x) { return (x * 100).toFixed(1) + "%"; }
function barRow(name, p, top) {
  const d = document.createElement("div"); d.className = "bar" + (top ? " top" : "");
  d.innerHTML = `<span class="name"></span><span class="track"><span class="fill" style="display:block;width:${(p * 100).toFixed(1)}%"></span></span><span class="val">${fmtPct(p)}</span>`;
  d.querySelector(".name").textContent = name; d.querySelector(".name").title = name;
  return d;
}

function renderAnswers(res) {
  const box = $("answers"); box.innerHTML = "";
  for (const [qid, a] of Object.entries(res.answers)) {
    const q = document.createElement("div"); q.className = "q";
    const head = document.createElement("div"); head.className = "qhead";
    const headline = a.type === "choice" ? a.choice : a.type === "score" ? `score ${a.score.toFixed(2)}` : `P(true) ${fmtPct(a.noul)}`;
    head.innerHTML = `<span><span class="qname"></span> <span class="qtype">${a.type}</span></span><span><b></b></span>`;
    head.querySelector(".qname").textContent = qid; head.querySelector("b").textContent = headline;
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
    const m = document.createElement("div"); m.className = "meta"; m.textContent = `confidence ${a.confidence.toFixed(3)}`; q.appendChild(m);
    box.appendChild(q);
  }
}

function renderOutcome(o) {
  const el = $("outcome");
  const label = o.status === AUTO ? "Acted automatically" : o.status === BLOCK ? (o.title.startsWith("Escalated") || o.title.startsWith("Paging") ? "Escalated" : "Blocked automatically") : "Held for a person";
  el.innerHTML = `<div class="outcome ${o.status}"><div class="label"></div><div class="title"></div><div class="detail"></div><div class="chips"></div></div>`;
  el.querySelector(".label").textContent = label;
  el.querySelector(".title").textContent = o.title;
  el.querySelector(".detail").textContent = o.detail;
  const chips = el.querySelector(".chips");
  for (const c of o.chips) { const s = document.createElement("span"); s.className = "chip"; s.textContent = c; chips.appendChild(s); }
}

async function run() {
  if (!laya) return;
  const wf = WORKFLOWS[$("domain").value];
  const text = $("message").value.trim();
  if (!text) { setStatus("Type a message first.", "warn"); return; }
  const state = wf.toState($("subject").value.trim(), text);
  $("runBtn").disabled = true;
  try {
    const res = await laya.systemOne(state, wf.questions);
    $("resultCard").style.display = "block";
    renderOutcome(wf.decide(res.answers, Number($("thresh").value)));
    renderAnswers(res);
    $("raw").textContent = JSON.stringify(res, null, 2);
  } catch (e) {
    console.error(e); setStatus("Run failed: " + (e?.message || e), "warn");
  } finally { $("runBtn").disabled = false; }
}

function applyDomain() {
  const wf = WORKFLOWS[$("domain").value];
  $("domainDesc").textContent = wf.description;
  $("subjectField").style.display = wf.needsSubject ? "block" : "none";
  $("message").value = ""; $("subject").value = "";
  $("resultCard").style.display = "none";
  exampleIdx[$("domain").value] = 0;
}

function applyExample() {
  const key = $("domain").value, wf = WORKFLOWS[key];
  const i = (exampleIdx[key] || 0) % wf.examples.length;
  const ex = wf.examples[i];
  $("message").value = ex.text;
  if (wf.needsSubject) $("subject").value = ex.subject || "";
  exampleIdx[key] = i + 1;
}

for (const k of ORDER) $("domain").add(new Option(WORKFLOWS[k].label, k));
$("domain").addEventListener("change", applyDomain);
applyDomain(); applyExample();

$("thresh").addEventListener("input", () => { $("threshVal").textContent = Number($("thresh").value).toFixed(2); });
$("exampleBtn").addEventListener("click", applyExample);
$("loadBtn").addEventListener("click", loadModel);
$("runBtn").addEventListener("click", run);
document.addEventListener("keydown", (e) => { if ((e.ctrlKey || e.metaKey) && e.key === "Enter") run(); });

$("loadBtn").disabled = true;
$("baseModel").addEventListener("change", () => applyBase($("baseModel").value));
initManifests();
