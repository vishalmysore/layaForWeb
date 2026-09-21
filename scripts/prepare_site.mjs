// Assemble the static site into dist/: web app + ONNX Runtime Web files + tokenizer library + built model.
//   npm ci && node scripts/prepare_site.mjs
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const arg = (n) => { const i = process.argv.indexOf(n); return i > 0 ? process.argv[i + 1] : undefined; };
const dist = path.resolve(root, arg("--out") || "dist");
const external = arg("--external-model-base"); // e.g. https://huggingface.co/<user>/<repo>/resolve/main/
const model = path.join(root, "build", "model");
const need = (p, hint) => { if (!fs.existsSync(p)) { console.error(`Missing ${path.relative(root, p)}. ${hint}`); process.exit(1); } };

need(path.join(model, "manifest.json"), "Build the model first: python scripts/build_model.py");
need(path.join(root, "node_modules", "onnxruntime-web"), "Install dependencies first: npm ci");

fs.rmSync(dist, { recursive: true, force: true });
fs.mkdirSync(path.join(dist, "vendor"), { recursive: true });

fs.cpSync(path.join(root, "web"), dist, { recursive: true });

const ort = path.join(root, "node_modules", "onnxruntime-web", "dist");
for (const f of ["ort.min.mjs", "ort-wasm-simd-threaded.jsep.mjs", "ort-wasm-simd-threaded.jsep.wasm"]) {
  need(path.join(ort, f), "Unexpected onnxruntime-web layout; check the pinned version in package.json.");
  fs.copyFileSync(path.join(ort, f), path.join(dist, "vendor", f));
}
const tk = path.join(root, "node_modules", "@huggingface", "tokenizers", "dist", "tokenizers.min.mjs");
need(tk, "Install dependencies first: npm ci");
fs.copyFileSync(tk, path.join(dist, "vendor", "tokenizers.min.mjs"));

if (external) {
  fs.writeFileSync(path.join(dist, "site-config.json"), JSON.stringify({ modelBase: external }, null, 2));
  console.log(`model files are NOT copied; the page will load them from ${external}`);
} else {
  fs.cpSync(model, path.join(dist, "model"), { recursive: true });
}
fs.copyFileSync(path.join(root, "NOTICE.md"), path.join(dist, "NOTICE.md"));
fs.writeFileSync(path.join(dist, ".nojekyll"), "");

let total = 0, biggest = 0;
const walk = (d) => fs.readdirSync(d, { withFileTypes: true }).forEach((e) => {
  const p = path.join(d, e.name);
  if (e.isDirectory()) return walk(p);
  const s = fs.statSync(p).size; total += s; biggest = Math.max(biggest, s);
});
walk(dist);
console.log(`dist/ ready: ${(total / 1e6).toFixed(0)} MB total, largest file ${(biggest / 1e6).toFixed(0)} MB`);
if (total > 0.95e9) console.warn("Warning: GitHub Pages sites are limited to 1 GB. Build fewer variants (--variants q8e8) or host the model on Hugging Face (--external-model-base).");
