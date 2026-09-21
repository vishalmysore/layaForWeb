// The JavaScript port must build exactly the same token sequences as the Python reference.
//   node tests/seq_parity.mjs [dir with tokenizer.json]      (default: build/model)
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { Tokenizer } from "@huggingface/tokenizers";
import { buildSequence, toInternal, specialIds } from "../web/laya-core.js";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const dir = path.resolve(process.argv[2] || path.join(root, "build", "model"));
const tok = new Tokenizer(JSON.parse(fs.readFileSync(path.join(dir, "tokenizer.json"), "utf8")), JSON.parse(fs.readFileSync(path.join(dir, "tokenizer_config.json"), "utf8")));
const sp = specialIds(tok);
const tests = JSON.parse(fs.readFileSync(path.join(root, "tests", "seq_tests.json"), "utf8"));
let bad = 0;
tests.forEach((t, i) => {
  const r = buildSequence(tok, sp, t.state, toInternal(t.q));
  if (JSON.stringify(r.ids) !== JSON.stringify(t.ids) || JSON.stringify(r.markers) !== JSON.stringify(t.markers)) {
    bad++; console.error(`MISMATCH in case ${i}: ${JSON.stringify(t.state).slice(0, 70)}`);
  }
});
console.log(`${tests.length - bad}/${tests.length} token sequences identical to the Python reference`);
process.exit(bad ? 1 : 0);
