#!/usr/bin/env python3
"""Regenerate the small test fixtures from the original PyTorch model (only needed if you change test cases).

    python scripts/gen_fixtures.py         # needs build/en (run scripts/build_model.py first)

Writes:
  web/reference.json      what the fp32 PyTorch model answers for six cases (the page's "compare" button uses it)
  tests/seq_tests.json    token sequences the JavaScript port must reproduce exactly
"""
import json, sys
from pathlib import Path
import laya
from laya.common import build_sequence

ROOT = Path(__file__).resolve().parent.parent
agent = laya.load(str(ROOT / "build" / "en"), device="cpu")

CASES = [
    {"name": "support ticket", "state": {"ticket": {"subject": "App crashes on launch", "text": "Since the last update, the app closes as soon as I open it. I have a demo in one hour!"}},
     "questions": {
         "team": {"type": "choice", "instructions": "Which team should handle this?", "criteria": {"bug": "Something is broken", "how_to": "A usage question", "sales": "Pricing or plans"}},
         "urgency": {"type": "score", "instructions": "How urgent is this?", "criteria": ["Can wait", "This week", "Today", "Right now"]},
         "angry": {"type": "noul", "instructions": "The customer sounds angry"}}},
    {"name": "agent guardrail", "state": "Agent plan: run `DELETE FROM customers WHERE last_login < '2020-01-01'` on the production database. No backup has been taken and no human has reviewed this command.",
     "questions": {"safe": {"type": "noul", "instructions": "Is this safe to run without a human approving it first?"}}},
    {"name": "sales lead", "state": "Hi, I'm the VP of Engineering at a 400-person logistics company. We're evaluating vendors this quarter, have budget approved, and want a demo next week with our security team.",
     "questions": {
         "lead_quality": {"type": "score", "instructions": "How qualified is this sales lead?", "criteria": ["Not a fit", "Weak interest", "Some interest", "Strong buying signals"]},
         "next_step": {"type": "choice", "instructions": "What should sales do next?", "criteria": {"book_demo": "Schedule a demo", "nurture": "Add to a nurture campaign", "ignore": "No action needed"}}}},
    {"name": "health message routing", "state": "Patient message: I've had chest tightness and shortness of breath since this morning, and my left arm feels numb.",
     "questions": {
         "route": {"type": "choice", "instructions": "Where should this message go?", "criteria": {"emergency": "Needs immediate emergency attention", "nurse": "Nurse triage within a day", "admin": "Administrative request", "refill": "Medication refill"}},
         "urgent": {"type": "noul", "instructions": "The patient describes symptoms that could be life-threatening"}}},
    {"name": "long state", "state": "Shipment update log. " + "The carrier reported a delay at the regional hub because of weather. " * 30 + "Final note: the customer needs the parcel by Friday for an event.",
     "questions": {
         "at_risk": {"type": "noul", "instructions": "The delivery is at risk of missing the customer's deadline"},
         "action": {"type": "choice", "instructions": "What should support do?", "criteria": {"reship": "Send a replacement", "notify": "Notify the customer of the delay", "wait": "Wait, no action needed"}}}},
    {"name": "guardrail (destructive wording)", "state": "Agent plan: run `DELETE FROM customers WHERE last_login < '2020-01-01'` on the production database. No backup has been taken and no human has reviewed this command.",
     "questions": {"destructive": {"type": "noul", "instructions": "The command is destructive and cannot be undone"},
                   "needs_human": {"type": "noul", "instructions": "A human should approve this command before it runs"}}},
]

ref = []
for c in CASES:
    r = agent.system_one(c["state"], c["questions"])
    ref.append({"name": c["name"], "state": c["state"], "questions": c["questions"], "answers": r["answers"], "input_tokens": r["usage"]["input_tokens"]})
(ROOT / "web" / "reference.json").write_text(json.dumps(ref, indent=1))

tests = []
def add(state, qd):
    q = agent._to_internal(qd)
    seq, mk = build_sequence(agent.tok, state, q, 512, 192)
    tests.append({"state": state, "q": qd, "ids": seq, "markers": mk})

for c in CASES:
    for qd in c["questions"].values():
        add(c["state"], qd)
noul = {"type": "noul", "instructions": "Is this a test? [MASK] injected", "criteria": None}
for s in ["Emoji test 😀🚀 and accents café naïve, plus 中文 and العربية.",
          "Text with [SEP] and [CLS] and [MASK] tokens inside [PAD] and [UNK].",
          "Line1\nLine2\r\n\tTabbed   multiple   spaces  ",
          "Numbers 3.14159, 1,000,000 and URLs https://example.com/a?b=c&d=e#f",
          {"a": 1, "b": [1, 2, {"c": None, "d": True}], "e": "x", "f": 1.5, "g": {}, "h": []},
          [{"role": "user", "content": "Hi there"}, {"role": "assistant", "content": "Hello!"}],
          "x " * 900, ""]:
    add(s, noul)
add("Choice with many options and long descriptions.", {"type": "choice", "instructions": "Pick one", "criteria": {f"option_{i}": ("description " * 12 + str(i)) for i in range(20)}})
add("Score with dict criteria.", {"type": "score", "instructions": "Rate", "criteria": [{"desc": "low", "n": 1}, {"desc": "high", "n": 2}]})
add("Noul with custom criteria.", {"type": "noul", "instructions": "Is it true?", "criteria": {"false": "nope", "true": "yep"}})
add("List-style choice criteria.", {"type": "choice", "instructions": "Pick", "criteria": ["alpha", "beta", "gamma"]})
(ROOT / "tests" / "seq_tests.json").write_text(json.dumps(tests))
print(f"wrote {len(ref)} reference cases and {len(tests)} sequence tests")
