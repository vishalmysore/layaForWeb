# Test data

108 labeled cases in 9 domains, for trying the model on more than the six built-in examples. Each domain is one JSON file with a shared set of typed questions and 12 cases.

| File | Domain | Questions |
|---|---|---|
| `support-tickets.json` | Software support tickets (JSON state) | team (choice), urgency (score), angry (yes/no) |
| `product-reviews.json` | Household product reviews | sentiment (score), topic (choice), recommends (yes/no) |
| `agent-guardrails.json` | Actions an AI agent plans to take | destructive, needs_human, safe_without_approval (yes/no), risk (score) |
| `it-incidents.json` | Monitoring alerts and incident reports | severity (score), owner (choice), customer_impact (yes/no) |
| `content-moderation.json` | Blog comments | verdict (choice), toxicity (score), spam (yes/no) |
| `email-triage.json` | Workplace emails | action (choice), needs_reply (yes/no), urgency (score) |
| `delivery-exceptions.json` | Parcel delivery problems | action (choice), urgency (score), upset (yes/no) |
| `sales-leads.json` | Inbound sales messages | lead_quality (score), next_step (choice) |
| `patient-messages.json` | Clinic portal messages (synthetic, routing only) | route (choice), urgent (yes/no) |

`index.json` lists all files.

## File format

```json
{
  "domain": "product-reviews",
  "description": "...",
  "questions": { "...": "same shape the demo and the Python SDK use" },
  "cases": [
    {
      "id": "pr-04",
      "state": "Broke after two days. Cheap plastic, the hinge snapped. Do not buy.",
      "expected": { "sentiment": 0, "topic": "quality", "recommends": false },
      "hard": ["recommends"]
    }
  ]
}
```

`expected` holds the label for each question: an option name for `choice`, a level index (0 is the first entry of `criteria`) for `score`, and `true` or `false` for `noul`. `hard` lists the questions where the label is a judgement call; everything else is meant to be clear.

## Use it in the demo

Copy a file's `questions` object into the **Questions** box and one case's `state` into the **State** box, then press Run. Nothing here appears in the Example dropdown.

## Score it against the original PyTorch model

```
pip install laya
python scripts/eval_dataset.py                    # downloads convaiinnovations/laya, CPU is enough
python scripts/eval_dataset.py --domain it-incidents
python scripts/eval_dataset.py --out results.json # every answer, with probabilities
```

A choice is correct when the top option matches, a yes/no when P(yes) is on the right side of 0.5, and a score when the most probable level matches ("within one" allows one level off).

## What the first run showed

Original fp32 PyTorch model on a CPU, 312 answers (`--threshold 0.90`):

| Domain | Correct | Clear labels only |
|---|---|---|
| content-moderation | 77.8% | 87.1% |
| patient-messages | 75.0% | 72.7% |
| product-reviews | 75.0% | 76.0% |
| it-incidents | 63.9% | 66.7% |
| support-tickets | 58.3% | 64.5% |
| delivery-exceptions | 55.6% | 60.7% |
| email-triage | 50.0% | 50.0% |
| agent-guardrails | 41.7% | 42.2% |
| sales-leads | 41.7% | 47.4% |
| **All** | **59.3%** | **62.3%** |

By type: yes/no 75.8%, choice 60.4%, score 37.5% exact (81.2% within one level).

The confidence value is what makes the model usable on this data. Of 312 answers, 55 had confidence of at least 0.90, and 94.5% of those were correct. The other 257 were right 51.8% of the time, so routing them to a person is the right call. Three of the 55 confident answers were wrong, so a high confidence is a strong signal and not a guarantee: `ag-11` (deleting a whole storage bucket with versioning off, called safe to run at 93%), `em-07` (a flooding server room, "no reply expected" at 93%) and `pm-03` (a very sleepy child with a fever, "not life-threatening" at 93%).

Two patterns stand out. The score head stays near the middle of the scale: it answered "Medium" for the risk of all 12 agent actions, including the ones labeled Low and Critical. Use the probabilities and not only the top level. Agent-guardrail questions worded as "is this safe to run" are the weakest case; check them on your own wording before relying on them.

## Read this before quoting the numbers

- The labels were written by the author of this repository and have not been reviewed by other people.
- The cases and question wordings were not tuned for the model. Different wording of the same question changes the result.
- Twelve cases per domain is too few for a benchmark. Treat the table as a smoke test that shows where the model is confident and where it is not.
- The figures are for the PyTorch model. The browser builds differ from it by about 0.01 on average (see the main README).
- All text is synthetic.
