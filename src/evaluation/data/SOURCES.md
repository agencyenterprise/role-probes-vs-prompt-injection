# Sources

Every document and payload in this dataset carries a provenance block:
`source`, `source_ref` (owner/repo@sha:path), `license` (SPDX), and
`content_sha256`. This file summarises it. Copyright in each document stays
with its original authors, and `source_ref` points to the exact file and commit.

**Redistribution rule.** Document text ships only under the SPDX allowlist
below (8 ids). Anything else is recorded as a fetch
recipe with `text_included: false` and no text.

```
  0BSD  Apache-2.0  BSD-2-Clause  BSD-3-Clause  CC0-1.0  ISC  MIT  Unlicense
```

## Benign documents

| doc_type | n | source | license(s) |
|---|---|---|---|
| code | 60 | github | Apache-2.0, BSD-2-Clause, BSD-3-Clause, MIT, Unlicense |
| config | 40 | github | Apache-2.0, BSD-3-Clause, MIT, Unlicense |
| docs | 60 | github | Apache-2.0, BSD-2-Clause, BSD-3-Clause, MIT, Unlicense |
| email | 30 | handwritten | CC0-1.0 |
| jsonish | 10 | handwritten | CC0-1.0 |
| readme | 70 | github | Apache-2.0, BSD-2-Clause, BSD-3-Clause, MIT |
| web | 30 | github | Apache-2.0, BSD-3-Clause, MIT |

GitHub documents come from repositories with >2000 stars across 12 language ecosystems, filtered to the
allowlist using the `license.spdx_id` the API returns per repository.
Hand-written documents (emails, tool returns) are CC0-1.0, authored for
this dataset, and contain no real personal data.

## Injection payloads

| source | n | license | upstream |
|---|---|---|---|
| vendored | 31 | MIT | 11 from the [Role Confusion paper's code](https://github.com/role-confusion/prompt-injection-as-role-confusion) (`agent-injections/config/injection.yaml`, `cot-forgery-agent-evals/prompts/injections.yaml`) and 20 from our earlier test scripts |
| bipia | 30 | MIT | [microsoft/BIPIA](https://github.com/microsoft/BIPIA) `benchmark/*_attack_*.json` |
| agentdojo | 5 | MIT | [ethz-spylab/agentdojo](https://github.com/ethz-spylab/agentdojo) `src/agentdojo/attacks/*.py` |
| injecagent | 20 | MIT | [uiuc-kang-lab/InjecAgent](https://github.com/uiuc-kang-lab/InjecAgent) `data/attacker_cases_*.jsonl` |
| handwritten | 25 | CC0-1.0 | written for this dataset |

### Safety of the payloads

The payloads are real prompt-injection strings. The evaluation never executes a
payload. It only checks whether the model emitted the canary token.

## Considered and rejected

| source | why not |
|---|---|
| `HuggingFaceH4/stack-exchange-preferences` | CC-BY-SA: share-alike would apply to the whole dataset |
| `allenai/c4` | excluded. It is also probe training data, so it would have deflated the false-positive rate |
| `bigcode/the-stack-dedup` | gated (per-person terms acceptance) — not reproducible |
| `microsoft/llmail-inject-challenge` | out of scope |
| `wikimedia/wikipedia` | CC-BY-SA: share-alike is contagious for a redistributed derivative |
