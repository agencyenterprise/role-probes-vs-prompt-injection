# evaluation

Does the userness probe detect prompt injections in tool output? ROC over the
labelled eval set in `data/`.

`data/` is the bare minimum of that dataset: 300 benign host documents
(`documents.jsonl`, provenance in `SOURCES.md`) and 1,800 assembly recipes
(`samples.jsonl`: 1,300 injected / 300 clean / 200 placebo, all label 0/1).
Sample text is rebuilt from the recipe and sha256-asserted; the payload text
travels inside the recipe, so nothing else from that experiment is needed.

`roc.py` rebuilds each sample, renders it as the result of a `fetch` tool call,
scores per-token P(user) with a trained probe (the original harness's operative
detector: max of a rolling window-10 mean), and writes a single ROC curve.
Two context modes, verified character-identical to the original harness:
`in_situ` (default) puts the payload inside the fixed agent conversation — the
product setting; `--context-mode isolated` scores it alone in a bare tool
message — the probe's training distribution.

```bash
uv run python src/evaluation/roc.py gptoss-20b --layer 8               # real run
uv run python src/evaluation/roc.py qwen3-0.6b --layer 14 --limit 120  # local smoke
```

## steering.py — does moving userness change attack success?

The causal counterpart, ported from our original steering experiment
and pointed at the *same* dataset: 200 of these samples are tier-A, carrying a
canary witness token, which makes "did the attack work?" a string match.

Instead of reading userness off the injected tokens, it adds
`alpha * ref_norm * unit(direction)` to them during prefill and grades whether
the canary still gets emitted. `alpha` is a fraction of the mean activation
norm at the steer site, so it is comparable across layers; positive steers
*toward* userness, negative *away*. The control is a matched-norm random vector
orthogonalized against the user direction — if both curves move together, the
effect is generic disruption, not role steering.

```bash
uv run python src/evaluation/steering.py gptoss-20b --layer 8          # real run
uv run python src/evaluation/steering.py qwen3-0.6b --layer 14 --limit 48 \
    --max-new-tokens 900 --oracle in_reply                         # local smoke
```

Only the steer vector differs from upstream's `user_dir` arm: theirs comes from
the pickled cuml probe, ours from the probe this repo trains. `--site` selects
`resid_pre` (default, as upstream's final config) or `post_attn_ln` (where the
probe reads). Steering is applied on the prefill pass only — generated tokens
are new positions.
