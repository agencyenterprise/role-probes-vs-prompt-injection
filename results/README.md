# Results behind "Testing role probes as a prompt-injection defense"

This folder holds the result files behind the probe and steering numbers in our LessWrong post. Each file is copied unchanged from our experiment runs, except for the small changes listed under "Changes from the originals".

The detection dataset these results use is `src/evaluation/data/` in this repo. Every `sample_id` in the files below matches a sample there, and `role-probe-detection/test_sample_ids.txt` lists the 1,209 held-out samples.

## Which file backs which number

### Role probes as detectors (gpt-oss-20b and Apriel-1.6-15B)

| Number in the post | File | How to read it |
|---|---|---|
| 25.2% of all injections and 22.6% of successful ones caught at a 10% false positive rate (userness probe, gpt-oss-20b, layer 8) | `role-probe-detection/detection_at_10fpr.json` | `gpt-oss-20b L8`, `recall_all_at_10fpr` and `recall_success_at_10fpr` |
| 32.6% caught at a 10% false positive rate (Apriel-1.6-15B, layer 24) | `role-probe-detection/detection_at_10fpr.json` | `apriel L24`, `recall_all_at_10fpr` |
| Original probe: detection AUC 0.661, recall 4.7% at a 1% false positive rate, precision 0.47% | `role-probe-detection/probe_comparison_userness.json` | `userness probe, original (L8)`, `test` block |
| Retrained probe at layer 8: detection AUC 0.700, recall 13.8%, precision 1.36%, success AUC 0.556 | `role-probe-detection/gptoss-20b-retrained/` | Computed from `doc_score` in `scores-in_situ-L8.sample.jsonl` on the 1,209 test samples, with success labels from `effectiveness-in_situ.jsonl` |
| Success AUCs 0.522 (original) and 0.524 (Apriel) | `role-probe-detection/gptoss-20b/` and `role-probe-detection/apriel-1.6-15b/` | `doc_score` from the scores file against the success labels in `effectiveness-in_situ.jsonl`, on the 200 graded samples |

### The injection-labelled probe (gpt-oss-20b)

| Number in the post | File | How to read it |
|---|---|---|
| Detection AUC 0.976, recall 62.6% at a 1% false positive rate | `injection-labelled-probe/step3_results.json` | `taps`, `post_attention_layernorm`, `test` block: `auc` and `recall_at_fpr_0.01` |
| Precision 5.9% at one attack per 1,000 tool calls | `injection-labelled-probe/step3_results.json` | Computed from that recall and the 1% false positive rate: 0.626 × 0.001 / (0.626 × 0.001 + 0.01 × 0.999) |
| 0.955 when trained at layer 8 | `injection-labelled-probe/e1_results.json` | Variant `PIShield, L8 only` |
| 0.98 on payloads never seen in training | `injection-labelled-probe/step7_results.json` | `auc_mean` over five folds that hold out documents and payloads |
| 0.97 against the controls with ordinary text spliced in | `injection-labelled-probe/step4_results.json` | `test`, `placebo_vs_injected` |
| Success AUC 0.735 | `injection-labelled-probe/e4_results.json` | `e4a_success`, the out-of-fold row |

### Steering (gpt-oss-20b, layer 10)

"wiki" is the Wikipedia dataset and "ood" is the detection dataset. In the `direction` column, `user_dir` suppresses userness over the injected text, `user_dir_whole` suppresses it over the whole tool result, `user_dir_pos` amplifies it, and `random_0` to `random_2` are the three random directions.

| Number in the post | File | How to read it |
|---|---|---|
| Wikipedia dataset at strength 0.25: 19.8% without steering, 2.5% (whole tool result), 5.0% (injected text), 38% (amplified), 12.1% (random directions, averaged) | `steering/robust-wikiL10.csv` | `asr` column. Each row covers 400 replies |
| Detection dataset at strength 0.2: 31% without steering, 14.5%, 17.5%, 26.1% (random directions, averaged) | `steering/robust-oodL10.csv` | `asr` column. Each row covers 600 replies |
| Strengths 0.15 and 0.2 on the Wikipedia dataset | `steering/robust-wikiL10.csv` | Rows with `alpha` 0.15 and 0.2 |
| Comprehension score 0.488 to 0.475 | `steering/robust-wikiL10.csv` | `benign` column, `baseline` row against `user_dir_whole` at 0.25 |
| McNemar p-values 5.5e-10, 3.1e-8 and 1.4e-7 | `steering/robust-stats-wikiL10.json` and `steering/robust-stats-oodL10.json` | First reply per sample, each steered condition against no steering |
| Layer sweep: 16.5% without steering, 1.5% at layer 8, 3.5% at layer 10, comprehension 0.478 to 0.395 at layer 8 and 0.472 at layer 10 | `steering/sweep-L6.csv` to `steering/sweep-L22.csv` | Strength 0.3, one reply per sample |
| The direction with role cues and politeness words removed: 16.5% to 15.5%, against 12.0% for a random direction | `steering/sweep-L10.csv` | The `destyle` and `random` rows at strength 0.3 |
| The layer-23 attention direction: 45% to 47%, 51% and 51% (100 samples) | `other-steering-directions/e7_results.json` | `injection`, the `tool alpha` rows |

## Not in this folder

The Wikipedia dataset used in the steering runs isn't included. We built it from the `wikimedia/wikipedia` snapshot `20231101.en` on Hugging Face.

## Changes from the originals

- `other-steering-directions/e7_results.json` recorded an absolute path on our machine in `config.loader_file`. We replaced it with a relative path.
- The three `.stats.json` files under `role-probe-detection/` recorded the name of the machine each run used, in a `host` field. We removed that field.
- `injection-labelled-probe/e1_results.json` and `step4_results.json` had informal `verdict` notes written by our analysis scripts. We removed those fields.

No other file in this folder was changed.
