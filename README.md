# Testing role probes as a prompt-injection defense

Code and results for our LessWrong post [Testing role probes as a prompt-injection defense](https://www.lesswrong.com/posts/xsWjzToE6ksmzwovk). It builds on the code for [Prompt Injection as Role Confusion](https://role-confusion.github.io/). Work by [AE Studio](https://ae.studio).

## In one paragraph

The `roleprobe` package trains a role probe, a small classifier on a model's activations that reads which role (system, user, assistant, tool or reasoning) the model treats each token as coming from. We use its score for the user role, which we call userness, in two ways. As a detector, a high userness score inside a tool result flags text the model may be treating as the user's own request. For steering, we subtract the userness direction from the model's activations while it reads tool output, and measure how often injections still succeed.

## Repository layout

```
src/roleprobe/          Probe training: data, activations, probe fitting and export
src/evaluation/         Detection (roc.py) and steering (steering.py) evaluations
src/evaluation/data/    The detection dataset (300 documents, 1,800 samples)
tests/                  Tests for the data pipeline, the dataset and the probe
results/                The result files behind the post's probe and steering numbers
```

## Train

```bash
uv sync

# smoke test
uv run roleprobe-train qwen3-0.6b --n-docs 16 --seq-len 128 --corpus c4 --batch-size 8

# example full run
uv run --extra gptoss roleprobe-train gptoss-20b --batch-size 8
```

Probes are written to `runs/<model>/`. See
`roleprobe-train --help` for options.

Tested on GPT OSS and Qwen3.

## Eval

```bash
# ROC: does userness detect injections in tool output?
uv run python src/evaluation/roc.py gptoss-20b --layer 8

# Steering: does moving userness change attack success?
uv run python src/evaluation/steering.py gptoss-20b --layer 10
```

## Results

`results/README.md` lists the post's probe and steering numbers, the file each one comes from and how to read it.

## Notes on this repository

- The numbers in the post come from our original experiment scripts, which aren't included. This package is a cleaned-up version of the same methods. Its detection evaluation follows the method in the post. Its steering evaluation is simpler, since it steers only the injected text and uses one random direction.
- Trained probe weights, model activations and per-token scores aren't included.
- The Wikipedia dataset used in the steering runs isn't included. `results/README.md` says how we built it.
- For a supervised prompt-injection probe to compare against, see [PIShield](https://github.com/weizou52/PIShield).

## License

MIT, see `LICENSE`. The code for "Prompt Injection as Role Confusion" (Ye, Cui and Hadfield-Menell) that this builds on is also MIT. The dataset includes third-party documents and injection payloads under their own licenses, listed in `src/evaluation/data/SOURCES.md`. Copyright in that material stays with its authors.
