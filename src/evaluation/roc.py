"""Score the injection eval set with a trained role probe and plot a ROC curve.

Data (evaluation/data/) is our labelled injection eval set: benign
host documents plus assembly recipes that splice attack payloads into them.
Each sample is rebuilt from its recipe (sha256-asserted), rendered as the tool
result — either inside the original harness's fixed agent conversation
(`in_situ`, the product setting and its default) or alone in a bare tool
message (`isolated`, the probe's training distribution) — and scored with the
harness's operative detector: max over a rolling mean (window 10) of per-token
P(user). Labels are per document.

    uv run python src/evaluation/roc.py gptoss-20b --layer 8
    uv run python src/evaluation/roc.py qwen3-0.6b --layer 14 --limit 120   # local smoke
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from common import assemble, context_ids, load_dataset, pick_probe, userness
from roleprobe.activations import collect_hidden_states
from roleprobe.data import Sequence
from roleprobe.export import load_probes
from roleprobe.models import MODELS, load_model


def window_max(p, w):
    w = max(1, min(w, len(p)))
    return float(np.convolve(p, np.ones(w) / w, mode='valid').max())


def roc(scores, labels):
    order = np.argsort(-scores, kind='stable')
    y = labels[order]
    tpr = np.concatenate([[0.0], np.cumsum(y) / y.sum()])
    fpr = np.concatenate([[0.0], np.cumsum(1 - y) / (1 - y).sum()])
    return fpr, tpr, float(np.trapezoid(tpr, fpr))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('model', choices=sorted(MODELS))
    ap.add_argument('--layer', type=int, required=True)
    ap.add_argument('--probes', help='probe dir (default: runs/<model>)')
    ap.add_argument('--limit', type=int, help='evaluate a label-balanced subset')
    ap.add_argument('--context-mode', choices=('in_situ', 'isolated'), default='in_situ')
    ap.add_argument('--window', type=int, default=10)
    ap.add_argument('--max-tokens', type=int, default=1024)
    ap.add_argument('--tool-name', default='fetch')
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--chunk', type=int, default=256, help='samples per capture chunk')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--out', default='outputs/roc.png')
    args = ap.parse_args()

    docs, samples = load_dataset()
    if args.limit:
        samples = samples[::max(1, len(samples) // args.limit)][:args.limit]
    labels = np.array([s['label'] for s in samples])
    print(f'{len(samples)} samples ({labels.sum()} injected, {len(labels) - labels.sum()} benign)')

    spec = MODELS[args.model]
    probe = pick_probe(load_probes(args.model, args.probes or f'runs/{args.model}'), args.layer)
    print(f'probe: layer {args.layer}, roles {probe["role_space"]}')
    tok, model = load_model(spec, args.device)

    pre, post = context_ids(tok, spec.fmt, args.context_mode, args.tool_name)
    budget = args.max_tokens - len(pre) - len(post)

    scores = np.zeros(len(samples))
    for lo in range(0, len(samples), args.chunk):
        chunk = samples[lo:lo + args.chunk]
        texts = [assemble(s, docs[s['doc_id']]) for s in chunk]
        seqs = [Sequence(lo + i, 'tool', pre + tuple(ids[:budget]) + post,
                         slice(len(pre), len(pre) + min(len(ids), budget)))
                for i, ids in enumerate(tok(texts, add_special_tokens=False).input_ids)]
        hs, doc_ix, _ = collect_hidden_states(model, seqs, [args.layer],
                                              tok.pad_token_id, args.batch_size, args.device)
        p_user = userness(hs[args.layer].float().numpy(), probe)
        for i in np.unique(doc_ix):
            scores[i] = window_max(p_user[doc_ix == i], args.window)

    fpr, tpr, auc = roc(scores, labels)
    print(f'AUC: {auc:.3f}')

    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr)
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title(f'ROC — userness probe, {args.model} layer {args.layer}, '
              f'{args.context_mode} (AUC {auc:.3f})', fontsize=10)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=150)
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
