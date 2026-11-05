"""Train role probes: corpus -> role-wrapped sequences -> hidden states -> probes."""
import argparse

import torch

from .activations import collect_hidden_states
from .data import build_sequences, load_texts, parse_mix
from .export import export_probes
from .models import MODELS, default_layers, load_model
from .probe import train_probes

ROLE_SPACES = [
    ('user', 'assistant'),
    ('user', 'assistant', 'tool'),
    ('user', 'cot', 'assistant'),
    ('user', 'cot', 'assistant', 'tool'),
    ('system', 'user', 'assistant'),
    ('system', 'user', 'assistant', 'tool'),
    ('system', 'user', 'cot', 'assistant'),
    ('system', 'user', 'cot', 'assistant', 'tool'),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('model', choices=sorted(MODELS))
    ap.add_argument('--n-docs', type=int, default=250)
    ap.add_argument('--seq-len', type=int, default=1024)
    ap.add_argument('--corpus', default='c4=0.25,dolma3=0.75')
    ap.add_argument('--layers', help='comma-separated indices (default: every 2nd/4th)')
    ap.add_argument('--tool-names', default='', help='comma-separated pool for the tool role name')
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--C', type=float, help='override the registry inverse L2 strength')
    ap.add_argument('--test-frac', type=float, default=0.1)
    ap.add_argument('--seed', type=int, default=123)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--out', default='runs')
    args = ap.parse_args()

    spec = MODELS[args.model]
    layers = ([int(x) for x in args.layers.split(',')] if args.layers
              else default_layers(spec.n_layers))
    C = args.C if args.C is not None else spec.C

    texts = load_texts(parse_mix(args.corpus), args.n_docs, args.seed)
    tok, model = load_model(spec, args.device)
    seqs = build_sequences(texts, tok, spec.fmt, args.seq_len,
                           tool_names=args.tool_names.split(','), seed=args.seed)
    print(f'{len(texts)} docs -> {len(seqs)} sequences; layers {layers}; C={C} scale={spec.scale}')

    hs, doc_ix, role = collect_hidden_states(model, seqs, layers, tok.pad_token_id,
                                             args.batch_size, args.device)
    del model
    if args.device != 'cpu':
        torch.cuda.empty_cache()
    print(f'{len(doc_ix)} content tokens x {len(layers)} layers captured')

    probes = train_probes(hs, doc_ix, role, ROLE_SPACES, C=C, scale=spec.scale,
                          seed=args.seed, device=args.device, test_frac=args.test_frac)

    out = export_probes(probes, args.model, f'{args.out}/{args.model}',
                        run_meta={'config': vars(args), 'layers': layers, 'C': C,
                                  'scale': spec.scale, 'role_spaces': ROLE_SPACES})
    print(f'wrote {len(probes)} probes -> {out}')


if __name__ == '__main__':
    main()
