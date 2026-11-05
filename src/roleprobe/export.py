"""Portable probe export: safetensors + JSON sidecar + metrics CSVs.

File layout matches the legacy utils/probe_io.py format, so existing loaders
(PortableProbe et al.) keep working.
"""
import csv
import json
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file, save_file

FORMAT_VERSION = 1


def load_probes(model_prefix, out_dir):
    """Inverse of export_probes: probe metadata dicts with weight arrays attached."""
    out = Path(out_dir)
    tensors = load_file(out / f'{model_prefix}.safetensors')
    meta = json.loads((out / f'{model_prefix}.json').read_text())
    for m in meta['probes']:
        i = m['index']
        m['coef'] = tensors[f'probe_{i}.coef']
        m['intercept'] = tensors[f'probe_{i}.intercept']
        m['scaler_mean'] = tensors.get(f'probe_{i}.scaler_mean')
        m['scaler_scale'] = tensors.get(f'probe_{i}.scaler_scale')
    return meta['probes']


def export_probes(probes, model_prefix, out_dir, run_meta=None):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    tensors, meta = {}, []
    for i, p in enumerate(probes):
        tensors[f'probe_{i}.coef'] = np.ascontiguousarray(p.coef, dtype=np.float32)
        tensors[f'probe_{i}.intercept'] = np.ascontiguousarray(p.intercept, dtype=np.float32)
        if p.scaler_mean is not None:
            tensors[f'probe_{i}.scaler_mean'] = np.ascontiguousarray(p.scaler_mean, dtype=np.float32)
            tensors[f'probe_{i}.scaler_scale'] = np.ascontiguousarray(p.scaler_scale, dtype=np.float32)
        meta.append({
            'index': i, 'layer_ix': int(p.layer), 'role_space': list(p.roles),
            'roles_map': {r: j for j, r in enumerate(p.roles)},
            'binary': p.binary, 'has_scaler': p.scaler_mean is not None,
            'hidden_dim': int(p.coef.shape[1]),
            'acc': p.acc, 'nll': p.nll, 'n_inputs': p.n_inputs,
        })

    save_file(tensors, out / f'{model_prefix}.safetensors')
    (out / f'{model_prefix}.json').write_text(json.dumps({
        'format_version': FORMAT_VERSION, 'model_prefix': model_prefix,
        'n_probes': len(meta), 'probes': meta, **(run_meta or {}),
    }, indent=2, default=str))

    with open(out / f'{model_prefix}_metrics.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['layer_ix', 'role_space', 'acc', 'nll', 'n_inputs'])
        for p in probes:
            w.writerow([p.layer, ','.join(p.roles), round(p.acc, 4), round(p.nll, 4), p.n_inputs])

    with open(out / f'{model_prefix}_confusion.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['layer_ix', 'role_space', 'role', 'pred', 'count'])
        for p in probes:
            for i, r in enumerate(p.roles):
                for j, pr in enumerate(p.roles):
                    w.writerow([p.layer, ','.join(p.roles), r, pr, int(p.confusion[i, j])])

    return out
