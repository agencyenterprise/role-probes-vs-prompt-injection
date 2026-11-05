import json

import numpy as np
import torch
from safetensors.numpy import load_file

from roleprobe.export import export_probes
from roleprobe.probe import train_probes


def _synthetic(k=3, n=900, d=8, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, k, n)
    centers = rng.normal(size=(k, d)) * 4
    X = centers[y] + rng.normal(size=(n, d))
    roles = np.array(['user', 'assistant', 'tool'])[y]
    return {0: torch.tensor(X, dtype=torch.float16)}, np.arange(n) % 50, roles


def test_probes_recover_separable_classes():
    hs, doc_ix, role = _synthetic()
    spaces = [('user', 'assistant', 'tool'), ('user', 'assistant')]
    probes = train_probes(hs, doc_ix, role, spaces, C=1.0, scale=True, seed=0, device='cpu')

    assert len(probes) == 2
    for p in probes:
        assert p.acc > 0.9
        assert p.confusion.sum() > 0
    assert probes[1].binary and probes[1].coef.shape[0] == 1
    assert probes[0].coef.shape[0] == 3


def test_export_round_trip(tmp_path):
    hs, doc_ix, role = _synthetic()
    probes = train_probes(hs, doc_ix, role, [('user', 'assistant')],
                          C=1.0, scale=False, seed=0, device='cpu')
    out = export_probes(probes, 'toy', tmp_path, run_meta={'note': 'test'})

    tensors = load_file(out / 'toy.safetensors')
    meta = json.loads((out / 'toy.json').read_text())
    assert tensors['probe_0.coef'].shape == (1, 8)
    assert meta['probes'][0]['binary'] and not meta['probes'][0]['has_scaler']
    assert meta['probes'][0]['roles_map'] == {'user': 0, 'assistant': 1}
    assert (out / 'toy_metrics.csv').exists() and (out / 'toy_confusion.csv').exists()
