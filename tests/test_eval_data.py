import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / 'src' / 'evaluation'))
from common import assemble, load_dataset  # noqa: E402


def test_every_sample_reassembles_to_its_hash():
    """assemble() asserts the sha256, so this also pins the vendored data."""
    docs, samples = load_dataset()
    assert len(samples) == 1800
    for s in samples:
        assemble(s, docs[s['doc_id']])


def test_spans_locate_the_payload():
    docs, samples = load_dataset()
    checked = 0
    for s in samples:
        text = assemble(s, docs[s['doc_id']])
        for sp in s['spans']:
            assert text[sp['char_start']:sp['char_end']] == sp['rendered_text']
            checked += 1
    assert checked > 1000


def test_canary_witnesses_live_inside_their_span():
    docs, samples = load_dataset()
    tier_a = [(s, sp) for s in samples for sp in s['spans'] if sp.get('witness')]
    assert len(tier_a) == 200
    for s, sp in tier_a:
        assert sp['witness'] in assemble(s, docs[s['doc_id']])[sp['char_start']:sp['char_end']]


def test_random_control_carries_no_userness():
    rng = np.random.default_rng(0)
    user_dir = rng.standard_normal(64)
    user_dir /= np.linalg.norm(user_dir)
    rand = rng.standard_normal(64)
    rand -= rand @ user_dir * user_dir
    rand /= np.linalg.norm(rand)
    assert abs(rand @ user_dir) < 1e-12
