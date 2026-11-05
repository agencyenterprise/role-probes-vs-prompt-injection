from types import SimpleNamespace

from roleprobe.data import build_sequences, parse_mix
from roleprobe.models import HARMONY, ROLES


class FakeTok:
    """Tokenizes each character to its codepoint."""

    def __call__(self, text, add_special_tokens=False, truncation=False, max_length=None):
        if isinstance(text, list):
            return SimpleNamespace(input_ids=[[ord(c) for c in t][:max_length] for t in text])
        return SimpleNamespace(input_ids=[ord(c) for c in text])


def test_parse_mix():
    assert parse_mix('c4=0.25,dolma3=0.75') == {'c4': 0.25, 'dolma3': 0.75}
    assert parse_mix('c4') == {'c4': 1.0}


def test_content_slices_are_exact():
    texts = ['hello world', 'x' * 50]
    seqs = build_sequences(texts, FakeTok(), HARMONY, seq_len=8, tool_names=('fetch',))

    assert len(seqs) == len(texts) * len(ROLES)
    for s in seqs:
        doc = [ord(c) for c in texts[s.doc_ix]][:8]
        assert list(s.ids[s.content]) == doc
        prefix = ''.join(chr(i) for i in s.ids[:s.content.start])
        assert prefix == HARMONY.wrap[s.role][0].format(tool_name='fetch')
    assert {s.role for s in seqs} == set(ROLES)
