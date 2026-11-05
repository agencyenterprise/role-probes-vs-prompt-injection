"""Hidden-state capture: each probed layer's MLP input (post-attention layernorm)."""
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


@torch.inference_mode()
def _sanity_ppl(model, seqs, pad_id, device):
    batch = sorted(seqs, key=lambda s: len(s.ids))[:4]
    T = max(len(s.ids) for s in batch)
    input_ids = torch.full((len(batch), T), pad_id, dtype=torch.long)
    mask = torch.zeros(len(batch), T, dtype=torch.bool)
    for r, s in enumerate(batch):
        input_ids[r, :len(s.ids)] = torch.tensor(s.ids)
        mask[r, :len(s.ids)] = True
    input_ids, mask = input_ids.to(device), mask.to(device)
    logits = model(input_ids=input_ids, attention_mask=mask).logits[:, :-1]
    m = mask[:, 1:]
    nll = F.cross_entropy(logits[m].float(), input_ids[:, 1:][m])
    print(f'sanity ppl (4 shortest seqs): {nll.exp().item():.1f}')


@torch.inference_mode()
def collect_hidden_states(model, seqs, layers, pad_id, batch_size, device):
    """Forward-pass all sequences and keep content-token states at the probed layers.

    Returns (hs, doc_ix, role): hs maps layer -> [n_tokens, D] fp16 cpu; doc_ix
    and role are parallel per-token label arrays.
    """
    grabbed = {}
    hooks = [model.model.layers[L].mlp.register_forward_pre_hook(
        lambda _mod, args, L=L: grabbed.__setitem__(L, args[0])) for L in layers]

    _sanity_ppl(model, seqs, pad_id, device)

    order = sorted(range(len(seqs)), key=lambda i: -len(seqs[i].ids))  # minimize padding
    hs, doc_ix, role = {L: [] for L in layers}, [], []
    try:
        for start in tqdm(range(0, len(order), batch_size), desc='forward'):
            batch = [seqs[i] for i in order[start:start + batch_size]]
            T = max(len(s.ids) for s in batch)
            input_ids = torch.full((len(batch), T), pad_id, dtype=torch.long)
            mask = torch.zeros(len(batch), T, dtype=torch.bool)
            content = torch.zeros(len(batch), T, dtype=torch.bool)
            for r, s in enumerate(batch):
                input_ids[r, :len(s.ids)] = torch.tensor(s.ids)
                mask[r, :len(s.ids)] = True
                content[r, s.content] = True
                doc_ix += [s.doc_ix] * (s.content.stop - s.content.start)
                role += [s.role] * (s.content.stop - s.content.start)
            model(input_ids=input_ids.to(device), attention_mask=mask.to(device), logits_to_keep=1)
            content = content.to(device)
            for L in layers:
                hs[L].append(grabbed.pop(L)[content].to(torch.float16).cpu())
    finally:
        for h in hooks:
            h.remove()

    return ({L: torch.cat(chunks) for L, chunks in hs.items()},
            np.array(doc_ix), np.array(role))
