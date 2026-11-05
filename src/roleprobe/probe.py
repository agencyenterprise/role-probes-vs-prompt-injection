"""L2-regularized logistic-regression probes, full-batch LBFGS."""
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class Probe:
    roles: tuple
    layer: int
    coef: np.ndarray               # (n_roles, D); (1, D) sigmoid form when binary
    intercept: np.ndarray
    scaler_mean: np.ndarray | None
    scaler_scale: np.ndarray | None
    acc: float
    nll: float
    n_inputs: int
    confusion: np.ndarray          # test counts, rows = true role, cols = predicted

    @property
    def binary(self):
        return len(self.roles) == 2


def _fit(X, y, n_classes, C, max_iter):
    """sklearn objective: min 0.5||W||^2 + C * sum_i CE_i, rescaled by 1/(C n)."""
    n, d = X.shape
    k = 1 if n_classes == 2 else n_classes
    W = torch.zeros(k, d, device=X.device, requires_grad=True)
    b = torch.zeros(k, device=X.device, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], max_iter=max_iter, history_size=20,
                            line_search_fn='strong_wolfe')
    yf = y.float()

    def closure():
        opt.zero_grad(set_to_none=True)
        z = X @ W.T + b
        data = (F.binary_cross_entropy_with_logits(z[:, 0], yf) if k == 1
                else F.cross_entropy(z, y))
        loss = data + W.square().sum() / (2 * C * n)
        loss.backward()
        return loss

    opt.step(closure)
    return W.detach(), b.detach()


def _eval(X, y, W, b, n_classes):
    z = X @ W.T + b
    if n_classes == 2:
        logp = torch.stack([F.logsigmoid(-z[:, 0]), F.logsigmoid(z[:, 0])], dim=1)
    else:
        logp = F.log_softmax(z, dim=1)
    pred = logp.argmax(1)
    acc = (pred == y).float().mean().item()
    nll = F.nll_loss(logp, y).item()
    conf = torch.bincount(y * n_classes + pred, minlength=n_classes ** 2)
    return acc, nll, conf.reshape(n_classes, n_classes).cpu().numpy()


def train_probes(hs, doc_ix, role, spaces, *, C, scale, seed, device,
                 test_frac=0.1, max_iter=300):
    """Fit one probe per (role space x layer); train/test split by document."""
    docs = np.unique(doc_ix)
    test_docs = np.random.default_rng(seed).permutation(docs)[:max(1, round(len(docs) * test_frac))]
    in_test = np.isin(doc_ix, test_docs)

    prepared = []
    for roles in spaces:
        role_ix = {r: i for i, r in enumerate(roles)}
        y = np.array([role_ix.get(r, -1) for r in role])
        sel = y >= 0
        prepared.append((roles, y, np.where(sel & ~in_test)[0], np.where(sel & in_test)[0]))

    probes = []
    for layer, X_cpu in hs.items():
        X = X_cpu.to(device)
        for roles, y, tr, te in prepared:
            Xtr = X[torch.from_numpy(tr).to(device)].float()
            ytr = torch.from_numpy(y[tr]).to(device)
            assert len(ytr.unique()) == len(roles), f'missing roles in train: {roles}'
            if scale:
                mean, std = Xtr.mean(0), Xtr.std(0, correction=0).clamp_min(1e-8)
                Xtr = (Xtr - mean) / std
            W, b = _fit(Xtr, ytr, len(roles), C, max_iter)
            del Xtr

            Xte = X[torch.from_numpy(te).to(device)].float()
            if scale:
                Xte = (Xte - mean) / std
            acc, nll, conf = _eval(Xte, torch.from_numpy(y[te]).to(device), W, b, len(roles))
            del Xte

            probes.append(Probe(
                roles=roles, layer=layer,
                coef=W.cpu().numpy(), intercept=b.cpu().numpy(),
                scaler_mean=mean.cpu().numpy() if scale else None,
                scaler_scale=std.cpu().numpy() if scale else None,
                acc=acc, nll=nll, n_inputs=len(tr) + len(te), confusion=conf))
            print(f'layer {layer:>3} {"/".join(roles):<40} acc {acc:.3f}  nll {nll:.3f}')
        del X
        if device != 'cpu':
            torch.cuda.empty_cache()
    return probes
