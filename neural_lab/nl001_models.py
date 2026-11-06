"""Unfrozen NL-001 dense calibration model and update primitive.

Requires PyTorch. This module deliberately has no CLI, data loading, CUDA
selection, checkpoint selection, or queue integration. It is not runnable as a
scientific experiment until its protocol and complete data flow are reviewed.
"""
import torch
from torch import nn
import torch.nn.functional as F
from .nl001_model_input import MAX_NODES, MAX_TOKENS, VOCAB


class CalibrationModel(nn.Module):
    def __init__(self, arm, hidden=128, embedding=32):
        super().__init__()
        if arm not in ('A', 'B'):
            raise ValueError('arm must be A or B')
        self.arm = arm
        self.tokens = nn.Embedding(max(VOCAB.values()) + 1, embedding, padding_idx=0)
        # The positional pointer is needed to preserve subject/object identity.
        self.pointers = nn.Embedding(MAX_NODES + 1, embedding, padding_idx=0)
        self.encoder = nn.Sequential(nn.Linear(MAX_NODES * MAX_NODES + 2 * embedding * MAX_TOKENS + 2, hidden),
                                     nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        if arm == 'A':
            self.answer = nn.Linear(hidden, 3)
        else:
            self.property = nn.Linear(hidden, 3)
            self.subject = nn.Linear(hidden, MAX_NODES)
            self.object = nn.Linear(hidden, MAX_NODES)
            self.edges = nn.Linear(hidden, MAX_NODES * MAX_NODES)

    def forward(self, inputs):
        tokens, pointers = inputs['tokens'].long(), inputs['pointers'].long()
        # Retain word and pointer positions; mean pooling loses argument order.
        token_features = self.tokens(tokens).flatten(1)
        pointer_features = self.pointers(pointers).flatten(1)
        adj = inputs['adjacency'].float().flatten(1)
        extra = torch.stack((inputs['node_count'].float() / MAX_NODES, inputs['complete'].float()), dim=1)
        hidden = self.encoder(torch.cat((token_features, pointer_features, adj, extra), dim=1))
        if self.arm == 'A':
            return {'status': self.answer(hidden)}
        valid = torch.arange(MAX_NODES, device=hidden.device)[None, :] < inputs['node_count'][:, None]
        return {'property': self.property(hidden),
                'subject': self.subject(hidden).masked_fill(~valid, float('-inf')),
                'object': self.object(hidden).masked_fill(~valid, float('-inf')),
                'edges': self.edges(hidden).reshape(-1, MAX_NODES, MAX_NODES)}


def training_loss(model, inputs, targets, *, with_cross_entropy=False):
    """Loss for one batch; witness loss uses observed edges of positive examples."""
    out = model(inputs)
    if model.arm == 'A':
        loss = F.cross_entropy(out['status'], targets['status'].long())
        return (loss, loss) if with_cross_entropy else loss
    loss = sum(F.cross_entropy(out[k], targets[k].long()) for k in ('property', 'subject', 'object'))
    cross_entropy = loss
    mask = inputs['adjacency'].bool() & targets['has_witness'].bool()[:, None, None]
    if mask.any():
        loss = loss + F.binary_cross_entropy_with_logits(out['edges'][mask], targets['witness'].float()[mask])
    return (loss, cross_entropy) if with_cross_entropy else loss


def update(model, optimizer, inputs, targets, *, observe=None):
    """Perform exactly one optimizer update; caller owns timing and scheduling."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss, cross_entropy = training_loss(model, inputs, targets, with_cross_entropy=True)
    if not torch.isfinite(loss) or not torch.isfinite(cross_entropy):
        raise ValueError('non-finite training loss')
    if observe is not None:
        observe(model, float(cross_entropy.detach()))
    loss.backward()
    optimizer.step()
    return float(loss.detach())
