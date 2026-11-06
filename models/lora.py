"""Minimal LoRA (Low-Rank Adaptation) for the PVGround grounding model.

The adapters are injected **in place**: the wrapped layer keeps the *same*
parameter names as the original one, so a checkpoint produced by the original
network (``data/PV-Ground_ScanRefer_single.pth``) loads with ``strict=False``
and *zero* renaming::

    nn.Linear            -> weight, bias                  (+ lora_A, lora_B)
    nn.MultiheadAttention-> in_proj_weight, in_proj_bias,
                            out_proj.weight, out_proj.bias (+ lora_in_A/B,
                                                            lora_out_A/B)

Initialisation follows the standard LoRA recipe: ``lora_B = 0`` (and therefore
``delta = 0``) while ``lora_A`` is Kaiming-uniform.  Consequently a freshly
adapted model is **bit-identical** to the pretrained one (up to fp round-off),
so training starts exactly from the pretrained behaviour.

Only the ``lora_*`` parameters (plus explicitly whitelisted new parameters, see
:func:`mark_only_lora_trainable`) are trainable; every other parameter of the
network is frozen.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    'LoRALinear', 'LoRAMultiheadAttention', 'apply_lora',
    'mark_only_lora_trainable', 'lora_state_dict', 'lora_named_parameters',
    'count_trainable_parameters', 'LORA_PREFIX',
]

LORA_PREFIX = 'lora_'


def _kaiming_uniform(weight, a=math.sqrt(5)):
    nn.init.kaiming_uniform_(weight, a=a)


class LoRALinear(nn.Module):
    """``nn.Linear`` + low-rank update ``B @ A`` (``B`` zero-initialised)."""

    def __init__(self, base, r=8, alpha=16, dropout=0.0):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError(f'LoRALinear expects nn.Linear, got {type(base)}')
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.r = int(r)
        self.alpha = float(alpha)
        self.scaling = self.alpha / max(self.r, 1)

        # same attribute names as nn.Linear -> checkpoint keys are preserved
        self.weight = nn.Parameter(base.weight.detach().clone(), requires_grad=False)
        if base.bias is not None:
            self.bias = nn.Parameter(base.bias.detach().clone(), requires_grad=False)
        else:
            self.register_parameter('bias', None)

        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        if self.r > 0:
            ref = self.weight
            kw = {'device': ref.device, 'dtype': ref.dtype}
            self.lora_A = nn.Parameter(torch.zeros(self.r, self.in_features, **kw))
            self.lora_B = nn.Parameter(torch.zeros(self.out_features, self.r, **kw))
            _kaiming_uniform(self.lora_A)
            # lora_B stays zero -> delta == 0 at initialisation
        # keep the mode (train/eval) of the wrapped layer
        self.train(base.training)

    def extra_repr(self):
        return (f'in_features={self.in_features}, out_features={self.out_features}, '
                f'r={self.r}, alpha={self.alpha}')

    def forward(self, x):
        out = F.linear(x, self.weight, self.bias)
        if self.r > 0:
            delta = F.linear(F.linear(self.lora_dropout(x), self.lora_A), self.lora_B)
            out = out + delta * self.scaling
        return out


class LoRAMultiheadAttention(nn.Module):
    """``nn.MultiheadAttention`` (self/cross attention) + LoRA on QKV and out.

    Supports the two call patterns used in this project: sequence-first
    ``(L, B, E)`` and ``batch_first`` ``(B, L, E)``, with an optional boolean
    ``key_padding_mask`` (``True`` == padded/ignored, as in ``nn.MultiheadAttention``).
    ``need_weights`` is ignored (``None`` is returned, as the callers only read
    ``out[0]``), ``attn_mask`` is not used anywhere in this codebase.
    """

    def __init__(self, base, r=8, alpha=16, dropout=0.0):
        super().__init__()
        if not isinstance(base, nn.MultiheadAttention):
            raise TypeError(
                f'LoRAMultiheadAttention expects nn.MultiheadAttention, got {type(base)}')
        if not getattr(base, '_qkv_same_embed_dim', True):
            raise NotImplementedError('only q/k/v with the same embed_dim is supported')

        self.embed_dim = base.embed_dim
        self.num_heads = base.num_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.dropout = float(base.dropout)
        self.batch_first = bool(base.batch_first)
        self.r = int(r)
        self.alpha = float(alpha)
        self.scaling = self.alpha / max(self.r, 1)

        # identical parameter names -> pretrained checkpoint stays compatible
        self.in_proj_weight = nn.Parameter(
            base.in_proj_weight.detach().clone(), requires_grad=False)
        if base.in_proj_bias is not None:
            self.in_proj_bias = nn.Parameter(
                base.in_proj_bias.detach().clone(), requires_grad=False)
        else:
            self.register_parameter('in_proj_bias', None)

        self.out_proj = nn.Linear(
            self.embed_dim, self.embed_dim, bias=base.out_proj.bias is not None
        ).to(device=base.out_proj.weight.device, dtype=base.out_proj.weight.dtype)
        with torch.no_grad():
            self.out_proj.weight.copy_(base.out_proj.weight)
            if base.out_proj.bias is not None:
                self.out_proj.bias.copy_(base.out_proj.bias)
        for p in self.out_proj.parameters():
            p.requires_grad = False

        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        if self.r > 0:
            ref = self.in_proj_weight
            kw = {'device': ref.device, 'dtype': ref.dtype}
            self.lora_in_A = nn.Parameter(torch.zeros(self.r, self.embed_dim, **kw))
            self.lora_in_B = nn.Parameter(torch.zeros(3 * self.embed_dim, self.r, **kw))
            self.lora_out_A = nn.Parameter(torch.zeros(self.r, self.embed_dim, **kw))
            self.lora_out_B = nn.Parameter(torch.zeros(self.embed_dim, self.r, **kw))
            _kaiming_uniform(self.lora_in_A)
            _kaiming_uniform(self.lora_out_A)
        self.train(base.training)

    def extra_repr(self):
        return (f'embed_dim={self.embed_dim}, num_heads={self.num_heads}, '
                f'r={self.r}, alpha={self.alpha}, batch_first={self.batch_first}')

    # ------------------------------------------------------------------ #
    def _project(self, x, start, end):
        """Linear projection of ``x`` for the q/k/v slice ``[start:end]``."""
        w = self.in_proj_weight[start:end]
        b = self.in_proj_bias[start:end] if self.in_proj_bias is not None else None
        out = F.linear(x, w, b)
        if self.r > 0:
            delta = F.linear(
                F.linear(self.lora_dropout(x), self.lora_in_A),
                self.lora_in_B[start:end])
            out = out + delta * self.scaling
        return out

    def forward(self, query, key, value, key_padding_mask=None,
                need_weights=True, attn_mask=None, average_attn_weights=True,
                is_causal=False):
        if attn_mask is not None:
            raise NotImplementedError('attn_mask is not supported by LoRAMultiheadAttention')

        if self.batch_first:
            q, k, v = query, key, value
        else:
            q, k, v = query.transpose(0, 1), key.transpose(0, 1), value.transpose(0, 1)

        E = self.embed_dim
        H = self.num_heads
        dh = self.head_dim
        B, L, _ = q.shape
        S = k.shape[1]

        qp = self._project(q, 0, E).view(B, L, H, dh).transpose(1, 2)
        kp = self._project(k, E, 2 * E).view(B, S, H, dh).transpose(1, 2)
        vp = self._project(v, 2 * E, 3 * E).view(B, S, H, dh).transpose(1, 2)

        mask = None
        if key_padding_mask is not None:
            # nn.MultiheadAttention: True == ignore; SDPA: True == keep
            keep = ~key_padding_mask.to(torch.bool)[:, None, None, :]
            mask = keep.expand(B, H, L, S) if keep.shape[-2] == 1 else keep

        attn = F.scaled_dot_product_attention(
            qp, kp, vp, attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False)
        out = attn.transpose(1, 2).reshape(B, L, E)

        w = self.out_proj.weight
        if self.r > 0:
            w = w + self.scaling * (self.lora_out_B @ self.lora_out_A)
        out = F.linear(out, w, self.out_proj.bias)

        if not self.batch_first:
            out = out.transpose(0, 1)
        return out, None


# --------------------------------------------------------------------------- #
# injection
# --------------------------------------------------------------------------- #
def _inject_into(module, r, alpha, dropout, _depth=0):
    """Recursively replace ``nn.Linear`` / ``nn.MultiheadAttention`` children."""
    if _depth > 12:                                    # safety against recursion bugs
        return 0
    n = 0
    for name, child in list(module.named_children()):
        if isinstance(child, (LoRALinear, LoRAMultiheadAttention)):
            continue
        if isinstance(child, nn.MultiheadAttention):
            setattr(module, name, LoRAMultiheadAttention(child, r, alpha, dropout))
            n += 1
        elif isinstance(child, nn.Linear):
            setattr(module, name, LoRALinear(child, r, alpha, dropout))
            n += 1
        else:
            n += _inject_into(child, r, alpha, dropout, _depth + 1)
    return n


DEFAULT_LORA_TARGETS = ('cross_encoder', 'gumbel', 'decoder')


def apply_lora(model, r=8, alpha=16, dropout=0.0, targets=DEFAULT_LORA_TARGETS,
               verbose=True):
    """Freeze ``model`` and inject LoRA into ``targets`` (submodule names).

    Returns:
        int: number of adapted layers.
    """
    for p in model.parameters():
        p.requires_grad = False

    n_layers = 0
    for target in targets:
        module = getattr(model, target, None)
        if module is None:
            raise AttributeError(f'LoRA target {target!r} not found on the model')
        n_layers += _inject_into(module, r, alpha, dropout)

    if verbose:
        n_param = count_trainable_parameters(model, only_lora=True)
        print(f'[lora] injected into {n_layers} layers '
              f'({", ".join(targets)}), rank={r}, alpha={alpha}, '
              f'dropout={dropout}, trainable={n_param / 1e6:.3f}M')
    return n_layers


def mark_only_lora_trainable(model, extra_names=('roi_type_embedding',)):
    """Set ``requires_grad`` on LoRA params (and whitelisted new params) only."""
    extra = tuple(extra_names) if extra_names else ()
    n_train, n_total = 0, 0
    for name, p in model.named_parameters():
        keep = (LORA_PREFIX in name) or any(k in name for k in extra)
        p.requires_grad_(keep)
        if keep:
            n_train += p.numel()
        n_total += p.numel()
    return n_train, n_total


def lora_named_parameters(model):
    return [(n, p) for n, p in model.named_parameters() if LORA_PREFIX in n]


def lora_state_dict(model, extra_names=('roi_type_embedding',)):
    """Only the freshly trained parameters (LoRA + new fusion params)."""
    extra = tuple(extra_names) if extra_names else ()
    return {
        k: v.detach().clone() for k, v in model.state_dict().items()
        if (LORA_PREFIX in k) or any(name in k for name in extra)
    }


def count_trainable_parameters(model, only_lora=False):
    if only_lora:
        return sum(p.numel() for n, p in model.named_parameters() if LORA_PREFIX in n)
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
