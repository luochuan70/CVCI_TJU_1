import math
import re
import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Low-rank adapter for an existing Linear layer.

    The wrapped base Linear is frozen. Only lora_A/lora_B are trainable.
    """

    def __init__(self, base, r=8, alpha=16, dropout=0.0):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError(f"LoRALinear expects nn.Linear, got {type(base)}")
        self.base = base
        self.r = int(r)
        self.alpha = float(alpha)
        self.scaling = self.alpha / max(self.r, 1)
        self.dropout = nn.Dropout(float(dropout)) if dropout and float(dropout) > 0 else nn.Identity()
        for p in self.base.parameters():
            p.requires_grad = False
        if self.r <= 0:
            self.lora_A = None
            self.lora_B = None
        else:
            self.lora_A = nn.Parameter(torch.empty(self.r, base.in_features))
            self.lora_B = nn.Parameter(torch.empty(base.out_features, self.r))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def forward(self, x):
        y = self.base(x)
        if self.r <= 0:
            return y
        # Keep this explicit so it works with fp16 autocast and arbitrary leading dims.
        update = torch.matmul(self.dropout(x), self.lora_A.t())
        update = torch.matmul(update, self.lora_B.t()) * self.scaling
        return y + update

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # Backward compatibility: old checkpoints store Linear params as
        # ``<prefix>weight``/``<prefix>bias``. After wrapping, the frozen base
        # Linear lives under ``<prefix>base.weight``/``<prefix>base.bias``.
        old_w = prefix + 'weight'
        old_b = prefix + 'bias'
        new_w = prefix + 'base.weight'
        new_b = prefix + 'base.bias'
        if old_w in state_dict and new_w not in state_dict:
            state_dict[new_w] = state_dict.pop(old_w)
        if old_b in state_dict and new_b not in state_dict:
            state_dict[new_b] = state_dict.pop(old_b)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)


def _matches(name, patterns):
    return any(p in name or re.search(p, name) for p in patterns)


def _rank_for_name(name, rank_rules, default_rank):
    for pattern, rank in rank_rules:
        if pattern in name or re.search(pattern, name):
            return int(rank)
    return int(default_rank)


def _alpha_for_rank(rank, alpha):
    return float(alpha if alpha is not None else max(2 * int(rank), 1))


def apply_lora(model, cfg):
    """Replace selected Linear modules with LoRALinear.

    cfg keys:
      include: list of substrings/regexes to target
      exclude: list of substrings/regexes to skip
      rank: default rank
      alpha: default alpha, or None for 2*rank
      dropout: LoRA dropout
      rank_rules: list[(pattern, rank)] applied before default rank
      train_bias: optional bool, keep selected base biases trainable
    """
    cfg = dict(cfg or {})
    include = list(cfg.get('include', []))
    exclude = list(cfg.get('exclude', []))
    rank = int(cfg.get('rank', 8))
    alpha = cfg.get('alpha', None)
    dropout = float(cfg.get('dropout', 0.0))
    rank_rules = list(cfg.get('rank_rules', []))
    train_bias = bool(cfg.get('train_bias', False))
    if not include:
        raise ValueError('lora_config.include must not be empty')

    for p in model.parameters():
        p.requires_grad = False

    replaced = []

    def visit(module, prefix=''):
        for child_name, child in list(module.named_children()):
            full = f'{prefix}.{child_name}' if prefix else child_name
            if isinstance(child, LoRALinear):
                continue
            if isinstance(child, nn.Linear) and _matches(full, include) and not _matches(full, exclude):
                r = _rank_for_name(full, rank_rules, rank)
                wrapped = LoRALinear(child, r=r, alpha=_alpha_for_rank(r, alpha), dropout=dropout)
                if train_bias and wrapped.base.bias is not None:
                    wrapped.base.bias.requires_grad = True
                setattr(module, child_name, wrapped)
                replaced.append((full, r))
            else:
                visit(child, full)

    visit(model)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {
        'replaced': replaced,
        'num_replaced': len(replaced),
        'trainable_params': trainable,
        'total_params': total,
    }
