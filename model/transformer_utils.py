from collections import defaultdict
import torch.nn.functional as F
import torch.nn as nn


def count_params(m: nn.Module):
    total_trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    total_all       = sum(p.numel() for p in m.parameters())
    return total_trainable, total_all

def bytes_human(nbytes: int):
    for unit in ['B','KB','MB','GB','TB']:
        if nbytes < 1024.0:
            return f"{nbytes:,.1f} {unit}"
        nbytes /= 1024.0
    return f"{nbytes:.1f} PB"

def params_of(module: nn.Module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)

def params_direct_of(module: nn.Module):
    # Solo los parámetros directamente en este módulo (no hijos) para evitar doble conteo
    return sum(p.numel() for p in module.parameters(recurse=False) if p.requires_grad)

def breakdown_by_child(model: nn.Module, topk=10):
    sizes = []
    for name, child in model.named_children():
        sizes.append((name, params_of(child)))
    sizes.sort(key=lambda x: x[1], reverse=True)
    return sizes[:topk]

def breakdown_by_type(model: nn.Module):
    agg = defaultdict(int)
    for module in model.modules():
        t = type(module).__name__
        agg[t] += params_direct_of(module)
    return sorted(agg.items(), key=lambda x: x[1], reverse=True)

def breakdown_by_prefix(model: nn.Module, prefixes=('encoder','decoder','src_embed','tgt_embed','embed','generator','output_projection')):
    agg = defaultdict(int)
    for name, module in model.named_modules():
        for pref in prefixes:
            if name.startswith(pref):
                agg[pref] += params_of(module)
                break
    return sorted(agg.items(), key=lambda x: x[1], reverse=True)