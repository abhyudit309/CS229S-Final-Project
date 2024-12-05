import torch
import torch.nn as nn


def l1_unstructured_prune(module, name, pruning_percentage):
    assert 0 <= pruning_percentage <= 1
    tensor = getattr(module, name)
    num_params_to_prune = round(pruning_percentage * tensor.nelement())

    if num_params_to_prune != 0:
        l1_norm = torch.abs(tensor.view(-1))
        threshold = torch.topk(l1_norm, num_params_to_prune, largest=False).values[-1]
        mask = torch.gt(l1_norm, threshold).float().view_as(tensor)
    else:
        mask = torch.ones_like(tensor)

    module.register_parameter(name + "_orig", torch.nn.Parameter(tensor.detach()))
    module.register_buffer(name + "_mask", mask)

    def apply_mask_hook(module, inputs):
        orig_tensor = getattr(module, name + "_orig")
        mask = getattr(module, name + "_mask")
        pruned_tensor = orig_tensor * mask
        setattr(module, name, torch.nn.Parameter(pruned_tensor))

    hook = module.register_forward_pre_hook(apply_mask_hook)
    return hook


def l2_structured_prune(model, name1, name2, pruning_percentage, n_head):
    c_attn_layer = getattr(model, name1)
    print("Original Shape: ", getattr(model, name1).weight.data.shape)

    qkv_weights = c_attn_layer.weight.data
    n_embd = qkv_weights.shape[0] // 3
    q, k, v = qkv_weights.split(n_embd, dim=0)

    ql2_norm = torch.norm(q, p=2, dim=1)
    qnum_rows_to_keep = int((1 - pruning_percentage) * ql2_norm.size(0))
    qnum_rows_to_keep -= qnum_rows_to_keep % n_head
    qrows_to_keep = torch.topk(ql2_norm, qnum_rows_to_keep, largest=True).indices

    kl2_norm = torch.norm(k, p=2, dim=1)
    knum_rows_to_keep = int((1 - pruning_percentage) * kl2_norm.size(0))
    knum_rows_to_keep -= knum_rows_to_keep % n_head
    krows_to_keep = torch.topk(kl2_norm, knum_rows_to_keep, largest=True).indices

    vl2_norm = torch.norm(v, p=2, dim=1)
    vnum_rows_to_keep = int((1 - pruning_percentage) * vl2_norm.size(0))
    vnum_rows_to_keep -= vnum_rows_to_keep % n_head
    vrows_to_keep = torch.topk(vl2_norm, vnum_rows_to_keep, largest=True).indices

    new_current_layer = nn.Linear(c_attn_layer.in_features, qnum_rows_to_keep + knum_rows_to_keep + vnum_rows_to_keep, bias=c_attn_layer.bias is not None)
    new_current_layer.weight.data = torch.cat([q[qrows_to_keep, :] , k[krows_to_keep, :] , v[vrows_to_keep, :] ], dim=0)

    setattr(model, name1, new_current_layer)
    print("Pruned Shape: ", getattr(model, name1).weight.data.shape)

    c_proj_layer = getattr(model, name2)
    new_c_proj_layer = nn.Linear(vnum_rows_to_keep, c_proj_layer.out_features, bias=c_proj_layer.bias is not None)
    new_c_proj_layer.weight.data = c_proj_layer.weight.data[:, vrows_to_keep]
    if c_proj_layer.bias is not None:
        new_c_proj_layer.bias.data = c_proj_layer.bias.data[vrows_to_keep]

    setattr(model, name2, new_c_proj_layer)


def prune(model, prune_type, pruning_percentage, n_head):
    if prune_type == 'l1':
        for block in model.transformer.h:
            l1_unstructured_prune(block.attn.c_attn, 'weight', pruning_percentage)
    elif prune_type == 'l2':
        for block in model.transformer.h:
            l2_structured_prune(block.attn, 'c_attn', 'c_proj', pruning_percentage, n_head)
    else:
        raise ValueError(f'Invalid prune type: {prune_type}')