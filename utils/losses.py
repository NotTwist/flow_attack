import torch
import torch.nn.functional as F
import re
from typing import Union, List, Tuple, Callable, Dict
from functools import wraps

EPS = 1e-8

FLOAT_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

def safe_sqrt(x, eps: float = EPS):
    return torch.sqrt(torch.clamp(x, min=eps))

def epe(flow1, flow2):

    diff_squared = (flow1 - flow2)**2
    if len(diff_squared.size()) == 3:
        # here, dim=0 is the 2-dimension (u and v direction of flow [2,M,N]) , which needs to be added BEFORE taking the square root. To get the length of a flow vector, we need to do sqrt(u_ij^2 + v_ij^2)
        epe = safe_sqrt(torch.sum(diff_squared, dim=0))
    elif len(diff_squared.size()) == 4:
        # here, dim=0 is the 2-dimension (u and v direction of flow [b,2,M,N]) , which needs to be added BEFORE taking the square root. To get the length of a flow vector, we need to do sqrt(u_ij^2 + v_ij^2)
        epe = safe_sqrt(torch.sum(diff_squared, dim=1))
    else:
        raise ValueError("The flow tensors for which the EPE should be computed do not have a valid number of dimensions (either [b,2,M,N] or [2,M,N]). Here: " + str(
            flow1.size()) + " and " + str(flow1.size()))
    return epe


def avg_epe(flow1, flow2, mask=None):
    """"
    Compute the average endpoint errors (AEE) between two flow fields.
    The epe measures the euclidean- / 2-norm of the difference of two optical flow vectors
    (u0, v0) and (u1, v1) and is defined as sqrt((u0 - u1)^2 + (v0 - v1)^2).

    Args:
        flow1 (tensor):
            represents a flow field with dimension (2,M,N) or (b,2,M,N) where M ~ u-component and N ~v-component
        flow2 (tensor):
            represents a flow field with dimension (2,M,N) or (b,2,M,N) where M ~ u-component and N ~v-component

    Raises:
        ValueError: dimensons not valid

    Returns:
        float: scalar average endpoint error
    """
    diff_squared = (flow1 - flow2)**2
    if len(diff_squared.size()) == 3:
        epe_map = safe_sqrt(torch.sum(diff_squared, dim=0))  # [H,W]
        if mask is not None:
            m = (mask == 1).float().squeeze()
            epe = (epe_map * m).sum() / m.sum().clamp_min(EPS)
        else:
            epe = epe_map.mean()
    elif len(diff_squared.size()) == 4:
        epe_map = safe_sqrt(torch.sum(diff_squared, dim=1))  # [B,H,W]
        if mask is not None:
            m = (mask == 1).float()
            if m.dim() == 4:
                m = m.squeeze(1)
            epe = (epe_map * m).sum() / m.sum().clamp_min(EPS)
        else:
            epe = epe_map.mean()
    else:
        raise ValueError("The flow tensors for which the EPE should be computed do not have a valid number of dimensions (either [b,2,M,N] or [2,M,N]). Here: " + str(
            flow1.size()) + " and " + str(flow1.size()))
    return epe

import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms.functional as TF

def save_heatmap(tensor, filename='focal_weight_heatmap.png'):
    # Detach and move to CPU
    heatmap = tensor.detach().cpu()
    
    # If batch exists, take first element
    if heatmap.ndim == 3:
        heatmap = heatmap[0]
    
    # Normalize to [0, 1]
    heatmap -= heatmap.min()
    heatmap /= heatmap.max() + 1e-8

    # Convert to numpy
    heatmap_np = heatmap.numpy()

    # Save heatmap
    plt.imsave(filename, heatmap_np, cmap='magma')  # Or 'viridis', 'hot', etc.


def focal_epe(flow1, flow2, gamma=1, eps=1e-1, mask=None):
    """
    Focal Endpoint Error (Focal-EPE) loss.
    Down-weights pixels with large flow error to focus on hard (low-error) pixels.

    Args:
        flow1 (tensor): Predicted flow [2, H, W] or [B, 2, H, W]
        flow2 (tensor): Ground-truth flow [2, H, W] or [B, 2, H, W]
        gamma (float): Focusing parameter, e.g., 2.0
        eps (float): Small constant to avoid division by zero
        mask (tensor, optional): Binary mask to specify valid pixels

    Returns:
        float: Scalar focal EPE loss
    """
    d = epe(flow1, flow2)  # [H,W] or [B,H,W]
    if mask is not None:
        m = mask
        if mask.dim() == 4 and d.dim() == 3:
            m = mask.squeeze(1)
        m = (m == 1).float()
    else:
        m = None

    # focal weight: (1 - d/(d+eps))^gamma  (clamped >=0)
    weight = (1.0 - d / (d + eps)).clamp(min=0.0) ** gamma
    out = weight * d
    if m is not None:
        denom = m.sum().clamp_min(EPS)
        return (out * m).sum() / denom
    else:
        return out.mean()

def charbonnier_epe(flow1, flow2, epsilon=1e-3, alpha=0.45, mask=None):
    """
    Charbonnier applied to d (||diff||), возвращает скаляр в пикселях^(alpha) —
    но поскольку alpha < 1, величины остаются в сравнимом масштабе с EPE.
    (Если хотите строгие пиксельные единицы — используйте alpha ~ 1).
    """
    d = epe(flow1, flow2)
    if mask is not None:
        m = mask
        if mask.dim() == 4 and d.dim() == 3:
            m = mask.squeeze(1)
        m = (m == 1).float()
        out = (d + epsilon).pow(alpha)
        denom = m.sum().clamp_min(EPS)
        return (out * m).sum() / denom
    else:
        return (d + epsilon).pow(alpha).mean()


def huber_epe(flow1, flow2, delta=1.0, mask=None):
    """
    Huber (on d) — работает в пикселях. Возвращает скаляр.
    delta — переход в тех же единицах пикселей.
    """
    d = epe(flow1, flow2)
    hub = torch.where(d <= delta, 0.5 * d ** 2, delta * (d - 0.5 * delta))
    if mask is not None:
        m = mask
        if mask.dim() == 4 and d.dim() == 3:
            m = mask.squeeze(1)
        m = (m == 1).float()
        denom = m.sum().clamp_min(EPS)
        return (hub * m).sum() / denom
    else:
        return hub.mean()


def mse(flow1, flow2, mask=None):
    """Computes mean squared error between two flow fields.

    Args:
        flow1 (tensor):
            flow field, which must have the same dimension as flow2
        flow2 (tensor):
            flow field, which must have the same dimension as flow1

    Returns:
        float: scalar average squared end-point-error
    """
    return (flow1 - flow2)**2

def rmse(flow1, flow2, mask=None):
    """
    RMSE: sqrt(mean(||flow1-flow2||^2)).
    Это переводит MSE (пиксели^2) в пиксели.
    Возвращает скаляр.
    """
    diff_squared = (flow1 - flow2) ** 2
    if mask is not None:
        m = mask
        if mask.dim() == 4 and diff_squared.dim() == 3:
            m = mask.squeeze(1)
        m = (m == 1).float()
        # суммируем по всем осям кроме batch
        if diff_squared.dim() == 3:
            # [2,H,W] -> sum over channel -> [H,W]
            d2 = torch.sum(diff_squared, dim=0)
            denom = m.sum().clamp_min(EPS)
            return safe_sqrt(torch.sum(d2 * m) / denom)
        else:
            # [B,2,H,W] -> sum over channel -> [B,H,W]
            d2 = torch.sum(diff_squared, dim=1)
            denom = m.sum().clamp_min(EPS)
            return safe_sqrt(torch.sum(d2 * m) / denom)
    else:
        # no mask: mean then sqrt
        if diff_squared.dim() == 3:
            d2 = torch.sum(diff_squared, dim=0)  # [H,W]
            return safe_sqrt(torch.mean(d2))
        else:
            d2 = torch.sum(diff_squared, dim=1)  # [B,H,W]
            return safe_sqrt(torch.mean(d2))

def avg_mse(flow1, flow2, mask=None):
    """Computes mean squared error between two flow fields.

    Args:
        flow1 (tensor):
            flow field, which must have the same dimension as flow2
        flow2 (tensor):
            flow field, which must have the same dimension as flow1

    Returns:
        float: scalar average squared end-point-error
    """
    return torch.mean((flow1 - flow2)**2)


def f_epe(pred, target, mask=None):
    """Wrapper function to compute the average endpoint error between prediction and target

    Args:
        pred (tensor):
            predicted flow field (must have same dimensions as target)
        target (tensor):
            specified target flow field (must have same dimensions as prediction)

    Returns:
        float: scalar average endpoint error 
    """
    return avg_epe(pred, target, mask)


def f_mse(pred, target, mask=None):
    """Wrapper function to compute the mean squared error between prediction and target

    Args:
        pred (tensor):
            predicted flow field (must have same dimensions as target)
        target (tensor):
            specified target flow field (must have same dimensions as prediction)

    Returns:
        float: scalar average squared end-point-error
    """
    return rmse(pred, target, mask)


def f_cosim(pred, target, mask=None):
    """Compute the mean cosine similarity between the two flow fields prediction and target

    Args:
        pred (tensor):
            predicted flow field (must have same dimensions as target)
        target (tensor):
            specified target flow field (must have same dimensions as prediction)

    Returns:
        float: scalar mean cosine similarity
    """
    return 1 - torch.sum(pred * target) / torch.sqrt(torch.sum(pred*pred)) * torch.sqrt(torch.sum(target*target))


def down_hinge_loss(
    pred,
    target,
    mask=None,
    min_mag_ratio: float = 0.8,
    horizontal_weight: float = 0.1,
    magnitude_weight: float = 0.5,
    vertical_weight: float = 1.0,
):
    """Directional hinge loss for absolute downward flow targets.

    This is meant for target='down'. It avoids the failure mode where a large
    patch collapses the flow prediction to zero: vertical flow below the target
    and magnitude below a target-ratio margin are explicitly penalized.
    """
    pred_u = pred[:, 0:1]
    pred_v = pred[:, 1:2]
    target_u = target[:, 0:1].detach()
    target_v = target[:, 1:2].detach()

    target_mag = safe_sqrt(target_u.pow(2) + target_v.pow(2)).detach()
    pred_mag = safe_sqrt(pred_u.pow(2) + pred_v.pow(2))

    vertical = F.relu(target_v - pred_v).pow(2)
    horizontal = (pred_u - target_u).abs()
    min_mag = float(min_mag_ratio) * target_mag
    magnitude = F.relu(min_mag - pred_mag).pow(2)

    loss_map = (
        float(vertical_weight) * vertical
        + float(horizontal_weight) * horizontal
        + float(magnitude_weight) * magnitude
    )

    if mask is not None:
        m = (mask == 1).float()
        if m.dim() == 3:
            m = m.unsqueeze(1)
        if m.shape[-2:] != loss_map.shape[-2:]:
            m = F.interpolate(m, size=loss_map.shape[-2:], mode="nearest")
        return (loss_map * m).sum() / m.sum().clamp_min(EPS)

    return loss_map.mean()


def two_norm_avg_delta(delta1, delta2):
    """Computes the mean of the L2-norm of two perturbations used during PCFA.

    Args:
        delta1 (tensor):
            perturbation applied to the first image
        delta2 (tensor):
            perturbation applied to the second image

    Returns:
        float: scalar average L2-norm of two perturbations
    """
    numels_delta1 = torch.numel(delta1)
    numels_delta2 = torch.numel(delta2)
    sqrt_numels = (numels_delta1 + numels_delta2)**(0.5)
    two_norm = torch.sqrt(torch.sum(torch.pow(torch.flatten(
        delta1), 2)) + torch.sum(torch.pow(torch.flatten(delta2), 2)))
    return two_norm / sqrt_numels


def two_norm_avg_delta_squared(delta1, delta2):
    """Computes the mean of the squared L2-norm of two perturbations used during PCFA.

    Args:
        delta1 (tensor):
            perturbation applied to the first image
        delta2 (tensor):
            perturbation applied to the second image

    Returns:
        float: scalar average squared L2-norm of two perturbations
    """
    numels_delta1 = torch.numel(delta1)
    numels_delta2 = torch.numel(delta2)
    numels = numels_delta1 + numels_delta2
    two_norm = torch.sum(torch.pow(torch.flatten(delta1), 2)) + \
        torch.sum(torch.pow(torch.flatten(delta2), 2))
    return two_norm / numels


def two_norm_avg(x):
    """Computes the L2-norm of the input normalized by the root of the number of elements.

    Args:
        x (tensor):
            input tensor with variable dimensions

    Returns:
        float: normalized L2-norm
    """
    numels_x = torch.numel(x)
    sqrt_numels = numels_x**0.5
    two_norm = torch.sqrt(torch.sum(torch.pow(torch.flatten(x), 2)))
    return two_norm / sqrt_numels


def parse_loss_specs(specs: Union[str, List[str]]) -> List[Tuple[str, float]]:
    """
    Parse CLI loss specs.
    Accepts:
      "aee" -> [("aee",1.0)]
      "aee:1.0,mse:0.5" -> [("aee",1.0),("mse",0.5)]
      ["aee:1.0","mse:0.5"] -> same
    Returns list of (name, weight).
    """
    if isinstance(specs, list):
        s = ",".join(specs)
    else:
        s = str(specs or "")

    s = s.replace(" ", "")
    if s == "":
        return []

    parts = [p for p in re.split(r"[,;]+", s) if p]
    out = []
    for p in parts:
        if ":" in p:
            name, w = p.split(":", 1)
            try:
                w = float(w)
            except ValueError:
                raise ValueError(f"Invalid weight in loss spec '{p}'")
        else:
            name = p
            w = 1.0
        if not name:
            raise ValueError(f"Empty loss name in spec '{p}'")
        out.append((name, float(w)))
    return out


# def get_loss(f_type, mask=None):
#     """Wrapper to return a specified loss metric. 

#     Args:
#         f_type (str):
#             specifies the returned metric. Options: [aee | mse | cosim]
#         pred (tensor):
#             predicted flow field (must have same dimensions as target)
#         target (tensor):
#             specified target flow field (must have same dimensions as prediction)

#     Raises:
# 		NotImplementedError: Unknown metric.

#     Returns:
#         float: scalar representing the loss measured with the specified norm
#     """

#     similarity_term = None

#     if f_type == "aee":
#         similarity_term = f_epe
#     elif f_type == "cosim":
#         similarity_term = f_cosim
#     elif f_type == "mse":
#         similarity_term = f_mse
#     elif f_type == 'epe':
#         similarity_term = epe
#     elif f_type == 'focal':
#         similarity_term = focal_epe
#     elif f_type == 'huber':
#         similarity_term = huber_epe
#     elif f_type == 'charbonnier':
#         similarity_term = charbonnier_epe
#     else:
#         raise (NotImplementedError,
#                "The requested loss type %s does not exist. Please choose one of 'aee', 'mse' or 'cosim'" % (f_type))

#     return similarity_term

def get_loss(
    f_type: Union[str, List[str]],
    mask=None,
    normalize: bool = False,
    untargeted: bool = False
) -> Callable:
    """
    Возвращает функцию потерь, которая может быть комбинацией нескольких метрик.
    При наличии mask, итоговый loss вычисляется только по пикселям внутри маски.
    """

    loss_fn_map = {
        "aee": f_epe,
        "cosim": f_cosim,
        "mse": f_mse,
        "epe": epe,
        "focal": focal_epe,
        "huber": huber_epe,
        "charbonnier": charbonnier_epe,
        "down_hinge": down_hinge_loss,
    }

    specs = parse_loss_specs(f_type)
    if not specs:
        raise ValueError("No loss specs provided to get_loss()")

    collected = []
    for name, w in specs:
        if name not in loss_fn_map:
            raise KeyError(
                f"Requested loss '{name}' is not available. "
                f"Available: {', '.join(sorted(loss_fn_map.keys()))}"
            )
        if w < 0:
            raise ValueError(
                f"Weight for loss '{name}' must be non-negative (got {w})")
        collected.append((loss_fn_map[name], float(w), name))

    total_w = sum(w for _, w, _ in collected)
    if normalize and total_w > 0:
        collected = [(fn, w / total_w, name) for fn, w, name in collected]

    def combined_loss(pred, target, mask_override=None, **kwargs):
        used_mask = mask_override if mask_override is not None else mask

        if used_mask is not None:
            # Приводим маску к нужной форме
            m = (used_mask == 1).float()
            if m.dim() == 4 and pred.dim() == 3:
                m = m.squeeze(1)
            # Применяем маску только к видимым областям
            pred_masked = pred * m
            target_masked = target * m
        else:
            pred_masked, target_masked = pred, target

        total = 0
        for fn, w, name in collected:
            # если функция поддерживает mask, передаем её
            try:
                val = fn(pred_masked, target_masked, mask=used_mask, **kwargs)
            except TypeError:
                # если функция не принимает mask, просто используем маскированные тензоры
                val = fn(pred_masked, target_masked, **kwargs)
            total = total + (val * w)

        if untargeted:
            total = -total
        return total

    combined_loss.specs = [(name, w) for _, w, name in collected]
    combined_loss.component_fns = {name: fn for fn, _, name in collected}
    combined_loss.untargeted = bool(untargeted)

    return combined_loss

# def get_loss(f_type: Union[str, List[str]], mask=None, normalize: bool = False, normalize_components_by_running_avg: bool = True, untargeted: bool = False, running_avg_eps: float = 1e-6) -> Callable:
#     """
#     Return a callable loss(pred, target, **kwargs) that computes the (weighted) combination
#     of losses specified by `f_type`.

#     f_type may be:
#       - a single name: "aee"
#       - a spec string: "aee:1.0,mse:0.5"
#       - a list of specs: ["aee:1.0", "mse:0.5"]

#     normalize: if True, weights are normalized to sum to 1 before combining.
#     untargeted: if True, the final returned loss will be negated (i.e. multiplied by -1).

#     The returned callable signature:
#         loss_val = loss_fn(pred, target, mask_override=None, **kwargs)
#     If mask_override is provided it will override the mask passed to get_loss.
#     """
#     # Map CLI names to actual loss-callables in your code.
#     loss_fn_map = {
#         "aee": f_epe,
#         "cosim": f_cosim,
#         "mse": f_mse,
#         "epe": epe,
#         "focal": focal_epe,
#         "huber": huber_epe,
#         "charbonnier": charbonnier_epe,
#     }

#     # Parse f_type to list of (name, weight)
#     specs = parse_loss_specs(f_type)  # assume parse_loss_specs returns List[Tuple[str,float]]
#     if not specs:
#         raise ValueError("No loss specs provided to get_loss()")

#     # Validate and collect functions as tuples (fn, weight, name)
#     collected = []
#     for name, w in specs:
#         if name not in loss_fn_map:
#             raise KeyError(
#                 f"Requested loss '{name}' is not available. "
#                 f"Available: {', '.join(sorted(loss_fn_map.keys()))}"
#             )
#         if w < 0:
#             raise ValueError(f"Weight for loss '{name}' must be non-negative (got {w})")
#         collected.append((loss_fn_map[name], float(w), name))

#     # Normalize weights if requested
#     total_w = sum(w for _, w, _ in collected)
#     if normalize and total_w > 0:
#         collected = [(fn, w / total_w, name) for fn, w, name in collected]


#     specs_processed = [(name, w) for _, w, name in collected]
#     component_fns: Dict[str, Callable] = {name: fn for fn, _, name in collected}
#     original_weights = {name: float(w) for name, w in specs_processed}
#     original_total_weight = sum(original_weights.values())

#     # SIMPLE running stats stored in closure
#     running_mean = {name: 0.0 for name in original_weights}
#     running_count = {name: 0 for name in original_weights}
#     eps = float(running_avg_eps)

#     def _to_scalar(val):
#         if isinstance(val, torch.Tensor):
#             # reduce to scalar (mean if multi-element)
#             return float(val.detach().cpu().mean().item())
#         return float(val)

#     def combined_loss(pred, target, mask_override=None, update_running: bool = True, **kwargs):
#         used_mask = mask_override if (mask_override is not None) else mask

#         # compute component tensors
#         component_vals = {}
#         for name, _ in specs_processed:
#             fn = component_fns[name]
#             v = fn(pred, target, mask=used_mask, **kwargs)
#             # ensure tensor for autograd; if fn returned numpy/float convert to tensor on pred device
#             if not isinstance(v, torch.Tensor):
#                 device = pred.device if isinstance(pred, torch.Tensor) else None
#                 v = torch.tensor(v, device=device)
#             component_vals[name] = v

#         # choose weights based on previous running means (no circular dependency)
#         if normalize_components_by_running_avg and not all(running_count[n] == 0 for n in running_mean):
#             inv = {n: 1.0 / (running_mean[n] + eps) for n in running_mean}
#             scaled = {n: original_weights[n] * inv[n] for n in original_weights}
#             sum_scaled = sum(scaled.values())
#             if sum_scaled != 0 and original_total_weight > 0:
#                 factor = original_total_weight / sum_scaled
#                 weights_to_use = {n: scaled[n] * factor for n in scaled}
#             elif sum_scaled != 0:
#                 # if original total was zero, normalize to sum 1
#                 weights_to_use = {n: scaled[n] / sum_scaled for n in scaled}
#             else:
#                 weights_to_use = original_weights.copy()
#         else:
#             weights_to_use = original_weights.copy()

#         # build combined tensor
#         total_tensor = None
#         for name, val in component_vals.items():
#             w = float(weights_to_use[name])
#             weighted = val * w
#             print(name, weighted)
#             total_tensor = weighted if total_tensor is None else (total_tensor + weighted)
#         combined = total_tensor if total_tensor is not None else torch.tensor(0.0, device=(pred.device if isinstance(pred, torch.Tensor) else None))

#         if untargeted:
#             combined = -combined

#         # update running means AFTER computing combined so first call uses original weights
#         if update_running:
#             for name, val in component_vals.items():
#                 scalar = _to_scalar(val)
#                 running_count[name] += 1
#                 n = running_count[name]
#                 # incremental mean update
#                 running_mean[name] = running_mean[name] + (scalar - running_mean[name]) / n

#         return combined

#     # attach a few helpers for inspection
#     combined_loss.specs = [(name, w) for name, w in specs_processed]
#     combined_loss.component_fns = component_fns
#     combined_loss._running_mean = running_mean
#     combined_loss._running_count = running_count
#     combined_loss._simple_config = dict(normalize_components_by_running_avg=bool(normalize_components_by_running_avg),
#                                         running_avg_eps=eps,
#                                         untargeted=bool(untargeted))
#     return combined_loss


def f1_loss(depth, target, mask=None):
    diff = (depth - target).abs()

    if mask is not None:
        mask = mask.float()
        if mask.shape != depth.shape:
            mask = mask.expand_as(depth)
        diff = diff * mask
        denom = mask.sum().clamp(min=1.0)
    else:
        denom = diff.numel()

    return diff.sum() / denom

def get_mde_loss(f_type=None, untargeted: bool = False) -> Callable:
    """
    untargeted: если True — возвращаем функцию с отрицательным знаком.
    """
    base_loss = f1_loss

    if not untargeted:
        return base_loss

    @wraps(base_loss)
    def neg_loss(*args, **kwargs):
        return - base_loss(*args, **kwargs)

    # optional: attach flag so caller can introspect
    neg_loss._negated = True
    return neg_loss 


# def ss_base_loss(logits, target_mask, mask=None):
#     loss = F.cross_entropy(logits, target_mask, reduction='none')  # (B,H,W)

#     if mask is not None:
#         mask = mask.float()
#         if mask.dim() == 4 and mask.size(1) == 1:
#             mask = mask.squeeze(1)
#         loss = (loss * mask).sum() / mask.sum().clamp_min(EPS)
#     else:
#         loss = loss.mean()

#     return loss


def ss_base_loss(logits, target, mask=None, gamma=2.0):
    ce = F.cross_entropy(logits, target, reduction='none')
    if gamma > 0:
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** gamma) * ce
    else:
        loss = ce
    if mask is not None:
        loss = (loss * mask).sum() / mask.sum().clamp_min(1e-6)
    else:
        loss = loss.mean()
    return loss


def get_ss_loss(untargeted: bool = False, gamma: float = 2.0) -> Callable:
    """
    Возвращает loss-функцию для semantic segmentation.
    gamma: focal loss gamma. 0 = plain cross-entropy, >0 = focal loss.
    Если untargeted=True → loss будет инвертирован (используется для untargeted атак).
    """
    def loss_fn(logits, target, mask=None):
        return ss_base_loss(logits, target, mask=mask, gamma=gamma)

    if not untargeted:
        return loss_fn

    @wraps(loss_fn)
    def neg_loss(*args, **kwargs):
        return -loss_fn(*args, **kwargs)

    neg_loss._negated = True
    return neg_loss


def get_loss_cospgd(f_type, pred, target):
    """Wrapper to return a specified loss metric. 

    Args:
        f_type (str):
            specifies the returned metric. Options: [aee | mse | cosim]
        pred (tensor):
            predicted flow field (must have same dimensions as target)
        target (tensor):
            specified target flow field (must have same dimensions as prediction)

    Raises:
		NotImplementedError: Unknown metric.

    Returns:
        float: scalar representing the loss measured with the specified norm
    """

    similarity_term = None

    if f_type == "aee":
        similarity_term = epe
    elif f_type == "cosim":
        similarity_term = f_cosim
    elif f_type == "mse":
        similarity_term = mse
    else:
        raise (NotImplementedError,
               "The requested loss type %s does not exist. Please choose one of 'aee', 'mse' or 'cosim'" % (f_type))

    return similarity_term


def relu_penalty(delta1, delta2, device, delta_bound=0.001):
    """Implementation of the penalty term.
    The penalty function linearly penalizes deviations from a constraint and is otherwise zero.
    This is implemented using the ReLU function.

    Args:
        delta1 (tensor):
            perturbation for image1
        delta2 (tensor):
            perturbation for image2
        device (torch.device):
            changes the selected device
        delta_bound (float, optional):
            L2-constraint for the perturbation. Defaults to 0.001.

    Returns:
        float: scalar penalty value
    """
    zero_tensor = torch.tensor(0.).to(device)
    delta_minus_bound = two_norm_avg_delta_squared(
        delta1, delta2) - torch.tensor(delta_bound**2).to(device)
    # This is relu( ||delta||**2-delta_bond**2).
    return torch.max(zero_tensor, delta_minus_bound)


def loss_delta_constraint(pred, target, delta1, delta2, device, delta_bound=0.001, mu=100., f_type="aee"):
    """Penalty method to optimize the perturbations.
    An exact penalty function is used to transform the inequality constrained problem into an
    unconstrained optimization problem.

    Args:
        pred (tensor):
            predicted flow field (must have same dimensions as target)
        target (tensor):
            specified target flow field (must have same dimensions as prediction)
        delta1 (tensor):
            perturbation for image1
        delta2 (tensor):
            perturbation for image2
        device (torch.device):
            changes the selected device
        delta_bound (float, optional):
            L2-constraint for the perturbation. Defaults to 0.001.
        mu (_type_, optional):
            penalty parameter which enforces the unconstrained the specified constraint. Defaults to 100..
        f_type (str, optional):
            specifies the metric used for comparing prediction and target. Options: [aee | mse | cosim]. Defaults to "aee".

    Returns:
        _type_: _description_
    """

    similarity_term = get_loss(f_type, pred, target)
    # This is relu( ||delta||**2-delta_bond**2).
    penalty_term = relu_penalty(delta1, delta2, device, delta_bound)

    return similarity_term + mu * penalty_term
