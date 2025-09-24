# Flow attack framework

## TODOS
1. Add better argument import to attacks
<s>2. Rerun Cospgd, Sobel and high freq on full dataset since we changed the code</s>
3. Add attack-time logging to attack_metrics.py
4. Find optimal num_samples for new attacks
5. Write grpah plotter in a easy to use way instead of whatever i am doing now 😁
6. Run new FGSMs on full dataset (??? if there is time)
<s>7. Check different neighbour sample amounts for new fgsm attacks</s>


Final AEE Metrics:
AEE (init vs attack): 25.4606
AEE (attacked vs target): 10.0171


def get_loss(f_type: Union[str, List[str]], mask=None, normalize: bool = False, untargeted: bool = False) -> Callable:
    """
    Return a callable loss(pred, target, **kwargs) that computes the (weighted) combination
    of losses specified by `f_type`.

    f_type may be:
      - a single name: "aee"
      - a spec string: "aee:1.0,mse:0.5"
      - a list of specs: ["aee:1.0", "mse:0.5"]

    normalize: if True, weights are normalized to sum to 1 before combining.
    untargeted: if True, the final returned loss will be negated (i.e. multiplied by -1).

    The returned callable signature:
        loss_val = loss_fn(pred, target, mask_override=None, **kwargs)
    If mask_override is provided it will override the mask passed to get_loss.
    """
    # Map CLI names to actual loss-callables in your code.
    loss_fn_map = {
        "aee": f_epe,
        "cosim": f_cosim,
        "mse": f_mse,
        "epe": epe,
        "focal": focal_epe,
        "huber": huber_epe,
        "charbonnier": charbonnier_epe,
    }

    # Parse f_type to list of (name, weight)
    specs = parse_loss_specs(f_type)  # assume parse_loss_specs returns List[Tuple[str,float]]
    if not specs:
        raise ValueError("No loss specs provided to get_loss()")

    # Validate and collect functions as tuples (fn, weight, name)
    collected = []
    for name, w in specs:
        if name not in loss_fn_map:
            raise KeyError(
                f"Requested loss '{name}' is not available. "
                f"Available: {', '.join(sorted(loss_fn_map.keys()))}"
            )
        if w < 0:
            raise ValueError(f"Weight for loss '{name}' must be non-negative (got {w})")
        collected.append((loss_fn_map[name], float(w), name))

    # Normalize weights if requested
    total_w = sum(w for _, w, _ in collected)
    if normalize and total_w > 0:
        collected = [(fn, w / total_w, name) for fn, w, name in collected]

    def combined_loss(pred, target, mask_override=None, **kwargs):
        """
        Compute weighted sum of the configured losses.
        mask_override takes precedence over the mask provided when get_loss() was called.
        Extra kwargs are forwarded to individual loss functions if they accept them.
        """
        used_mask = mask_override if (mask_override is not None) else mask

        # Start from zero (works with numpy/scalars and torch tensors)
        total = 0
        for fn, w, name in collected:
            val = fn(pred, target, mask=used_mask, **kwargs)
            total = total + (val * w)

        # If untargeted mode, invert the loss sign
        if untargeted:
            return -total
        return total

    # attach metadata for debugging/inspection
    combined_loss.specs = [(name, w) for _, w, name in collected]  # list of (name, weight)
    combined_loss.component_fns = {name: fn for fn, _, name in collected}
    combined_loss.untargeted = bool(untargeted)

    return combined_loss
