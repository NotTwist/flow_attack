import ptlflow
from functools import partial
from tqdm import tqdm
import torch
import torch.nn.functional as F
import numpy as np
import os
import os.path
import math


def zero_flow(flow):
    """Create a zero tensor with the same size as flow

    Args:
            flow (tensor): input

    Returns:
            tensor: containing zeros with same dimension as the input
    """
    return torch.zeros_like(flow)


def neg_flow(flow):
    """Mirror the input flow by 180 degree

    Args:
            flow (tensor): input flow field

    Returns:
            tensor: reversed flow field
    """
    return - flow.detach()


def untargeted_flow(flow):
    return flow.detach()


def flow_to_camera(flow):
    """Create flow vectors directed toward the camera center.

    Args:
        flow (torch.Tensor): shape (B, 2, H, W)
            optical flow tensor [u, v]

    Returns:
        torch.Tensor: flow directed toward image center
    """
    B, C, H, W = flow.shape
    device = flow.device

    y, x = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing="ij"
    )

    cx = W / 2
    cy = H / 2

    # вектор к центру
    u = cx - x
    v = cy - y

    flow_to_center = torch.stack([u, v], dim=0).unsqueeze(0).repeat(B, 1, 1, 1)
    # print(flow_to_center)
    return flow_to_center


def scene_flow(flow, x, y):
    """
    Fill the flow map with the vector value from coordinate (x, y).

    Args:
        flow (torch.Tensor): [B, 2, H, W]
        x (int or tensor): x coordinate
        y (int or tensor): y coordinate
    """
    H, W = flow.shape[-2:]
    device = flow.device

    # Convert to tensors if needed
    if not torch.is_tensor(x):
        x = torch.tensor(x, device=device)
    if not torch.is_tensor(y):
        y = torch.tensor(y, device=device)

    # Clamp safely
    y = torch.clamp(y, 0, H - 1)
    x = torch.clamp(x, 0, W - 1)

    # Get the flow vector at (x, y)
    value = flow[:, :, y, x].view(-1, 2, 1, 1)
    return value.expand_as(flow)


def down_flow(flow, magnitude=1.0):
    """Create flow vectors directed straight downward.

    Args:
        flow (torch.Tensor): shape (B, 2, H, W)
        magnitude (float): magnitude of the downward flow vector

    Returns:
        torch.Tensor: flow where u = 0, v = magnitude
    """
    B, C, H, W = flow.shape
    device = flow.device

    u = torch.zeros((B, 1, H, W), device=device)
    v = torch.full((B, 1, H, W), magnitude, device=device)

    return torch.cat([u, v], dim=1)


def direction_flow(flow, magnitude=1.0, angle_deg=90.0):
    """Create constant flow vectors at a requested image-plane angle.

    Angle convention follows image coordinates: 0 degrees points right
    (+u), 90 degrees points down (+v), 180 degrees points left, and
    270 degrees points up.
    """
    B, C, H, W = flow.shape
    device = flow.device
    dtype = flow.dtype

    angle_rad = math.radians(float(angle_deg))
    u_value = float(magnitude) * math.cos(angle_rad)
    v_value = float(magnitude) * math.sin(angle_rad)

    u = torch.full((B, 1, H, W), u_value, device=device, dtype=dtype)
    v = torch.full((B, 1, H, W), v_value, device=device, dtype=dtype)

    return torch.cat([u, v], dim=1)


def relative_down_flow(flow, magnitude=1.0):
    """Shift the clean flow target downward by a fixed amount.

    Unlike down_flow(), this does not ask the model for an absolute flow of
    (0, +magnitude). It asks for the clean flow plus a downward displacement,
    so a near-zero prediction under the patch is not accidentally close unless
    the clean flow itself is already near (0, -magnitude).
    """
    target = flow.detach().clone()
    target[:, 1:2, :, :] = target[:, 1:2, :, :] + float(magnitude)
    return target


def zero_depth(depth):
    return torch.full_like(depth, 100)


def zero_raw_depth(depth):
    return torch.zeros_like(depth)


def untargeted_depth(depth):
    return depth.detach()


def infinite_depth(depth):
    """Simulate infinite depth (far away objects)."""
    # Use the maximum representable depth value or a large constant
    d_max = depth.detach().max()
    return torch.ones_like(depth) * d_max


def near_raw_depth(depth, margin: float = 0.1):
    """Target larger raw MDE values, interpreted as visually closer objects.

    Depth-Anything-style relative outputs are not metric depth in this wrapper.
    Empirically they are often inverse-depth-like: larger raw values correspond
    to closer structures. This target is per-sample and slightly above the
    current max, so minimising L1 pushes all masked pixels upward.
    """
    B = depth.shape[0]
    flat = depth.detach().view(B, -1)
    d_min = flat.min(dim=1).values.view(B, 1, 1, 1)
    d_max = flat.max(dim=1).values.view(B, 1, 1, 1)
    d_range = (d_max - d_min).clamp_min(1e-6)
    target = d_max + float(margin) * d_range
    return target.expand_as(depth)


def far_raw_depth(depth, margin: float = 0.1):
    """Target smaller raw MDE values, the opposite of near_raw_depth()."""
    B = depth.shape[0]
    flat = depth.detach().view(B, -1)
    d_min = flat.min(dim=1).values.view(B, 1, 1, 1)
    d_max = flat.max(dim=1).values.view(B, 1, 1, 1)
    d_range = (d_max - d_min).clamp_min(1e-6)
    target = d_min - float(margin) * d_range
    return target.expand_as(depth)


def near_global_raw_depth(depth, dataset_min: float, dataset_max: float, margin: float = 0.1):
    """Global-depth variant of near_raw_depth().

    The target value is fixed from dataset-level raw MDE min/max estimates
    instead of being recomputed for every input image.
    """
    d_range = max(float(dataset_max) - float(dataset_min), 1e-6)
    value = float(dataset_max) + float(margin) * d_range
    return torch.full_like(depth, value)


def far_global_raw_depth(depth, dataset_min: float, dataset_max: float, margin: float = 0.1):
    """Global-depth variant of far_raw_depth()."""
    d_range = max(float(dataset_max) - float(dataset_min), 1e-6)
    value = float(dataset_min) - float(margin) * d_range
    return torch.full_like(depth, value)


def scene_depth(depth, x, y):
    """
    Create a depth map filled with the value from coordinate (x, y).

    Args:
        depth (torch.Tensor): input depth map, shape (B, 1, H, W)
        x (int): x pixel coordinate
        y (int): y pixel coordinate

    Returns:
        torch.Tensor: depth map where all pixels equal depth[:, :, y, x]
    """
    H, W = depth.shape[-2:]
    device = depth.device

    # Convert to tensors if needed
    if not torch.is_tensor(x):
        x = torch.tensor(x, device=device)
    if not torch.is_tensor(y):
        y = torch.tensor(y, device=device)

    # Clamp safely
    y = torch.clamp(y, 0, H - 1)
    x = torch.clamp(x, 0, W - 1)
    value = depth[:, :, y, x].view(-1, 1, 1, 1)
    return value.expand_as(depth)


def percentile_depth(depth, value: float):
    """
    Возвращает depth-карту, полностью заполненную одной глубиной (value).
    """
    return torch.full_like(depth, fill_value=value)


def get_mde_target(
    target_name: str = 'zero',
    *,
    data_loader=None,
    flow_model=None,
    mde_model=None,
    device=None,
    q: float = 0.9,
    near_margin: float = 0.1,
):
    """
    Возвращает функцию-таргет для глубины.

    Для target_name='p90' перцентиль считается лениво при первом вызове таргета,
    затем кэшируется внутри замыкания.
    """
    if target_name == 'zero':
        def target(depth):
            return zero_depth(depth)

    elif target_name == 'zero_raw':
        def target(depth):
            return zero_raw_depth(depth)

    elif target_name == 'untargeted':
        def target(depth):
            return untargeted_depth(depth)

    elif target_name == 'infinite':
        def target(depth):
            return infinite_depth(depth)

    elif target_name == 'near':
        def target(depth):
            return near_raw_depth(depth, margin=near_margin)

    elif target_name == 'far':
        def target(depth):
            return far_raw_depth(depth, margin=near_margin)

    elif target_name in ('near_global', 'far_global'):
        if data_loader is None or flow_model is None or mde_model is None or device is None:
            raise ValueError(
                f"For target_name='{target_name}' you must provide data_loader, flow_model, mde_model and device"
            )

        dataset_min = None
        dataset_max = None

        def target(depth):
            nonlocal dataset_min, dataset_max
            if dataset_min is None or dataset_max is None:
                print(f"Computing dataset depth min/max for {target_name} target...")
                dataset_min, dataset_max = estimate_depth_minmax(
                    data_loader=data_loader,
                    flow_model=flow_model,
                    mde_model=mde_model,
                    device=device,
                )
                print(f"Dataset raw depth min={dataset_min:.4f}, max={dataset_max:.4f}")
            if target_name == 'near_global':
                return near_global_raw_depth(
                    depth, dataset_min=dataset_min, dataset_max=dataset_max, margin=near_margin
                )
            return far_global_raw_depth(
                depth, dataset_min=dataset_min, dataset_max=dataset_max, margin=near_margin
            )

    elif target_name == 'scene':
        # здесь исходная сигнатура scene_depth(depth, x, y) —
        # если ты её используешь, оставь как было
        def target(depth):
            # пример: берём точку по центру (можешь адаптировать)
            B, _, H, W = depth.shape
            x = W // 2
            y = H // 2
            return scene_depth(depth, x, y)

    elif target_name == 'p90':
        # ленивый вариант: считаем перцентиль при первом вызове

        if data_loader is None or flow_model is None or mde_model is None or device is None:
            raise ValueError(
                "For target_name='p90' you must provide data_loader, flow_model, mde_model and device"
            )

        depth_p90 = None  # будет хранить найденный перцентиль

        def target(depth):
            nonlocal depth_p90
            if depth_p90 is None:
                # ленивый подбор глубины по всему датасету
                print("Computing depth 90th percentile inside target...")
                depth_p90 = estimate_depth_percentile(
                    data_loader=data_loader,
                    flow_model=flow_model,
                    mde_model=mde_model,
                    device=device,
                    q=q
                )
                print(f"Depth p{int(q * 100)} = {depth_p90:.4f}")
            return percentile_depth(depth, depth_p90)

    else:
        raise ValueError(
            f'The specified mde target type "{target_name}" is not defined.'
        )

    return target


def untargeted_ss(pred):
    """Return the same segmentation prediction (no explicit target mask).

    Args:
        pred (torch.Tensor): segmentation logits or probabilities of shape (B, C, H, W)
    Returns:
        torch.Tensor: detached copy of the same tensor
    """
    return pred.detach()


def targeted_ss(reference: torch.Tensor, target_class: int = 13, device: torch.device | None = None):
    if reference.ndim == 4:
        B, C, H, W = reference.shape
    elif reference.ndim == 3:
        B, H, W = reference.shape
    else:
        raise ValueError(
            "reference tensor must have shape (B,C,H,W) or (B,H,W)")
    target = torch.full((B, H, W), fill_value=int(
        target_class), dtype=torch.long, device=device)
    return target


def get_ss_target(target_name='untargeted'):
    if target_name == 'untargeted':
        target = untargeted_ss
    elif target_name == 'targeted':
        target = targeted_ss
    return target


def get_target(target_name, custom_target_path="", device=None, magnitude=1.0, angle_deg=90.0):
    """Getter method which yields a specified target flow used during PCFA 

    Args:
            target_name (str):
                    description the attack target. Options: [zero | negative | custom]
            flow_pred_init (tensor):
                    unattacked flow field
            custom_target_path (str, optional):
                    if custom target is desired provide the path to a .npy perturbation file. Defaults to "".
            device (_type_, optional): _description_. Defaults to None.
            magnitude (float): magnitude for directional targets (e.g. 'down').
            angle_deg (float): image-plane angle for target_name='direction'.

    Raises:
            ValueError: Undefined choice for target.

    Returns:
            tensor: target flow field used during PCFA
    """
    if target_name == 'zero':
        target = zero_flow
    elif target_name == 'neg_flow':
        target = neg_flow
    elif target_name == 'untargeted':
        target = untargeted_flow
    elif target_name == 'camera':
        target = flow_to_camera
    elif target_name == 'scene':
        target = scene_flow
    elif target_name == 'down':
        target = partial(down_flow, magnitude=magnitude)
    elif target_name == 'relative_down':
        target = partial(relative_down_flow, magnitude=magnitude)
    elif target_name == 'direction':
        target = partial(direction_flow, magnitude=magnitude, angle_deg=angle_deg)
    else:
        raise ValueError('The specified target type "' + target_name +
                         '" is not defined and cannot be used. Select one of "zero", "neg_flow" or "custom". Aborting.')
    return target


def estimate_depth_percentile(
    data_loader,
    flow_model,
    mde_model,
    device,
    q: float = 0.1,
    max_pixels_per_image: int = 50000
) -> float:
    """
    Оценивает q-перцентиль глубины по всему датасету (по предсказаниям mde_model).

    Args:
        data_loader: DataLoader, выдаёт (images, flow, valid, ...)
        flow_model: модель оптического потока (нужна для IOAdapter'а)
        mde_model: модель глубины
        device: torch.device
        q: перцентиль (0..1)
        max_pixels_per_image: сколько пикселей максимум брать из одного изображения

    Returns:
        float: значение глубины на q-перцентиле
    """
    depth_samples = []

    # Берём размер входа из первого батча
    first_batch = next(iter(data_loader))
    images, flow, valid, _, _ = first_batch
    input_size = images.shape[-2:]  # (H, W)

    io_adapter = ptlflow.utils.io_adapter.IOAdapter(
        flow_model, input_size=input_size, cuda=torch.cuda.is_available()
    )

    with torch.no_grad():
        for images, flow, valid, _, _ in tqdm(
            data_loader, desc=f"Estimating depth p{int(q * 100)}"
        ):
            wrapped_inputs = {'images': images, 'flows': flow, 'valids': valid}
            inputs = io_adapter.prepare_inputs(inputs=wrapped_inputs)

            depth = mde_model(inputs)           # (B,1,H,W)
            depth = depth.detach().cpu().view(-1)

            # Подсэмплируем, чтобы не копить слишком много
            if depth.numel() > max_pixels_per_image:
                idx = torch.randperm(depth.numel())[:max_pixels_per_image]
                depth = depth[idx]

            depth_samples.append(depth)

    all_depths = torch.cat(depth_samples, dim=0)
    depth_p = torch.quantile(all_depths, q).item()
    return depth_p


def estimate_depth_minmax(
    data_loader,
    flow_model,
    mde_model,
    device,
) -> tuple[float, float]:
    """Estimate global raw MDE min/max over a dataloader."""
    first_batch = next(iter(data_loader))
    images, flow, valid, _, _ = first_batch
    input_size = images.shape[-2:]

    io_adapter = ptlflow.utils.io_adapter.IOAdapter(
        flow_model, input_size=input_size, cuda=torch.cuda.is_available()
    )

    global_min = float("inf")
    global_max = float("-inf")

    with torch.no_grad():
        for images, flow, valid, _, _ in tqdm(data_loader, desc="Estimating depth min/max"):
            wrapped_inputs = {'images': images, 'flows': flow, 'valids': valid}
            inputs = io_adapter.prepare_inputs(inputs=wrapped_inputs)
            depth = mde_model(inputs).detach()
            global_min = min(global_min, float(depth.min().item()))
            global_max = max(global_max, float(depth.max().item()))

    if not math.isfinite(global_min) or not math.isfinite(global_max):
        raise RuntimeError("Failed to estimate finite dataset depth min/max")
    return global_min, global_max
