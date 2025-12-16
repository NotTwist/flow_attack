import ptlflow
from tqdm import tqdm
import torch
import torch.nn.functional as F
import numpy as np
import os
import os.path


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


def down_flow(flow):
    """Create flow vectors directed straight downward.

    Args:
        flow (torch.Tensor): shape (B, 2, H, W)

    Returns:
        torch.Tensor: flow where u = 0, v = 1 (downward unit vectors)
    """
    B, C, H, W = flow.shape
    device = flow.device

    # Горизонтальная компонента = 0
    u = torch.zeros((B, 1, H, W), device=device)
    # Вертикальная компонента > 0 — вниз
    v = torch.ones((B, 1, H, W), device=device)

    return torch.cat([u, v], dim=1)


def zero_depth(depth):
    return torch.full_like(depth, 100)


def untargeted_depth(depth):
    return depth.detach()


def infinite_depth(depth):
    """Simulate infinite depth (far away objects)."""
    # Use the maximum representable depth value or a large constant
    d_max = depth.detach().max()
    return torch.ones_like(depth) * d_max


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
    q: float = 0.9
):
    """
    Возвращает функцию-таргет для глубины.

    Для target_name='p90' перцентиль считается лениво при первом вызове таргета,
    затем кэшируется внутри замыкания.
    """
    if target_name == 'zero':
        def target(depth):
            return zero_depth(depth)

    elif target_name == 'untargeted':
        def target(depth):
            return untargeted_depth(depth)

    elif target_name == 'infinite':
        def target(depth):
            return infinite_depth(depth)

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


def get_target(target_name, custom_target_path="", device=None):
    """Getter method which yields a specified target flow used during PCFA 

    Args:
            target_name (str):
                    description the attack target. Options: [zero | negative | custom]
            flow_pred_init (tensor):
                    unattacked flow field
            custom_target_path (str, optional):
                    if custom target is desired provide the path to a .npy perturbation file. Defaults to "".
            device (_type_, optional): _description_. Defaults to None.

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
        target = down_flow
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
