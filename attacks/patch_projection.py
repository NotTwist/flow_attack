import torch.nn.functional as F
import os
import cv2
import numpy as np
import torch


def _get_fx_fy_cx_cy_from_K_entry(K_entry, device=None):
    """
    Универсальный хелпер:
    - Если K_entry — dict от _make_camera_config_from_kitti → берём fx, fy, cx, cy.
    - Если K_entry — тензор 3x3 → берём из матрицы.
    """
    if isinstance(K_entry, dict):
        fx = float(K_entry["fx"])
        fy = float(K_entry["fy"])
        cx = float(K_entry["cx"])
        cy = float(K_entry["cy"])
        if device is not None:
            fx = torch.tensor(fx, device=device)
            fy = torch.tensor(fy, device=device)
            cx = torch.tensor(cx, device=device)
            cy = torch.tensor(cy, device=device)
        return fx, fy, cx, cy

    # fallback: тензор 3x3
    if isinstance(K_entry, torch.Tensor):
        if K_entry.ndim != 2 or K_entry.shape != (3, 3):
            raise ValueError(f"Unexpected K_entry shape: {K_entry.shape}")
        fx = K_entry[0, 0]
        fy = K_entry[1, 1]
        cx = K_entry[0, 2]
        cy = K_entry[1, 2]
        return fx, fy, cx, cy

    raise TypeError(f"Unsupported K_entry type: {type(K_entry)}")


def fit_plane_from_depth(depth, K, mask=None):
    """
    depth: Tensor [B,1,H,W] — предсказанная глубина (м)
    K:     либо:
           - список длиной B из camera_config-словарей (как у тебя в KITTI),
           - либо тензор [B,3,3],
           - либо один dict/3x3 для всего батча.
    mask:  Tensor [B,1,H,W] или [B,H,W] bool — какие пиксели использовать
           (например, низ кадра или маска дороги).

    Возвращает:
        planes: list длиной B, каждый элемент — (normal:[3], d:скаляр)
                уравнение ax+by+cz+d=0 в системе камеры.
    """
    B, _, H, W = depth.shape
    device = depth.device

    # Нормализуем формат K к "по одному на элемент батча"
    if isinstance(K, dict) or (isinstance(K, torch.Tensor) and K.ndim == 2):
        K_list = [K for _ in range(B)]
    else:
        # предполагаем список/кортеж/тензор [B,3,3]
        K_list = K

    if isinstance(K_list, torch.Tensor):
        if K_list.ndim != 3 or K_list.shape[1:] != (3, 3):
            raise ValueError(f"Unexpected K tensor shape: {K_list.shape}")
    else:
        if len(K_list) != B:
            raise ValueError(
                f"len(K_list)={len(K_list)} != batch size B={B}"
            )

    # создаём сетку индексов пикселей (u = x, v = y)
    ys, xs = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing='ij'
    )  # [H,W]
    xs = xs.unsqueeze(0).expand(B, -1, -1)  # [B,H,W]
    ys = ys.unsqueeze(0).expand(B, -1, -1)  # [B,H,W]

    planes = []

    for b in range(B):
        d_b = depth[b, 0]   # [H,W] — карта глубины одного примера
        K_b = K_list[b]

        fx, fy, cx, cy = _get_fx_fy_cx_cy_from_K_entry(K_b, device=device)

        # ---------- 1. Проецируем пиксели в 3D ----------
        Z = d_b                         # [H,W] — глубина
        X = (xs[b] - cx) * Z / fx       # [H,W]
        Y = (ys[b] - cy) * Z / fy       # [H,W]

        # ---------- 2. Определяем, какие точки брать ----------
        if mask is not None:
            if mask.dim() == 4:
                m_b = mask[b, 0] if mask.shape[1] == 1 else mask[b].any(dim=0)
            else:
                m_b = mask[b]              # [H,W]
        else:
            # если маска не задана — берём нижнюю часть кадра
            m_b = torch.zeros_like(Z, dtype=torch.bool)
            m_b[int(H * 0.6):, :] = True   # нижние 40%

        # фильтруем по валидной глубине
        valid = torch.isfinite(Z) & (Z > 0)
        m_b = m_b & valid

        Xv = X[m_b]   # [N]
        Yv = Y[m_b]   # [N]
        Zv = Z[m_b]   # [N]

        pts = torch.stack([Xv, Yv, Zv], dim=-1)  # [N,3] — набор 3D точек
        if pts.shape[0] < 3:
            # если точек мало — возвращаем заглушку (нормаль вверх)
            planes.append((
                torch.tensor([0., 1., 0.], device=device),
                torch.tensor(0., device=device)
            ))
            continue

        # ---------- 3. LS-подгонка плоскости через SVD ----------
        # центрируем точки
        mean = pts.mean(dim=0)          # [3] — средняя точка
        pts_centered = pts - mean       # [N,3]

        # SVD матрицы точек
        U, S, Vh = torch.linalg.svd(pts_centered, full_matrices=False)
        # последняя строка Vh — направление с минимальной дисперсией → нормаль плоскости
        normal = Vh[-1]                 # [3]
        normal = normal / (normal.norm() + 1e-8)

        # уравнение плоскости: n·x + d = 0 => d = -n·p0
        d_plane = -torch.dot(normal, mean)

        planes.append((normal, d_plane))

    return planes


def plane_mask_from_depth(depth, K, planes, thresh=0.3, mask=None):
    """
    depth: [B,1,H,W]
    K:     как и в fit_plane_from_depth (список dict или [B,3,3] и т.п.)
    planes: список из (normal:[3], d)
    thresh: макс. расстояние до плоскости (в метрах)
    mask:  [B,1,H,W] или [B,H,W] bool — ограничение области (например, дорога).
           Если None — считаем по всему кадру.
    """
    B, _, H, W = depth.shape
    device = depth.device

    # Нормализуем K в список
    if isinstance(K, dict) or (isinstance(K, torch.Tensor) and K.ndim == 2):
        K_list = [K for _ in range(B)]
    else:
        K_list = K

    if isinstance(K_list, torch.Tensor):
        if K_list.ndim != 3 or K_list.shape[1:] != (3, 3):
            raise ValueError(f"Unexpected K tensor shape: {K_list.shape}")
    else:
        if len(K_list) != B:
            raise ValueError(
                f"len(K_list)={len(K_list)} != batch size B={B}"
            )

    ys, xs = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing='ij'
    )
    xs = xs.unsqueeze(0).expand(B, -1, -1)
    ys = ys.unsqueeze(0).expand(B, -1, -1)

    masks = torch.zeros_like(depth, dtype=torch.bool)

    for b in range(B):
        d_b = depth[b, 0]
        K_b = K_list[b]

        fx, fy, cx, cy = _get_fx_fy_cx_cy_from_K_entry(K_b, device=device)

        Z = d_b
        X = (xs[b] - cx) * Z / fx
        Y = (ys[b] - cy) * Z / fy

        normal, d_plane = planes[b]
        a, b_n, c = normal

        dist = (a * X + b_n * Y + c * Z + d_plane).abs()  # [H,W]

        # базовая маска: валидные глубины
        m = torch.isfinite(Z) & (Z > 0)

        # если есть маска (дорога) — стягиваемся к ней
        if mask is not None:
            if mask.dim() == 4:
                road_m = mask[b, 0] if mask.shape[1] == 1 else mask[b].any(
                    dim=0)
            else:
                road_m = mask[b]
            m = m & road_m

        masks[b, 0] = m

        # для отладки считаем статистику ТОЛЬКО по m
        if m.any():
            dist_valid = dist[m]
            print(f"[b={b}] dist stats (masked):",
                  dist_valid.min().item(),
                  dist_valid.mean().item(),
                  dist_valid.max().item())
        else:
            print(f"[b={b}] no valid pixels in mask")

    return masks


def overlay_floor_mask_on_frame(rgb_tensor, mask_tensor, out_path):
    """
    rgb_tensor: [3,H,W], float в [0,1]
    mask_tensor: [H,W] bool
    """
    # в numpy, HWC, [0,255]
    rgb = rgb_tensor.detach().cpu().permute(1, 2, 0).numpy()
    rgb = (rgb * 255.0).clip(0, 255).astype(np.uint8)

    mask = mask_tensor.detach().cpu().numpy().astype(bool)  # [H,W]

    # делаем копию для заливки плоскости
    overlay = rgb.copy()
    # цвет плоскости, например зелёный
    color = np.array([0, 255, 0], dtype=np.uint8)

    # заливаем только там, где маска True
    overlay[mask] = (0.3 * overlay[mask] + 0.7 * color).astype(np.uint8)

    # лёгкий альфа-блендинг целиком (по желанию)
    vis = cv2.addWeighted(rgb, 0.6, overlay, 0.4, 0)

    # OpenCV ожидает BGR
    vis_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
    cv2.imwrite(out_path, vis_bgr)


def project_patch_on_plane(
    images,          # [B,3,H,W] фон (I1_batch)
    surface_xyz,     # [B,3,H,W] 3D-точки в координатах камеры
    planes,          # list длиной B: (normal:[3], d)
    patch_tex,       # [B,3,P,P] текстура патча (из A.P)
    # [B,1,H,W] bool — где вообще хотим патч (дорога∧плоскость)
    floor_masks=None,
    # физический размер области на плоскости (м) по (t1,t2)
    plane_size=(3.0, 3.0),
):
    """
    Возвращает:
        I_p: [B,3,H,W] — изображение с патчем
        M:   [B,1,H,W] bool — где патч реально нарисован
    """
    B, C, H, W = images.shape
    device = images.device
    Lx, Ly = plane_size

    # grid для grid_sample: [B,H,W,2]
    grid = torch.zeros(B, H, W, 2, device=device)

    # если floor_masks нет — считаем, что можно везде
    if floor_masks is None:
        floor_masks = torch.ones(B, 1, H, W, dtype=torch.bool, device=device)

    for b in range(B):
        n, d_plane = planes[b]         # нормаль и d
        n = n / (n.norm() + 1e-8)

        # выбираем "вверх", почти не коллинеарный с нормалью
        up = torch.tensor([0., 1., 0.], device=device)
        if torch.abs(torch.dot(n, up)) > 0.9:
            up = torch.tensor([1., 0., 0.], device=device)

        # базис в плоскости: t1, t2
        t1 = torch.cross(n, up)
        t1 = t1 / (t1.norm() + 1e-8)
        t2 = torch.cross(n, t1)
        t2 = t2 / (t2.norm() + 1e-8)

        # берём точки пола (floor_masks) и считаем их среднее — центр патча
        mask_b = floor_masks[b, 0]     # [H,W]
        xyz_b = surface_xyz[b]         # [3,H,W]
        pts_b = xyz_b.permute(1, 2, 0)[mask_b]  # [N,3]

        if pts_b.numel() == 0:
            # fallback: центрим в начале координат
            origin = torch.zeros(3, device=device)
        else:
            origin = pts_b.mean(dim=0)  # [3]

        # все точки кадра в 3D, чтобы посчитать (u,v)
        P = xyz_b.view(3, -1).t()      # [H*W,3]
        rel = P - origin               # [Npix,3]

        u = rel @ t1                   # [Npix]
        v = rel @ t2                   # [Npix]

        # нормируем в [-1,1] так, что [-Lx/2, Lx/2] → [-1,1]
        u_norm = (u / (Lx / 2.0)).clamp(-1.0, 1.0)
        v_norm = (v / (Ly / 2.0)).clamp(-1.0, 1.0)

        grid_b = torch.stack([u_norm, v_norm], dim=-1).view(H, W, 2)
        grid[b] = grid_b

    # grid_sample: берем патч [B,3,P,P] и размазываем его по [B,3,H,W]
    # grid в формате (x, y), уже нормализован
    patch_on_img = F.grid_sample(
        patch_tex,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )  # [B,3,H,W]

    # маска где патч не ноль и где floor_masks True
    M = (patch_on_img.abs().sum(dim=1, keepdim=True) > 1e-6) & floor_masks

    # финальное изображение: фон + патч
    M_f = M.float()
    I_p = images * (1.0 - M_f) + patch_on_img * M_f

    return I_p, M


def plane_basis(normal: torch.Tensor):
    """
    normal: [3] (нормализованная или нет)
    Возвращает два ортогональных вектора t1, t2 в плоскости и нормализованный n.
    """
    n = normal / (normal.norm() + 1e-8)
    up = torch.tensor([0., 1., 0.], device=n.device)
    if torch.abs(torch.dot(n, up)) > 0.9:
        up = torch.tensor([1., 0., 0.], device=n.device)

    t1 = torch.cross(n, up)
    t1 = t1 / (t1.norm() + 1e-8)
    t2 = torch.cross(n, t1)
    t2 = t2 / (t2.norm() + 1e-8)

    return t1, t2, n


def plane_basis(normal: torch.Tensor):
    n = normal / (normal.norm() + 1e-8)
    up = torch.tensor([0., 1., 0.], device=n.device)
    if torch.abs(torch.dot(n, up)) > 0.9:
        up = torch.tensor([1., 0., 0.], device=n.device)

    t1 = torch.cross(n, up)
    t1 = t1 / (t1.norm() + 1e-8)
    t2 = torch.cross(n, t1)
    t2 = t2 / (t2.norm() + 1e-8)

    return t1, t2, n


def project_patch_on_plane_homography(
    image: torch.Tensor,     # [3,H,W]
    patch: torch.Tensor,     # [3,Hp,Wp]  (A.P)
    K: torch.Tensor,         # [3,3]
    normal: torch.Tensor,    # [3]
    d_plane: torch.Tensor,   # скаляр
    plane_center_uv=None,    # (u0,v0) пиксели центра патча
    plane_size_m=(2.0, 2.0),  # (Wm,Hm) физический размер патча в метрах
    road_mask=None,          # [H,W] bool или [1,H,W], опционально
):
    """
    Возвращает:
        I_out: [3,H,W] — изображение с патчем
        M:     [1,H,W] bool — маска патча
    """
    assert image.dim() == 3, f"image must be [3,H,W], got {image.shape}"
    assert patch.dim() == 3, f"patch must be [3,Hp,Wp], got {patch.shape}"

    C, H, W = image.shape
    device = image.device

    # 1) базис в плоскости
    t1, t2, n = plane_basis(normal)

    # 2) центр патча – пересечение луча через (u0,v0) с плоскостью
    if plane_center_uv is None:
        u0 = W // 2
        v0 = int(H * 0.7)   # чуть ниже центра
    else:
        u0, v0 = plane_center_uv

    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    # луч из камеры
    x = (u0 - cx) / fx
    y = (v0 - cy) / fy
    ray_dir = torch.tensor([x, y, 1.0], device=device)
    ray_dir = ray_dir / (ray_dir.norm() + 1e-8)

    denom = torch.dot(n, ray_dir)
    t = -d_plane / (denom + 1e-8)
    p0 = t * ray_dir  # [3]

    # 3) лучи для всех пикселей
    ys, xs = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing='ij'
    )

    x_cam = (xs - cx) / fx   # [H,W]
    y_cam = (ys - cy) / fy   # [H,W]
    z_cam = torch.ones_like(x_cam)

    dirs = torch.stack([x_cam, y_cam, z_cam], dim=0)  # [3,H,W]
    dirs = dirs / (dirs.norm(dim=0, keepdim=True) + 1e-8)

    ndot = (n.view(3, 1, 1) * dirs).sum(dim=0)          # [H,W]
    t_all = -d_plane / (ndot + 1e-8)                  # [H,W]
    P = dirs * t_all.unsqueeze(0)                     # [3,H,W]

    # 4) координаты в базисе плоскости
    rel = P - p0.view(3, 1, 1)
    xi = (rel * t1.view(3, 1, 1)).sum(dim=0)          # [H,W]
    eta = (rel * t2.view(3, 1, 1)).sum(dim=0)          # [H,W]

    Wm, Hm = plane_size_m
    xi_norm = (xi / (Wm / 2.0)).clamp(-1.0, 1.0)
    eta_norm = (eta / (Hm / 2.0)).clamp(-1.0, 1.0)

    # grid: [1,H,W,2] float
    grid = torch.stack([xi_norm, eta_norm], dim=-1).unsqueeze(0).float()

    # 5) grid_sample: patch -> изображение
    patch_in = patch.unsqueeze(0)  # [1,3,Hp,Wp]

    patch_on_img = F.grid_sample(
        patch_in,        # [1,3,Hp,Wp]
        grid,            # [1,H,W,2]
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )[0]  # [3,H,W]

    M = (patch_on_img.abs().sum(dim=0, keepdim=True) > 1e-6)  # [1,H,W] bool

    if road_mask is not None:
        if road_mask.dim() == 2:
            road_m = road_mask.unsqueeze(0)
        else:
            road_m = road_mask
        M = M & road_m

    M_f = M.float()
    I_out = image * (1.0 - M_f) + patch_on_img * M_f

    return I_out, M


def keep_largest_component(road_mask: torch.Tensor) -> torch.Tensor:
    """
    Given a road_mask [B,1,H,W] or [B,H,W], keep only the largest connected
    component per batch element (8-connectivity). Returns a mask of same shape.
    """
    # ensure shape [B,1,H,W]
    if road_mask.dim() == 3:
        road_mask = road_mask.unsqueeze(1)  # [B,1,H,W]
    B, C, H, W = road_mask.shape
    assert C == 1, f"Expected road_mask with C=1, got {C}"

    device = road_mask.device
    out = torch.zeros_like(road_mask, dtype=torch.float32)

    for b in range(B):
        # to numpy uint8: 0 or 1
        m = road_mask[b, 0].detach().cpu().numpy().astype(np.uint8)

        # connected components (0 = background)
        num_labels, labels = cv2.connectedComponents(m, connectivity=8)

        if num_labels <= 1:
            # no foreground components
            continue

        # compute area of each label
        areas = np.bincount(labels.ravel())
        # ignore label 0 (background)
        best_label = 1 + np.argmax(areas[1:])

        biggest = (labels == best_label).astype(np.uint8)

        out[b, 0] = torch.from_numpy(biggest).to(device)

    # return in the same dtype as input (0/1 float or bool)
    if road_mask.dtype == torch.bool:
        return out.bool()
    else:
        return out

import torch
import torch.nn.functional as F
from .adversarial_manhole.adv_manhole.texture_mapping.depth_utils import disp_to_depth


def project_patch_on_scene(
    I1_batch,
    I2_batch,
    K,
    A,                       # PatchAdversary
    mde_model=None,
    ss_model=None,
    io_adapter=None,
    device="cuda",
    plane_aug=True,
    plane_aug_angle=5.0,
    road_class_id=0,
    precomputed_depth=None,      # [B,1,H,W] – skip MDE forward if provided
    precomputed_road_mask=None,  # [B,1,H,W] – skip SS forward if provided
    precomputed_planes=None,     # list of (normal, d) – skip plane fitting if provided
    flow_shift=0.0,
):
    """
    Универсальная функция проецирования патча A на дорожную плоскость
    с учетом:
        - mde depth (depth → plane fit)
        - semantic segmentation mask (road region)
        - plane tilt augmentation
        - patch projection PatchAdversary(A)

    Precomputed arguments can be supplied to skip the corresponding
    model forward passes and plane fitting (useful when calling this
    function multiple times on the same clean batch, e.g. inside an
    inner optimisation loop).

    Возвращает:
        I1_p_batch, I2_p_batch  — изображения с патчем
        M_batch                 — маска патча
        ys_batch, xs_batch      — координаты центра патча на исходном изображении
        road_mask               — маска дороги
        planes                  — fitted planes
    """

    B = I1_batch.shape[0]

    # ---------------------------------------------------------
    # 1. Compute depth via MDE model (skip if precomputed)
    # ---------------------------------------------------------
    if precomputed_depth is not None:
        depth_pred = precomputed_depth
    elif mde_model is not None:
        inp = io_adapter.prepare_inputs(
            inputs={'images': torch.stack([I1_batch, I2_batch], dim=1)}
        )

        with torch.no_grad():
            mde_out_unatt = mde_model(inp)

        if mde_out_unatt is not None:
            # Ensure shape [B,1,H,W]
            if mde_out_unatt.dim() == 2:
                depth_pred = mde_out_unatt.unsqueeze(0).unsqueeze(0)
            elif mde_out_unatt.dim() == 3:
                depth_pred = mde_out_unatt.unsqueeze(1)
            else:
                depth_pred = mde_out_unatt

            depth_pred = disp_to_depth(depth_pred.to(device).float())
        else:
            depth_pred = None
    else:
        depth_pred = None

    # ---------------------------------------------------------
    # 2. Compute road mask from segmentation model (skip if precomputed)
    # ---------------------------------------------------------
    if precomputed_road_mask is not None:
        road_mask = precomputed_road_mask
    elif ss_model is not None:
        inp = io_adapter.prepare_inputs(
            inputs={'images': torch.stack([I1_batch, I2_batch], dim=1)}
        )
        with torch.no_grad():
            ss_logits = ss_model(inp, return_logits=True)

        ss_pred = ss_logits.argmax(dim=1)  # [B,H,W]
        road_mask = (ss_pred == road_class_id).unsqueeze(1)  # [B,1,H,W]
        road_mask = keep_largest_component(road_mask)        # stabilize
    else:
        road_mask = None

    # ---------------------------------------------------------
    # 3. Fit plane from depth (skip if precomputed)
    # ---------------------------------------------------------
    if precomputed_planes is not None:
        planes = precomputed_planes
    elif depth_pred is not None:
        planes = fit_plane_from_depth(depth_pred, K, road_mask)
    else:
        planes = None  # still allow PatchAdversary to run (flat projection)

    # ---------------------------------------------------------
    # 4. Random plane tilt augmentation (optional)
    # ---------------------------------------------------------
    if plane_aug and planes is not None:
        planes = random_tilt_plane(planes, max_angle_deg=plane_aug_angle)

    # ---------------------------------------------------------
    # 5. Project patch using PatchAdversary(A)
    # ---------------------------------------------------------
    if planes is not None:
        I1_p, I2_p, M_batch, ys_batch, xs_batch = A(
            I1_batch, I2_batch,
            K=K,
            planes=planes,
            road_masks=road_mask,
            flow_shift=flow_shift,
        )
    else:
        I1_p, I2_p, M_batch, ys_batch, xs_batch = A(
            I1_batch, I2_batch, flow_shift=flow_shift)

    return (
        I1_p,       # patched I1
        I2_p,       # patched I2
        M_batch,    # mask of patch
        ys_batch,   # y-center
        xs_batch,   # x-center
        road_mask,
        planes
    )

def random_tilt_plane(plane, max_angle_deg=5.0):
    """
    Добавляет небольшой случайный наклон нормали плоскости.
    plane: tensor [B,4] or [4]
    max_angle_deg: максимальный угол наклона в градусах
    """
    if plane.ndim == 1:
        plane = plane.unsqueeze(0)  # → [1,4]

    B = plane.shape[0]
    device = plane.device
    max_angle = max_angle_deg * torch.pi / 180.0

    # исходная нормаль
    n = plane[:, :3]
    n = n / (n.norm(dim=1, keepdim=True) + 1e-9)

    # случайные векторы
    rand_vec = torch.randn((B, 3), device=device)
    rand_vec = rand_vec / (rand_vec.norm(dim=1, keepdim=True) + 1e-9)

    # случайные углы
    angles = (torch.rand(B, 1, device=device) * 2 - 1) * max_angle  # [-max_angle, max_angle]

    # формула вращения вектора вокруг случайной оси: Rodrigues rotation
    k = rand_vec
    k = k / (k.norm(dim=1, keepdim=True) + 1e-9)

    cos = torch.cos(angles)
    sin = torch.sin(angles)

    n_rot = (
        n * cos +
        torch.cross(k, n, dim=1) * sin +
        k * (torch.sum(k * n, dim=1, keepdim=True) * (1 - cos))
    )

    # пересборка плоскости (d оставляем прежним)
    new_plane = torch.cat([n_rot, plane[:, 3:].clone()], dim=1)
    return new_plane
