from metrics.attack_metrics import AttackMetricsTracker
from utils.targets import get_target, get_mde_target, get_ss_target
from utils.losses import get_mde_loss, get_ss_loss, get_loss
from os import path as op
import os
from tqdm import tqdm
import torch
import torch.optim as optim
from .patch_projection import fit_plane_from_depth, keep_largest_component, project_patch_on_scene
from .adversarial_manhole.adv_manhole.texture_mapping.depth_utils import disp_to_depth
from .DetectionDefenses.helper_functions.defenses import LGS, ILP
from .DetectionDefenses.helper_functions.losses import aee_masked, acs_masked, mse_masked
from .DetectionDefenses.helper_functions.ownutilities import preprocess_img as oa_preprocess_img, postprocess_flow as oa_postprocess_flow
from .DetectionDefenses.helper_functions.custom_optimizer import IFGSM, ClippedPGD
from .DetectionDefenses.helper_functions.patch_adversary import PatchAdversary
import sys
import pathlib
from datasets_utils.dataset_utils import prepare_dataloader
from defenses.temporal_filter import TemporalPredictionFilter

sys.path.append(str(pathlib.Path(__file__).resolve().parent))




def project_simplex(v, z=1.0):
    """Project vector v onto the z-simplex: w >= 0, sum(w) = z.

    Implements the algorithm from Duchi et al., ICML 2008.
    """
    with torch.no_grad():
        v = v.view(-1)
        n = v.shape[0]
        if n == 1:
            return torch.full_like(v, z)
        mu, _ = torch.sort(v, descending=True)
        cumsum = torch.cumsum(mu, dim=0)
        j = torch.arange(1, n + 1, dtype=v.dtype, device=v.device)
        rho = int((mu * j - cumsum + z > 0).sum().item()) - 1
        theta = (cumsum[rho] - z) / (rho + 1.0)
        return torch.clamp(v - theta, min=0.0)


def total_variation_loss(patch, tv_weight=1e-4):
    """
    patch: [C, H, W] or [B,C,H,W]
    """
    if patch.dim() == 3:
        patch = patch.unsqueeze(0)

    diff_x = torch.abs(patch[:, :, :, 1:] - patch[:, :, :, :-1])
    diff_y = torch.abs(patch[:, :, 1:, :] - patch[:, :, :-1, :])
    return tv_weight * (diff_x.mean() + diff_y.mean())

def printability_loss(patch, nps_weight=1.0, grid_size=6):
    """
    Non-Printability Score (NPS) loss.
    Генерирует сетку печатаемых цветов внутри функции.

    patch: [C,H,W] или [B,C,H,W], значения в [0,1]
    grid_size: количество дискретизаций RGB канала (6 → 216 цветов)
    """
    if patch.dim() == 3:
        patch = patch.unsqueeze(0)

    B, C, H, W = patch.shape

    # --- 1. генерируем "printable" RGB-сетку ---
    vals = torch.linspace(0.05, 0.95, steps=grid_size, device=patch.device)
    printable_colors = torch.stack(torch.meshgrid(vals, vals, vals), dim=-1).reshape(-1, 3)  
    # shape: [grid_size^3, 3], например [216, 3]

    # --- 2. получаем пиксели патча ---
    pixels = patch.permute(0, 2, 3, 1).reshape(-1, 3)  # [N,3]

    # --- 3. расстояние каждого пикселя до ближайшего печатаемого цвета ---
    diff = pixels.unsqueeze(1) - printable_colors.unsqueeze(0)  # [N, K, 3]
    dist = torch.sqrt((diff ** 2).sum(dim=2))                   # [N, K]
    min_dist, _ = dist.min(dim=1)                               # [N]
    return nps_weight * min_dist.mean()



def build_pedestrian_like_mask(
    xs_batch,
    ys_batch,
    patch_size: int,
    pred_shape: torch.Size,   # shape предсказания потока [B,C,H,W]
    device,
    width_factor: float = 0.2,   # относительная ширина прямоугольника от размера патча
    height_factor: float = 2.0   # относительная высота
):
    """
    Строит маску прямоугольника, похожего по размеру на человека,
    который стоит над патчем (в координатах ПРЕДСКАЗАНИЯ, а не исходного изображения).

    Прямоугольник:
      - центр по X совпадает с центром патча,
      - нижняя граница по Y == центр патча,
      - верхняя граница выше на patch_size * height_factor.
    """

    B, _, H, W = pred_shape
    ped_mask = torch.zeros((B, 1, H, W), device=device, dtype=torch.float32)

    # приводим xs_batch / ys_batch к спискам длины B
    def to_list(v):
        if isinstance(v, (list, tuple)):
            return [int(x) for x in v]
        if isinstance(v, torch.Tensor):
            if v.ndim == 0:
                return [int(v.item())] * B
            else:
                return [int(x.item()) for x in v]
        # скаляр
        return [int(v)] * B

    xs_list = to_list(xs_batch)
    ys_list = to_list(ys_batch)

    for b in range(B):
        cx = xs_list[b]
        cy = ys_list[b]

        ped_width = int(patch_size * width_factor)
        ped_height = int(patch_size * height_factor)

        # прямоугольник над патчем:
        # низ прямоугольника в центре патча (cy),
        # верх — выше на ped_height
        x0 = cx - ped_width // 2
        x1 = cx + ped_width // 2
        y1 = cy
        y0 = cy - ped_height

        # не выходить за границы карты предсказания
        x0 = max(0, min(W, x0))
        x1 = max(0, min(W, x1))
        y0 = max(0, min(H, y0))
        y1 = max(0, min(H, y1))

        if x1 > x0 and y1 > y0:
            ped_mask[b, 0, y0:y1, x0:x1] = 1.0

    return ped_mask


def masked_mse_consistency(pred_a, pred_b, mask, eps: float = 1e-6):
    """
    MSE между pred_a и pred_b только в области mask.
    pred_*: [B,C,H,W], mask: [B,1,H,W] с 0/1.
    """
    if mask is None:
        return torch.tensor(0.0, device=pred_a.device)

    mask = mask.float()
    if mask.shape[1] == 1 and pred_a.shape[1] > 1:
        mask = mask.expand(-1, pred_a.shape[1], -1, -1)

    diff = (pred_a - pred_b) * mask
    num = (diff ** 2).sum()
    denom = mask.sum() * 1.0 + eps
    return num / denom

def train_patch_ptlflow(
    args,
    model,
    data_loader,
    device,
    io_adapter,
    metrics_tracker: AttackMetricsTracker = None,
    mde_model=None,
    ss_model=None
):
    """
    Train one universal patch across the whole dataset (PTLFlow-style) + outside-consistency.
    """

    import torch
    import numpy as np
    from itertools import islice

    # adversary
    A = PatchAdversary(
        None,
        size=args.patch_size,
        angle=[-10, 10],
        scale=[0.95, 1.05],
        change_of_variable=args.change_of_variables,
        random_location=args.random_loc,
        image_size=(375, 1242), 
        ellipse_scale_y=args.y_scale
    ).to(device)

    depth_loader, _ = prepare_dataloader(
        dataset_name=args.dataset, small_run=args.small_run,
        subset_size=getattr(args, 'subset_size', 0))

    # optimizer
    if args.optimizer == "adam":
        optimizer = optim.Adam(A.parameters(), lr=args.lr)
    elif args.optimizer == "sgd":
        optimizer = optim.SGD(A.parameters(), lr=args.lr, momentum=0.9)
    elif args.optimizer == "clipped-pgd":
        optimizer = ClippedPGD(A.parameters(), lr=args.lr,
                               min_=0, max_=1, max_delta=args.max_delta)
    elif args.optimizer == "ifgsm":
        if args.change_of_variables:
            optimizer = IFGSM(A.parameters(), lr=args.lr, min_=-100, max_=100)
        else:
            optimizer = IFGSM(A.parameters(), lr=args.lr, min_=0, max_=1)
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")

    # defense
    D = None
    if args.defense == "lgs":
        D = LGS(args.k, args.o, args.t, args.s, "forward")
    elif args.defense == "ilp":
        D = ILP(args.k, args.o, args.t, args.s, args.r, "forward")
    

    temporal_mode_map = {
        'temporal-avg': 'average',
        'temporal-median': 'median',
        'temporal-bilateral': 'bilateral',
        'temporal-domain-transform': 'domain_transform'
    }

    # TF_flow = None
    # TF_mde = None
    # TF_ss = None

    # temporal_modes = ["temporal-avg", "temporal-median", "temporal-bilateral", "temporal-domain-transform"]
    # if args.defense in temporal_modes:
    #     mode = temporal_mode_map[args.defense]
    #     # Создаем фильтры (параметры можно брать общие или разные из args)
    #     TF_flow = TemporalPredictionFilter(mode=mode, window_size=args.temp_window, sigma_color=args.sigma_color).to(device)
    #     TF_mde = TemporalPredictionFilter(mode=mode, window_size=args.temp_window, sigma_color=args.sigma_color).to(device)
    #     TF_ss = TemporalPredictionFilter(mode=mode, window_size=args.temp_window, sigma_color=args.sigma_color).to(device)
    # targets & losses for optical flow
    flow_target_fn = get_target(args.target, magnitude=args.flow_target_magnitude)
    flow_loss_fn = get_loss(args.loss, untargeted=args.target == 'untargeted')

    # targets and losses for mde
    if args.attack_mde:
        mde_loss_fn = get_mde_loss(untargeted=args.mde_target == 'untargeted')
        mde_target_fn = get_mde_target(
            target_name=args.mde_target,
            data_loader=depth_loader,
            flow_model=model,
            mde_model=mde_model,
            device=device,
            q=0.9
        )
    else:
        mde_loss_fn = None
        mde_target_fn = None


    # targets and losses for semantic segmentation
    if args.attack_ss:
        ss_loss_fn = get_ss_loss(
            untargeted=args.ss_target == 'untargeted',
            gamma=getattr(args, 'ss_focal_gamma', 2.0)
        )
        ss_target_fn = get_ss_target(args.ss_target)
    else:
        ss_loss_fn = None
        ss_target_fn = None

    flow_w_init, mde_w_init, ss_w_init = args.loss_weights

    weight_strategy = getattr(args, "weight_strategy", "fixed")

    # Build the list of *active* task indices (only tasks that are actually
    # attacked contribute to the dynamic weight vector).
    active_tasks = ["flow"]
    if args.attack_mde and mde_model is not None:
        active_tasks.append("mde")
    if args.attack_ss and ss_model is not None:
        active_tasks.append("ss")
    n_tasks = len(active_tasks)

    # For 'fixed' strategy the weights never change.
    flow_w, mde_w, ss_w = flow_w_init, mde_w_init, ss_w_init

    # For 'minmax' strategy we maintain a weight vector on the probability
    # simplex P = {w | sum(w)=1, w_i>=0}.  Algorithm 1, Guo et al. ICASSP 2025.
    # Initialised to uniform w^(0) = 1/k as in the paper.
    if weight_strategy == "minmax":
        W = torch.ones(n_tasks, device=device) / n_tasks
        alpha_w = getattr(args, "minmax_alpha_w", 0.03)   # α₂ in Eq.4
        gamma_w = getattr(args, "minmax_gamma", 5.0)      # γ  in Eq.2

    # коэффициенты для outside-consistency 
    flow_cons_w = getattr(args, "flow_cons_w", 0)
    mde_cons_w = getattr(args, "mde_cons_w", 0)
    ss_cons_w = getattr(args, "ss_cons_w", 0)

    # удобный helper для io_adapter
    def prepare_inputs_safe(images_tensor, flows_tensor=None, valids_tensor=None):
        """Try passing batched tensors, if fails convert to list-of-numpy per-sample and retry."""
        try:
            return io_adapter.prepare_inputs(
                inputs={'images': images_tensor,
                        'flows': flows_tensor, 'valids': valids_tensor}
            )
        except Exception:
            imgs = []
            Bs = images_tensor.shape[0]
            for bi in range(Bs):
                arr = images_tensor[bi].detach().cpu().numpy()  # [2,C,H,W]
                imgs_sample = []
                for t in range(arr.shape[0]):
                    frame = arr[t].transpose(1, 2, 0)  # C,H,W -> H,W,C
                    imgs_sample.append(frame)
                imgs.append(imgs_sample)

            flows_list = None
            valids_list = None
            if flows_tensor is not None:
                flows_list = [flows_tensor[b].detach().cpu().numpy()
                              for b in range(flows_tensor.shape[0])]
            if valids_tensor is not None:
                valids_list = [valids_tensor[b].detach().cpu().numpy()
                               for b in range(valids_tensor.shape[0])]
            return io_adapter.prepare_inputs(
                inputs={'images': imgs, 'flows': flows_list,
                        'valids': valids_list}
            )

    # prepare models
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    if mde_model is not None:
        mde_model.eval()
        for p in mde_model.parameters():
            p.requires_grad = False

    if ss_model is not None:
        ss_model.eval()
        for p in ss_model.parameters():
            p.requires_grad = False

    mapping = None

    tv_w = getattr(args, "tv_weight", 0.0)
    nps_w = getattr(args, "nps_weight", 0.0)

    for epoch in range(args.n):
        print(f"Epoch {epoch+1}/{args.n}")

        epoch_accum = {
            "loss": 0.0, "flow_adv": 0.0, "mde_adv": 0.0, "ss_adv": 0.0,
            "flow_cons": 0.0, "mde_cons": 0.0, "ss_cons": 0.0,
            "tv": 0.0, "nps": 0.0,
        }
        n_batches = 0

        for batch_idx, (images, flow_gt, valid, meta, K) in enumerate(tqdm(data_loader)):
            images = images.to(device)          # [B,2,C,H,W]
            flow_gt = flow_gt.to(device)

            if isinstance(valid, torch.Tensor):
                valid = valid.to(device)

            B = images.shape[0]

            I1_batch = images[:, 0, :, :, :]  # [B,C,H_img,W_img]
            I2_batch = images[:, 1, :, :, :]

            # ------------------------------------------------------------------
            # Pre-compute per-batch quantities that are invariant across the
            # inner optimisation loop (unattacked predictions, depth, road mask,
            # and plane fits all depend only on the clean images and don't change
            # between inner steps).
            # ------------------------------------------------------------------
            _unatt_imgs = torch.stack([I1_batch, I2_batch], dim=1)
            unattacked_inputs_cache = prepare_inputs_safe(
                _unatt_imgs, flows_tensor=flow_gt, valids_tensor=valid)

            with torch.no_grad():
                pred_unattacked = model(unattacked_inputs_cache)['flows'].squeeze(0)

                # MDE unattacked prediction (reused for loss and plane fitting)
                mde_pred_unatt = None
                if mde_model is not None and args.attack_mde:
                    mde_pred_unatt = mde_model(unattacked_inputs_cache)

                # SS unattacked prediction (reused for loss and road mask)
                ss_pred_unatt = None
                if ss_model is not None and args.attack_ss:
                    ss_pred_unatt = ss_model(unattacked_inputs_cache, return_logits=True)

                # Depth + road mask + planes for patch projection (computed once
                # per batch; reused by project_patch_on_scene every inner step)
                cached_depth = None
                cached_road_mask = None
                cached_planes = None
                if args.patch_projection:
                    if mde_model is not None:
                        # Reuse mde_pred_unatt when available to avoid a 2nd forward pass
                        _mde_raw = mde_pred_unatt if mde_pred_unatt is not None \
                            else mde_model(unattacked_inputs_cache)
                        if _mde_raw is not None:
                            _d = _mde_raw
                            if _d.dim() == 2:
                                _d = _d.unsqueeze(0).unsqueeze(0)
                            elif _d.dim() == 3:
                                _d = _d.unsqueeze(1)
                            cached_depth = disp_to_depth(_d.to(device).float())

                    if ss_model is not None:
                        # Reuse ss_pred_unatt when available
                        _ss_logits = ss_pred_unatt if ss_pred_unatt is not None \
                            else ss_model(unattacked_inputs_cache, return_logits=True)
                        _road_pred = _ss_logits.argmax(dim=1)
                        cached_road_mask = (_road_pred == 0).unsqueeze(1)
                        cached_road_mask = keep_largest_component(cached_road_mask)

                    if cached_depth is not None:
                        cached_planes = fit_plane_from_depth(cached_depth, K, cached_road_mask)

            H_flow, W_flow = pred_unattacked.shape[-2:]
            B_flow = pred_unattacked.shape[0]

            # Pre-compute SS softmax probabilities for outside-consistency
            ss_prob_unatt = None
            if ss_model is not None and args.attack_ss and ss_pred_unatt is not None:
                with torch.no_grad():
                    ss_prob_unatt = torch.softmax(ss_pred_unatt, dim=1)

            # ----------------------------------------------------------
            # Weight strategy: per-batch initialisation
            # ----------------------------------------------------------
            if weight_strategy == "normalized":
                # Compute each task's clean-image loss and use 1/L_clean
                # as a normaliser so that all tasks contribute equally at
                # the start, then scaled by the user-provided ratios.
                with torch.no_grad():
                    _M_ones = torch.ones(
                        (B_flow, 1, H_flow, W_flow), device=device)

                    _flow_target = flow_target_fn(pred_unattacked) \
                        if args.target != 'scene' \
                        else flow_target_fn(pred_unattacked, 0, 0)
                    _flow_target = _flow_target.to(device)
                    L_flow_clean = flow_loss_fn(
                        pred_unattacked, _flow_target, _M_ones
                    ).clamp(min=1e-8)

                    L_mde_clean = torch.tensor(1.0, device=device)
                    if mde_model is not None and args.attack_mde and mde_pred_unatt is not None:
                        _mde_target = mde_target_fn(mde_pred_unatt) \
                            if args.mde_target != 'scene' \
                            else mde_target_fn(mde_pred_unatt)
                        _mde_target = _mde_target.to(device)
                        _M_mde = torch.ones_like(mde_pred_unatt[:, :1, :, :])
                        L_mde_clean = mde_loss_fn(
                            mde_pred_unatt, _mde_target, _M_mde
                        ).clamp(min=1e-8)

                    L_ss_clean = torch.tensor(1.0, device=device)
                    if ss_model is not None and args.attack_ss and ss_pred_unatt is not None:
                        _ss_target = ss_target_fn(ss_pred_unatt).to(device)
                        _M_ss = torch.ones(
                            ss_pred_unatt.shape[0], 1,
                            *ss_pred_unatt.shape[-2:], device=device)
                        L_ss_clean = ss_loss_fn(
                            ss_pred_unatt, _ss_target, _M_ss
                        ).clamp(min=1e-8)

                flow_w = flow_w_init / L_flow_clean.item()
                mde_w = mde_w_init / L_mde_clean.item()
                ss_w = ss_w_init / L_ss_clean.item()

            elif weight_strategy == "minmax":
                # Reset to uniform at the start of each batch
                W = torch.ones(n_tasks, device=device) / n_tasks

            for inner_step in range(args.steps):
                optimizer.zero_grad()

                # --- патч-проекция / текстурирование по плоскости дороги ---
                if args.patch_projection:
                    (
                        I1_p_batch,
                        I2_p_batch,
                        M_batch,
                        ys_batch,
                        xs_batch,
                        road_mask,
                        planes
                    ) = project_patch_on_scene(
                        I1_batch, I2_batch, K,
                        A=A,
                        mde_model=mde_model,
                        ss_model=ss_model,
                        io_adapter=io_adapter,
                        device=device,
                        plane_aug=args.plane_aug,
                        precomputed_depth=cached_depth,
                        precomputed_road_mask=cached_road_mask,
                        precomputed_planes=cached_planes,
                        flow_shift=args.flow_shift,
                    )
                else:
                    I1_p_batch, I2_p_batch, M_batch, ys_batch, xs_batch = A(
                        I1_batch, I2_batch, flow_shift=args.flow_shift)
                    road_mask, planes = None, None

                # --- defense ---
                if D is not None:
                    I1_att_def_batch, I2_att_def_batch = D(
                        I1_p_batch, I2_p_batch, M_batch)
                    I1_unatt_def_batch, I2_unatt_def_batch = D(
                        I1_batch, I2_batch, torch.zeros_like(M_batch))
                    # When a defense is active the "unattacked" images are
                    # defense-filtered, so re-compute inputs/predictions for
                    # that branch (cannot reuse the clean-image cache).
                    unattacked_images_tensor = torch.stack(
                        [I1_unatt_def_batch, I2_unatt_def_batch], dim=1)
                    unattacked_inputs = prepare_inputs_safe(
                        unattacked_images_tensor, flows_tensor=flow_gt, valids_tensor=valid)
                    with torch.no_grad():
                        pred_unattacked = model(unattacked_inputs)['flows'].squeeze(0)
                        if mde_model is not None and args.attack_mde:
                            mde_pred_unatt = mde_model(unattacked_inputs)
                        if ss_model is not None and args.attack_ss:
                            ss_pred_unatt = ss_model(unattacked_inputs, return_logits=True)
                            ss_prob_unatt = torch.softmax(ss_pred_unatt, dim=1)
                else:
                    I1_att_def_batch, I2_att_def_batch = I1_p_batch, I2_p_batch
                    # Reuse the pre-computed cache (no defense applied)
                    unattacked_inputs = unattacked_inputs_cache

                attacked_images_tensor = torch.stack(
                    [I1_att_def_batch, I2_att_def_batch], dim=1)

                attacked_inputs = prepare_inputs_safe(
                    attacked_images_tensor, flows_tensor=flow_gt, valids_tensor=valid)

                # --- forward through flow model (attacked only; unattacked is cached) ---
                pred_attacked = model(attacked_inputs)['flows'].squeeze(0)

                # if TF_flow is not None:
                #     pred_attacked = TF_flow(pred_attacked, current_image=I1_att_def_batch)

                if M_batch is not None and M_batch.dim() == 4 and M_batch.shape[1] == 1:
                    M_flow_patch = torch.nn.functional.interpolate(
                        M_batch, size=(H_flow, W_flow), mode='nearest'
                    )
                else:
                    M_flow_patch = torch.ones(
                        (B_flow, 1, H_flow, W_flow), device=device, dtype=torch.float32
                    )

                M_attack_flow = M_flow_patch.float()
                M_outside_flow = 1.0 - M_attack_flow

                # --- target для flow ---
                if args.target == 'scene':
                    target = flow_target_fn(
                        pred_unattacked, xs_batch, ys_batch + args.patch_size // 2
                    )
                else:
                    target = flow_target_fn(pred_unattacked)
                target = target.to(device)

                # --- adversarial flow loss (только в области атаки) ---
                flow_adv_loss = flow_loss_fn(
                    pred_attacked, target, M_attack_flow
                )

                # --- outside-consistency для flow ---
                flow_cons_loss = masked_mse_consistency(
                    pred_attacked, pred_unattacked, M_outside_flow
                )

                # --- MDE branch ---
                if mde_model is not None and args.attack_mde:
                    mde_pred_att = mde_model(attacked_inputs)
                    # mde_pred_unatt is pre-computed above (cached or defense-adjusted)

                    if args.mde_target == 'scene':
                        mde_target_batch = mde_target_fn(
                            mde_pred_unatt, xs_batch, ys_batch + args.patch_size // 2
                        )
                    else:
                        mde_target_batch = mde_target_fn(mde_pred_unatt)

                    mde_target_batch = mde_target_batch.to(device)

                    # маски в разрешении depth
                    M_attack_mde = torch.nn.functional.interpolate(
                        M_attack_flow, size=mde_pred_att.shape[-2:], mode='nearest'
                    )
                    M_outside_mde = 1.0 - M_attack_mde

                    mde_adv_loss = mde_loss_fn(
                        mde_pred_att, mde_target_batch, M_attack_mde
                    )

                    mde_cons_loss = masked_mse_consistency(
                        mde_pred_att, mde_pred_unatt, M_outside_mde
                    )
                else:
                    mde_adv_loss = torch.tensor(0.0, device=device)
                    mde_cons_loss = torch.tensor(0.0, device=device)

                # --- semantic segmentation branch ---
                if ss_model is not None and args.attack_ss:
                    ss_pred_att = ss_model(attacked_inputs, return_logits=True)
                    # ss_pred_unatt and ss_prob_unatt are pre-computed above

                    # if TF_ss is not None:
                        # ss_pred_att = TF_ss(ss_pred_att, current_image=I1_att_def_batch)

                    ss_target_batch = ss_target_fn(ss_pred_unatt).to(device)

                    M_attack_ss = torch.nn.functional.interpolate(
                        M_attack_flow, size=ss_pred_att.shape[-2:], mode='nearest'
                    )
                    M_outside_ss = 1.0 - M_attack_ss

                    # adversarial seg loss (ломаем внутри)
                    ss_adv_loss = ss_loss_fn(
                        ss_pred_att, ss_target_batch, M_attack_ss
                    )

                    # outside-consistency по вероятностям
                    ss_prob_att = torch.softmax(ss_pred_att, dim=1)
                    ss_cons_loss = masked_mse_consistency(
                        ss_prob_att, ss_prob_unatt, M_outside_ss
                    )
                else:
                    ss_adv_loss = torch.tensor(0.0, device=device)
                    ss_cons_loss = torch.tensor(0.0, device=device)

                # --- TV loss (skip when weight is 0) ---
                tv_loss = total_variation_loss(A.P, tv_w) if tv_w > 0.0 \
                    else torch.tensor(0.0, device=device)

                # --- Printability (NPS) loss (skip when weight is 0) ---
                nps_loss = printability_loss(A.P, nps_w) if nps_w > 0.0 \
                    else torch.tensor(0.0, device=device)

                # --- Resolve per-task weights for this step ------
                if weight_strategy == "minmax":
                    # Map simplex vector W back to per-task scalars.
                    # W is ordered according to active_tasks.
                    _wi = {t: W[i] for i, t in enumerate(active_tasks)}
                    flow_w = _wi.get("flow", torch.tensor(0.0, device=device))
                    mde_w = _wi.get("mde", torch.tensor(0.0, device=device))
                    ss_w = _wi.get("ss", torch.tensor(0.0, device=device))
                # (for 'fixed' and 'normalized', flow_w/mde_w/ss_w are already set)

                # --- TOTAL LOSS ---
                total_loss = (
                    flow_w * flow_adv_loss +
                    mde_w * mde_adv_loss +
                    ss_w * ss_adv_loss +
                    flow_cons_w * flow_cons_loss +
                    mde_cons_w * mde_cons_loss +
                    ss_cons_w * ss_cons_loss +
                    tv_loss +
                    nps_loss
                )

                total_loss.backward()
                optimizer.step()

                # --- Inner minimisation: update w  (Alg.1 step 4, Eq.4) ---
                # Guo et al. ICASSP 2025, Eq.2:
                #   max_δ  min_{w∈P}  Σ w_i L_i(x+δ) + γ/2 ||w − 1/k||²
                #
                # In the paper L_i is *maximised* (PGD ascent), so the inner
                # min over w uses gradient descent:
                #   ∇_w f = L_vec + γ(w − 1/k)
                #   w_new = proj_P(w − α₂ · ∇_w f)            ... Eq.4
                #
                # In our patch attack we *minimise* L_i (lower = better attack),
                # so the task with the HIGHEST loss is the hardest to attack.
                # The equivalent formulation is:
                #   min_patch  max_{w∈P}  Σ w_i L_i − γ/2 ||w − 1/k||²
                # which gives gradient ASCENT on w:
                #   ∇_w g = L_vec − γ(w − 1/k)
                #   w_new = proj_P(w + α₂ · ∇_w g)
                if weight_strategy == "minmax":
                    with torch.no_grad():
                        task_losses = []
                        for t in active_tasks:
                            if t == "flow":
                                task_losses.append(flow_adv_loss.detach())
                            elif t == "mde":
                                task_losses.append(mde_adv_loss.detach())
                            elif t == "ss":
                                task_losses.append(ss_adv_loss.detach())
                        L_vec = torch.stack(task_losses)
                        grad_w = L_vec - gamma_w * (W - 1.0 / n_tasks)
                        W = project_simplex(W + alpha_w * grad_w)

                step = epoch * len(data_loader) + batch_idx

                is_last_inner = (inner_step == args.steps - 1)

                if is_last_inner:
                    epoch_accum["loss"] += total_loss.detach()
                    epoch_accum["flow_adv"] += flow_adv_loss.detach()
                    epoch_accum["mde_adv"] += mde_adv_loss.detach()
                    epoch_accum["ss_adv"] += ss_adv_loss.detach()
                    epoch_accum["flow_cons"] += flow_cons_loss.detach()
                    epoch_accum["mde_cons"] += mde_cons_loss.detach()
                    epoch_accum["ss_cons"] += ss_cons_loss.detach()
                    epoch_accum["tv"] += tv_loss.detach()
                    epoch_accum["nps"] += nps_loss.detach()
                    n_batches += 1

                if metrics_tracker is not None and is_last_inner:
                    metrics_tracker.log_metric(
                        "epoch_batch_loss", float(total_loss.item()), step=step)
                    metrics_tracker.log_metric(
                        "epoch_batch_flow_adv_loss", float(flow_adv_loss.item()), step=step)
                    metrics_tracker.log_metric(
                        "epoch_batch_flow_cons_loss", float(flow_cons_loss.item()), step=step)
                    metrics_tracker.log_metric(
                        "epoch_batch_mde_adv_loss", float(mde_adv_loss.item()), step=step)
                    metrics_tracker.log_metric(
                        "epoch_batch_mde_cons_loss", float(mde_cons_loss.item()), step=step)
                    metrics_tracker.log_metric(
                        "epoch_batch_ss_adv_loss", float(ss_adv_loss.item()), step=step)
                    metrics_tracker.log_metric(
                        "epoch_batch_ss_cons_loss", float(ss_cons_loss.item()), step=step)
                    metrics_tracker.log_metric("epoch_batch_tv_loss", float(tv_loss.item()), step=step)
                    metrics_tracker.log_metric("epoch_batch_nps_loss", float(nps_loss.item()), step=step)
                    if weight_strategy in ("normalized", "minmax"):
                        metrics_tracker.log_metric("w_flow", float(flow_w) if isinstance(flow_w, float) else float(flow_w.item()), step=step)
                        metrics_tracker.log_metric("w_mde", float(mde_w) if isinstance(mde_w, float) else float(mde_w.item()), step=step)
                        metrics_tracker.log_metric("w_ss", float(ss_w) if isinstance(ss_w, float) else float(ss_w.item()), step=step)
                # clamp patch values
                if (not args.change_of_variables) and args.optimizer not in ["ifgsm", "pgd"]:
                    with torch.no_grad():
                        A.P.clamp_(0, 1)

            # logging артефактов
            if metrics_tracker is not None and args.save_artifacts:
                metrics_tracker.save_artifact(
                    I1_att_def_batch, f"batch_{batch_idx:04d}_attacked_image", artifact_type="image")
                metrics_tracker.save_artifact(
                    I1_batch, f"batch_{batch_idx:04d}_unattacked_image", artifact_type="image")
                metrics_tracker.save_artifact(
                    pred_attacked, f"batch_{batch_idx:04d}_attacked_flow", artifact_type="flow")
                metrics_tracker.save_artifact(
                    pred_unattacked, f"batch_{batch_idx:04d}_unattacked_flow", artifact_type="flow")

                if args.attack_mde and mde_model is not None:
                    metrics_tracker.save_artifact(
                        mde_pred_att, f"batch_{batch_idx:04d}_attacked_depth", artifact_type="depth")
                    metrics_tracker.save_artifact(
                        mde_pred_unatt, f"batch_{batch_idx:04d}_unattacked_depth", artifact_type="depth")
                    metrics_tracker.save_artifact(
                        torch.abs(mde_pred_unatt - mde_pred_att),
                        f"batch_{batch_idx:04d}_depth_diff",
                        artifact_type="depth"
                    )


                if args.attack_ss and ss_model is not None:
                    # можно сохранить logits или argmax
                    ss_pred_att_vis = ss_pred_att.argmax(dim=1)
                    ss_pred_unatt_vis = ss_pred_unatt.argmax(dim=1)
                    metrics_tracker.save_artifact(
                        ss_pred_att_vis, f"batch_{batch_idx:04d}_attacked_ss", artifact_type="ss")
                    metrics_tracker.save_artifact(
                        ss_pred_unatt_vis, f"batch_{batch_idx:04d}_unattacked_ss", artifact_type="ss")
                    metrics_tracker.save_artifact(
                        ss_target_batch, f"batch_{batch_idx:04d}_target_ss", artifact_type="ss")

        # Log epoch-level averages (one value per epoch for clear trend)
        if metrics_tracker is not None and n_batches > 0:
            for key, val in epoch_accum.items():
                avg = float(val) / n_batches
                metrics_tracker.log_metric(f"epoch_avg_{key}", avg, step=epoch)

        # save patch
        A.save_png(op.join(
            args.output_dir, f"patch_{metrics_tracker.run_name}_{epoch+1}.png"))
        if metrics_tracker is not None and args.save_artifacts:
                metrics_tracker.save_artifact(
                    A, f"{epoch+1}_patch", artifact_type="patch")
        print(f"Saved patch for epoch {epoch+1}")

    return A
