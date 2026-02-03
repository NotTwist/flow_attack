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
        dataset_name=args.dataset, small_run=args.small_run)

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

    TF_flow = None
    TF_mde = None
    TF_ss = None

    temporal_modes = ["temporal-avg", "temporal-median", "temporal-bilateral", "temporal-domain-transform"]
    if args.defense in temporal_modes:
        mode = temporal_mode_map[args.defense]
        # Создаем фильтры (параметры можно брать общие или разные из args)
        TF_flow = TemporalPredictionFilter(mode=mode, window_size=args.temp_window, sigma_color=args.sigma_color).to(device)
        TF_mde = TemporalPredictionFilter(mode=mode, window_size=args.temp_window, sigma_color=args.sigma_color).to(device)
        TF_ss = TemporalPredictionFilter(mode=mode, window_size=args.temp_window, sigma_color=args.sigma_color).to(device)
    # targets & losses for optical flow
    flow_target_fn = get_target(args.target)
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
        ss_loss_fn = get_ss_loss(untargeted=args.ss_target == 'untargeted')
        ss_target_fn = get_ss_target(args.ss_target)
    else:
        ss_loss_fn = None
        ss_target_fn = None

    flow_w, mde_w, ss_w = args.loss_weights

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

    for epoch in range(args.n):
        print(f"Epoch {epoch+1}/{args.n}")
        if TF_flow is not None:
            TF_flow.reset()
            TF_mde.reset()
            TF_ss.reset()
        for batch_idx, (images, flow_gt, valid, meta, K) in enumerate(tqdm(data_loader)):
            images = images.to(device)          # [B,2,C,H,W]
            flow_gt = flow_gt.to(device)

            if isinstance(valid, torch.Tensor):
                valid = valid.to(device)

            B = images.shape[0]

            for inner_step in range(args.steps):
                optimizer.zero_grad()

                I1_batch = images[:, 0, :, :, :]  # [B,C,H_img,W_img]
                I2_batch = images[:, 1, :, :, :]

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
                    )
                else:
                    I1_p_batch, I2_p_batch, M_batch, ys_batch, xs_batch, road_mask, planes = \
                    A(I1_batch, I2_batch), None, None

                # --- defense ---
                if D is not None:
                    I1_att_def_batch, I2_att_def_batch = D(
                        I1_p_batch, I2_p_batch, M_batch)
                    I1_unatt_def_batch, I2_unatt_def_batch = D(
                        I1_batch, I2_batch, torch.zeros_like(M_batch))
                else:
                    I1_att_def_batch, I2_att_def_batch = I1_p_batch, I2_p_batch
                    I1_unatt_def_batch, I2_unatt_def_batch = I1_batch, I2_batch

                attacked_images_tensor = torch.stack(
                    [I1_att_def_batch, I2_att_def_batch], dim=1)
                unattacked_images_tensor = torch.stack(
                    [I1_unatt_def_batch, I2_unatt_def_batch], dim=1)

                attacked_inputs = prepare_inputs_safe(
                    attacked_images_tensor, flows_tensor=flow_gt, valids_tensor=valid)
                unattacked_inputs = prepare_inputs_safe(
                    unattacked_images_tensor, flows_tensor=flow_gt, valids_tensor=valid)

                # --- forward through flow model ---
                pred_attacked = model(attacked_inputs)['flows'].squeeze(0)
                pred_unattacked = model(unattacked_inputs)['flows'].squeeze(0)

                if TF_flow is not None:
                    pred_attacked = TF_flow(pred_attacked, current_image=I1_att_def_batch)

                H_flow, W_flow = pred_unattacked.shape[-2:]
                B_flow = pred_attacked.shape[0]

                # # --- построение маски "человека" в координатах flow ---
                # if ys_batch is not None and xs_batch is not None:
                #     # координаты центра патча в координатах исходного изображения
                #     img_H, img_W = I1_batch.shape[-2:]
                #     scale_x = W_flow / float(img_W)
                #     scale_y = H_flow / float(img_H)

                #     xs_scaled = xs_batch * scale_x
                #     ys_scaled = ys_batch * scale_y

                #     patch_size_flow = max(1, int(args.patch_size * scale_y))

                #     ped_mask = build_pedestrian_like_mask(
                #         xs_scaled,
                #         ys_scaled,
                #         patch_size=patch_size_flow,
                #         pred_shape=pred_attacked.shape,
                #         device=device,
                #         width_factor=getattr(args, "ped_width_factor", 0.5),
                #         height_factor=getattr(args, "ped_height_factor", 1.5),
                #     )
                # else:
                #     ped_mask = torch.zeros(
                #         (B_flow, 1, H_flow, W_flow), device=device, dtype=torch.float32
                #     )

                if M_batch is not None and M_batch.dim() == 4 and M_batch.shape[1] == 1:
                    M_flow_patch = torch.nn.functional.interpolate(
                        M_batch, size=(H_flow, W_flow), mode='nearest'
                    )
                else:
                    M_flow_patch = torch.ones(
                        (B_flow, 1, H_flow, W_flow), device=device, dtype=torch.float32
                    )

                # маска атаки: "человек над патчем"
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
                    mde_pred_unatt = mde_model(unattacked_inputs)

                    if TF_mde is not None:
                        mde_pred_att = TF_mde(mde_pred_att, current_image=I1_att_def_batch)

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
                    ss_pred_unatt = ss_model(
                        unattacked_inputs, return_logits=True)
                    
                    if TF_ss is not None:
                        ss_pred_att = TF_ss(ss_pred_att, current_image=I1_att_def_batch)
                    
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
                    with torch.no_grad():
                        ss_prob_unatt = torch.softmax(ss_pred_unatt, dim=1)
                    ss_prob_att = torch.softmax(ss_pred_att, dim=1)

                    ss_cons_loss = masked_mse_consistency(
                        ss_prob_att, ss_prob_unatt, M_outside_ss
                    )
                else:
                    ss_adv_loss = torch.tensor(0.0, device=device)
                    ss_cons_loss = torch.tensor(0.0, device=device)

                # --- TV loss ---
                tv_w = getattr(args, "tv_weight", 0.0)
                tv_loss = total_variation_loss(A.P, tv_w)

                # --- Printability (NPS) loss ---
                nps_w = getattr(args, "nps_weight", 0.0)
                nps_loss = printability_loss(A.P, nps_w)


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

                step = epoch * len(data_loader) * args.steps + \
                    batch_idx * args.steps + inner_step

                if metrics_tracker is not None:
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

        # save patch
        A.save_png(op.join(
            args.output_dir, f"patch_{metrics_tracker.run_name}_{epoch+1}.png"))
        if metrics_tracker is not None and args.save_artifacts:
                metrics_tracker.save_artifact(
                    A, f"{epoch+1}_patch", artifact_type="patch")
        print(f"Saved patch for epoch {epoch+1}")

    return A
