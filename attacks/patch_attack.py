from metrics.attack_metrics import AttackMetricsTracker
from utils.targets import get_target, get_mde_target, get_ss_target
from utils.losses import get_mde_loss, get_ss_loss, get_loss
from os import path as op
import os
from tqdm import tqdm
import torch
import torch.optim as optim
from .patch_projection import fit_plane_from_depth, keep_largest_component
from .adversarial_manhole.adv_manhole.texture_mapping.depth_utils import disp_to_depth
from .DetectionDefenses.helper_functions.defenses import LGS, ILP
from .DetectionDefenses.helper_functions.losses import aee_masked, acs_masked, mse_masked
from .DetectionDefenses.helper_functions.ownutilities import preprocess_img as oa_preprocess_img, postprocess_flow as oa_postprocess_flow
from .DetectionDefenses.helper_functions.custom_optimizer import IFGSM, ClippedPGD
from .DetectionDefenses.helper_functions.patch_adversary import PatchAdversary
import sys
import pathlib
from datasets_utils.dataset_utils import prepare_dataloader
sys.path.append(str(pathlib.Path(__file__).resolve().parent))


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
    который стоит над патчем.

    Прямоугольник:
      - центр по X совпадает с центром патча,
      - нижняя граница по Y == центр патча,
      - верхняя граница выше на patch_size * height_factor.

    xs_batch, ys_batch: центры патча (то, что возвращает PatchAdversary),
                        могут быть скалярами, списком или тензорами.
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

        # можно немного "вылезать" из патча, но не из изображения
        x0 = max(0, min(W, x0))
        x1 = max(0, min(W, x1))
        y0 = max(0, min(H, y0))
        y1 = max(0, min(H, y1))

        if x1 > x0 and y1 > y0:
            ped_mask[b, 0, y0:y1, x0:x1] = 1.0

    return ped_mask



def train_patch_ptlflow(args, model, data_loader, device, io_adapter, metrics_tracker: AttackMetricsTracker = None, mde_model=None, ss_model=None):
    """
    Train one universal patch across the whole dataset (PTLFlow-style).
    - A single PatchAdversary A and one optimizer are created.
    - Run args.n epochs; each epoch = full pass over data_loader.
    - For each batch: perform args.steps internal optimization steps.
    Supports joint optimization against optical-flow model `model` and optional `mde_model`.
    """
    import torch
    import numpy as np
    from itertools import islice

    # create adversary once
    A = PatchAdversary(None, size=args.patch_size,
                       angle=[-10, 10], scale=[0.95, 1.05],
                       change_of_variable=args.change_of_variables,
                       random_location=args.random_loc, image_size=(375, 1242)).to(device)
    
    depth_loader, _ = prepare_dataloader(
        dataset_name=args.dataset, small_run=args.small_run)

    # with torch.no_grad():
    #     A.P[:] = 0
    #     A.P[:, 0, :, :] = 1.0

    # optimizer
    if args.optimizer == "adam":
        optimizer = optim.Adam(A.parameters(), lr=args.lr)
    elif args.optimizer == "sgd":
        optimizer = optim.SGD(A.parameters(), lr=args.lr, momentum=0.9)
    elif args.optimizer == "clipped-pgd":
        from .DetectionDefenses.helper_functions.custom_optimizer import ClippedPGD
        optimizer = ClippedPGD(A.parameters(), lr=args.lr,
                               min_=0, max_=1, max_delta=args.max_delta)
    elif args.optimizer == "ifgsm":
        if args.change_of_variables:
            # if change of variables is used, the range of the patch is [-100,100] and not [0,1] because the patch is scaled to [0,1] before the forward pass
            optimizer = IFGSM(A.parameters(), lr=args.lr, min_=-100, max_=100)
        else:
            optimizer = IFGSM(A.parameters(), lr=args.lr, min_=0, max_=1)

    # defense
    D = None
    if args.defense == "lgs":
        D = LGS(args.k, args.o, args.t, args.s, "forward")
    elif args.defense == "ilp":
        D = ILP(args.k, args.o, args.t, args.s, args.r, "forward")

    # targets & losses
    flow_target_fn = get_target(args.target)
    flow_loss_fn = get_loss(args.loss, untargeted=args.target == 'untargeted')
    # get_mde_loss may accept args.loss or a separate arg name; try both
    if args.attack_mde:
        mde_loss_fn = get_mde_loss(untargeted=args.mde_target == 'untargeted')
        mde_target_fn = get_mde_target(
            target_name=args.mde_target,
            data_loader=depth_loader,   # отдельный loader!
            flow_model=model,
            mde_model=mde_model,
            device=device,
            q=0.9
        )

    if args.attack_mde:
        ss_loss_fn = get_ss_loss(untargeted=args.ss_target == 'untargeted')
        ss_target_fn = get_ss_target(args.ss_target)

    flow_w, mde_w, ss_w = args.loss_weights
    # Convenience helper: safe call to io_adapter.prepare_inputs with various accepted formats

    def prepare_inputs_safe(images_tensor, flows_tensor=None, valids_tensor=None):
        """Try passing batched tensors, if fails convert to list-of-numpy per-sample and retry."""
        try:
            # try direct call (most efficient)
            return io_adapter.prepare_inputs(inputs={'images': images_tensor, 'flows': flows_tensor, 'valids': valids_tensor})
        except Exception:
            # convert to list of numpy per sample (io_adapter.transform often accepts list)
            imgs = []
            Bs = images_tensor.shape[0]
            for bi in range(Bs):
                # convert [2,C,H,W] -> HWC list because many wrappers expect HWC arrays or list-of-images
                arr = images_tensor[bi].detach().cpu().numpy()  # [2,C,H,W]
                # transform to list of numpy images with HWC per-frame if needed
                # prefer shape (2,H,W,C) list-of-arrays
                imgs_sample = []
                for t in range(arr.shape[0]):
                    frame = arr[t].transpose(1, 2, 0)  # C,H,W -> H,W,C
                    imgs_sample.append(frame)
                imgs.append(imgs_sample)
            # flows/valids similar: pass None or list
            flows_list = None
            valids_list = None
            if flows_tensor is not None:
                flows_list = [flows_tensor[b].detach().cpu().numpy()
                              for b in range(flows_tensor.shape[0])]
            if valids_tensor is not None:
                valids_list = [valids_tensor[b].detach().cpu().numpy()
                               for b in range(valids_tensor.shape[0])]
            return io_adapter.prepare_inputs(inputs={'images': imgs, 'flows': flows_list, 'valids': valids_list})

    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    if mde_model is not None:
        mde_model.eval()
        for param in mde_model.parameters():
            param.requires_grad = False

    if ss_model is not None:
        ss_model.eval()
        for param in ss_model.parameters():
            param.requires_grad = False

    mapping = None
    # if args.patch_projection:

    #     mapping = DepthTextureMapping(
    #         texture_res=100,
    #         tex_scale=0.5,
    #         tex_offset=[0.0, 0.0],
    #         random_scale=(0, 0),
    #         random_shift_x=(0, 0),
    #         random_shift_y=(0, 0),
    #         with_circle_mask=False,
    #         device=device,
    #     )
    # main loops
    for epoch in range(args.n):
        print(f"Epoch {epoch+1}/{args.n}")
        for batch_idx, (images, flow_gt, valid, meta, K) in enumerate(tqdm(data_loader)):
            # expected shape [B, 2, C, H, W]
            images = images.to(device)
            flow_gt = flow_gt.to(device)

            if isinstance(valid, torch.Tensor):
                valid = valid.to(device)

            B = images.shape[0]

            for inner_step in range(args.steps):
                optimizer.zero_grad()
                I1_batch = images[:, 0, :, :, :]  # [B,C,H,W]
                I2_batch = images[:, 1, :, :, :]
                if args.patch_projection:
                    K_b = K
                    if isinstance(K_b, dict):
                        fx = K_b["fx"]
                        fy = K_b["fy"]
                        cx = K_b["cx"]
                        cy = K_b["cy"]
                        K_mat = torch.tensor([[fx, 0., cx],
                                              [0., fy, cy],
                                              [0., 0., 1.]], device=device)
                    else:
                        K_mat = K_b.to(device)
                    unattacked_images_tensor = torch.stack(
                        [I1_batch, I2_batch], dim=1)
                    unattacked_inputs = prepare_inputs_safe(
                        unattacked_images_tensor, flows_tensor=flow_gt, valids_tensor=valid)
                    if mde_model is not None:
                        with torch.no_grad():
                            mde_out_unatt = mde_model(unattacked_inputs)

                        if mde_out_unatt is not None:
                            depth_pred = mde_out_unatt
                            if depth_pred.dim() == 2:
                                depth_pred = depth_pred.unsqueeze(
                                    0).unsqueeze(0)
                            elif depth_pred.dim() == 3:
                                depth_pred = depth_pred.unsqueeze(1)
                            # invert depth predictions: close to camera - low, far from camera - high
                            depth_pred = disp_to_depth(depth_pred.to(device).float())
                    if ss_model is not None:
                        with torch.no_grad():
                            ss_logits = ss_model(
                                unattacked_inputs, return_logits=True)
                        ss_pred = ss_logits.argmax(
                            dim=1)                            # [B,H,W]

                        ROAD_CLASS_ID = 0   
                        road_mask = (ss_pred == ROAD_CLASS_ID)
                        road_mask = road_mask.unsqueeze(
                            1)            
                        road_mask = keep_largest_component(road_mask)               
                    else:
                        road_mask = None
                    planes = fit_plane_from_depth(depth_pred, K, road_mask)

                    ys_batch = xs_batch = None
                    I1_p_batch, I2_p_batch, M_batch, ys_batch, xs_batch = A(
                        I1_batch, I2_batch,
                        K=K,
                        planes=planes,
                        road_masks=road_mask,
                    )
                else:
                    I1_p_batch, I2_p_batch, M_batch, ys_batch, xs_batch = A(
                        I1_batch, I2_batch)

                # apply defense (batched)
                if D is not None:
                    I1_att_def_batch, I2_att_def_batch = D(
                        I1_p_batch, I2_p_batch, M_batch)
                    I1_unatt_def_batch, I2_unatt_def_batch = D(
                        I1_batch, I2_batch, torch.zeros_like(M_batch))
                else:
                    I1_att_def_batch, I2_att_def_batch = I1_p_batch, I2_p_batch
                    I1_unatt_def_batch, I2_unatt_def_batch = I1_batch, I2_batch

                # stack [B,2,C,H,W]
                attacked_images_tensor = torch.stack(
                    [I1_att_def_batch, I2_att_def_batch], dim=1)
                unattacked_images_tensor = torch.stack(
                    [I1_unatt_def_batch, I2_unatt_def_batch], dim=1)

                # prepare inputs (safe)
                attacked_inputs = prepare_inputs_safe(
                    attacked_images_tensor, flows_tensor=flow_gt, valids_tensor=valid)
                unattacked_inputs = prepare_inputs_safe(
                    unattacked_images_tensor, flows_tensor=flow_gt, valids_tensor=valid)

                # forward pass (batched)
                pred_attacked = model(attacked_inputs)['flows'].squeeze(0)
                pred_unattacked = model(unattacked_inputs)['flows'].squeeze(0)
                H, W = pred_unattacked.shape[-2:]
                B = pred_attacked.shape[0]

                # маска "человека" над патчем
                if ys_batch is not None and xs_batch is not None:
                    ped_mask = build_pedestrian_like_mask(
                        xs_batch,
                        ys_batch,
                        patch_size=args.patch_size,
                        pred_shape=pred_attacked.shape,
                        device=device,
                        width_factor=getattr(args, "ped_width_factor", 0.2),
                        height_factor=getattr(args, "ped_height_factor", 1.5),
                    )
                else:
                    ped_mask = torch.zeros(
                        (B, 1, H, W), device=device, dtype=torch.float32
                    )

                # target = flow_target_fn(pred_unattacked).to(device)
                if args.target == 'scene':
                    target = flow_target_fn(
                        pred_unattacked, xs_batch, ys_batch+args.patch_size//2)
                else:
                    target = flow_target_fn(pred_unattacked)
                target = target.to(device)

                # M -> resize to flow spatial size if needed
                if M_batch is not None and M_batch.dim() == 4 and M_batch.shape[1] == 1:
                    M_flow = torch.nn.functional.interpolate(
                        M_batch, size=pred_attacked.shape[-2:], mode='nearest')
                else:
                    M_flow = M_batch
                M_flow = ped_mask
                save_dir = os.path.join(args.output_dir, "masks")

                os.makedirs(save_dir, exist_ok=True)
                from PIL import Image
                for b in range(B):
                    mask_np = (ped_mask[b, 0].detach(
                    ).cpu().numpy() * 255).astype(np.uint8)
                    mask_img = Image.fromarray(mask_np, mode="L")

                    mask_img.save(
                        os.path.join(
                            save_dir,
                            f"mask_epoch{epoch:02d}_batch{batch_idx:04d}_step{inner_step:03d}_b{b}.png"
                        )
                    )
                flow_loss_batch = flow_loss_fn(pred_attacked, target, M_flow)

                # MDE branch
                if mde_model is not None:
                    mde_pred_att = mde_model(attacked_inputs)
                    mde_pred_unatt = mde_model(unattacked_inputs)

                    if args.mde_target == 'scene':
                        mde_target_batch = mde_target_fn(
                            mde_pred_unatt, xs_batch, ys_batch+args.patch_size//2)
                    else:
                        mde_target_batch = mde_target_fn(mde_pred_unatt)

                    mde_target_batch = mde_target_batch.to(device)
                    mde_loss_batch = mde_loss_fn(
                        mde_pred_att, mde_target_batch, M_flow)

                    # except Exception:
                    #     mde_acc = 0.0
                    #     for b in range(B):
                    #         ma = mde_pred_att[b:b+1]
                    #         mt = mde_target_batch[b:b+1]
                    #         mde_acc = mde_acc + mde_loss_fn(ma, mt)
                    #     mde_loss_batch = mde_acc / float(B)
                else:
                    mde_loss_batch = torch.tensor(0.0, device=device)

                if ss_model is not None:
                    ss_pred_att = ss_model(attacked_inputs, return_logits=True)
                    ss_pred_unatt = ss_model(
                        unattacked_inputs,  return_logits=True)
                    ss_target_batch = ss_target_fn(ss_pred_unatt).to(device)
                    ss_loss_batch = ss_loss_fn(
                        ss_pred_att, ss_target_batch, M_flow)
                else:
                    ss_loss_batch = torch.tensor(0.0, device=device)

                # backward grad check for patch
                optimizer.zero_grad()
                total_loss = flow_w * flow_loss_batch + \
                    mde_w * mde_loss_batch + ss_w * ss_loss_batch
                # временно удерживаем граф для отладки
                total_loss.backward(retain_graph=True)
                optimizer.step()
                step = epoch * len(data_loader) * args.steps + \
                    batch_idx * args.steps + inner_step
                metrics_tracker.log_metric("epoch_batch_loss", float(
                    total_loss.item()), step=step)
                metrics_tracker.log_metric("epoch_batch_of_loss", float(
                    flow_loss_batch.item()), step=step)
                metrics_tracker.log_metric("epoch_batch_mde_loss", float(
                    mde_loss_batch.item()), step=step)
                metrics_tracker.log_metric("epoch_batch_ss_loss", float(
                    ss_loss_batch.item()), step=step)

                # clamp patch values
                if not args.change_of_variables and args.optimizer not in ["ifgsm", "pgd"]:
                    with torch.no_grad():
                        A.P.clamp_(0, 1)

            # logging
            if metrics_tracker is not None and args.save_artifacts:
                # try:

                metrics_tracker.save_artifact(
                    I1_att_def_batch, f"batch_{batch_idx:04d}_attacked_image", artifact_type="image")
                metrics_tracker.save_artifact(
                    I1_batch, f"batch_{batch_idx:04d}_unattacked_image", artifact_type="image")
                metrics_tracker.save_artifact(
                    pred_attacked, f"batch_{batch_idx:04d}_attacked_flow", artifact_type="flow")
                metrics_tracker.save_artifact(
                    pred_unattacked, f"batch_{batch_idx:04d}_unattacked_flow", artifact_type="flow")

                if args.attack_mde:
                    metrics_tracker.save_artifact(
                        mde_pred_att, f"batch_{batch_idx:04d}_attacked_depth", artifact_type="depth")
                    metrics_tracker.save_artifact(
                        mde_pred_unatt, f"batch_{batch_idx:04d}_unattacked_depth", artifact_type="depth")
                    metrics_tracker.save_artifact(
                        torch.abs(mde_pred_unatt-mde_pred_att), f"batch_{batch_idx:04d}_depth_diff", artifact_type="depth")
                if args.attack_ss:
                    ss_pred_att = ss_model(attacked_inputs)
                    ss_pred_unatt = ss_model(
                        unattacked_inputs)
                    metrics_tracker.save_artifact(
                        ss_pred_att, f"batch_{batch_idx:04d}_attacked_ss", artifact_type="ss")
                    metrics_tracker.save_artifact(
                        ss_pred_unatt, f"batch_{batch_idx:04d}_unattacked_ss", artifact_type="ss")
                    metrics_tracker.save_artifact(
                        ss_target_batch, f"batch_{batch_idx:04d}_target_ss", artifact_type="ss")
                    # except Exception:
                #     pass

        # save patch at epoch end
        A.save_png(op.join(
            args.output_dir, f"patch_{metrics_tracker.run_name}_{epoch+1}.png"))
        print(f"Saved patch for epoch {epoch+1}")

    return A
