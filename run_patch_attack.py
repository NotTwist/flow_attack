import ptlflow.utils.io_adapter as io_adapter_lib
import cv2
import os
import ptlflow.models
import ptlflow.utils
import ptlflow.utils.io_adapter
import torch
import numpy as np
from tqdm import tqdm
from datasets_utils.dataset_utils import prepare_dataloader
from models.model_utils import load_mde_model, load_seg_model
from utils.process_images import preprocess_img, postprocess_flow, get_image_tensors, replace_images_dic
from metrics.attack_metrics import AttackMetricsTracker
from utils.seed import set_seed
from utils.args import parse_args
from attacks.get_attacks import get_attack
# import your patch training function
from attacks.patch_attack import train_patch_ptlflow
import ptlflow
import cv2 as cv2
from utils.targets import get_target, get_mde_target, get_ss_target
from attacks.DetectionDefenses.helper_functions.patch_adversary import PatchAdversary, circ_mask
from attacks.patch_projection import (
    project_patch_on_scene,
    fit_plane_from_depth,
    keep_largest_component,
)
from attacks.adversarial_manhole.adv_manhole.texture_mapping.depth_utils import disp_to_depth
from defenses.temporal_filter import init_temporal_filters, reset_temporal_filters, apply_temporal_filters

def load_model(model_name, dataset):
    model_ref = ptlflow.get_model_reference(model_name)
    checkpoints = model_ref.pretrained_checkpoints.keys()
    
    for c in checkpoints:
        if c in dataset:
            model = ptlflow.get_model(model_name, c)
            return model
        else:
            print(f"Using checkpoint from other dataset!: {c}")
            model = ptlflow.get_model(model_name, c)
            return model
    print(f"No pre-trained model available for {model}/{dataset}.")
    return None


def _make_baseline_patch(args, image_size, device):
    """Return a fixed (non-optimized) PatchAdversary for baseline evaluation."""
    import os
    import torchvision.transforms.functional as tvf
    from PIL import Image

    size = args.patch_size
    parametrization = getattr(args, 'patch_parametrization', 'pixel')

    if parametrization == 'diffusion':
        base_image_path = getattr(args, 'diffusion_base_image', '') or os.path.join(
            os.path.dirname(__file__),
            'attacks', 'adversarial_manhole', 'adversarial_example', 'full_manhole.png',
        )
        img = Image.open(base_image_path).convert('RGB')
        rgb = tvf.to_tensor(img).unsqueeze(0)
        rgb = torch.nn.functional.interpolate(rgb, size=(size, size), mode='bilinear', align_corners=False)
    else:
        rgb = torch.rand(1, 3, size, size)

    mask = circ_mask(rgb)
    patch_tensor = torch.cat([rgb, mask], dim=1)

    patch = PatchAdversary(
        patch_tensor,
        size=size,
        angle=0,
        scale=1,
        change_of_variable=args.change_of_variables,
        random_location=args.random_loc,
        image_size=image_size,
        ellipse_scale_y=args.y_scale,
    ).to(device)
    patch.P.requires_grad_(False)
    return patch


def main():
    # Parse arguments using the separate args.py file
    args = parse_args()

    args.attack_type = "patch"
    args.effective_flow_loss = (
        "directional_hinge"
        if args.target in ("down", "direction") and getattr(args, "down_loss", "hinge") == "hinge"
        else ",".join(args.loss)
    )
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    set_seed(42)

    # Prepare dataloader
    data_loader, has_gt = prepare_dataloader(mode='training',
                                             dataset_name=args.dataset, small_run=args.small_run,
                                             subset_size=getattr(args, 'subset_size', 0), n_images=1)
    image_size = data_loader.image_size
    # Отдельный loader только для оценки перцентиля глубины (если хотим взять цель атаки, которая зависит от датасета)
    depth_loader, _ = prepare_dataloader(
        dataset_name=args.dataset, small_run=args.small_run,
        subset_size=getattr(args, 'subset_size', 0))
    
    # Load optical flow model
    model = load_model(args.model_name, args.dataset.lower()).to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    # Load MDE model (only when it will be attacked)
    mde_model = None
    of_target_fn = get_target(
        args.target,
        magnitude=args.flow_target_magnitude,
        angle_deg=getattr(args, "flow_target_angle_deg", 90.0),
    )

    if args.attack_mde:
        mde_model = load_mde_model(model_name=args.mde_model, device=device)
        mde_target_fn = get_mde_target(
            target_name=args.mde_target,
            data_loader=depth_loader,
            flow_model=model,
            mde_model=mde_model,
            device=device,
            q=0.9,
            near_margin=getattr(args, "mde_near_margin", 0.1),
        )

    # Load semantic segmentation model (only when it will be attacked)
    ss_model = None
    if args.attack_ss:
        ss_model = load_seg_model(model_name=args.ss_model, device=device)
        for param in ss_model.parameters():
            param.requires_grad = False
        ss_target_fn = get_ss_target('targeted')

    io_adapter = ptlflow.utils.io_adapter.IOAdapter(
        model, input_size=data_loader, cuda=torch.cuda.is_available())
    

    # defences
    temporal_filters = init_temporal_filters(args, device=device)
    reset_temporal_filters(temporal_filters)

    # Train new patch or use patch from args.trained_patch
    if getattr(args, 'baseline', False):
        label = 'random noise' if getattr(args, 'patch_parametrization', 'pixel') == 'pixel' else 'base image'
        print(f"Baseline mode: using fixed {label} patch (no training).")
        trained_patch = _make_baseline_patch(args, image_size, device)
        os.makedirs(args.output_dir, exist_ok=True)
        trained_patch.save_png(os.path.join(args.output_dir, "baseline_patch.png"))
    elif args.trained_patch == '':
        print("Starting patch training...")
        train_tracker = AttackMetricsTracker(
            output_dir=args.output_dir,
            experiment_name=args.experiment_name,
            args=args,
            train=True,
        )
        # train patch on train dataloader
        trained_patch = train_patch_ptlflow(
            args, model, data_loader, device, io_adapter, train_tracker, mde_model, ss_model)
        # Build the evaluation patch in memory. This avoids an unnecessary
        # root-level patch.png write, which is fragile during large sweeps on
        # nearly-full disks.
        patch_tensor = trained_patch.get_P(Mask=True).detach().cpu()
        trained_patch = PatchAdversary(
            patch_tensor,
            size=args.patch_size,
            angle=0,
            scale=1,
            change_of_variable=False,
            random_location=args.random_loc,
            image_size=image_size,
            ellipse_scale_y=args.y_scale,
        ).to(device)
        train_tracker.finalize()
    else:
        print(f"Using trained patch from {args.trained_patch}...")
        trained_patch = PatchAdversary(args.trained_patch, size=args.patch_size,
                                       angle=0, scale=1, change_of_variable=args.change_of_variables,
                                       random_location=args.random_loc, image_size=image_size, ellipse_scale_y=args.y_scale).to(device)

    print("Evaluating trained patch...")
    eval_loader, has_gt = prepare_dataloader(mode=args.eval_mode,
                                             dataset_name=args.dataset,
                                             small_run=args.small_run,
                                             subset_size=getattr(args, 'subset_size', 0),
                                             n_images=1,
                                             has_depth=False)
    eval_tracker = AttackMetricsTracker(
        output_dir=args.output_dir,
        experiment_name=args.experiment_name,
        args=args,
    )
    try:
        eval_tracker.save_artifact(trained_patch, "evaluated_patch", artifact_type="patch")
    except Exception as e:
        print(f"Warning: failed to log evaluated patch to MLflow: {e}")
    
    # Evaluate path on eval dataset
    for batch, (images, flow, valid, meta, K) in enumerate(tqdm(eval_loader)):
        io_adapter = ptlflow.utils.io_adapter.IOAdapter(
            model, input_size=images.shape[-2:], cuda=torch.cuda.is_available()
        )
        wrapped_inputs = {'images': images, 'flows': flow, 'valids': valid}
        inputs = io_adapter.prepare_inputs(inputs=wrapped_inputs)
        images = images.to(device)

        I1_batch = images[:, 0, :, :, :]
        I2_batch = images[:, 1, :, :, :]

        # ---------------------------------------------------------------
        # Pre-compute clean-image predictions for ALL models in one pass.
        # These are used both for metrics (original_*) and for patch
        # projection (precomputed_depth / road_mask / planes), so we only
        # need to run each model once on the clean input.
        # ---------------------------------------------------------------
        original_depth = None
        mde_target     = None
        original_ss    = None
        ss_target      = None

        proj_depth     = None   # metric depth [B,1,H,W] for plane fitting
        proj_road_mask = None   # binary road mask [B,1,H,W]
        proj_planes    = None   # list of (normal, d) per batch element

        with torch.no_grad():
            original_flow = model(inputs)['flows'].squeeze(0)
            of_target = of_target_fn(original_flow)

            if args.attack_mde:
                original_depth = mde_model(inputs)
                mde_target = mde_target_fn(original_depth)
                # Convert raw model output → metric depth for plane fitting
                _d = original_depth
                if _d.dim() == 2:
                    _d = _d.unsqueeze(0).unsqueeze(0)
                elif _d.dim() == 3:
                    _d = _d.unsqueeze(1)
                proj_depth = disp_to_depth(_d.to(device).float())

            if args.attack_ss:
                # Single forward pass: logits used for road mask + argmax for metrics
                _ss_logits = ss_model(inputs, return_logits=True)
                original_ss = _ss_logits.argmax(dim=1)   # [B,H,W] class map
                ss_target = ss_target_fn(_ss_logits)
                # Road mask for patch projection
                proj_road_mask = (_ss_logits.argmax(dim=1) == 0).unsqueeze(1)
                proj_road_mask = keep_largest_component(proj_road_mask)

            if proj_depth is not None:
                proj_planes = fit_plane_from_depth(proj_depth, K, proj_road_mask)

        # Project patch onto the road plane; all expensive per-clean-image
        # work is already done above — project_patch_on_scene only needs to
        # apply the PatchAdversary with the pre-fitted geometry.
        if args.patch_projection:
            attacked_image1, attacked_image2, mask, y, x, road_mask, planes = project_patch_on_scene(
                I1_batch,
                I2_batch,
                K,
                A=trained_patch,
                mde_model=mde_model,
                ss_model=ss_model,
                io_adapter=io_adapter,
                device=device,
                plane_aug=False,    # disable randomness in evaluation
                precomputed_depth=proj_depth,
                precomputed_road_mask=proj_road_mask,
                precomputed_planes=proj_planes,
                flow_shift=args.flow_shift,
                clean_flow=original_flow,
                flow_shift_mode=getattr(args, "flow_shift_mode", "fixed"),
                flow_shift_scale=getattr(args, "flow_shift_scale", 1.0),
                flow_shift_max=getattr(args, "flow_shift_max", 80.0),
            )
        else:
            attacked_image1, attacked_image2, mask, y, x = trained_patch(
                images[:, 0, :, :, :], images[:, 1, :, :, :],
                flow_shift=args.flow_shift,
                clean_flow=original_flow,
                flow_shift_mode=getattr(args, "flow_shift_mode", "fixed"),
                flow_shift_scale=getattr(args, "flow_shift_scale", 1.0),
                flow_shift_max=getattr(args, "flow_shift_max", 80.0),
            )
        attacked_images = torch.stack(
            [attacked_image1, attacked_image2], dim=1).squeeze(0)

        inputs = replace_images_dic(inputs, attacked_images)

        # Get attacked predictions for models
        with torch.no_grad():
            flow_pred = model(inputs)['flows'].squeeze(0)
            flow_pred = apply_temporal_filters(temporal_filters, flow_pred, attacked_image1, 'flow')
            depth_pred = None
            ss_pred = None
            if args.attack_mde:
                depth_pred = mde_model(inputs)
                depth_pred = apply_temporal_filters(temporal_filters, depth_pred, attacked_image1, 'mde')

            if args.attack_ss:
                ss_pred = ss_model(inputs)
                ss_pred = apply_temporal_filters(temporal_filters, ss_pred, attacked_image1, 'ss')


        # update metrics (чистый выход модели — для AEE/multitask; GT только если есть)
        eval_tracker.update(
            original_flow,
            flow_pred,
            gt_flow=flow if has_gt else None,
            target_flow=of_target,
            inverse_flow=None,
            valid=valid,
            # tracked_flows=tracked_flows,
            mask=(mask).cpu(),
            original_depth=original_depth,
            attacked_depth=depth_pred,
            target_depth=mde_target,
            tracked_depths=None,
            original_seg=original_ss, attacked_seg=ss_pred, target_seg=ss_target
        )

        artifact_limit = int(getattr(args, "eval_artifact_limit", 0) or 0)
        should_save_eval_artifacts = (
            (getattr(args, "save_artifacts", False) or getattr(args, "save_diploma_artifacts", False))
            and (artifact_limit <= 0 or batch < artifact_limit)
        )

        if getattr(args, "save_artifacts", False) and should_save_eval_artifacts:
            eval_tracker.save_artifact(
                I1_batch, f"eval_{batch:04d}_clean_image", artifact_type="image")
            eval_tracker.save_artifact(
                attacked_image1, f"eval_{batch:04d}_attacked_image", artifact_type="image")
            eval_tracker.save_artifact(
                original_flow, f"eval_{batch:04d}_clean_flow", artifact_type="flow")
            eval_tracker.save_artifact(
                flow_pred, f"eval_{batch:04d}_attacked_flow", artifact_type="flow")
            eval_tracker.save_artifact(
                of_target, f"eval_{batch:04d}_target_flow", artifact_type="flow")
            if mask is not None:
                eval_tracker.save_artifact(
                    mask, f"eval_{batch:04d}_patch_mask", artifact_type="image")
            if depth_pred is not None:
                eval_tracker.save_artifact(
                    original_depth, f"eval_{batch:04d}_clean_depth", artifact_type="depth")
                eval_tracker.save_artifact(
                    depth_pred, f"eval_{batch:04d}_attacked_depth", artifact_type="depth")
                eval_tracker.save_artifact(
                    torch.abs(original_depth - depth_pred),
                    f"eval_{batch:04d}_depth_diff", artifact_type="depth")
            if args.attack_ss and ss_pred is not None:
                eval_tracker.save_artifact(
                    original_ss, f"eval_{batch:04d}_clean_ss", artifact_type="ss")
                eval_tracker.save_artifact(
                    ss_pred, f"eval_{batch:04d}_attacked_ss", artifact_type="ss")
                eval_tracker.save_artifact(
                    ss_target, f"eval_{batch:04d}_target_ss", artifact_type="ss")

        if getattr(args, "save_diploma_artifacts", False) and should_save_eval_artifacts:
            eval_tracker.save_diploma_artifacts(
                f"eval_{batch:04d}",
                clean_image=I1_batch,
                attacked_image=attacked_image1,
                clean_flow=original_flow,
                attacked_flow=flow_pred,
                mask=mask,
                clean_depth=original_depth,
                attacked_depth=depth_pred,
            )
                
    eval_tracker.finalize()


if __name__ == '__main__':
    main()
