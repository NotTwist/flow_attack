import ptlflow.utils.io_adapter as io_adapter_lib
import cv2
import ptlflow.models
import ptlflow.utils
import ptlflow.utils.io_adapter
import torch
import numpy as np
from tqdm import tqdm
from datasets_utils.dataset_utils import prepare_dataloader
from models.model_utils import load_mde_model, load_seg_model
from utils.process_images import preprocess_img, postprocess_flow, get_image_tensors
from metrics.attack_metrics import AttackMetricsTracker
from utils.seed import set_seed
from utils.args import parse_args
from attacks.get_attacks import get_attack
import ptlflow
import cv2 as cv
from utils.targets import get_mde_target

def load_model(model_name, dataset):
    model_ref = ptlflow.get_model_reference(model_name)
    checkpoints = model_ref.pretrained_checkpoints.keys()
    for c in checkpoints:
        if c in dataset:
            model = ptlflow.get_model(model_name, c)
            return model
    print(f"No pre-trained model available for {model}/{dataset}.")
    return None



def main():
    # Parse arguments using the separate args.py file
    args = parse_args()

    # Initialize metrics tracker
    metrics_tracker = AttackMetricsTracker(
        output_dir=args.output_dir, args=args)
    set_seed(42)

    # Prepare data loader
    data_loader, has_gt = prepare_dataloader(
        dataset_name=args.dataset, small_run=args.small_run)

    
    # Отдельный loader только для оценки перцентиля глубины
    depth_loader, _ = prepare_dataloader(
        dataset_name=args.dataset, small_run=args.small_run)
    # Set device (CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Setting Device to {device}\n")

    # Import and load the model
    # Get available checkpoints for a specific model (e.g., RAFT)
    
    model = load_model(args.model_name, args.dataset.lower()).to(device)
    model.eval()
    
    for param in model.parameters():
        param.requires_grad = False
    
    mde_model = None
    depth_pred = None
    original_depth = None
    mde_target = None
    tracked_depths = None
    if args.attack_mde:
        mde_model = load_mde_model(model_name=args.mde_model, device=device)
        # создаём функцию-таргет, которая внутри себя лениво посчитает p90
        mde_target_fn = get_mde_target(
            target_name=args.mde_target,
            data_loader=depth_loader,   # отдельный loader!
            flow_model=model,
            mde_model=mde_model,
            device=device,
            q=0.9
        )

    ss_model = None
    original_ss = None
    ss_pred = None
    ss_target = None
    tracked_ss = None
    if args.attack_ss:
        ss_model = load_seg_model(model_name=args.ss_model, device=device)
        for param in ss_model.parameters():
            param.requires_grad = False
        

    # Set the attack based on argument
    attack = get_attack(args.attack_type, model, mde_model=mde_model, mde_target=mde_target_fn, ss_model=ss_model, ss_target=args.ss_target, num_steps=args.steps, no_softmax=args.no_softmax,
                        target=args.target, epsilon=args.epsilon, save_iterations=args.saved_iterations, alpha=args.alpha, target_layer=args.target_layer, use_map_scaling=args.use_map_scaling, scaling_type=args.scaling_type, num_samples=args.num_samples, loss=args.loss, loss_weights=args.loss_weights)

    # Loop over data batches
    for batch, (images, flow, valid, _, _) in enumerate(tqdm(data_loader)):
        io_adapter = ptlflow.utils.io_adapter.IOAdapter(
            model, input_size=images.shape[-2:], cuda=torch.cuda.is_available())
        wrapped_inputs = {'images': images, 'flows': flow, 'valids': valid}
        inputs = io_adapter.prepare_inputs(inputs=wrapped_inputs)
        # Compute original flow
        import time
        with torch.no_grad():
            original_flow = model(inputs)['flows'].squeeze(0)
            of_target = attack.target(original_flow)
            if args.attack_mde:
                original_depth = mde_model(inputs)
                mde_target = attack.mde_target(original_depth)
            if args.attack_ss:
                original_ss = ss_model(inputs)
                ss_target = attack.ss_target(original_ss)
        torch.cuda.empty_cache()

        # Perform the attack
        attack_result = attack.attack(inputs)

        tracked_flows = attack_result["tracked_flows"]
        
        if args.attack_mde:
            tracked_depths = attack_result["tracked_depths"]
        if args.attack_ss:
            tracked_ss = attack_result["tracked_ss"]
        attacked_images = attack_result["final_images"]
        masks = attack_result.get('tracked_masks', None)
        deltas = torch.clamp(get_image_tensors(attacked_images).detach().cpu(), 0, 1) - images.squeeze(0)
        with torch.no_grad():
            flow_pred = model(attacked_images)['flows'].squeeze(0)
            if args.attack_mde:
                depth_pred = mde_model(attacked_images)
            if args.attack_ss:
                ss_pred = ss_model(attacked_images)

        inverse_flow = None
        # Update metrics
        metrics_tracker.update(original_flow, flow_pred,
                               gt_flow=flow, target_flow=of_target, inverse_flow=inverse_flow, valid=valid, tracked_flows=tracked_flows, original_depth=original_depth, attacked_depth = depth_pred, target_depth=mde_target, tracked_depths = tracked_depths, original_seg=original_ss, attacked_seg=ss_pred, target_seg=ss_target, tracked_segs=tracked_ss)

        # Save artifacts if required
        if args.save_artifacts:
            if masks is not None:
                for step in masks:
                    metrics_tracker.save_artifact(
                        masks[step], f"batch_{batch:04d}_mask_step_{step}", artifact_type="image")
            metrics_tracker.save_artifact(
                get_image_tensors(inputs)[0], f"batch_{batch:04d}_attacked_image", artifact_type="image")
            metrics_tracker.save_artifact(
                flow_pred, f"batch_{batch:04d}_attacked_flow", artifact_type="flow")
            metrics_tracker.save_artifact(
                original_flow, f"batch_{batch:04d}_init_flow", artifact_type="flow")
            metrics_tracker.save_artifact(
                deltas[0], f"batch_{batch:04d}_delta", artifact_type="image")
            if args.attack_mde:
                metrics_tracker.save_artifact(
                    original_depth, f"batch_{batch:04d}_init_depth", artifact_type="depth")
                metrics_tracker.save_artifact(
                    depth_pred, f"batch_{batch:04d}_attacked_depth", artifact_type="depth")
            if args.attack_ss:
                metrics_tracker.save_artifact(
                    original_ss, f"batch_{batch:04d}_init_ss", artifact_type="ss")
                metrics_tracker.save_artifact(
                    ss_pred, f"batch_{batch:04d}_attacked_ss", artifact_type="ss")

    metrics_tracker.finalize()


if __name__ == '__main__':
    main()
