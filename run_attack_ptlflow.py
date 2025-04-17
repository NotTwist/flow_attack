import ptlflow.models
import ptlflow.utils
import ptlflow.utils.io_adapter
import torch
import numpy as np
from tqdm import tqdm
from datasets_utils.dataset_utils import prepare_dataloader
from models.model_utils import import_and_load, compute_flow, model_takes_unit_input
from utils.process_images import preprocess_img, postprocess_flow, get_image_tensors
from metrics.attack_metrics import AttackMetricsTracker
from utils.seed import set_seed
from utils.args import parse_args
from attacks.get_attacks import get_attack
import ptlflow
import cv2 as cv


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

    # Set device (CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Setting Device to {device}\n")

    # Import and load the model
    # Get available checkpoints for a specific model (e.g., RAFT)
    model = load_model(args.model_name, args.dataset.lower()).to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    # Set the attack based on argument
    attack = get_attack(args.attack_type, model, num_steps=args.steps, no_softmax=args.no_softmax,
                        target=args.target, epsilon=args.epsilon, save_iterations=args.saved_iterations, alpha=args.alpha, target_layer=args.target_layer, use_map_scaling=args.use_map_scaling, scaling_type=args.scaling_type)

    # Loop over data batches
    for batch, (images, flow, valid) in enumerate(tqdm(data_loader)):
        io_adapter = ptlflow.utils.io_adapter.IOAdapter(
            model, input_size=images.shape[-2:], cuda=torch.cuda.is_available())
        wrapped_inputs = {'images': images, 'flows': flow, 'valids': valid}
        inputs = io_adapter.prepare_inputs(inputs=wrapped_inputs)
        # Compute original flow
        import time
        with torch.no_grad():
            original_flow = model(inputs)['flows'].squeeze(0)
        torch.cuda.empty_cache()

        # Perform the attack
        attack_result = attack.attack(inputs)

        tracked_flows = attack_result["tracked_flows"]
        attacked_images = attack_result["final_images"]
        masks = attack_result.get('tracked_masks', None)
        deltas = torch.clamp(get_image_tensors(attacked_images).detach().cpu(), 0, 1) - images.squeeze(0)
        with torch.no_grad():
            flow_pred = model(attacked_images)['flows'].squeeze(0)

        inverse_flow = None
        # Update metrics
        metrics_tracker.update(original_flow, flow_pred,
                               gt_flow=flow, target_flow=attack.target(original_flow), inverse_flow=inverse_flow, valid=valid, tracked_flows=tracked_flows)

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

    metrics_tracker.finalize()


if __name__ == '__main__':
    main()
