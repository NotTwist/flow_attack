import torch
from tqdm import tqdm
from datasets_utils.dataset_utils import prepare_dataloader
from models.model_utils import import_and_load, compute_flow, model_takes_unit_input
from utils.process_images import preprocess_img, postprocess_flow
from metrics.attack_metrics import AttackMetricsTracker
from utils.seed import set_seed
from utils.args import parse_args
from attacks.get_attacks import get_attack


def main():
    # Parse arguments using the separate args.py file
    parsed_args = parse_args()

    # Initialize metrics tracker
    metrics_tracker = AttackMetricsTracker(
        output_dir=parsed_args.output_dir, args=parsed_args)
    set_seed(42)

    # Prepare data loader
    data_loader, has_gt = prepare_dataloader(
        dataset_name=parsed_args.dataset, small_run=parsed_args.small_run)

    # Set device (CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Setting Device to {device}\n")

    # Import and load the model
    model = import_and_load(parsed_args.net, make_unit_input=not model_takes_unit_input(parsed_args.net),
                            make_scaled_input_model=True, device=device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    # Set the attack based on argument
    attack = get_attack(parsed_args.attack, model, num_steps=parsed_args.steps, no_softmax=parsed_args.no_softmax,
                        target=parsed_args.target, epsilon=parsed_args.epsilon, save_iterations=parsed_args.save_iterations)

    # Loop over data batches
    for batch, (images, flow, valid) in enumerate(tqdm(data_loader)):
        images = images.permute(1, 0, 2, 3, 4)
        images = images / 255.0
        padder, images = preprocess_img(parsed_args.net, images)
        images = images.detach().to(device)
        images.requires_grad = True

        # Compute original flow
        original_flow = compute_flow(model, "scaled_input_model", images)
        [original_flow] = postprocess_flow(
            parsed_args.net, padder, original_flow)

        # Perform the attack
        attack_result = attack.attack(images)
        tracked_flows = attack_result["tracked_flows"]
        attacked_images = attack_result["final_images"]
        flow_pred = compute_flow(model, "scaled_input_model", attacked_images)
        [flow_pred] = postprocess_flow(parsed_args.net, padder, flow_pred)

        for step in tracked_flows:
            tracked_flows[step] = postprocess_flow(parsed_args.net, padder, tracked_flows[step])[0]
        
        inverse_flow = None
        # Update metrics
        metrics_tracker.update(original_flow, flow_pred,
                               gt_flow=flow, target_flow=attack.target(original_flow), inverse_flow=inverse_flow, valid=valid, tracked_flows=tracked_flows)

        # Save artifacts if required
        if parsed_args.save_artifacts:
            metrics_tracker.save_artifact(
                attacked_images[0], f"batch_{batch:04d}_attacked_image", artifact_type="image")
            metrics_tracker.save_artifact(
                flow_pred, f"batch_{batch:04d}_attacked_flow", artifact_type="flow")
            metrics_tracker.save_artifact(
                original_flow, f"batch_{batch:04d}_init_flow", artifact_type="flow")

    metrics_tracker.finalize()


if __name__ == '__main__':
    main()
