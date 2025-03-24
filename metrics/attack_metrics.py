import os
import torch
import numpy as np
import mlflow
import cv2
from typing import Literal
from utils.process_images import quickvis_flow
from datetime import datetime


class AttackMetricsTracker:
    def __init__(self, output_dir="experiment_data", experiment_name="attack_experiment", run_name=None, args=None):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        mlflow.set_experiment(experiment_name)

        if run_name is None:
            current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            run_name = f"{args.net}_{args.attack}_{args.dataset}_{current_date}"

        mlflow.start_run(run_name=run_name)
        print(
            f"Running experiment: {experiment_name}, with run name: {run_name}")

        # Log parameters
        mlflow.log_param("model_name", args.net)
        mlflow.log_param("attack_type", args.attack)
        mlflow.log_param("target", args.target)
        mlflow.log_param("steps", args.steps)
        mlflow.log_param("dataset", args.dataset)
        mlflow.log_param("small_run", args.small_run)
        mlflow.log_param("output_dir", args.output_dir)
        mlflow.log_param("no_softmax", args.no_softmax)
        mlflow.log_param("epsilon", args.epsilon)
        mlflow.log_param("alpha", args.alpha)
        mlflow.log_param("saved_iterations", args.save_iterations)

        # Store save_iterations as a list of steps
        # Ensure it’s a list
        self.save_iterations = args.save_iterations if args.save_iterations else []
        self.reset()

    def reset(self):
        """Reset cumulative metric values and counts."""
        self.cumulative_metrics = {
            "aee_init_attack": 0.0,
            "aee_init_gt": 0.0,
            "aee_attacked_gt": 0.0,
            "gt_diff_error": 0.0,
            "reconstruction_error": 0.0,
            "aee_target_attack": 0.0,
        }
        self.count = 0

        # Dynamically add metric keys for custom iterations
        for step in self.save_iterations:
            self.cumulative_metrics.update({
                f"aee_init_attack_step_{step}": 0.0,
                f"aee_init_gt_step_{step}": 0.0,
                f"aee_attacked_gt_step_{step}": 0.0,
                f"gt_diff_error_step_{step}": 0.0,
                f"reconstruction_error_step_{step}": 0.0,
                f"aee_target_attack_step_{step}": 0.0,
            })

    def compute_aee(self, flow1, flow2, mask=None):
        """
        Compute the Average Endpoint Error (AEE) between two flow fields.
        Args:
            flow1 (torch.Tensor or np.ndarray): Flow field [B,2,H,W] or [2,H,W].
            flow2 (torch.Tensor or np.ndarray): Flow field [B,2,H,W] or [2,H,W].
        Returns:
            float: Average endpoint error.
        """
        if torch.is_tensor(flow1):
            flow1 = flow1.detach().cpu().numpy()
        if torch.is_tensor(flow2):
            flow2 = flow2.detach().cpu().numpy()
        # Compute per-pixel error, assuming flows have shape [2,H,W] or [B,2,H,W]
        diff_squared = (flow1 - flow2)**2
        if mask is not None:
            diff_squared = np.where(mask == 1, diff_squared, 0)
        if diff_squared.ndim == 3:
            # here, dim=0 is the 2-dimension (u and v direction of flow [2,M,N]) , which needs to be added BEFORE taking the square root. To get the length of a flow vector, we need to do sqrt(u_ij^2 + v_ij^2)
            epe = np.mean(np.sqrt(np.sum(diff_squared, axis=0)))
        elif diff_squared.ndim == 4:
            # here, dim=0 is the 2-dimension (u and v direction of flow [b,2,M,N]) , which needs to be added BEFORE taking the square root. To get the length of a flow vector, we need to do sqrt(u_ij^2 + v_ij^2)
            epe = np.mean(np.sqrt(np.sum(diff_squared, axis=1)))
        else:
            raise ValueError("The flow tensors for which the EPE should be computed do not have a valid number of dimensions (either [b,2,M,N] or [2,M,N]). Here: " + str(
                flow1.size()) + " and " + str(flow1.size()))
        return epe

    def compute_reconstruction_error(self, flow, inverse_flow):
        """
        Compute reconstruction error using inverse flow.
        Here we assume a simple L2 error between the original flow and the inverse flow.
        Args:
            flow (torch.Tensor or np.ndarray): Original flow [B,2,H,W] or [2,H,W].
            inverse_flow (torch.Tensor or np.ndarray): Inverse flow [B,2,H,W] or [2,H,W].
        Returns:
            float: Reconstruction error.
        """
        if torch.is_tensor(flow):
            flow = flow.detach().cpu().numpy()
        if torch.is_tensor(inverse_flow):
            inverse_flow = inverse_flow.detach().cpu().numpy()
        error = np.linalg.norm(flow - inverse_flow, axis=0)
        return np.mean(error)


    def update(self, original_flow, attacked_flow, gt_flow=None, target_flow=None, inverse_flow=None, valid=None, tracked_flows=None):
        """Update metrics for the current batch."""
        aee_attack = self.compute_aee(original_flow, attacked_flow)
        aee_attack_target = self.compute_aee(
            attacked_flow, target_flow)

        aee_gt = self.compute_aee(
            original_flow, gt_flow, valid) if gt_flow is not None else 0.0
        aee_attacked_gt = self.compute_aee(
            attacked_flow, gt_flow, valid) if gt_flow is not None else 0.0
        diff_error = aee_attacked_gt - aee_gt if gt_flow is not None else 0.0
        rec_error = self.compute_reconstruction_error(
            original_flow, inverse_flow) if inverse_flow is not None else 0.0

        # Update cumulative metrics
        self.cumulative_metrics["aee_init_attack"] += aee_attack
        self.cumulative_metrics["aee_init_gt"] += aee_gt
        self.cumulative_metrics["aee_attacked_gt"] += aee_attacked_gt
        self.cumulative_metrics["gt_diff_error"] += diff_error
        self.cumulative_metrics["reconstruction_error"] += rec_error
        self.cumulative_metrics["aee_target_attack"] += aee_attack_target
        self.count += 1

        # Log main metrics
        mlflow.log_metric("aee_init_attack", aee_attack, step=self.count)
        mlflow.log_metric("aee_target_attack",
                          aee_attack_target, step=self.count)

        if gt_flow is not None:
            mlflow.log_metric("aee_init_gt", aee_gt, step=self.count)
            mlflow.log_metric("aee_attacked_gt",
                              aee_attacked_gt, step=self.count)
            mlflow.log_metric("gt_diff_error", diff_error, step=self.count)
        if inverse_flow is not None:
            mlflow.log_metric("reconstruction_error",
                              rec_error, step=self.count)

        # Process dynamically defined iterations
        if self.save_iterations:
            for step in self.save_iterations:
                attacked_flow = tracked_flows[step]
                aee_attack = self.compute_aee(original_flow, attacked_flow)
                aee_attack_target = self.compute_aee(
                    attacked_flow, target_flow)

                aee_gt = self.compute_aee(
                    original_flow, gt_flow, valid) if gt_flow is not None else 0.0
                aee_attacked_gt = self.compute_aee(
                    attacked_flow, gt_flow, valid) if gt_flow is not None else 0.0
                diff_error = aee_attacked_gt - aee_gt if gt_flow is not None else 0.0
                rec_error = self.compute_reconstruction_error(
                    original_flow, inverse_flow) if inverse_flow is not None else 0.0

                mlflow.log_metric(
                    f"aee_init_attack_step_{step}", aee_attack, step=self.count)
                mlflow.log_metric(
                    f"aee_target_attack_step_{step}", aee_attack_target, step=self.count)
                if gt_flow is not None:
                    mlflow.log_metric(
                        f"aee_gt_step_{step}", aee_gt, step=self.count)
                    mlflow.log_metric(
                        f"aee_attacked_gt_step_{step}", aee_attacked_gt, step=self.count)
                    mlflow.log_metric(
                        f"gt_diff_error_step_{step}", diff_error, step=self.count)
                if inverse_flow is not None:
                    mlflow.log_metric(
                        f"reconstruction_error_step_{step}", rec_error, step=self.count)

                # Update cumulative metrics dynamically
                self.cumulative_metrics[f"aee_init_attack_step_{step}"] += aee_attack
                self.cumulative_metrics[f"aee_init_gt_step_{step}"] += aee_gt
                self.cumulative_metrics[f"aee_attacked_gt_step_{step}"] += aee_attacked_gt
                self.cumulative_metrics[f"gt_diff_error_step_{step}"] += diff_error
                self.cumulative_metrics[f"reconstruction_error_step_{step}"] += rec_error
                self.cumulative_metrics[f"aee_target_attack_step_{step}"] += aee_attack_target

        self.save_mean_metrics(step=self.count)
    def get_mean_metrics(self):
        """
        Compute mean metrics over all batches.
        Returns:
            dict: Mean metrics.
        """
        if self.count == 0:
            return {}
        return {k: v / self.count for k, v in self.cumulative_metrics.items()}

    def save_mean_metrics(self, step):
        mean_metrics = self.get_mean_metrics()
        if not mean_metrics:
            print("No metrics to save.")

        for metric, value in mean_metrics.items():
            mlflow.log_metric(f"mean_{metric}", value, step)

    def save_image(self, image, filename):
        if torch.is_tensor(image):
            image = image.detach().cpu().numpy().squeeze(0)
            if image.ndim == 3:
                image = np.transpose(image, (1, 2, 0))
        image = np.array(image)

        # convert to bgr
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Check the min and max values
        min_val, max_val = image.min(), image.max()

        if min_val >= 0.0 and max_val <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)

        return cv2.imwrite(filename, image)

    def save_artifact(self, artifact, name, artifact_type: Literal["image", "flow", "tensor"] = "image"):
        """
        Save an artifact (e.g., flow tensor, image, attack noise) and log it with mlflow.
        Args:
            artifact: The artifact to save (torch.Tensor or np.ndarray).
            name (str): Name identifier for the artifact.
            artifact_type (str): 'image' or 'flow' etc.
        """
        artifact_path = os.path.join(self.output_dir, f"{name}.png")
        # Convert tensor to numpy if needed.

        if artifact_type == "image":
            success = self.save_image(artifact, artifact_path)
            if not success:
                raise RuntimeError(
                    f"Failed to write artifact image to {artifact_path}")
        elif artifact_type == "tensor":
            artifact_path = artifact_path.replace(".png", ".npy")
            np.save(artifact_path, artifact)
        elif artifact_type == 'flow':
            quickvis_flow(artifact, artifact_path)
        mlflow.log_artifact(artifact_path)
        # print(f"Artifact saved to {artifact_path}")

    def finalize(self):
        """
        Final step: Compute and log final mean metrics at the end of the run.
        """
        mean_metrics = self.get_mean_metrics()
        if not mean_metrics:
            print("No final metrics to save.")
            return

        # Print final metrics once
        print("\nFinal AEE Metrics:")
        print(
            f"AEE (init vs attack): {mean_metrics.get('aee_init_attack', 'N/A'):.4f}")
        print(
            f"AEE (attacked vs target): {mean_metrics.get('aee_target_attack', 'N/A'):.4f}")
