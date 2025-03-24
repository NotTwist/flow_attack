import torch
import torch.nn.functional as F
from .attack_base import OpticalFlowAttack
import numpy as np
from models.model_utils import compute_flow
from cospgd import functions
from pytorch_grad_cam import GradCAM
from typing import Literal

class OpticalFlowRewardTarget:
    def __init__(self, unattacked_flow: torch.Tensor):
        """
        Args:
            ground_truth (torch.Tensor): Ground truth optical flow tensor of shape [B, 2, H, W].
        """
        self.unatacked_flow = unattacked_flow

    def __call__(self, model_output: torch.Tensor) -> torch.Tensor:
        # Compute endpoint error (EPE) per pixel: sqrt((F_x - GT_x)^2 + (F_y - GT_y)^2)
        # shape: [B, H, W]
        epe = torch.sqrt(
            torch.sum((model_output - self.unatacked_flow) ** 2, dim=1))
        # Negative mean error so that lower error gives a higher reward
        return torch.mean(epe)


class GradCAMOpticalFlowAttack(OpticalFlowAttack):
    """
    Implements the Fast Gradient Sign Method (FGSM) attack.
    This is a non-learned attack.
    """

    def __init__(self, model, target: Literal['zero', 'neg_flow', 'untargeted'], epsilon = 0.03, device=None, num_steps=20, common_perturb=False, clipping=True, image_min=0, image_max=1, no_softmax=False):
        super().__init__(model, epsilon, alpha, device, target=target, learned=False)
        self.num_steps = num_steps
        self.common_perturb = common_perturb
        self.clipping = clipping
        self.image_max = image_max
        self.image_min = image_min
        self.no_softmax = no_softmax
        print(f"no_softmax set to {self.no_softmax}")

    def attack(self, images: torch.Tensor):
        """
        Generates adversarial images using FGSM.

        Args:
            images (torch.Tensor): Original images [B, C, H, W].
            target_flows (torch.Tensor): Ground-truth optical flow [B, 2, H, W].

        Returns:
            torch.Tensor: Adversarial images.
        """
        flow_pred = compute_flow(
            self.model, "scaled_input_model", images)
        flow_pred = flow_pred.to(self.device)
        target = self.target(flow_pred)
        target = target.to(self.device)
        target.requires_grad = False
        for step in range(1, self.num_steps + 1):
            loss = self.scaled_loss(flow_pred, target, images)

            loss.backward()
            grads = images.grad.data
            images = self.step(images, grads)
            images = images.detach()
            images.requires_grad = True
            flow_pred = compute_flow(
                self.model, "scaled_input_model", images)
            flow_pred = flow_pred.to(self.device)

        return images

    def step(self, images, grads):
        # alpha = self.epsilon / self.num_steps
        if not self.common_perturb:
            signs = grads.sign()  # Element-wise sign of gradients
        else:
            # Averaged sign across batch
            signs = grads.mean(dim=0, keepdim=True).sign()

        perturbed_images = images - self.alpha * signs  # Apply perturbation

        if self.clipping:
            perturbed_images = torch.clamp(
                perturbed_images, self.image_min, self.image_max)

        return perturbed_images

    def scaled_loss(self, flow_pred, target, images):
        """
        Compute the loss scaled by the GradCAM mask.
        Assumes self.loss returns an element-wise loss (e.g. with reduction='none').
        """
        gradcam_mask = self.apply_gradcam(
            images, target)  # shape: [B, 1, H, W]
        raw_loss = self.loss(
            flow_pred, target)  # raw_loss expected shape: [B, C, H, W] or [B, H, W]

        # If necessary, match the number of channels of raw_loss to the mask.
        if raw_loss.dim() == 4 and raw_loss.size(1) != gradcam_mask.size(1):
            gradcam_mask = gradcam_mask.expand(-1, raw_loss.size(1), -1, -1)

        # Multiply element-wise and then average
        return (raw_loss * gradcam_mask).mean()

    def apply_gradcam(self, images, target):
        """
        Compute the GradCAM mask using pytorch-gradcam with a wrapped model that applies compute_flow.
        """
        target_layer = self.model.model_loaded.module.cnet.conv2  # Choose the appropriate layer

        # Wrap the model so Grad-CAM applies compute_flow instead of calling model()
        wrapped_model = FlowModelWrapper(self.model)

        cam = GradCAM(model=wrapped_model, target_layers=[target_layer])

        # Define target function for Grad-CAM
        target_reward = OpticalFlowRewardTarget(target)

        grayscale_cam = cam(input_tensor=images, targets=[
                            target_reward])

        # Convert numpy to torch tensor
        gradcam_mask = torch.tensor(
            grayscale_cam, device=images.device, dtype=images.dtype)

        # Reshape to [B, 1, H, W]
        gradcam_mask = gradcam_mask.unsqueeze(1)

        # Normalize to [0, 1]
        gradcam_mask = (gradcam_mask - gradcam_mask.min()) / \
            (gradcam_mask.max() - gradcam_mask.min() + 1e-8)

        return gradcam_mask

class FlowModelWrapper(torch.nn.Module):
    def __init__(self, model, mode="scaled_input_model"):
        super().__init__()
        self.model = model
        self.mode = mode

    def forward(self, x):
        """Use compute_flow instead of the model's default forward method."""
        return compute_flow(self.model, self.mode, x)



