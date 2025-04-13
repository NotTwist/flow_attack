import torch
import torch.nn.functional as F
from .attack_base import OpticalFlowAttack
import numpy as np
from models.model_utils import compute_flow
from cospgd import functions
from pytorch_grad_cam import GradCAM
from typing import Literal, Dict
from utils.process_images import get_image_tensors, get_image_grads, get_flow_tensors, replace_images_dic
from torch import nn
from .attack_utils.utils import apply_exponential_transformation

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


class ModelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        """Wraps BaseModel forward pass."""
        x = x[None, :]
        output = self.model({'images': x})  # RAFT expects a dict
        return output['flows'][-1]  # Extract final flow prediction


class GradCAMOpticalFlowAttack(OpticalFlowAttack):
    """
    Implements the Fast Gradient Sign Method (FGSM) attack.
    This is a non-learned attack.
    """

    def __init__(self, model, target: Literal['zero', 'neg_flow', 'untargeted'], epsilon=0.03, alpha: float = 0.01, device=None, num_steps=20, common_perturb=False, clipping=True, image_min=0, image_max=1, target_layer='update_block.flow_head', use_map_scaling :bool = True,  save_iterations: list = []):
        super().__init__(model, epsilon, alpha, device, target=target, learned=False)
        self.num_steps = num_steps
        self.common_perturb = common_perturb
        self.clipping = clipping
        self.image_max = image_max
        self.image_min = image_min
        self.save_iterations = save_iterations
        self.target_layer = target_layer
        self.target_layers = {
            'update_block.mask': self.model.update_block.mask,
            'cnet.conv2': self.model.cnet.conv2,
            'update_block.flow_head': self.model.update_block.flow_head
        }
        self.use_map_scaling = use_map_scaling

    def attack(self, inputs: Dict[str, torch.Tensor]):
        """
        Generates adversarial images using FGSM.

        Args:
            images (torch.Tensor): Original images [B, C, H, W].
            target_flows (torch.Tensor): Ground-truth optical flow [B, 2, H, W].

        Returns:
            torch.Tensor: Adversarial images.
        """
        orig_images = get_image_tensors(inputs, clone=True)
        inputs['images'].requires_grad_(True)
        flow_pred = self.model(inputs)['flows'].squeeze(0)
        target = self.target(flow_pred)
        target = target.to(self.device)
        target.requires_grad = False

        tracked_flows = {}
        tracked_masks = {}
        
        target_mask = self.apply_gradcam(inputs, )
        for step in range(1, self.num_steps + 1):
            loss, mask = self.scaled_loss(
                flow_pred, target, get_image_tensors(inputs))
            self.model.zero_grad()
            loss.backward()
            images = get_image_tensors(inputs)
            grads = get_image_grads(inputs)
            images = self.step(images, grads, orig_images)
            inputs = replace_images_dic(inputs, images)
            inputs['images'].requires_grad_(True)
            flow_pred = self.model(inputs)['flows'].squeeze(0)

            # Store flow at specific attack steps
            if step in self.save_iterations:
                tracked_masks[step] = mask.clone().detach()
                tracked_flows[step] = flow_pred.clone().detach()

        return {
            "final_images": inputs,
            "tracked_flows": tracked_flows,
            "tracked_masks": tracked_masks
        }

    def step(self, images, grads, orig_images):
        # alpha = self.epsilon / self.num_steps
        perturbed_images = functions.step_inf(
            perturbed_image=images,
            epsilon=self.epsilon,
            data_grad=grads,
            orig_image=orig_images,  # Use the original unmodified images
            alpha=self.alpha,
            targeted=True,
            clamp_min=self.image_min,
            clamp_max=self.image_max,
            grad_scale=None,
        )

        return perturbed_images

    def scaled_loss(self, flow_pred, target, images):
        """
        Compute the loss scaled by the GradCAM mask.
        Assumes self.loss returns an element-wise loss (e.g. with reduction='none').
        """
        gradcam_mask = self.apply_gradcam(
            images, target)  # shape: [B, 1, H, W]
        if self.use_map_scaling:
            gradcam_mask = apply_exponential_transformation(gradcam_mask, gamma=0.5)
            
        return -gradcam_mask.sum(), gradcam_mask
        raw_loss = self.loss(
            flow_pred, target)  # raw_loss expected shape: [B, C, H, W] or [B, H, W]

        # If necessary, match the numberP of channels of raw_loss to the mask.
        if raw_loss.dim() == 4 and raw_loss.size(1) != gradcam_mask.size(1):
            gradcam_mask = gradcam_mask.expand(-1, raw_loss.size(1), -1, -1)

        # Multiply element-wise and then average
        return (raw_loss * gradcam_mask).mean(), gradcam_mask


    def apply_gradcam(self, images, target):
        """
        Compute the Grad-CAM mask using pytorch-gradcam.
        Ensure it is connected to the computation graph.
        """
        images.requires_grad_(True)  # Ensure images can be differentiated

        model = ModelWrapper(self.model)
        target_layer = self.target_layers[self.target_layer]

        cam = GradCAM(model=model, target_layers=[target_layer])
        target_reward = OpticalFlowRewardTarget(target)

        grayscale_cam = cam(input_tensor=images, targets=[target_reward])

        # Convert numpy to tensor
        gradcam_mask = torch.tensor(
            grayscale_cam, device=images.device, dtype=images.dtype, requires_grad=True)

        # Reshape to [B, 1, H, W]
        gradcam_mask = gradcam_mask.unsqueeze(1)

        # Normalize
        gradcam_mask = (gradcam_mask - gradcam_mask.min()) / \
            (gradcam_mask.max() - gradcam_mask.min() + 1e-8)

        return gradcam_mask


