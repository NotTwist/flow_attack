import torch
import torch.nn.functional as F
from .attack_base import OpticalFlowAttack
import numpy as np
from models.model_utils import compute_flow

class FGSMOpticalFlowAttack(OpticalFlowAttack):
    """
    Implements the Fast Gradient Sign Method (FGSM) attack.
    This is a non-learned attack.
    """

    def __init__(self, model, epsilon=0.03, device=None, num_steps=20, common_perturb=False, clipping=True, image_min=0, image_max=1):
        super().__init__(model, epsilon, device, learned=False)
        self.num_steps = num_steps
        self.common_perturb = common_perturb
        self.clipping = clipping
        self.image_max = image_max
        self.image_min = image_min
        
    def attack(self, images: torch.Tensor):
        """
        Generates adversarial images using FGSM.

        Args:
            images (torch.Tensor): Original images [B, C, H, W].
            target_flows (torch.Tensor): Ground-truth optical flow [B, 2, H, W].

        Returns:
            torch.Tensor: Adversarial images.
        """
        images = images.clone().detach().to(self.device)
        images.requires_grad = True
        flow_pred = compute_flow(
            self.model, "scaled_input_model", images)

        flow_pred = flow_pred.to(self.device)
        target = self.target(flow_pred)
        target = target.to(self.device)
        target.requires_grad = False
        for step in range(self.num_steps):
            loss = self.loss(flow_pred, target)
            self.model.zero_grad()
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
        alpha = self.epsilon / self.num_steps
        if not self.common_perturb:
            signs = grads.sign()  # Element-wise sign of gradients
        else:
            # Averaged sign across batch
            signs = grads.mean(dim=0, keepdim=True).sign()

        perturbed_images = images - alpha * signs  # Apply perturbation

        if self.clipping:
            perturbed_images = torch.clamp(
                perturbed_images, self.image_min, self.image_max)

        return perturbed_images
