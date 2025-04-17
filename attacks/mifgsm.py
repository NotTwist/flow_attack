import torch
import torch.nn.functional as F

from .attack_utils.utils import apply_high_frequency_mask, apply_sobel
from .attack_base import OpticalFlowAttack
import numpy as np
from models.model_utils import compute_flow
from typing import Literal
from cospgd import functions
from utils.process_images import get_image_tensors, get_image_grads, get_flow_tensors, replace_images_dic

class MIFGSMOpticalFlowAttack(OpticalFlowAttack):
    """
    Implements the Momentum Iterative FGSM (MI-FGSM) attack.
    This is a non-learned, iterative attack that accumulates a momentum term over iterations.
    """

    def __init__(self, model, target: Literal['zero', 'neg_flow', 'untargeted'], epsilon=0.03, alpha: float = 0.01,
                 decay: float = 1.0, device=None, num_steps=20, common_perturb=False, clipping=True,
                 image_min=0, image_max=1, scaling_type: Literal['sobel', 'high_freq', 'cospgd', 'none'] = 'none', save_iterations: list = []):
        super().__init__(model, epsilon, alpha, device, target=target, learned=False, loss='epe')
        self.num_steps = num_steps
        self.decay = decay  # momentum decay factor
        self.common_perturb = common_perturb
        self.clipping = clipping
        self.image_max = image_max
        self.image_min = image_min
        self.save_iterations = save_iterations
        self.scaling_type = scaling_type

    def attack(self, inputs: torch.Tensor):
        """
        Generates adversarial images using MI-FGSM.

        Args:
            images (torch.Tensor): Original images [B, C, H, W].

        Returns:
            dict: Contains the final adversarial images and optionally tracked intermediate flows.
        """
        orig_images = get_image_tensors(inputs, clone=True)
        inputs['images'].requires_grad_(True)
        
        momentum = torch.zeros_like(orig_images).to(self.device)

        flow_pred = self.model(inputs)['flows'].squeeze(0)

        target = self.target(flow_pred)
        target = target.to(self.device)
        target.requires_grad = False

        # Dictionary to store flow outputs at specific iterations
        tracked_flows = {}

        for step in range(1, self.num_steps + 1):
            loss, _ = self.scaled_loss(
                flow_pred, target, get_image_tensors(inputs))        
            self.model.zero_grad()
            loss.backward()
            grads = get_image_grads(inputs)
            images = get_image_tensors(inputs)
        
            grad_norm = torch.norm(grads, p=1) + 1e-8
            momentum = self.decay * momentum + grads / grad_norm
            
            images = self.step(images, momentum, orig_images)
            inputs = replace_images_dic(inputs, images)
            inputs['images'].requires_grad_(True)
            flow_pred = self.model(inputs)['flows'].squeeze(0)

            # Store flow at specific attack steps
            if step in self.save_iterations:
                tracked_flows[step] = flow_pred.clone().detach()

        return {
            "final_images": inputs,
            "tracked_flows": tracked_flows,
        }

    def scaled_loss(self, flow_pred, target, images):
        """
        Домножает лосс на карту границ, полученную с фильтром Собеля.
        """
        raw_loss = self.loss(flow_pred, target)
        if self.scaling_type == 'sobel':
            mask = apply_sobel(
                images).detach()  # Генерируем карту границ
        elif self.scaling_type == 'high_freq':
            mask = apply_high_frequency_mask(images).detach()
        elif self.scaling_type == 'low_freq':
            pass
        elif self.scaling_type == 'cospgd':
            loss = functions.cospgd_scale(
                predictions=flow_pred, labels=target.float(), loss=raw_loss, targeted=True, one_hot=False
            )
            return loss.mean(), None
        elif self.scaling_type == 'none':
            return raw_loss.mean(), None

        if raw_loss.dim() == 4 and raw_loss.size(1) != mask.size(1):
            mask = mask.expand(-1, raw_loss.size(1), -1, -1)

        return (raw_loss * mask).mean(), mask

    def step(self, images, momentum, orig_images):
        """
        Performs MI-FGSM step update.

        Args:
            images (torch.Tensor): Current adversarial images.
            momentum (torch.Tensor): Accumulated momentum.
            orig_images (torch.Tensor): Original images.

        Returns:
            torch.Tensor: Updated adversarial images.
        """
        perturbed_images = functions.step_inf(
            perturbed_image=images,
            epsilon=self.epsilon,
            data_grad=momentum,  # Use accumulated momentum instead of raw gradients
            orig_image=orig_images,
            alpha=self.alpha,
            targeted=True,
            clamp_min=self.image_min,
            clamp_max=self.image_max,
            grad_scale=None,
        )

        return perturbed_images
