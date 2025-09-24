import torch
import torch.nn.functional as F
from .attack_base import OpticalFlowAttack
import numpy as np
from models.model_utils import compute_flow
from typing import Literal, Dict
from cospgd import functions
from utils.process_images import get_image_tensors, get_image_grads, get_flow_tensors, replace_images_dic
import torch.autograd.profiler as profiler
class FGSMOpticalFlowMDEDAttack(OpticalFlowAttack):

    """
    Implements the Fast Gradient Sign Method (FGSM) attack.
    This is a non-learned attack.
    """

    def __init__(self, model, mde_model, target: Literal['zero', 'neg_flow', 'untargeted'], mde_target: Literal['zero','untargeted'], loss_weights=None, epsilon=0.03, alpha: float = 0.01, device=None, num_steps=20, common_perturb=False, clipping=True, image_min=0, image_max=1, save_iterations: list = [], loss='aee'):
        super().__init__(model, epsilon, alpha, device, target=target, learned=False, loss=loss, mde_model=mde_model, mde_target=mde_target)
        self.num_steps = num_steps
        self.common_perturb = common_perturb
        self.clipping = clipping
        self.image_max = image_max
        self.image_min = image_min
        self.save_iterations = save_iterations
        self.flow_w = 1
        self.mde_w = 0.1
        if loss_weights is not None:
            self.flow_w, self.mde_w = loss_weights
    def attack(self, inputs: Dict[str, torch.Tensor]):
        """
        Generates adversarial images using FGSM.

        Args:
            images (torch.Tensor): Original images [B, C, H, W].

        Returns:
            torch.Tensor: Adversarial images.
        """

        orig_images = get_image_tensors(inputs, clone=True)


        inputs['images'] = inputs['images'].detach()
        inputs['images'].requires_grad_(True)
        
        flow_pred = self.model(inputs)['flows'].squeeze(0)
        mde_pred = self.mde_model(inputs)
        
        target = self.target(flow_pred)
        target = target.to(self.device)
        target.requires_grad = False


        mde_target = self.mde_target(mde_pred)
        mde_target = mde_target.to(self.device)
        mde_target.requires_grad = False
        # Dictionary to store flow outputs at specific iterations
        tracked_flows = {}
        tracked_depths = {}
        
        for step in range(1, self.num_steps + 1):
            flow_loss = self.loss(flow_pred, target)
            mde_loss = self.mde_loss(mde_pred, mde_target)
            loss = mde_loss * self.mde_w + flow_loss * self.flow_w
            # print(loss.item(), flow_loss.item(), mde_loss.item())
            # print("loss.requires_grad:", loss.requires_grad)
            # print("flow_pred.requires_grad:", flow_pred.requires_grad)
            # print("mde_pred.requires_grad:", mde_pred.requires_grad)
            # print("Model gradients enabled:", torch.is_grad_enabled())

            self.model.zero_grad()
            self.mde_model.zero_grad()
            loss.backward()
            grads = get_image_grads(inputs)
            images = get_image_tensors(inputs)
            # Pass original images
            images = self.step(images, grads, orig_images)
            inputs = replace_images_dic(inputs, images)
            
            inputs['images'] = inputs['images'].detach()
            inputs['images'].requires_grad_(True)
            
            flow_pred = self.model(inputs)['flows'].squeeze(0)
            mde_pred = self.mde_model(inputs)

            # Store flow at specific attack steps
            if step in self.save_iterations:
                tracked_flows[step] = flow_pred.clone().detach()
                tracked_depths[step] = mde_pred.clone().detach()

        return {
            "final_images": inputs,
            "tracked_flows": tracked_flows,
            "tracked_depths": tracked_depths
        }

    def step(self, images, grads, orig_images):
        # alpha =  self.epsilon / self.num_steps

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
