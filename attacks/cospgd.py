import torch
import torch.nn.functional as F
from .attack_base import OpticalFlowAttack
import numpy as np
from models.model_utils import compute_flow
from cospgd import functions
from typing import Literal

class CosPGDOpticalFlowAttack(OpticalFlowAttack):
    """
    Implements the Fast Gradient Sign Method (FGSM) attack.
    This is a non-learned attack.
    """

    def __init__(self, model, target: Literal['zero', 'neg_flow', 'untargeted'], epsilon = 0.03, alpha = 0.01, device=None, num_steps=20, common_perturb=False, clipping=True, image_min=0, image_max=1, no_softmax=False, save_iterations: list = []):
        super().__init__(model, epsilon, alpha, device,target=target, learned=False)
        self.num_steps = num_steps
        self.common_perturb = common_perturb
        self.clipping = clipping
        self.image_max = image_max
        self.image_min = image_min
        self.no_softmax = no_softmax
        self.save_iterations = save_iterations
        print(f"no_softmax set to {self.no_softmax}")
        

    def attack(self, images: torch.Tensor):
        """
        Generates adversarial images using CosPGD attack and returns flow results for specific iterations.

        Args:
            images (torch.Tensor): Original images [B, C, H, W].

        Returns:
            dict: Contains final adversarial images and optical flow results at selected iterations.
        """
        images = images.clone().detach().to(self.device)
        orig_images = images.clone().detach()  # Store the original images
        images.requires_grad = True
        flow_pred = compute_flow(
            self.model, "scaled_input_model", images).to(self.device)

        target = self.target(flow_pred).to(self.device)
        target.requires_grad = False

        # Dictionary to store flow outputs at specific iterations
        tracked_flows = {}

        for step in range(1, self.num_steps + 1):
            loss = self.loss(flow_pred, target)
            if self.no_softmax:
                loss = functions.cospgd_scale_no_softmax(
                    predictions=flow_pred, labels=target.float(), loss=loss, targeted=True, one_hot=False
                )
            else:
                loss = functions.cospgd_scale(
                    predictions=flow_pred, labels=target.float(), loss=loss, targeted=True, one_hot=False
                )
            loss = loss.mean()
            loss.backward()
            grads = images.grad.data
            images = self.step(images, grads, orig_images)
            images = images.detach()
            images.requires_grad = True


            flow_pred = compute_flow(
                self.model, "scaled_input_model", images)
            flow_pred = flow_pred.to(self.device)

            # Store flow at specific attack steps
            if step in self.save_iterations:
                tracked_flows[step] = flow_pred.clone().detach()

        return {
            "final_images": images,
            "tracked_flows": tracked_flows,
        }

    def step(self, images, grads, orig_images):
        #alpha = self.epsilon / self.num_steps

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
