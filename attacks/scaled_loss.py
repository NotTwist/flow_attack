import torch
import torch.nn.functional as F
import numpy as np
import cv2
from .attack_base import OpticalFlowAttack
from cospgd import functions
from utils.process_images import get_image_tensors, get_image_grads, get_flow_tensors, replace_images_dic
from torch import nn
from typing import Literal, Dict
from .attack_utils.utils import apply_exponential_transformation, apply_high_frequency_mask, apply_sobel

class ScaledLossOpticalFlowAttack(OpticalFlowAttack):
    def __init__(self, model, target: Literal['zero', 'neg_flow', 'untargeted'], epsilon=0.03, alpha=0.01, device=None, num_steps=20, common_perturb=False, clipping=True, image_min=0, image_max=1, scaling_type: Literal['sobel', 'high_freq', 'low_freq'] = 'sobel', use_map_scaling: bool = True, save_iterations=[]):
        super().__init__(model, epsilon, alpha, device, target=target, learned=False, loss='epe')
        self.num_steps = num_steps
        self.common_perturb = common_perturb
        self.clipping = clipping
        self.image_max = image_max
        self.image_min = image_min
        self.save_iterations = save_iterations
        self.use_map_scaling = use_map_scaling
        self.scaling_type = scaling_type

    def attack(self, inputs: dict):
        orig_images = get_image_tensors(inputs, clone=True)
        inputs['images'].requires_grad_(True)
        flow_pred = self.model(inputs)['flows'].squeeze(0)
        target = self.target(flow_pred).to(self.device)
        target.requires_grad = False

        tracked_flows, tracked_masks = {}, {}

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

            if step in self.save_iterations:
                tracked_masks[step] = mask[0].clone().detach()
                tracked_flows[step] = flow_pred.clone().detach()

        return {"final_images": inputs, "tracked_flows": tracked_flows, "tracked_masks": tracked_masks}

    def step(self, images, grads, orig_images):
        return functions.step_inf(
            perturbed_image=images,
            epsilon=self.epsilon,
            data_grad=grads,
            orig_image=orig_images,
            alpha=self.alpha,
            targeted=True,
            clamp_min=self.image_min,
            clamp_max=self.image_max,
            grad_scale=None,
        )

    def scaled_loss(self, flow_pred, target, images):
        """
        Домножает лосс на карту границ, полученную с фильтром Собеля.
        """
        if self.scaling_type == 'sobel':
            mask = apply_sobel(
                images).detach()  # Генерируем карту границ
        elif self.scaling_type == 'high_freq':
            mask = apply_high_frequency_mask(images).detach()
        elif self.scaling_type == 'low_freq':
            pass
        raw_loss = self.loss(flow_pred, target)
        print(raw_loss.shape)
        if raw_loss.dim() == 4 and raw_loss.size(1) != mask.size(1):
            mask = mask.expand(-1, raw_loss.size(1), -1, -1)

        if self.use_map_scaling:
            mask = apply_exponential_transformation(mask, gamma=0.5)

        return (raw_loss * mask).mean(), mask
