import torch
import torch.nn.functional as F
import random
import numpy as np
from typing import Literal
from .attack_base import OpticalFlowAttack
from models.model_utils import compute_flow
from cospgd import functions
from utils.process_images import get_image_tensors, get_image_grads, get_flow_tensors, replace_images_dic
from .attack_utils.utils import input_diversity

#############################
# VNI-FGSM (Variance-reduced Nesterov Iterative FGSM)
#############################


class VNIFGSMOpticalFlowAttack(OpticalFlowAttack):
    """
    VNI-FGSM: Combines Nesterov accelerated gradient with variance tuning.
    """

    def __init__(self, model, target: Literal['zero', 'neg_flow', 'untargeted'],
                 epsilon=0.03, alpha=0.01, decay=1.0, device=None,
                 num_steps=20, num_samples=5,
                 image_min=0, image_max=1, save_iterations: list = []):
        super().__init__(model, epsilon, alpha, device, target=target, learned=False)
        self.num_steps = num_steps
        self.decay = decay
        self.num_samples = num_samples
        self.image_max = image_max
        self.image_min = image_min
        self.save_iterations = save_iterations
        self.beta = 1.5

    def attack(self, inputs: torch.Tensor):
        orig_images = get_image_tensors(inputs, clone=True)
        x_adv = get_image_tensors(inputs)
        momentum = torch.zeros_like(x_adv).to(self.device)
        v = torch.zeros_like(x_adv).to(self.device)

        flow_pred = self.model(inputs)['flows'].squeeze(0)
        target = self.target(flow_pred).to(self.device)
        target.requires_grad = False

        tracked_flows = {}

        for t in range(self.num_steps):
            # === Шаг 1: Nesterov Look-ahead ===
            lookahead = x_adv + self.alpha * self.decay * momentum
            temp_inputs = inputs.copy()
            temp_inputs = replace_images_dic(temp_inputs, lookahead)

            temp_inputs['images'].requires_grad_(True)
            temp_inputs['images'].retain_grad()

            flow_pred = self.model(temp_inputs)['flows'].squeeze(0)
            loss = self.loss(flow_pred, target)
            self.model.zero_grad()
            loss.backward()
            grad_hat = get_image_grads(temp_inputs)

            # === Шаг 2: Variance Tuning ===
            grad_diffs = []
            for i in range(self.num_samples):
                r = torch.empty_like(
                    lookahead).uniform_(-self.beta * self.epsilon, self.beta * self.epsilon)
                x_sample = lookahead.detach() + r
                x_sample.requires_grad_(True)

                temp_sample = replace_images_dic(inputs, x_sample)
                temp_sample['images'].requires_grad_(True)
                temp_sample['images'].retain_grad()

                flow_sample = self.model(temp_sample)['flows'].squeeze(0)
                loss_sample = self.loss(flow_sample, target)
                self.model.zero_grad()
                loss_sample.backward()
                grad_sample = get_image_grads(temp_sample)

                grad_diff = grad_sample - grad_hat
                grad_diffs.append(grad_diff)

            v = torch.mean(torch.stack(grad_diffs, dim=0), dim=0)

            # === Шаг 3: Обновляем momentum ===
            grad_sum = grad_hat + v
            norm = torch.norm(grad_sum, p=1) + 1e-8
            momentum = self.decay * momentum + grad_sum / norm

            # === Шаг 4: Обновляем изображение ===
            x_adv = functions.step_inf(
                perturbed_image=x_adv,
                epsilon=self.epsilon,
                data_grad=momentum,
                orig_image=orig_images,
                alpha=self.alpha,
                targeted=True,
                clamp_min=self.image_min,
                clamp_max=self.image_max,
                grad_scale=None,
            )

            inputs = replace_images_dic(inputs, x_adv)
            x_adv.requires_grad_(True)

            if t+1 in self.save_iterations:
                flow_pred = self.model(inputs)['flows'].squeeze(0)
                tracked_flows[t+1] = flow_pred.clone().detach()

        return {"final_images": inputs, "tracked_flows": tracked_flows}
