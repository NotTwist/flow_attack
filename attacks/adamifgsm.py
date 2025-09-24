import torch
import torch.nn.functional as F

from .attack_utils.utils import apply_high_frequency_mask, apply_sobel
from .attack_base import OpticalFlowAttack
import numpy as np
from models.model_utils import compute_flow
from typing import Literal
from cospgd import functions
from utils.process_images import get_image_tensors, get_image_grads, get_flow_tensors, replace_images_dic

class AdaMIFGSMOpticalFlowAttack(OpticalFlowAttack):
    """
    Implements the Adaptive Momentum Iterative FGSM (AdaMI-FGSM) attack
    (Algorithm 2 in Tao et al., 2023).  Uses adaptive step-size (RMSProp/AdaGrad style)
    plus a heavy-ball momentum term, all within the familiar MI-FGSM structure.
    """

    def __init__(self,
                 model,
                 target: Literal['zero', 'neg_flow', 'untargeted'],
                 epsilon=0.03,
                 alpha: float = 0.01,
                 decay: float = 1.0,         # momentum µ
                 beta: float = 0.999,       # EMA for v_t
                 delta: float = 1e-6,       # stabilizer for V̂t
                 device=None,
                 num_steps=20,
                 scaling_type: Literal['sobel', 'high_freq', 'cospgd', 'none'] = 'none',
                 image_min=0, image_max=1,
                 save_iterations: list = []):
        super().__init__(model, epsilon, alpha, device, target=target, learned=False, loss='epe')
        self.num_steps   = num_steps
        self.decay       = decay
        self.beta        = beta
        self.delta       = delta
        self.scaling_type= scaling_type
        self.image_min   = image_min
        self.image_max   = image_max
        self.save_iterations = save_iterations
        self.prev_images = None

        # will hold per-iteration state
        self.v_t = None

    def attack(self, inputs: torch.Tensor):
        orig = get_image_tensors(inputs, clone=True).to(self.device)
        inputs['images'] = inputs['images'].detach().to(self.device)
        inputs['images'].requires_grad_(True)

        # init momentum & variance
        self.g_t = torch.zeros_like(orig)
        self.v_t = torch.zeros_like(orig)

        flow_pred = self.model(inputs)['flows'].squeeze(0)
        target    = self.target(flow_pred).detach().to(self.device)

        tracked = {}

        for t in range(1, self.num_steps+1):
            # forward + loss
            loss, _ = self.scaled_loss(flow_pred, target, get_image_tensors(inputs))
            self.model.zero_grad()
            loss.backward()

            grads = get_image_grads(inputs)
            imgs  = get_image_tensors(inputs)

            # perform AdaMI update
            imgs = self.step(imgs, grads, orig, step_idx=t)
            inputs = replace_images_dic(inputs, imgs)
            inputs['images'].requires_grad_(True)

            # next prediction
            flow_pred = self.model(inputs)['flows'].squeeze(0)

            if t in self.save_iterations:
                tracked[t] = flow_pred.clone().detach()

        return {"final_images": inputs, "tracked_flows": tracked}

    def scaled_loss(self, flow_pred, target, images):
        raw = self.loss(flow_pred, target)
        if self.scaling_type == 'sobel':
            mask = apply_sobel(images).detach()
        elif self.scaling_type == 'high_freq':
            mask = apply_high_frequency_mask(images).detach()
        elif self.scaling_type == 'cospgd':
            loss = functions.cospgd_scale(
                predictions=flow_pred, labels=target.float(),
                loss=raw, targeted=True, one_hot=False
            )
            return loss.mean(), None
        else:
            return raw.mean(), None

        # ensure same channels
        if raw.dim()==4 and raw.size(1)!=mask.size(1):
            mask = mask.expand(-1, raw.size(1), -1, -1)
        return (raw * mask).mean(), mask

    def step(self, images, grads, orig_images, step_idx):
        """
        AdaMI-FGSM step:
          - update momentum: g_t = µ g_{t-1} + grad
          - update variance:  v_t = β v_{t-1} + (1−β) grad^2
          - compute V̂ = sqrt(v_t) + δ / sqrt(t)
          - scaled_grad = g_t / V̂
          - feed scaled_grad into step_inf
        """

        # 2) second moment
        self.v_t = self.beta * self.v_t + (1 - self.beta) * grads.pow(2)
        v_hat    = self.v_t.sqrt() + (self.delta / (step_idx**0.5))

        # 3) normalized direction
        scaled = (self.alpha / (step_idx ** 0.5)) * grads / v_hat
        # if self.prev_images is not None:
        #     scaled += self.decay * (images - self.prev_images)

        # 4) FGSM‐style update via step_inf
        updated = functions.step_inf(
            perturbed_image=images,
            epsilon=self.epsilon,
            data_grad=scaled,            # ignored when grad_scale is provided
            orig_image=orig_images,
            alpha=1,
            targeted=True,
            clamp_min=self.image_min,
            clamp_max=self.image_max,
            grad_scale=None,
        )
        self.prev_images = images.clone().detach()
        return updated
