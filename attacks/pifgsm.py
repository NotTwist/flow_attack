import torch
import torch.nn.functional as F
import random
import numpy as np
from typing import Literal
from .attack_base import OpticalFlowAttack
from models.model_utils import compute_flow
from cospgd import functions
from utils.process_images import get_image_tensors, get_image_grads, get_flow_tensors, replace_images_dic

###########################################
# PI-FGSM: Pre-gradient guided Momentum Iterative FGSM
###########################################


class PIFGSMOpticalFlowAttack(OpticalFlowAttack):
    """
    Реализует PI-FGSM согласно оригинальной статье.

    На каждом шаге:
      1. Вычисляется предварительное (lookahead) изображение:
         ˜x_advᵗ = x_advᵗ + α · ˜gₜ₋₁, где ˜gₜ₋₁ – градиент, вычисленный на предыдущем шаге.
      2. Вычисляется градиент ˜gₜ для этого смещённого изображения:
         ˜gₜ = ∇₍˜x_advᵗ₎ J_f(˜x_advᵗ, y)
      3. Обновляется накопленный градиент (momentum):
         gₜ = μ · gₜ₋₁ + (˜gₜ / ||˜gₜ||₁)
      4. Обновляется изображение:
         x_advᵗ⁺¹ = x_advᵗ + α · sign(gₜ)
    """

    def __init__(self, model, target: Literal['zero', 'neg_flow', 'untargeted'],
                 epsilon=0.03, alpha=0.01, decay=1.0, device=None,
                 num_steps=20, image_min=0, image_max=1, save_iterations: list = []):
        super().__init__(model, epsilon, alpha, device, target=target, learned=False)
        self.num_steps = num_steps
        self.decay = decay  # μ - фактор затухания
        self.image_max = image_max
        self.image_min = image_min
        self.save_iterations = save_iterations

    def attack(self, inputs: torch.Tensor):
        # Сохраняем исходные изображения для ограничения отклонения (epsilon)
        orig_images = get_image_tensors(inputs, clone=True)
        # Начальное изображение для атаки
        x_adv = get_image_tensors(inputs)
        # Инициализируем momentum как нулевой тензор
        momentum = torch.zeros_like(x_adv).to(self.device)
        # Инициализируем pre-grad (˜g₀) как нулевой тензор; для первого шага будет использоваться x_adv без смещения
        pre_grad = torch.zeros_like(x_adv).to(self.device)

        # Вычисляем целевую величину (например, целевой поток) для расчёта функции потерь
        flow_pred = self.model(inputs)['flows'].squeeze(0)
        target = self.target(flow_pred).to(self.device)
        target.requires_grad = False

        tracked_flows = {}

        for step in range(1, self.num_steps + 1):
            # Lookahead: смещаем x_adv с учётом предыдущего градиента
            lookahead = x_adv + self.alpha * pre_grad

            # Подготавливаем вход для модели со смещённым изображением
            temp_inputs = inputs.copy()
            temp_inputs = replace_images_dic(temp_inputs, lookahead)
            temp_inputs['images'].requires_grad_(True)
            temp_inputs['images'].retain_grad()

            # Вычисляем градиент на смещённом изображении: ˜gₜ = ∇₍˜x_advᵗ₎ J_f(˜x_advᵗ, y)
            flow_pred = self.model(temp_inputs)['flows'].squeeze(0)
            loss = self.loss(flow_pred, target)
            self.model.zero_grad()
            loss.backward()
            current_grad = get_image_grads(temp_inputs)  # ˜gₜ

            # Нормируем текущий градиент по L1-норме
            grad_norm = torch.norm(current_grad, p=1) + 1e-8

            # Обновляем momentum: gₜ = μ · gₜ₋₁ + (˜gₜ / ||˜gₜ||₁)
            momentum = self.decay * momentum + current_grad / grad_norm

            # Обновляем изображение: x_advᵗ⁺¹ = x_advᵗ + α · sign(gₜ)
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

            # Сохраняем текущий градиент для использования в следующем шаге (˜gₜ станет ˜gₜ₋₁)
            pre_grad = current_grad.clone().detach()

            # Обновляем изображения в inputs
            inputs = replace_images_dic(inputs, x_adv)
            inputs['images'].requires_grad_(True)

            # Опционально сохраняем промежуточные результаты
            if step in self.save_iterations:
                flow_pred = self.model(inputs)['flows'].squeeze(0)
                tracked_flows[step] = flow_pred.clone().detach()

        return {"final_images": inputs, "tracked_flows": tracked_flows}
