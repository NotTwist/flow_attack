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

###########################################
# VMI-FGSM: Variance-reduced Momentum Iterative FGSM
###########################################


class VMIFGSMOpticalFlowAttack(OpticalFlowAttack):
    """
    VMI-FGSM использует ансамблирование градиентов с последующей коррекцией, учитывающей дисперсию 
    (variance reduction). Усреднённый градиент делится на (1 + среднее значение дисперсии),
    что позволяет уменьшить шумовые компоненты.
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
        # Исходное изображение, для контроля отклонения от оригинала.
        orig_images = get_image_tensors(inputs, clone=True)
        x_adv = get_image_tensors(inputs)
        x_adv.requires_grad_(True)

        # Инициализируем momentum и начальное "variance term" v как нули.
        momentum = torch.zeros_like(x_adv).to(self.device)
        # Здесь v будет обновляться по правилу, зависящему от вариации градиентов.
        # Изначально можно считать его нулевым.
        v = torch.zeros_like(x_adv).to(self.device)

        # Вычислим target один раз из исходного потока (или как требуется)
        flow_pred = self.model(inputs)['flows'].squeeze(0)
        target = self.target(flow_pred).to(self.device)
        target.requires_grad = False

        tracked_flows = {}

        for t in range(self.num_steps):
            # Обнуляем градиенты перед вычислением
            if x_adv.grad is not None:
                x_adv.grad.data.zero_()

            # --- Шаг 4: Вычисляем базовый градиент ˆgₜ₊₁ на x_adv ---
            temp_inputs = inputs.copy()
            temp_inputs = replace_images_dic(temp_inputs, x_adv)
            temp_inputs['images'].requires_grad_(True)
            # Если это не leaf-тензор, добавим retain_grad() для сохранения градиента.
            temp_inputs['images'].retain_grad()
            flow_pred = self.model(temp_inputs)['flows'].squeeze(0)
            loss = self.loss(flow_pred, target)
            self.model.zero_grad()
            loss.backward()
            # Получаем базовый градиент на x_adv.
            # Используется схема получения градиента,
            grad_hat = get_image_grads(temp_inputs)
            # как в ваших других атаках.
            # grad_hat имеет размерность [2, C, H, W] или [C, H, W] в зависимости от реализации.

            # --- Шаг 6: Вычисляем V(x_adv) ---
            # Для N сэмплов сгенерируем возмущения и вычислим соответствующие градиенты.
            grad_diffs = []
            for i in range(self.num_samples):
                # Сэмплируем r_i из равномерного распределения для каждого пикселя: размер x_adv.
                r = torch.empty_like(
                    x_adv).uniform_(-self.beta * self.epsilon, self.beta * self.epsilon)
                x_sample = x_adv.detach() + r  # создаём x_i
                x_sample.requires_grad_(True)
                # Вычисляем градиент для x_sample
                temp_inputs_sample = inputs.copy()
                temp_inputs_sample = replace_images_dic(temp_inputs_sample, x_sample)
                temp_inputs_sample['images'].requires_grad_(True)
                temp_inputs_sample['images'].retain_grad()
                flow_sample = self.model(temp_inputs_sample)[
                    'flows'].squeeze(0)
                loss_sample = self.loss(flow_sample, target)
                self.model.zero_grad()
                loss_sample.backward()
                grad_sample = get_image_grads(temp_inputs_sample)
                # Разница градиентов: grad(x_i) - grad(x_adv)
                grad_diff = grad_sample - grad_hat
                grad_diffs.append(grad_diff)
            # Усредняем разности по num_samples.
            v = torch.mean(torch.stack(grad_diffs, dim=0), dim=0)

            # --- Шаг 5: Обновляем момент ---
            # Суммарный градиент: (ˆgₜ₊₁ + v)
            grad_sum = grad_hat + v
            norm = torch.norm(grad_sum, p=1) + 1e-8
            # Обновляем momentum: gₜ₊₁ = μ * gₜ + (ˆgₜ₊₁ + v) / ||ˆgₜ₊₁ + v||₁
            momentum = self.decay * momentum + grad_sum / norm

            # --- Шаг 7: Обновляем изображение ---
            # Новый шаг: x_adv = x_adv + α * sign(momentum)
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
            # Обновляем inputs для следующей итерации.
            inputs = replace_images_dic(inputs, x_adv)
            x_adv.requires_grad_(True)

            if t+1 in self.save_iterations:
                flow_pred = self.model(inputs)['flows'].squeeze(0)
                tracked_flows[t+1] = flow_pred.clone().detach()

        return {"final_images": inputs, "tracked_flows": tracked_flows}
