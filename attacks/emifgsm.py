import torch
import torch.nn.functional as F
import random
import numpy as np
from typing import Literal
from .attack_base import OpticalFlowAttack
from models.model_utils import compute_flow
from cospgd import functions
from utils.process_images import get_image_tensors, get_image_grads, replace_images_dic


class EMIFGSMOpticalFlowAttack(OpticalFlowAttack):
    """
    Реализует Enhanced Momentum Iterative FGSM (EMI-FGSM) согласно:
    
      g₀ = 0;   ̄g₀ = 0;   x_adv₁ = x.
      Для t = 1, …, T:
         1) Сэмплируем N коэффициентов cᵢ ∈ [−η, η]
         2) Для каждого сэмпла вычисляем:
              ̄x_advₜ[i] = x_advₜ + cᵢ · ̄gₜ₋₁
         3) Усредняем градиенты по сэмплам:
              ̄gₜ = 1/N * Σᵢ ∇_{̄x_advₜ[i]} J(̄x_advₜ[i], y)
         4) Обновляем момент:
              gₜ = μ · gₜ₋₁ + (̄gₜ / ||̄gₜ||₁)
         5) Обновляем изображение:
              x_advₜ₊₁ = x_advₜ + α · sign(gₜ)
              
    Параметры:
      - epsilon: максимальное отклонение (ǫ)
      - num_steps: число итераций (T)
      - alpha: величина шага (если не задан, α = ǫ/T)
      - decay: коэффициент затухания (μ)
      - num_samples: число сэмплов (N) для оценки градиента
      - eta: параметр, задающий границы сэмплирования (η)
    """

    def __init__(self, model, target: Literal['zero', 'neg_flow', 'untargeted'],
                 epsilon=0.03, alpha: float = None, decay: float = 1.0, device=None,
                 num_steps=20, num_samples=5, eta=0.1, image_min=0, image_max=1,
                 save_iterations: list = []):
        # Если alpha не задан, вычисляем как epsilon / num_steps
        if alpha is None:
            alpha = epsilon / num_steps
        super().__init__(model, epsilon, alpha, device, target=target, learned=False)
        self.num_steps = num_steps
        self.decay = decay
        self.num_samples = num_samples
        self.eta = eta
        self.image_max = image_max
        self.image_min = image_min
        self.save_iterations = save_iterations

    def attack(self, inputs: torch.Tensor):
        # Исходные (неизменённые) изображения, для контроля ограничения по ǫ
        orig_images = get_image_tensors(inputs, clone=True)
        # Начинаем атаку с исходного изображения
        x_adv = get_image_tensors(inputs)
        x_adv.requires_grad_(True)

        # Инициализируем момент и enhanced gradient предыдущей итерации как нулевые тензоры:
        momentum = torch.zeros_like(x_adv).to(self.device)
        g_bar_prev = torch.zeros_like(x_adv).to(self.device)

        # Целевая величина (например, целевой оптический поток)
        flow_pred = self.model(inputs)['flows'].squeeze(0)
        target = self.target(flow_pred).to(self.device)
        target.requires_grad = False

        tracked_flows = {}

        for t in range(1, self.num_steps + 1):
            # На каждом шаге будем сэмплировать N коэффициентов cᵢ ∈ [−eta, eta]
            grads_samples = []
            for i in range(self.num_samples):
                # Сэмплируем коэффициент
                c = random.uniform(-self.eta, self.eta)
                # Вычисляем сэмплированное изображение: ̄x_adv_t[i] = x_adv_t + c * g_bar_prev
                # Заметим, что для t=1 g_bar_prev = 0, следовательно, ̄x_adv_t[i] = x_adv_t.
                x_adv_sample = x_adv.detach() + c * g_bar_prev
                x_adv_sample.requires_grad_(True)
                # Если требуется, сохраняем градиенты для не-leaf тензора:

                # Подготавливаем вход для модели с этим сэмплом
                temp_inputs = inputs.copy()
                temp_inputs = replace_images_dic(temp_inputs, x_adv_sample)
                temp_inputs['images'].requires_grad_(True)
                temp_inputs['images'].retain_grad()
                # Вычисляем потоки и функцию потерь на сэмплированном изображении
                flow_sample = self.model(temp_inputs)['flows'].squeeze(0)
                loss_sample = self.loss(flow_sample, target)
                self.model.zero_grad()
                loss_sample.backward()
                # Извлекаем градиент сэмпла.
                grad_sample = get_image_grads(temp_inputs)
                grads_samples.append(grad_sample)
            # Усредняем градиенты по N сэмплам: ̄gₜ
            g_bar = torch.mean(torch.stack(grads_samples, dim=0), dim=0)
            # Нормализуем усреднённый градиент (L1-норма)
            norm = torch.norm(g_bar, p=1) + 1e-8
            g_bar_normalized = g_bar / norm
            # Обновляем момент: gₜ = μ*gₜ₋₁ + g_bar_normalized
            momentum = self.decay * momentum + g_bar_normalized
            # Обновляем изображение: x_advₜ₊₁ = x_advₜ + α·sign(momentum)
            x_adv = functions.step_inf(
                perturbed_image=x_adv,
                epsilon=self.epsilon,
                data_grad=momentum,
                orig_image=orig_images,
                alpha=self.alpha,
                targeted=True,
                clamp_min=self.image_min,
                clamp_max=self.image_max,
                grad_scale=None
            )
            # Обновляем входные данные для следующей итерации
            inputs = replace_images_dic(inputs, x_adv)
            x_adv.requires_grad_(True)
            # Сохраняем текущий усреднённый градиент для следующей итерации
            g_bar_prev = g_bar.detach()

            if t in self.save_iterations:
                flow_pred = self.model(inputs)['flows'].squeeze(0)
                tracked_flows[t] = flow_pred.clone().detach()

        return {"final_images": inputs, "tracked_flows": tracked_flows}
