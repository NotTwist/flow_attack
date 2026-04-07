import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TemporalPredictionFilter(nn.Module):
    def __init__(self, mode='median', window_size=5, sigma_color=0.1, sigma_spatial=30.0):
        super().__init__()
        self.mode = mode
        self.window_size = window_size
        self.sigma_color = sigma_color      # Range sigma (насколько чувствительны к цвету)
        self.sigma_spatial = sigma_spatial  # Spatial/Temporal sigma (сила сглаживания)
        
        self.flow_buffer = [] 
        self.image_buffer = []
        # Состояние для рекурсивного фильтра (Domain Transform)
        self.last_filtered_output = None 

    def reset(self):
        self.flow_buffer = []
        self.image_buffer = []
        self.last_filtered_output = None

    def forward(self, current_flow, current_image=None):
        if self.mode == 'none':
            return current_flow

        # Обновление буферов
        self.flow_buffer.append(current_flow)
        if current_image is not None:
            self.image_buffer.append(current_image)
        
        if len(self.flow_buffer) > self.window_size:
            self.flow_buffer.pop(0)
            if current_image is not None:
                self.image_buffer.pop(0)

        # --- DYNAMIC DISPATCH ---
        if self.mode == 'average':
            return torch.mean(torch.stack([f.detach() if i < len(self.flow_buffer)-1 else f 
                                         for i, f in enumerate(self.flow_buffer)]), dim=0)

        elif self.mode == 'median':
            # Ошибка в прошлой версии: median в PyTorch возвращает tuple (values, indices)
            stack = torch.stack([f.detach() if i < len(self.flow_buffer)-1 else f 
                               for i, f in enumerate(self.flow_buffer)])
            return torch.median(stack, dim=0).values

        elif self.mode == 'bilateral':
            return self._bilateral_filter(current_flow, current_image)

        elif self.mode == 'domain_transform':
            return self._domain_transform_filter(current_flow, current_image)

        return current_flow

    def _bilateral_filter(self, current_flow, current_image):
        if len(self.image_buffer) < 2: return current_flow
        
        # Считаем веса относительно текущего изображения
        curr_img = self.image_buffer[-1]
        stack_flow = []
        weights = []
        
        for i in range(len(self.flow_buffer)):
            # Градиент только для текущего кадра
            f = self.flow_buffer[i] if i == len(self.flow_buffer)-1 else self.flow_buffer[i].detach()
            img = self.image_buffer[i]
            
            # Фотометрическое расстояние (разница цветов)
            diff_sq = torch.sum((curr_img - img)**2, dim=1, keepdim=True)
            w = torch.exp(-diff_sq / (2 * self.sigma_color**2))
            
            stack_flow.append(f)
            weights.append(w)
            
        weights_tensor = torch.stack(weights, dim=0)
        flow_tensor = torch.stack(stack_flow, dim=0)
        
        return torch.sum(flow_tensor * weights_tensor, dim=0) / (torch.sum(weights_tensor, dim=0) + 1e-6)

    def _domain_transform_filter(self, current_flow, current_image):
        """
        Реализация рекурсивного фильтра (основа Domain Transform).
        Использует формулу: J_t = (1 - alpha)*J_{t-1} + alpha*I_t
        """
        if self.last_filtered_output is None or self.image_buffer is None:
            self.last_filtered_output = current_flow
            return current_flow

        prev_img = self.image_buffer[-2].detach()
        curr_img = current_image
        
        # 1. Считаем "расстояние" в трансформированном домене
        # dist = 1 + (sigma_s / sigma_r) * |I_t - I_{t-1}|
        color_dist = torch.sqrt(torch.sum((curr_img - prev_img)**2, dim=1, keepdim=True) + 1e-8)
        
        # 2. Коэффициент альфа (зависит от расстояния)
        # alpha = a^dist, где a = exp(-sqrt(2) / sigma_spatial)
        a = torch.exp(torch.tensor(-math.sqrt(2) / self.sigma_spatial, device=current_flow.device))
        alpha = torch.pow(a, 1.0 + (self.sigma_spatial / self.sigma_color) * color_dist)
        
        # 3. Рекурсивное обновление
        # Важно: прошлый выход детачим, чтобы не хранить бесконечный граф градиентов
        new_output = (1.0 - alpha) * self.last_filtered_output.detach() + alpha * current_flow
        
        self.last_filtered_output = new_output
        return new_output
    

def init_temporal_filters(args, device):
    """Инициализирует временные фильтры для защиты на основе аргументов."""
    temporal_mode_map = {
        'temporal-avg': 'average',
        'temporal-median': 'median',
        'temporal-bilateral': 'bilateral',
        'temporal-domain-transform': 'domain_transform'
    }
    
    temporal_modes = ["temporal-avg", "temporal-median", 
                      "temporal-bilateral", "temporal-domain-transform"]
    
    filters = {
        'flow': None,
        'mde': None,
        'ss': None,
        'mode': None
    }
    
    if args.defense in temporal_modes:
        mode = temporal_mode_map[args.defense]
        filters['mode'] = mode
        
        # Создаем фильтры для каждой модели
        filters['flow'] = TemporalPredictionFilter(
            mode=mode, 
            window_size=args.temp_window, 
            sigma_color=args.sigma_color
        ).to(device)
        
        if args.attack_mde:
            filters['mde'] = TemporalPredictionFilter(
                mode=mode, 
                window_size=args.temp_window, 
                sigma_color=args.sigma_color
            ).to(device)
            
        if args.attack_ss:
            filters['ss'] = TemporalPredictionFilter(
                mode=mode, 
                window_size=args.temp_window, 
                sigma_color=args.sigma_color
            ).to(device)
            
            
        print(f"Initialized temporal defense: {args.defense} "
              f"(window={args.temp_window}, mode={mode})")
    
    return filters


def apply_temporal_filters(filters, predictions, current_image, model_type='flow'):
    """Применяет временные фильтры к предсказаниям моделей."""
    if filters[model_type] is not None:
        return filters[model_type](predictions, current_image)
    return predictions


def reset_temporal_filters(filters):
    """Сбрасывает состояние всех временных фильтров."""
    for key in ['flow', 'mde', 'ss']:
        if filters[key] is not None:
            filters[key].reset()