import cv2
import numpy as np
import torch
from typing import Dict
import copy
import torch.nn.functional as F
import random

def get_image_tensors(input_dic: Dict[str, torch.Tensor], clone=False):
    if clone:
        image_1 = copy.deepcopy(input_dic)["images"][0][0].unsqueeze(0)
        image_2 = copy.deepcopy(input_dic)["images"][0][1].unsqueeze(0)
    else:
        image_1 = input_dic["images"][0][0].unsqueeze(0)
        image_2 = input_dic["images"][0][1].unsqueeze(0)
    return image_1, image_2


def get_flow_tensors(input_dic: Dict[str, torch.Tensor]):
    flow = input_dic["flows"][0][0].unsqueeze(0)
    return flow


def get_image_grads(input_dic: Dict[str, torch.Tensor]):
    grad = input_dic["images"].grad
    image_1_grad = grad[0][0].unsqueeze(0)
    image_2_grad = grad[0][1].unsqueeze(0)
    return image_1_grad, image_2_grad


def replace_images_dic(
    input_dic: Dict[str, torch.Tensor],
    image_1: torch.Tensor,
    image_2: torch.Tensor,
    clone: bool = False,
):
    image_pair_tensor = torch.torch.cat((image_1, image_2)).unsqueeze(0)
    if clone:
        output_dic = copy.deepcopy(input_dic)
        output_dic["images"] = image_pair_tensor
        return output_dic
    else:
        input_dic["images"] = image_pair_tensor
        return input_dic


def get_input_format(input):
    if isinstance(input, dict):
        return input
    elif torch.is_tensor(input) and len(input.size()) == 4:
        input_dic = {"images": input.unsqueeze(0)}
        return input_dic
    elif torch.is_tensor(input) and len(input.size()) == 5:
        input_dic = {"images": input}
        return input_dic


# From FlowUnderAttack
def epe(flow1, flow2):
    """ "
    Compute the  endpoint errors (EPEs) between two flow fields.
    The epe measures the euclidean- / 2-norm of the difference of two optical flow vectors
    (u0, v0) and (u1, v1) and is defined as sqrt((u0 - u1)^2 + (v0 - v1)^2).

    Args:
        flow1 (tensor):
            represents a flow field with dimension (2,M,N) or (b,2,M,N) where M ~ u-component and N ~v-component
        flow2 (tensor):
            represents a flow field with dimension (2,M,N) or (b,2,M,N) where M ~ u-component and N ~v-component

    Raises:
        ValueError: dimensons not valid

    Returns:
        float: scalar average endpoint error
    """
    diff_squared = (flow1 - flow2) ** 2
    if len(diff_squared.size()) == 3:
        # here, dim=0 is the 2-dimension (u and v direction of flow [2,M,N]) , which needs to be added BEFORE taking the square root. To get the length of a flow vector, we need to do sqrt(u_ij^2 + v_ij^2)
        epe = torch.sum(diff_squared, dim=0).sqrt()
    elif len(diff_squared.size()) == 4:
        # here, dim=0 is the 2-dimension (u and v direction of flow [b,2,M,N]) , which needs to be added BEFORE taking the square root. To get the length of a flow vector, we need to do sqrt(u_ij^2 + v_ij^2)
        epe = torch.sum(diff_squared, dim=1).sqrt()
    else:
        raise ValueError(
            "The flow tensors for which the EPE should be computed do not have a valid number of dimensions (either [b,2,M,N] or [2,M,N]). Here: "
            + str(flow1.size())
            + " and "
            + str(flow1.size())
        )
    return epe


def apply_exponential_transformation(gradcam_map, gamma=2.0):
    """
    Применяет экспоненциальное преобразование к карте Grad-CAM с использованием PyTorch.

    :param gradcam_map: входная карта активации (torch tensor)
    :param gamma: коэффициент экспоненциального преобразования
    :return: преобразованная карта Grad-CAM (torch tensor)
    """
    gradcam_map = torch.clamp(
        gradcam_map, min=0, max=1)  # Ограничиваем значения
    # Применяем экспоненциальное преобразование
    exp_map = gradcam_map.pow(gamma)
    return exp_map / exp_map.max()  # Нормализация


#############################
# Вспомогательная функция для input diversity
#############################


def input_diversity(images: torch.Tensor, prob: float = 0.5, low: int = 224, high: int = 256) -> torch.Tensor:
    """
    Применяет случайную трансформацию (resize + pad) к батчу изображений с вероятностью prob.
    """
    if random.random() < prob:
        B, C, H, W = images.shape
        new_size = random.randint(low, high)
        images_resized = F.interpolate(images, size=(
            new_size, new_size), mode='bilinear', align_corners=False)
        pad_h = H - new_size
        pad_w = W - new_size
        pad_top = random.randint(0, pad_h)
        pad_bottom = pad_h - pad_top
        pad_left = random.randint(0, pad_w)
        pad_right = pad_w - pad_left
        images_padded = F.pad(images_resized, (pad_left, pad_right,
                              pad_top, pad_bottom), mode='constant', value=0)
        return images_padded
    else:
        return images


def apply_sobel(images):
    """
    Вычисляет карту границ изображения с помощью оператора Собеля.
    :param images: Входные изображения (тензор [B, C, H, W])
    :return: Карта границ (тензор [B, 1, H, W])
    """
    images_np = images.detach().cpu().numpy()  # Переводим в numpy для OpenCV
    edge_maps = []

    for img in images_np:
        img_gray = np.mean(img, axis=0)  # Градации серого
        sobel_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
        edge_map = np.sqrt(sobel_x**2 + sobel_y**2)  # Магнитуда градиента
        edge_map = edge_map / (edge_map.max() + 1e-8)  # Нормализация
        edge_maps.append(edge_map)

    edge_maps = np.array(edge_maps)[:, None, :, :]  # Добавляем канал
    return torch.tensor(edge_maps, device=images.device, dtype=images.dtype)


def apply_high_frequency_mask(images, kernel_size=5, sigma=1.0):
    """
    Вычисляет маску высоких частот путем вычитания размытых данных от исходного изображения.

    :param images: тензор изображений [B, C, H, W]
    :param kernel_size: размер ядра Гауссова размытия (нечетное число)
    :param sigma: стандартное отклонение для Гауссова размытия
    :return: тензор масок высоких частот [B, 1, H, W]
    """
    images_np = images.detach().cpu().numpy()
    hf_maps = []
    for img in images_np:
        # Усредняем по каналам для получения изображения в оттенках серого
        img_gray = np.mean(img, axis=0)
        # Применяем Гауссово размытие
        img_blur = cv2.GaussianBlur(
            img_gray, (kernel_size, kernel_size), sigma)
        # Вычисляем разницу (абсолютное значение) между исходным и размытым изображением
        hf = np.abs(img_gray - img_blur)
        # Нормализуем карту в диапазон [0, 1]
        hf = hf / (hf.max() + 1e-8)
        hf_maps.append(hf)
    hf_maps = np.array(hf_maps)[:, None, :, :]  # добавляем ось для канала
    return torch.tensor(hf_maps, device=images.device, dtype=images.dtype)
