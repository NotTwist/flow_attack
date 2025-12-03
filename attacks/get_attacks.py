# from .fgsm import FGSMOpticalFlowAttack
# from .pgd import PGDOpticalFlowAttack
# from .cospgd import CosPGDOpticalFlowAttack
# from .gradcam import GradCAMOpticalFlowAttack
# from .apgd import APGDOpticalFlowAttack
# from .mifgsm import MIFGSMOpticalFlowAttack
# from .scaled_loss import ScaledLossOpticalFlowAttack
# from .mifgsm_cospgd import MIFGSMCosPGDOpticalFlowAttack
# from .nifgsm import NIFGSMOpticalFlowAttack
# from .pifgsm import PIFGSMOpticalFlowAttack
# from .emifgsm import EMIFGSMOpticalFlowAttack
# from .vmifgsm import VMIFGSMOpticalFlowAttack
# from .vnifgsm import VNIFGSMOpticalFlowAttack
# from typing import Literal
# from .adamifgsm import AdaMIFGSMOpticalFlowAttack
# from .fgsm_mde import FGSMOpticalFlowMDEDAttack
# from .pdg_mde import PGDOpticalFlowMDEDAttack

# def get_attack(attack_name, model, target: Literal['zero', 'neg_flow', 'untargeted'], mde_model=None, mde_target='zero', loss_weights=None, epsilon=0.005, alpha=0.01, num_steps=20, num_samples=5, no_softmax=True, save_iterations: list = [], target_layer: str = None, use_map_scaling=False, scaling_type='', loss='aee'):
#     """
#     Select the attack based on the attack name string.

#     Args:
#         attack_name (str): The attack method name ('FGSM', 'PGD', 'CosPGD').
#         model: The model to use for the attack.
#         epsilon (float): Attack epsilon value.
#         no_softmax (bool): Whether to disable softmax during the attack.

#     Returns:
#         attack: The attack class instance corresponding to the selected attack.
#     """
#     if mde_model is not None:
#         if attack_name == "FGSM":
#             return FGSMOpticalFlowMDEDAttack(model=model, mde_model=mde_model, epsilon=epsilon, alpha=alpha, num_steps=num_steps, target=target, save_iterations=save_iterations, loss=loss, loss_weights = loss_weights, mde_target=mde_target)
#         elif attack_name == "PGD":
#             return PGDOpticalFlowMDEDAttack(model=model, mde_model=mde_model, epsilon=epsilon, alpha=alpha, num_steps=num_steps, target=target, save_iterations=save_iterations, loss=loss, loss_weights = loss_weights, mde_target=mde_target)
        
#     if attack_name == "FGSM":
#         return FGSMOpticalFlowAttack(model=model, epsilon=epsilon, alpha=alpha, num_steps=num_steps, target=target, save_iterations=save_iterations, loss=loss)

#     elif attack_name == "PGD":
#         return PGDOpticalFlowAttack(model=model, epsilon=epsilon, alpha=alpha, num_steps=num_steps, target=target, save_iterations=save_iterations)

#     elif attack_name == "CosPGD":
#         # Check if the attack requires 'no_softmax'
#         return CosPGDOpticalFlowAttack(model=model, epsilon=epsilon, alpha=alpha, num_steps=num_steps, no_softmax=no_softmax, target=target, save_iterations=save_iterations)
#     elif attack_name == 'GradCAM':
#         return GradCAMOpticalFlowAttack(model=model, epsilon=epsilon, alpha=alpha, num_steps=num_steps, target=target, save_iterations=save_iterations, target_layer=target_layer, use_map_scaling=use_map_scaling)
#     elif attack_name == 'APGD':
#         return APGDOpticalFlowAttack(model=model, epsilon=epsilon, num_steps=num_steps, target=target, save_iterations=save_iterations)
#     elif attack_name == 'MIFGSM':
#         return MIFGSMOpticalFlowAttack(model=model, epsilon=epsilon, alpha=alpha, num_steps=num_steps, target=target, scaling_type=scaling_type, save_iterations=save_iterations)
#     elif attack_name == 'Sobel':
#         return ScaledLossOpticalFlowAttack(model=model, epsilon=epsilon, alpha=alpha, num_steps=num_steps, target=target, save_iterations=save_iterations, scaling_type='sobel', use_map_scaling=use_map_scaling)
#     elif attack_name == 'HighFreq':
#         return ScaledLossOpticalFlowAttack(model=model, epsilon=epsilon, alpha=alpha, num_steps=num_steps, target=target, save_iterations=save_iterations, scaling_type='high_freq', use_map_scaling=use_map_scaling)
#     elif attack_name == 'MIFGSM_CosPGD':
#         return MIFGSMCosPGDOpticalFlowAttack(model=model, epsilon=epsilon, alpha=alpha, num_steps=num_steps, target=target, save_iterations=save_iterations)
#     elif attack_name == 'NIFGSM':
#         return NIFGSMOpticalFlowAttack(model=model, epsilon=epsilon, alpha=alpha, num_steps=num_steps, target=target, save_iterations=save_iterations)
#     elif attack_name == 'PIFGSM':
#         return PIFGSMOpticalFlowAttack(model=model, epsilon=epsilon, alpha=alpha, num_steps=num_steps, target=target, save_iterations=save_iterations)
#     elif attack_name == 'EMIFGSM':
#         return EMIFGSMOpticalFlowAttack(model=model, epsilon=epsilon, alpha=alpha, num_steps=num_steps, target=target, save_iterations=save_iterations)
#     elif attack_name == 'VMIFGSM':
#         return VMIFGSMOpticalFlowAttack(model=model, epsilon=epsilon, alpha=alpha, num_steps=num_steps, target=target, save_iterations=save_iterations, num_samples=num_samples)
#     elif attack_name == 'VNIFGSM':
#         return VNIFGSMOpticalFlowAttack(model=model, epsilon=epsilon, alpha=alpha, num_steps=num_steps, target=target, save_iterations=save_iterations)
#     elif attack_name == 'ADAMIFGSM':
#         return AdaMIFGSMOpticalFlowAttack(model=model, epsilon=epsilon, alpha=alpha, num_steps=num_steps, target=target, scaling_type=scaling_type, save_iterations=save_iterations)
#     else:
#         raise ValueError(f"Unknown attack name: {attack_name}")


# attacks/get_attack.py
import inspect
from typing import Dict
from .fgsm import FGSMOpticalFlowAttack
from .pgd import PGDOpticalFlowAttack
from .cospgd import CosPGDOpticalFlowAttack
from .gradcam import GradCAMOpticalFlowAttack
from .apgd import APGDOpticalFlowAttack
from .mifgsm import MIFGSMOpticalFlowAttack
from .scaled_loss import ScaledLossOpticalFlowAttack
from .mifgsm_cospgd import MIFGSMCosPGDOpticalFlowAttack
from .nifgsm import NIFGSMOpticalFlowAttack
from .pifgsm import PIFGSMOpticalFlowAttack
from .emifgsm import EMIFGSMOpticalFlowAttack
from .vmifgsm import VMIFGSMOpticalFlowAttack
from .vnifgsm import VNIFGSMOpticalFlowAttack
from .adamifgsm import AdaMIFGSMOpticalFlowAttack
from .fgsm_mde import FGSMOpticalFlowMDEDAttack
from .pdg_mde import PGDOpticalFlowMDEDAttack

# мапа базовых атак
_ATTACKS: Dict[str, type] = {
    "FGSM": FGSMOpticalFlowAttack,
    "PGD": PGDOpticalFlowAttack,
    "CosPGD": CosPGDOpticalFlowAttack,
    "GradCAM": GradCAMOpticalFlowAttack,
    "APGD": APGDOpticalFlowAttack,
    "MIFGSM": MIFGSMOpticalFlowAttack,
    "Sobel": ScaledLossOpticalFlowAttack,     # будет передан scaling_type='sobel'
    "HighFreq": ScaledLossOpticalFlowAttack,  # scaling_type='high_freq'
    "MIFGSM_CosPGD": MIFGSMCosPGDOpticalFlowAttack,
    "NIFGSM": NIFGSMOpticalFlowAttack,
    "PIFGSM": PIFGSMOpticalFlowAttack,
    "EMIFGSM": EMIFGSMOpticalFlowAttack,
    "VMIFGSM": VMIFGSMOpticalFlowAttack,
    "VNIFGSM": VNIFGSMOpticalFlowAttack,
    "ADAMIFGSM": AdaMIFGSMOpticalFlowAttack,
}

# специальные MDE-классы для FGSM/PGD
_MDE_ALTS = {
    "FGSM": FGSMOpticalFlowMDEDAttack,
    "PGD": PGDOpticalFlowMDEDAttack,
}


def _filter_kwargs_for_callable(callable_obj, kwargs):
    """Оставляет в kwargs только аргументы, которые принимает callable_obj (обычно __init__)."""
    sig = inspect.signature(callable_obj)
    allowed = {
        name for name, p in sig.parameters.items()
        if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    allowed.discard('self')
    return {k: v for k, v in kwargs.items() if k in allowed}


def get_attack(attack_name: str, model, **kwargs):
    """
    Универсальная фабрика атак: теперь не требует перечислять каждый параметр в сигнатуре.
    Примеры вызова:
        get_attack("FGSM", model, epsilon=0.01, num_steps=30)
        get_attack("FGSM", model, **asdict(cfg))  # если cfg - dataclass
        get_attack("PGD", model, mde_model=mde, mde_target='zero', epsilon=...)
    """
    # всегда передаём модель в kwargs (конструкторы обычно принимают model)
    kwargs = dict(kwargs)  # copy
    kwargs['model'] = model

    # спец. случаи: Sobel / HighFreq
    if attack_name == "Sobel":
        kwargs.setdefault('scaling_type', 'sobel')
    elif attack_name == "HighFreq":
        kwargs.setdefault('scaling_type', 'high_freq')

    # спец. логика: если указан mde_model и есть MDE-альтернатива для этой атаки -> используем её
    if 'mde_model' in kwargs and kwargs.get('mde_model') is not None and attack_name in _MDE_ALTS:
        cls = _MDE_ALTS[attack_name]
        filtered = _filter_kwargs_for_callable(cls.__init__, kwargs)
        return cls(**filtered)

    # стандартный путь через _ATTACKS
    cls = _ATTACKS.get(attack_name)
    if cls is None:
        raise ValueError(f"Unknown attack name: {attack_name}")

    filtered = _filter_kwargs_for_callable(cls.__init__, kwargs)
    return cls(**filtered)
