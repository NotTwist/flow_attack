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
from typing import Literal


def get_attack(attack_name, model, target: Literal['zero', 'neg_flow', 'untargeted'], epsilon=0.005, alpha=0.01, num_steps=20, no_softmax=True, save_iterations: list = [], target_layer: str = None, use_map_scaling=False, scaling_type=''):
    """
    Select the attack based on the attack name string.

    Args:
        attack_name (str): The attack method name ('FGSM', 'PGD', 'CosPGD').
        model: The model to use for the attack.
        epsilon (float): Attack epsilon value.
        no_softmax (bool): Whether to disable softmax during the attack.

    Returns:
        attack: The attack class instance corresponding to the selected attack.
    """
    if attack_name == "FGSM":
        return FGSMOpticalFlowAttack(model=model, epsilon=epsilon, alpha=alpha, num_steps=num_steps, target=target, save_iterations=save_iterations)

    elif attack_name == "PGD":
        return PGDOpticalFlowAttack(model=model, epsilon=epsilon, alpha=alpha, num_steps=num_steps, target=target, save_iterations=save_iterations)

    elif attack_name == "CosPGD":
        # Check if the attack requires 'no_softmax'
        return CosPGDOpticalFlowAttack(model=model, epsilon=epsilon, alpha=alpha, num_steps=num_steps, no_softmax=no_softmax, target=target, save_iterations=save_iterations)
    elif attack_name == 'GradCAM':
        return GradCAMOpticalFlowAttack(model=model, epsilon=epsilon, alpha=alpha, num_steps=num_steps, target=target, save_iterations=save_iterations, target_layer=target_layer, use_map_scaling=use_map_scaling)
    elif attack_name == 'APGD':
        return APGDOpticalFlowAttack(model=model, epsilon=epsilon, num_steps=num_steps, target=target, save_iterations=save_iterations)
    elif attack_name == 'MIFGSM':
        return MIFGSMOpticalFlowAttack(model=model, epsilon=epsilon, alpha=alpha, num_steps=num_steps, target=target, scaling_type=scaling_type, save_iterations=save_iterations)
    elif attack_name == 'Sobel':
        return ScaledLossOpticalFlowAttack(model=model, epsilon=epsilon, alpha=alpha, num_steps=num_steps, target=target, save_iterations=save_iterations, scaling_type='sobel', use_map_scaling=use_map_scaling)
    elif attack_name == 'HighFreq':
        return ScaledLossOpticalFlowAttack(model=model, epsilon=epsilon, alpha=alpha, num_steps=num_steps, target=target, save_iterations=save_iterations, scaling_type='high_freq', use_map_scaling=use_map_scaling)
    elif attack_name == 'MIFGSM_CosPGD':
        return MIFGSMCosPGDOpticalFlowAttack(model=model, epsilon=epsilon, alpha=alpha, num_steps=num_steps, target=target, save_iterations=save_iterations)
    elif attack_name == 'NIFGSM':
        return NIFGSMOpticalFlowAttack(model=model, epsilon=epsilon, alpha=alpha, num_steps=num_steps, target=target, save_iterations=save_iterations)
    elif attack_name == 'PIFGSM':
        return PIFGSMOpticalFlowAttack(model=model, epsilon=epsilon, alpha=alpha, num_steps=num_steps, target=target, save_iterations=save_iterations)
    elif attack_name == 'EMIFGSM':
        return EMIFGSMOpticalFlowAttack(model=model, epsilon=epsilon, alpha=alpha, num_steps=num_steps, target=target, save_iterations=save_iterations)
    elif attack_name == 'VMIFGSM':
        return VMIFGSMOpticalFlowAttack(model=model, epsilon=epsilon, alpha=alpha, num_steps=num_steps, target=target, save_iterations=save_iterations)
    elif attack_name == 'VNIFGSM':
        return VNIFGSMOpticalFlowAttack(model=model, epsilon=epsilon, alpha=alpha, num_steps=num_steps, target=target, save_iterations=save_iterations)
    else:
        raise ValueError(f"Unknown attack name: {attack_name}")
