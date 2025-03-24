from .fgsm import FGSMOpticalFlowAttack
from .pgd import PGDOpticalFlowAttack
from .cospgd import CosPGDOpticalFlowAttack
from .gradcam import GradCAMOpticalFlowAttack
from typing import Literal


def get_attack(attack_name, model, target: Literal['zero', 'neg_flow', 'untargeted'], epsilon=0.005, alpha = 0.01, num_steps=20, no_softmax=True, save_iterations: list = []):
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
        return GradCAMOpticalFlowAttack(model=model, epsilon=epsilon, alpha=alpha, num_steps=num_steps, target=target)
    else:
        raise ValueError(f"Unknown attack name: {attack_name}")
