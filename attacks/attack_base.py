import torch
import abc
from utils.targets import get_target
from utils.losses import get_loss
import sys
class OpticalFlowAttack(abc.ABC):
    """
    Abstract base class for attacking optical flow models.
    
    This class provides a common interface for both non-learned (e.g. FGSM) and learned attacks.
    """

    def __init__(self, model, epsilon=0.03, device=None, learned=False, target='zero', loss='aee'):
        """
        Args:
            model (torch.nn.Module): Optical flow model to attack.
            epsilon (float): Perturbation strength for gradient-based attacks.
            device (torch.device): Device to run the attack on.
            learned (bool): If True, indicates the attack has learnable parameters.
        """
        self.model = model.to(device)
        self.epsilon = epsilon
        self.device = device if device else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.learned = learned
        self.target = get_target(target) # TODO
        self.loss = get_loss(loss)  # TODO
        self.model.eval()  # set model to evaluation mode

    @abc.abstractmethod
    def attack(self, images, *args, **kwargs):
        """
        Perform the attack on input images.
        
        Args:
            images (torch.Tensor): Input images [B, C, H, W].
        
        Returns:
            torch.Tensor: Adversarial images.
        """
        pass

    def preprocess(self, images):
        """Convert images to the required format before attack (if needed)."""
        return images.to(self.device)

    def postprocess(self, adv_images):
        """Convert adversarial images back to the original format (if needed)."""
        return adv_images.clamp(0, 1)

    def compute_flow(self, adv_images):
        """Run adversarial images through the model to get flow outputs."""
        with torch.no_grad():
            return self.model(adv_images)

    def learn_attack(self, images, target_flows, optimizer, num_steps=10, *args, **kwargs):
        """
        Optionally implement a training loop to learn attack parameters.
        
        This method is only required if self.learned == True.
        
        Args:
            images (torch.Tensor): Original input images.
            target_flows (torch.Tensor): Target optical flow values.
            optimizer (torch.optim.Optimizer): Optimizer for attack parameters.
            num_steps (int): Number of training steps.
        
        Returns:
            torch.Tensor: Adversarial images after training.
        """
        # Default implementation: raise an error if not overridden.
        raise NotImplementedError(
            "This attack does not support learning. Override learn_attack() in your subclass if needed.")
