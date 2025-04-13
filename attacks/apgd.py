import torch
from .attack_base import OpticalFlowAttack
from models.model_utils import compute_flow
from .adversarial_attacks_pytorch.torchattacks import APGD
from typing import Literal, Dict
from utils.process_images import get_image_tensors, get_image_grads, get_flow_tensors, replace_images_dic


class FlowModelWrapper(torch.nn.Module):
    def __init__(self, model, mode="scaled_input_model"):
        super().__init__()
        self.model = model
        self.mode = mode

    def forward(self, x):
        """Use compute_flow instead of the model's default forward method."""
        return compute_flow(self.model, self.mode, x)


class APGDOpticalFlowAttack(OpticalFlowAttack):
    def __init__(self, model, target: Literal['zero', 'neg_flow', 'untargeted'], epsilon=0.03, device=None, num_steps=20, common_perturb=False, clipping=True, image_min=0, image_max=1, save_iterations: list = []):
        super().__init__(model, epsilon, device, learned=False, target=target)
        self.num_steps = num_steps
        self.common_perturb = common_perturb
        self.clipping = clipping
        self.image_max = image_max
        self.image_min = image_min
        self.save_iterations = save_iterations

    def attack(self, inputs: Dict[str, torch.Tensor]):
        """
        Generates adversarial images using FGSM.

        Args:
            images (torch.Tensor): Original images [B, C, H, W].
            target_flows (torch.Tensor): Ground-truth optical flow [B, 2, H, W].

        Returns:
            torch.Tensor: Adversarial images.
        """
        # inputs = inputs.clone().detach().to(self.device)
        # inputs.requires_grad = True
        flow_pred = self.model(inputs)['flows'].squeeze(0)

        target = self.target(flow_pred)
        target = target.to(self.device)
        target.requires_grad = False
        attack = APGD(self.model, loss='epe', norm="Linf", eps=self.epsilon,
                      verbose=False, steps=self.num_steps, n_restarts=1, seed=0, rho=0.75, eot_iter=1, save_iterations=self.save_iterations)
        # for targeted attacks
        attack.targeted = True
        attack.set_mode_targeted_by_label()
        images, tracked_flows = attack(inputs['images'], target)
        inputs['images'] = images
        return {"final_images": inputs, "tracked_flows": tracked_flows}
