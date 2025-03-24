from PIL import Image
import torch.nn.functional as F
import os
import sys
sys.path.append("flow_library")
from flow_library.flow_plot import colorplot_light
import numpy as np
import torch
import copy
from typing import Dict

class InputPadder:
    """Pads images such that dimensions are divisible by divisor

    This method is taken from https://github.com/princeton-vl/RAFT/blob/master/core/utils/utils.py
    """

    def __init__(self, dims, divisor=8, mode='sintel'):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // divisor) + 1) * divisor - self.ht) % divisor
        pad_wd = (((self.wd // divisor) + 1) * divisor - self.wd) % divisor
        if mode == 'sintel':
            self._pad = [pad_wd//2, pad_wd - pad_wd //
                         2, pad_ht//2, pad_ht - pad_ht//2]
        else:
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, 0, pad_ht]

    def pad(self, *inputs):
        """Pad a batch of input images such that the image size is divisible by the factor specified as divisor

        Returns:
                list: padded input images
        """
        return torch.stack([F.pad(x, self._pad, mode='replicate') for x in inputs])

    def get_dimensions(self):
        """get the original spatial dimension of the image

        Returns:
                int: original image height and width
        """
        return self.ht, self.wd

    def unpad(self, x):
        """undo the padding and restore original spatial dimension

        Args:
                x (tensor): a tensor with padded dimensions

        Returns:
                tesnor: tensor with removed padding (i.e. original spatial dimension)
        """
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]


def preprocess_img(network, images):
    """Manipulate input images, such that the specified network is able to handle them

    Args:
            network (str):
                    Specify the network to which the input images are adapted

    Returns:
            InputPadder, *tensor:
                    returns the Padder object used to adapt the image dimensions as well as the transformed images
    """
    if network == 'RAFT' or network == "GMA" or network == "FlowFormer" or network == "SEA-RAFT":
        padder = InputPadder(images[0].shape)
        output = padder.pad(*images)

    elif network == 'PWCNet':
        images = [img for img in images]
        padder = InputPadder(images[0].shape, divisor=64)
        output = padder.pad(*images)

    elif network == 'SpyNet':
        # normalize images to [0, 1]
        images = [img for img in images]
        # make image divisibile by 64
        padder = InputPadder(images[0].shape, divisor=64)
        output = padder.pad(*images)

    elif network[:7] == 'FlowNet':
        # normalization only for FlowNet, not FlowNet2
        if not network[:8] == 'FlowNet2':
            images = [img / 255. for img in images]
        # make image divisibile by 64
        padder = InputPadder(images[0].shape, divisor=64)
        output = padder.pad(*images)
    elif network == 'MeFlow':
        padder = InputPadder(images[0].shape, divisor=8)
        output = padder.pad(*images)
    elif network == 'MemFlow':
        padder = InputPadder(images[0].shape, divisor=8)
        output = padder.pad(*images)
    else:
        padder = None
        output = images
    return padder, output


def postprocess_flow(network, padder, *flows):
    """Manipulate the output flow by removing the padding

    Args:
            network (str): name of the network used to create the flow
            padder (InputPadder): instance of InputPadder class used during preprocessing
            flows (*tensor): (batch) of flow fields

    Returns:
            *tensor: output with removed padding
    """

    if padder != None:
        # remove padding
        return [padder.unpad(flow).cpu() for flow in flows]
    else:
        return flows


def model_takes_unit_input(model):
    """Boolean check if a network needs input in range [0,1] or [0,255]

    Args:
            model (str):
                    name of the model

    Returns:
            bool: True -> [0,1], False -> [0,255]
    """
    model_takes_unit_input = False
    if model in ["PWCNet", "SpyNet"]:
        model_takes_unit_input = True
    return model_takes_unit_input


def quickvis_flow(flow, filename, auto_scale=True, max_scale=-1):
    """Saves a flow field tensor with two dimensions as image to a specified file location.

    Args:
            flow (tensor):
                    2-dimensional tensor (c=2), following the dimension order (c,H,W) or (1,c,H,W)
            filename (str):
                    name for the image to save, including path and file extension.
            auto_scale (bool, optional):
                    automatically scale color values. Defaults to True.
            max_scale (int, optional):
                    if auto_scale is false, scale flow by this value. Defaults to -1.
    """
    valid = False
    if len(flow.size()) == 3:
        flow_img = flow.clone().detach().cpu().numpy()
        valid = True

    elif len(flow.size()) == 4 and flow.size()[0] == 1:
        flow_img = flow[0, :, :, :].clone().detach().cpu().numpy()
        valid = True

    else:
        print("Encountered invalid tensor dimensions %s, abort printing." %
              str(flow.size()))

    if valid:
        # make directory and ignore if it exists
        if not os.path.dirname(filename) == "":
            os.makedirs(os.path.dirname(filename), exist_ok=True)
        # write flow
        flow_img = np.rollaxis(flow_img, 0, 3)
        data = colorplot_light(
            flow_img, auto_scale=auto_scale, max_scale=max_scale, return_max=False)
        data = data.astype(np.uint8)
        data = Image.fromarray(data)
        data.save(filename)


# from flowbench

def get_image_tensors(input_dic: Dict[str, torch.Tensor], clone=False):
    if clone:
        images = input_dic["images"][0].clone()
    else:
        images = input_dic["images"][0]

    return torch.stack([images[0], images[1]], dim=0)  # Shape: (2, C, H, W)


def get_flow_tensors(input_dic: Dict[str, torch.Tensor]):
    return input_dic["flows"][0][0].unsqueeze(0)  # Shape: (1, C, H, W)


def get_image_grads(input_dic: Dict[str, torch.Tensor]):
    grad = input_dic["images"].grad
    return torch.stack([grad[0][0], grad[0][1]], dim=0)  # Shape: (2, C, H, W)


def replace_images_dic(
    input_dic: Dict[str, torch.Tensor],
    images: torch.Tensor,
    clone: bool = False,
):
    """
    Replaces the "images" key in input_dic with a new tensor containing both images.

    Args:
        input_dic (Dict[str, torch.Tensor]): The input dictionary.
        images (torch.Tensor): A tensor of shape (2, C, H, W) containing two images.
        clone (bool): If True, creates a cloned copy of input_dic before modifying.

    Returns:
        Dict[str, torch.Tensor]: Updated dictionary with replaced images.
    """
    if images.shape[0] != 2:
        raise ValueError("Expected images tensor of shape (2, C, H, W)")

    output_dic = {k: v.clone()
                  for k, v in input_dic.items()} if clone else input_dic
    # Ensures correct shape: (1, 2, C, H, W)
    output_dic["images"] = images.unsqueeze(0)

    return output_dic
