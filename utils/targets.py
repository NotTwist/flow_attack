import torch
import torch.nn.functional as F
import numpy as np
import os
import os.path


def zero_flow(flow):
	"""Create a zero tensor with the same size as flow

	Args:
		flow (tensor): input

	Returns:
		tensor: containing zeros with same dimension as the input
	"""
	return torch.zeros_like(flow)


def neg_flow(flow):
	"""Mirror the input flow by 180 degree

	Args:
		flow (tensor): input flow field

	Returns:
		tensor: reversed flow field
	"""
	return - flow.detach()

def get_target(target_name, custom_target_path="", device=None):
	"""Getter method which yields a specified target flow used during PCFA 

	Args:
		target_name (str):
			description the attack target. Options: [zero | negative | custom]
		flow_pred_init (tensor):
			unattacked flow field
		custom_target_path (str, optional):
			if custom target is desired provide the path to a .npy perturbation file. Defaults to "".
		device (_type_, optional): _description_. Defaults to None.

	Raises:
		ValueError: Undefined choice for target.

	Returns:
		tensor: target flow field used during PCFA
	"""
	if target_name == 'zero':
		target = zero_flow
	elif target_name == 'neg_flow':
		target = neg_flow
	elif target_name == 'untargeted':
		target = flow_gt
	else:
		raise ValueError('The specified target type "' + target_name +
		                 '" is not defined and cannot be used. Select one of "zero", "neg_flow" or "custom". Aborting.')
	return target
