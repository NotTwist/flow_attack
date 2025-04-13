import torch
from tqdm import tqdm
from datasets_utils.dataset_utils import prepare_dataloader
from models.model_utils import import_and_load, compute_flow, model_takes_unit_input
from utils.process_images import preprocess_img, postprocess_flow
from metrics.attack_metrics import AttackMetricsTracker
from utils.seed import set_seed
from utils.args import parse_args
from attacks.get_attacks import get_attack
import ptlflow.models
import ptlflow.utils
import ptlflow.utils.io_adapter
import torch
import numpy as np
from tqdm import tqdm
from datasets_utils.dataset_utils import prepare_dataloader
from models.model_utils import import_and_load, compute_flow, model_takes_unit_input
from utils.process_images import preprocess_img, postprocess_flow, get_image_tensors
from metrics.attack_metrics import AttackMetricsTracker
from utils.seed import set_seed
from utils.args import parse_args
from attacks.get_attacks import get_attack
import ptlflow
import cv2 as cv


def load_model(model_name, dataset):
    model_ref = ptlflow.get_model_reference(model_name)
    checkpoints = model_ref.pretrained_checkpoints.keys()
    for c in checkpoints:
        if c in dataset:
            model = ptlflow.get_model(model_name, c)
            return model
    print(f"No pre-trained model available for {model}/{dataset}.")
    return None
# my_weights_path = "models/_pretrained_weights/raft-sintel.pth"
# ptlflow_weights_path = "/home/28s_mur@lab.graphicon.ru/.cache/torch/hub/ptlflow/checkpoints/raft-sintel-fb44381e.ckpt"

# state_dict_pth = torch.load(my_weights_path)
# state_dict_pth = {k.replace('module.', ''): v for k, v in state_dict_pth.items()}
# state_dict_ckpt = torch.load(ptlflow_weights_path)['state_dict']
# # print(state_dict_pth.keys())
# # print()
# # print(state_dict_ckpt.keys())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = import_and_load('RAFT', make_unit_input=not model_takes_unit_input('RAFT'),
                        make_scaled_input_model=True, device=device)
model.eval()
for param in model.parameters():
    param.requires_grad = False
    
ptlmodel = load_model('raft', 'sintel').to(device)
ptlmodel.eval()
for param in ptlmodel.parameters():
    param.requires_grad = False


img1 = torch.from_numpy(cv.imread('analysis/frame_0001.png'))
img2 = torch.from_numpy(cv.imread('analysis/frame_0002.png'))
images = torch.stack((img1, img2), dim=0)
    # io_adapter = ptlflow.utils.io_adapter.IOAdapter(
    #     ptlmodel, input_size=images.shape[-2:], cuda=torch.cuda.is_available())
    # wrapped_inputs = {'images': images, 'flows': flow, 'valids': valid}
    # inputs = io_adapter.prepare_inputs(inputs=wrapped_inputs)
    # # Compute original flow
    # import time
    # with torch.no_grad():
    #     ptloriginal_flow = ptlmodel(inputs)['flows'].squeeze(0)
print(images.shape)
images = images.permute(0, 3, 1, 2)
images = images / 255.0
padder, images = preprocess_img('raft', images)
images = images.detach().to(device)
images.requires_grad = True
print(images.shape)

# Compute original flow

original_flow = compute_flow(model, "scaled_input_model", images)

    # print(original_flow==ptloriginal_flow)
