from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import torch
import numpy as np
from PIL import Image
import requests
import torch.nn.functional as F
from datasets_utils.dataset_utils import prepare_dataloader
from tqdm import tqdm
import torchvision.transforms.functional as TF
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Setting Device to {device}\n")


image_processor = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Large-hf", use_fast=True)
model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Large-hf").to(device)
model.eval()
for param in model.parameters():
    param.requires_grad = False
# prepare image for the model


data_loader, has_gt = prepare_dataloader(
    dataset_name='Kitti15', small_run=True)


for batch, (images, flow, valid, depth_gt) in enumerate(tqdm(data_loader)):
    image = images[0,0,...]
    image = TF.to_pil_image(image)
    # print(image.shape)
    print(depth_gt.shape, depth_gt.sum())
    
    inputs = image_processor(images=image, return_tensors="pt",device=device)
    inputs['pixel_values'].requires_grad = True
    outputs = model(**inputs)
    depth = outputs.predicted_depth
    print(depth.shape)
    zero_target = torch.zeros_like(depth)

    # Compute loss
    loss = F.l1_loss(depth, zero_target)
    loss.backward()

    epsilon = 0.1 
    perturbation = epsilon * inputs['pixel_values'].grad.sign()
    adv_image = inputs['pixel_values'] + perturbation
    adv_image = torch.clamp(adv_image, 0, 1)

    with torch.no_grad():
        adv_outputs = model(pixel_values=adv_image)

    post_processed_output = image_processor.post_process_depth_estimation(
        adv_outputs,
        target_sizes=[(image.height, image.width)],
    )
    adv_depth = post_processed_output[0]["predicted_depth"]
    depth = (adv_depth - adv_depth.min()) / (adv_depth.max() - adv_depth.min())
    depth = depth.detach().cpu().numpy() * 255
    depth = Image.fromarray(depth.astype("uint8"))
    depth.save(f'adv_test_{batch}.png')