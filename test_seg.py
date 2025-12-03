# Load torchvision deeplab
from PIL import Image
from models.model_utils import load_seg_model
import torch
import matplotlib.pyplot as plt
import numpy as np

seg = load_seg_model(device=torch.device("cuda:0"))

# single PIL image
img = Image.open("mlruns/914879025403898454/fab160d0840e4adaafec9152f77b1199/artifacts/batch_0000_attacked_image.png").convert("RGB")
# # returns torch.LongTensor [1,H,W] (class indices)
pred_mask = seg(img)
print(pred_mask.shape)
logits = seg(img, return_logits=True)  # returns logits [1,C,H,W]

# mmseg (PSPNet entry in models.yaml must exist)
# seg_mm = load_seg_model("pspnet_cityscapes", device=torch.device("cuda:0"))
# mask = seg_mm(img)  # for mmseg returns class mask (LongTensor)
mask = pred_mask.squeeze(0).cpu().numpy()  # shape [H,W]

# Choose a colormap (for Cityscapes, 19 classes example)
# You can define your own RGB colors for each class
CITYSCAPES_COLORS = np.array([
    [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
    [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
    [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
    [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
    [0, 80, 100], [0, 0, 230], [119, 11, 32]
], dtype=np.uint8)

# Map class indices to colors
color_mask = CITYSCAPES_COLORS[mask]  # shape [H,W,3]
color_mask_img = Image.fromarray(color_mask)
color_mask_img.save("segmentation_mask.png")

# original image
img_np = np.array(img)

# overlay
overlay = (0.5 * img_np + 0.5 * color_mask).astype(np.uint8)

# save
Image.fromarray(overlay).save("segmentation_overlay.png")
