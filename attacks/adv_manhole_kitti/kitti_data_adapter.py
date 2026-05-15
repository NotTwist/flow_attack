import os
import sys
import math
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset

# Make adv_manhole importable
_ADV_MANHOLE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'attacks', 'adversarial_manhole'
)
if _ADV_MANHOLE_DIR not in sys.path:
    sys.path.insert(0, _ADV_MANHOLE_DIR)

from adv_manhole.texture_mapping.depth_utils import depth_to_local_coordinates


def _camera_config_for_shape(K_dict, height, width):
    K_dict = dict(K_dict)
    old_width = float(K_dict.get("image_width", width) or width)
    old_fx = float(K_dict.get("fx", 0.0) or 0.0)

    K_dict["image_width"] = int(width)
    K_dict["image_height"] = int(height)

    if old_fx > 0.0 and old_width > 0.0:
        fx_scaled = old_fx * (float(width) / old_width)
        K_dict["fx"] = fx_scaled
        K_dict["fov"] = 2.0 * math.degrees(math.atan(float(width) / (2.0 * fx_scaled)))

    return K_dict


class KITTIAdvManholeDataset(Dataset):
    """
    Converts KITTI dataset items to adv-manhole batch format:
      rgb:                (3,H,W) float [0,1]
      local_surface_coors:(3,H,W) float in cm (UE4 coords, centerized)
      depth:              (1,H,W) float meters   — used in eval for scaling
      road_mask:          (1,H,W) float binary   — used in eval for ASR

    Only the first frame (I1) of each KITTI pair is used (single-frame attack).
    num_workers=0 is required because __getitem__ runs GPU inference.
    """

    def __init__(self, kitti_dataset, mde_model, ss_model, device, road_class_idx=0):
        self.kitti_dataset = kitti_dataset
        self.mde_model = mde_model
        self.ss_model = ss_model
        self.device = device
        self.road_class_idx = road_class_idx

    def __len__(self):
        return len(self.kitti_dataset)

    def __getitem__(self, idx):
        imgs, flow, valid, disp, K_dict = self.kitti_dataset[idx]
        # imgs: (2,3,H,W); K_dict: camera_config with fov/image_width/image_height/fx/fy/cx/cy
        I1 = imgs[0].to(self.device)  # (3,H,W)

        # Dense depth from MDE (no_grad: preprocessing, not optimization)
        # MDEModel.forward takes dict['images'] as (B,2,C,H,W) and uses [0,0]
        with torch.no_grad():
            depth = self.mde_model({'images': I1.unsqueeze(0).unsqueeze(0)})  # (1,1,H,W) meters

        depth_np = depth.squeeze().cpu().numpy()  # (H,W) in meters
        K_dict = _camera_config_for_shape(K_dict, depth_np.shape[0], depth_np.shape[1])

        # Surface coordinates via adv-manhole's depth_to_local_coordinates.
        # The function expects normalized depth in [0,1] and multiplies by 1000 to get meters.
        # K_dict from KITTI._make_camera_config_from_kitti has fov/image_width/image_height
        # compatible with depth_to_local_coordinates's K reconstruction from FOV.
        surface_xyz = depth_to_local_coordinates(
            depth_np / 1000.0, K_dict
        )  # (H,W,3) in cm, UE4 coords, centerized
        surface_coors = torch.from_numpy(surface_xyz).permute(2, 0, 1).float()  # (3,H,W)

        # Road mask from SS model
        with torch.no_grad():
            ss_logits = self.ss_model(I1, return_logits=True)
            if ss_logits.ndim == 3:
                ss_logits = ss_logits.unsqueeze(0)
        road_mask = (ss_logits.argmax(dim=1) == self.road_class_idx).float()  # (1,H,W)

        return {
            'rgb': I1.cpu(),
            'local_surface_coors': surface_coors.cpu(),
            'depth': depth.squeeze(0).cpu(),
            'road_mask': road_mask.squeeze(0).cpu(),
        }


def make_adv_manhole_dataset_dict(
    kitti_dataset,
    mde_model,
    ss_model,
    device,
    batch_size: int = 4,
    train_fraction: float = 0.8,
    road_class_idx: int = 0,
):
    """
    Returns {'train': DataLoader, 'validation': DataLoader} in adv-manhole format.
    Uses an 80/20 split of the KITTI training set.
    """
    wrapped = KITTIAdvManholeDataset(kitti_dataset, mde_model, ss_model, device, road_class_idx=road_class_idx)
    n = len(wrapped)
    n_train = int(n * train_fraction)

    train_ds = Subset(wrapped, list(range(n_train)))
    val_ds = Subset(wrapped, list(range(n_train, n)))

    return {
        'train': DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0),
        'validation': DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0),
    }
