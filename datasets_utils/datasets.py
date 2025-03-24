import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F
import random
from . import frame_utils
from PIL import Image
from glob import glob
import os.path as osp
import os
from argparse import Namespace
# from PCFA attack
class FlowDataset(data.Dataset):
    def __init__(self, aug_params=None, sparse=False, frames=2):
        self.sparse = sparse
        self.frames = frames
        self.has_gt = False
        self.init_seed = False
        self.flow_list = []
        self.image_list = []
        self.extra_info = []
        self.enforce_dimensions = False
        self.image_x_dim = 0
        self.image_y_dim = 0

    def __getitem__(self, index):

        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        index = index % len(self.image_list)

        imgs = []
        for i in range(self.frames):
            imgs.append(frame_utils.read_gen(self.image_list[index][i]))
            imgs[-1] = np.array(imgs[-1]).astype(np.uint8)

        if len(imgs[0].shape) == 2:
            for i in range(self.frames):
                imgs[i] = np.tile(imgs[i][..., None], (1, 1, 3))
        else:
            for i in range(self.frames):
                imgs[i] = imgs[i][..., :3]

        valid = None

        if self.has_gt:
            if self.sparse:
                flow, valid = frame_utils.readFlowKITTI(self.flow_list[index])
            else:
                flow = frame_utils.read_gen(self.flow_list[index])
            flow = np.array(flow).astype(np.float32)

            flow = torch.from_numpy(flow).permute(2, 0, 1).float()

            if valid is not None:
                valid = torch.from_numpy(valid)
            else:
                valid = (flow[0].abs() < 1000) & (flow[1].abs() < 1000)

        else:
            (img_x, img_y, img_chann) = imgs[0].shape

            # make correct size for flow (2 dimensions for u,v instead of 3 [r,g,b] for image )
            flow = np.zeros((img_x, img_y, 2))
            valid = False

            flow = torch.from_numpy(flow).permute(2, 0, 1).float()

        for i in range(self.frames):
            imgs[i] = torch.from_numpy(imgs[i]).permute(2, 0, 1)

        if self.enforce_dimensions:
            dims = imgs[0].size()
            x_dims = dims[-2]
            y_dims = dims[-1]

            diff_x = self.image_x_dim - x_dims
            diff_y = self.image_y_dim - y_dims

            for i in range(self.frames):
                imgs[i] = F.pad(imgs[i], (0, diff_y, 0, diff_x), "constant", 0)

            flow = F.pad(flow, (0, diff_y, 0, diff_x), "constant", 0)
            if self.has_gt:
                valid = F.pad(valid, (0, diff_y, 0, diff_x), "constant", False)

        return torch.stack(imgs, dim=0) / 255., flow, valid

    def __rmul__(self, v):
        self.flow_list = v * self.flow_list
        self.image_list = v * self.image_list
        return self

    def __len__(self):
        return len(self.image_list)

    def has_groundtruth(self):
        return self.has_gt


class MpiSintel(FlowDataset):
    def __init__(self, aug_params=None, split='training', root=None, dstype='clean', has_gt=False, frames=2):
        super(MpiSintel, self).__init__(aug_params)
        self.frames = frames
        flow_root = osp.join(root, split, 'flow')
        image_root = osp.join(root, split, dstype)

        self.has_gt = has_gt

        for scene in os.listdir(image_root):
            image_list = sorted(glob(osp.join(image_root, scene, '*.png')))
            for i in range(len(image_list)-1):
                self.image_list.append([])
                for d in range(-((self.frames - 1) // 2), self.frames // 2 + 1):
                    self.image_list[-1].append(
                        image_list[min(max(i + d, 0), len(image_list) - 1)])
                self.extra_info += [(scene, i)]  # scene and frame_id

            if split != 'test':
                self.flow_list += sorted(glob(osp.join(flow_root,
                                         scene, '*.flo')))

        if len(self.image_list) == 0:
            raise RuntimeWarning(
                "No MPI Sintel data found at dataset root '%s'. Check the configuration file under helper_functions/config_paths.py and add the correct path to the MPI Sintel dataset." % root)


class KITTI(FlowDataset):
    def __init__(self, aug_params=None, split='training', root=None, has_gt=False, frames=2):
        super(KITTI, self).__init__(aug_params, sparse=True)

        self.has_gt = has_gt
        self.frames = frames
        root = osp.join(root, split)
        images1 = sorted(glob(osp.join(root, 'image_2/*_10.png')))
        images2 = sorted(glob(osp.join(root, 'image_2/*_11.png')))

        for img1, img2 in zip(images1, images2):
            frame_id = img1.split('/')[-1]
            self.extra_info += [[frame_id]]
            self.image_list += [[img1, img2]]

        if self.has_gt:
            self.flow_list = sorted(glob(osp.join(root, 'flow_occ/*_10.png')))

        self.enforce_dimensions = True
        self.image_x_dim = 375
        self.image_y_dim = 1242

        if len(self.image_list) == 0:
            raise RuntimeWarning(
                "No KITTI data found at dataset root '%s'. Check the configuration file under helper_functions/config_paths.py and add the correct path to the KITTI dataset." % root)


class Demo(FlowDataset):
    def __init__(self, aug_params=None, split='eval', root=None, has_gt=False, n_images=-1, frames=2):
        super(Demo, self).__init__(aug_params)

        self.has_gt = False
        self.n_images = n_images
        self.frames = frames
        images1 = sorted(glob(osp.join(root, '*.png')))[:-1]
        images2 = sorted(glob(osp.join(root, '*.png')))[1:]

        if self.n_images != -1:
            images1 = images1[:self.n_images]
            images2 = images2[:self.n_images]

        image_list = sorted(glob(osp.join(root, '*.png')))
        for i in range(len(image_list)-1):
            self.image_list.append([])
            for d in range(-((self.frames - 1) // 2), self.frames // 2 + 1):
                self.image_list[-1].append(
                    image_list[min(max(i + d, 0), len(image_list) - 1)])
            self.extra_info += [i]

        if len(self.image_list) == 0:
            raise RuntimeWarning(
                "No  data found at dataset root '%s'. Check the configuration file under helper_functions/config_paths.py and add the correct path to the KITTI dataset." % root)

        self.enforce_dimensions = True
        image_path = images1[0]
        with Image.open(image_path) as img:
            self.image_y_dim, self.image_x_dim = img.size


