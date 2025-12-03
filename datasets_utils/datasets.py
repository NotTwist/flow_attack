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
        self.has_depth = False
        self.depth_list = []
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
        disp = False
        if self.has_depth:
            if self.sparse:
                disp, _ = frame_utils.readDispKITTI(self.depth_list[index])
            else:
                disp = None
                
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

        return torch.stack(imgs, dim=0) / 255., flow, valid, disp

    def __rmul__(self, v):
        self.flow_list = v * self.flow_list
        self.image_list = v * self.image_list
        return self

    def __len__(self):
        return len(self.image_list)

    def has_groundtruth(self):
        return self.has_gt
    
    def has_depth(self):
        return self.has_depth

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
    def __init__(self, aug_params=None, split='training', root=None, has_gt=False, frames=2, has_depth=True):
        super(KITTI, self).__init__(aug_params, sparse=True)
        self.has_depth = has_depth
        self.has_gt = has_gt
        self.frames = frames
        self.K_list = []
        root = osp.join(root, split)
        self.root = root
        images1 = sorted(glob(osp.join(root, 'image_2/*_10.png')))
        images2 = sorted(glob(osp.join(root, 'image_2/*_11.png')))

        for img1, img2 in zip(images1, images2):
            frame_id = img1.split('/')[-1]
            self.extra_info += [[frame_id]]
            self.image_list += [[img1, img2]]
            K = self._make_camera_config_from_kitti(frame_id)
            self.K_list.append(K)

        if self.has_gt:
            self.flow_list = sorted(glob(osp.join(root, 'flow_occ/*_10.png')))
        if self.has_depth:
            self.depth_list = sorted(glob(osp.join(root, 'disp_occ_0/*_10.png')))
        self.enforce_dimensions = True
        self.image_x_dim = 375
        self.image_y_dim = 1242

        if len(self.image_list) == 0:
            raise RuntimeWarning(
                "No KITTI data found at dataset root '%s'. Check the configuration file under helper_functions/config_paths.py and add the correct path to the KITTI dataset." % root)


    def _load_intrinsics_for_frame(self, frame_id, cam_id=2):
            """
            Load K for a single frame from its calibration file if available,
            otherwise fallback to a common calib_cam_to_cam.txt.
            """
            # try per-frame calibration first
            calib_dir = osp.join(self.root, "calib_cam_to_cam")
            per_frame_path = osp.join(calib_dir, f"{frame_id[:-7]}.txt")
            if osp.isfile(per_frame_path):
                calib_path = per_frame_path
            else:
                # fallback: single common file
                calib_path = osp.join(self.root, "calib_cam_to_cam.txt")
                if not osp.isfile(calib_path):
                    calib_path = osp.join(osp.dirname(
                        self.root), "calib_cam_to_cam.txt")
                if not osp.isfile(calib_path):
                    raise FileNotFoundError(
                        f"Calibration not found for frame {frame_id} (searched {per_frame_path} and {calib_path})"
                    )

            # parse file
            with open(calib_path, "r") as f:
                lines = f.readlines()

            key = f"P_rect_0{cam_id}"
            K = None
            for line in lines:
                if line.startswith(key):
                    vals = [float(x) for x in line.strip().split()[1:]]
                    P = np.array(vals, dtype=np.float32).reshape(3, 4)
                    K = P[:, :3].copy()
                    break

            if K is None:
                raise RuntimeError(
                    f"Camera intrinsics {key} not found in {calib_path}")

            return torch.from_numpy(K).float()


    def _make_camera_config_from_kitti(self, frame_id: int, cam_id: int = 2) -> dict:
        """
            Parse KITTI calib_cam_to_cam.txt and convert intrinsics to a camera_config
            dictionary compatible with depth_to_local_coordinates().

            Args:
                frame_id (str): frame name (e.g., '000000_10.png')
                cam_id (int): camera ID (usually 2 for left color camera)

            Returns:
                dict: camera_config with image size, field of view, and intrinsics.
            """

        # Путь до файла калибровки (либо на кадр, либо общий)
        calib_dir = osp.join(self.root, "calib_cam_to_cam")
        per_frame_path = osp.join(calib_dir, f"{frame_id[:-7]}.txt")
        if osp.isfile(per_frame_path):
            calib_path = per_frame_path
        else:
            # fallback — общий файл
            common_path = osp.join(self.root, "calib_cam_to_cam.txt")
            if not osp.isfile(common_path):
                common_path = osp.join(osp.dirname(
                    self.root), "calib_cam_to_cam.txt")
            calib_path = common_path

        # --- безопасный парсер файла калибровки ---
        calib = {}
        import re, math
        num_re = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")
        with open(calib_path, "r") as f:
            for line in f:
                if ":" not in line:
                    continue
                key, val = line.strip().split(":", 1)
                key = key.strip()
                val = val.strip()
                # извлекаем только числа
                nums = [float(m.group(0)) for m in num_re.finditer(val)]
                calib[key] = nums if nums else val  # строка или список чисел

        # --- Intrinsics из P_rect_0{cam_id} ---
        p_key = f"P_rect_0{cam_id}"
        if p_key not in calib or len(calib[p_key]) != 12:
            raise RuntimeError(
                f"{p_key} not found or malformed in {calib_path}")
        P = np.array(calib[p_key], dtype=np.float32).reshape(3, 4)
        fx, fy, cx, cy = P[0, 0], P[1, 1], P[0, 2], P[1, 2]

        # --- Image size ---
        s_key = f"S_rect_0{cam_id}" if f"S_rect_0{cam_id}" in calib else f"S_0{cam_id}"
        if s_key in calib and len(calib[s_key]) >= 2:
            width, height = calib[s_key][:2]
        else:
            width, height = 1242.0, 375.0  # defaults for KITTI

        # --- FOV ---
        fov = 2.0 * math.degrees(math.atan(width / (2.0 * fx)))

        # --- Extrinsics (не используется, но добавим для совместимости) ---
        transform_matrix = np.eye(4).tolist()

        camera_config = {
            "image_width": int(width),
            "image_height": int(height),
            "fov": fov,
            "transform_matrix": transform_matrix,
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
        }

        return camera_config

    def __getitem__(self, index):
        imgs, flow, valid, disp = super().__getitem__(index)
        K = self.K_list[index]  # return corresponding K
        return imgs, flow, valid, disp, K

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


