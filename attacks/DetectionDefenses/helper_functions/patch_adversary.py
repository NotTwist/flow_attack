import torch
import os
import torchvision.transforms.functional as tvf
from torch.nn.functional import pad
from numpy import load as load_npy
import torch.nn.functional as F


"""
A module representing an adversarial  patch adversary that performs input manipulations 

Inputs:
P (init) : None, Filepath to png or tensor
I1,I2    : Input image pair to attack as tensors of size (N x 3 x H x W)
y,x      : position of the patch center in image coordinates
angle    : a list [min, max] or a float specifying the rotation angle range
sx,sy    : lists[min,max] or float specifying the vertical/horizontal scaling factor

Especially considered sources:
https://discuss.pytorch.org/t/how-to-make-a-tensor-part-of-model-parameters/51037/6
"""


def circ_mask(X, /, k=0):
    """Creates a circular mask of size X.shape[-2:] with radius k"""
    from math import sqrt
    N, C, H, W = X.size()
    R = torch.zeros((N, 1, H, W))
    for i in range(-H//2, H//2):
        for j in range(-W//2, W//2):
            R[:, :, i+H//2, j+W //
                2] = 1 if sqrt((i+0.5)**2+(j+0.5)**2) < H//2-k else 0
    return R


class PatchAdversary(torch.nn.Module):
    def __init__(self, P, *, angle=[-10, 10], scale=[0.95, 1.05], size=None, change_of_variable=False, random_location=True, image_size=(256, 256),  ellipse_scale_x=1.0, ellipse_scale_y=3, patch_generator=None):
        """
        Adversarial patch that applies the patch to the image. This class is used for the patch attack with defense.

        Args:
            P (str): Path to png file or numpy array. If None, a zero patch is initialized.
            angle (list): [min, max] or float specifying the rotation angle range.
            scale (list): [min, max] or float specifying the vertical/horizontal scaling factor.
            size (int): Size of the patch (used if P is None).
            change_of_variable (bool): If True, a change of variable is used to ensure patch+image is in the range [0,1]. Defaults to False.
            random_location (bool): If True, the patch position will change for each forward pass and transformations are applied. Defaults to True.
            image_size (tuple): Dimensions (H, W) of the images being processed. Used to initialize the random position.
        """
        super(PatchAdversary, self).__init__()

        self.patch_generator = patch_generator
        self._runtime_patch = None
        self._runtime_final_latent = None
        self.last_diffusion_step_stats = {}

        # Initialize Patch from random, filepath, tensor, or diffusion generator.
        if self.patch_generator is not None:
            assert size is not None, "A diffusion patch still requires an explicit patch size"
            self.P = None
            self.M = circ_mask(torch.zeros((1, 3, size, size)))
        elif P is None:
            assert size is not None, "If no patch is given, a size must be specified"
            self.P = torch.zeros((1, 3, size, size))
            self.M = circ_mask(self.P)
        elif isinstance(P, str):
            if P.endswith('.png'):
                from PIL import Image
                P = tvf.to_tensor(Image.open(P)).unsqueeze_(0)
                if change_of_variable:
                    # PNG stores rendered pixels 0.5*(tanh(P_raw)+1).
                    # Recover unconstrained P_raw via inverse: atanh(2*pixel-1).
                    P = torch.atanh((P.clamp(1e-6, 1 - 1e-6) * 2) - 1)
            elif P.endswith('.npy'):
                P = torch.from_numpy(load_npy(P))

            if P.size(1) == 4:  # compatibility with old version
                self.P = P[:, :3]
                self.M = P[:, 3:]
            else:
                raise ValueError("Patch must have 4 channels (RGB and alpha)")
        elif isinstance(P, torch.Tensor):
            if P.size(1) == 4:
                self.P = P[:, :3]
                self.M = P[:, 3:]
            else:
                raise ValueError("Patch must have 4 channels (RGB and alpha)")
    
        if size is not None:
            current_h, current_w = self.M.shape[-2], self.M.shape[-1]
            if self.patch_generator is None and self.P is not None:
                current_h, current_w = self.P.shape[-2], self.P.shape[-1]
                if current_h != size or current_w != size:
                    self.P = F.interpolate(
                        self.P,
                        size=(size, size),
                        mode='bilinear',
                        align_corners=False
                    )
                    self.M = F.interpolate(
                        self.M,
                        size=(size, size),
                        mode='bilinear',
                        align_corners=False
                    )
            elif current_h != size or current_w != size:
                self.M = F.interpolate(
                    self.M,
                    size=(size, size),
                    mode='bilinear',
                    align_corners=False
                )
            self.M = torch.clamp(self.M, 0.0, 1.0)

        if self.patch_generator is None:
            self.P = torch.nn.Parameter(self.P, requires_grad=True)
        self.M = torch.nn.Parameter(self.M, requires_grad=False)
        self.angle = angle
        self.scale = scale
        self.cov = change_of_variable
        self.random_location = random_location
        # Image dimensions and patch size
        self.image_size = image_size
        if size is not None:
            self.patch_size = size
        elif self.patch_generator is not None:
            self.patch_size = self.patch_generator.patch_size
        else:
            self.patch_size = P.shape[-1]

        # Randomly initialize the patch position during object creation
        H, W = self.image_size
        h, w = self.patch_size, self.patch_size
        self.y = torch.randint(H - h, (1,)).item() + (h // 2)
        self.x = torch.randint(W - w, (1,)).item() + (w // 2)
        self.ellipse_scale_x = ellipse_scale_x
        self.ellipse_scale_y = ellipse_scale_y

    def get_P(self, Mask=False):
        if self._runtime_patch is not None:
            P = self._runtime_patch
        elif self.patch_generator is not None:
            P = self.patch_generator.render_patch()
        elif self.cov:
            P = .5 * (torch.tanh(self.P) + 1)
        else:
            P = self.P
        if Mask:
            return torch.concat([P, self.M], dim=1)
        else:
            return P

    def uses_diffusion_patch(self):
        return self.patch_generator is not None

    def prepare_runtime_patch(self):
        if self.patch_generator is None:
            return None
        self._runtime_patch, self._runtime_final_latent = (
            self.patch_generator.render_patch_for_optimization()
        )
        return self._runtime_patch

    def clear_runtime_patch(self):
        self._runtime_patch = None
        self._runtime_final_latent = None

    def step_runtime_patch(self, optimizer_name, lr, max_delta):
        if self.patch_generator is None:
            return
        if self._runtime_patch is None or self._runtime_final_latent is None:
            raise RuntimeError("No runtime diffusion patch is prepared for the current step")
        if self._runtime_patch.grad is None:
            raise RuntimeError("Runtime diffusion patch has no gradients")

        self.last_diffusion_step_stats = self.patch_generator.step(
            self._runtime_final_latent,
            self._runtime_patch.grad,
            optimizer_name=optimizer_name,
            lr=lr,
            max_delta=max_delta,
        )
        self.clear_runtime_patch()
        return self.last_diffusion_step_stats


    @staticmethod
    def _plane_basis(normal: torch.Tensor):
        """
        normal: [3] (может быть не нормирована)
        -> t1, t2, n: ортонормированный базис в плоскости

        n  — нормаль к плоскости (дорога)
        t2 — ось ВДОЛЬ дороги (как можно ближе к направлению камеры [0,0,1])
        t1 — ось ПОПЕРЁК дороги
        """
        # нормализуем нормаль
        n = normal / (normal.norm() + 1e-8)

        # направление вперёд камеры (оптическая ось)
        cam_forward = torch.tensor([0., 0., 1.], device=n.device)

        # проектируем cam_forward на плоскость, чтобы получить t2
        # вычитаем компоненту вдоль n
        t2 = cam_forward - torch.dot(cam_forward, n) * n

        # если вдруг нормаль почти совпала с cam_forward (почти смотрим перпендикулярно в плоскость)
        if t2.norm() < 1e-4:
            # пробуем альтернативное направление — ось X
            alt = torch.tensor([1., 0., 0.], device=n.device)
            t2 = alt - torch.dot(alt, n) * n

        t2 = t2 / (t2.norm() + 1e-8)

        # t1 = n × t2  — вторая ось в плоскости, поперёк дороги
        t1 = torch.cross(n, t2)
        t1 = t1 / (t1.norm() + 1e-8)

        return t1, t2, n
    
    @staticmethod
    def _build_K_mat(K_entry, device):
        """
        K_entry: либо dict с fx,fy,cx,cy, либо тензор [3,3]
        """
        if isinstance(K_entry, dict):
            fx = float(K_entry["fx"])
            fy = float(K_entry["fy"])
            cx = float(K_entry["cx"])
            cy = float(K_entry["cy"])
            K_mat = torch.tensor([[fx, 0., cx],
                                  [0., fy, cy],
                                  [0., 0., 1.]], device=device)
        elif isinstance(K_entry, torch.Tensor):
            if K_entry.ndim != 2 or K_entry.shape != (3, 3):
                raise ValueError(f"Unexpected K shape: {K_entry.shape}")
            K_mat = K_entry.to(device)
        else:
            raise TypeError(f"Unsupported K type: {type(K_entry)}")
        return K_mat




        cv2.imwrite(out_path, img)

    def _project_single_on_plane(
        self,
        image: torch.Tensor,   # [3,H,W]
        patch: torch.Tensor,   # [3,Hp,Wp]
        K_mat: torch.Tensor,   # [3,3]
        normal: torch.Tensor,  # [3]
        d_plane: torch.Tensor,  # скаляр
        uv_center,             # (x, y) центр патча в пикселях
        road_mask=None         # [H,W] bool или [1,H,W]
    ):
        C, H, W = image.shape
        device = image.device

        # базис в плоскости
        t1, t2, n = self._plane_basis(normal)

        # центр патча в пикселях
        u0, v0 = uv_center  # x, y

        fx = K_mat[0, 0]
        fy = K_mat[1, 1]
        cx = K_mat[0, 2]
        cy = K_mat[1, 2]

        # луч через (u0,v0)
        x = (u0 - cx) / fx
        y = (v0 - cy) / fy
        ray_dir = torch.tensor([x, y, 1.0], device=device)
        ray_dir = ray_dir / (ray_dir.norm() + 1e-8)

        denom = torch.dot(n, ray_dir)
        t = -d_plane / (denom + 1e-8)
        p0 = t * ray_dir  # [3] — центр патча в 3D
        Zc = p0[2].abs()  # глубина центра

        # размер патча в пикселях
        Hp, Wp = patch.shape[-2:]
        # физический размер патча в метрах
        Wm = Wp * Zc / fx
        Hm = Hp * Zc / fy
        # print(Wm, Hm)
        # лучи для всех пикселей
        ys, xs = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing='ij'
        )
        x_cam = (xs - cx) / fx
        y_cam = (ys - cy) / fy
        z_cam = torch.ones_like(x_cam)

        dirs = torch.stack([x_cam, y_cam, z_cam], dim=0)  # [3,H,W]
        dirs = dirs / (dirs.norm(dim=0, keepdim=True) + 1e-8)

        ndot = (n.view(3, 1, 1) * dirs).sum(dim=0)   # [H,W]
        t_all = -d_plane / (ndot + 1e-8)           # [H,W]
        P = dirs * t_all.unsqueeze(0)              # [3,H,W]

        # координаты в базисе плоскости
        rel = P - p0.view(3, 1, 1)
        xi = (rel * t1.view(3, 1, 1)).sum(dim=0)   # [H,W]
        eta = (rel * t2.view(3, 1, 1)).sum(dim=0)  # [H,W]

        # эллипсоидная маска: растягиваем патч в базисе (xi, eta)
        sx = self.ellipse_scale_x  # >1 => шире по t1
        sy = self.ellipse_scale_y  # >1 => длиннее по t2

        # сначала масштабируем физические координаты в плоскости
        xi_scaled = xi / sx
        eta_scaled = eta / sy

        # затем нормируем к [-1,1] для grid_sample
        xi_norm = xi_scaled / (Wm / 2.0)
        eta_norm = eta_scaled / (Hm / 2.0)

        # эллиптическая область: (xi_scaled, eta_scaled) попадает в круг
        r2 = xi_norm**2 + eta_norm**2
        inside = r2 <= 1.0
        M = inside.unsqueeze(0)

        grid = torch.stack([xi_norm, eta_norm], dim=-
                           1).unsqueeze(0).float()  # [1,H,W,2]

        patch_in = patch.unsqueeze(0)  # [1,3,Hp,Wp]
        patch_on_img = F.grid_sample(
            patch_in,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )[0]  # [3,H,W]


        M = inside.unsqueeze(0)  # [1,H,W] bool

        if road_mask is not None:
            if road_mask.dim() == 2:
                road_m = road_mask.unsqueeze(0)
            else:
                road_m = road_mask
            # M = M & road_m

        M_f = M.float()
        I_out = image * (1.0 - M_f) + patch_on_img * M_f
        # print("xi range:", float(xi.min()), float(xi.max()))
        # print("eta range:", float(eta.min()), float(eta.max()))
        # print("xi_norm range:", float(xi_norm.min()), float(xi_norm.max()))
        # print("eta_norm range:", float(eta_norm.min()), float(eta_norm.max()))
        # print("M mean:", float(M.float().mean()))

        return I_out, M

    def forward(self, I1, I2, y=None, x=None,
                K=None, planes=None, road_masks=None, flow_shift=0.0):
        if (K is not None) and (planes is not None):
            B, C, H, W = I1.shape
            device = I1.device

            P_tex = self.get_P(Mask=False)[0]  # [3,Hp,Wp]

            # список центров для возвращения
            y_centers = []
            x_centers = []

            I1_list, I2_list, M_list = [], [], []

            # приведение K к списку
            if isinstance(K, dict) or (isinstance(K, torch.Tensor) and K.ndim == 2):
                K_list = [K for _ in range(B)]
            else:
                K_list = K

            max_tries = 10  

            for b in range(B):
                normal_b, d_b = planes[b]
                K_mat_b = self._build_K_mat(K_list[b], device=device)

                road_mask_b = None
                if road_masks is not None:
                    if road_masks.dim() == 4:
                        road_mask_b = road_masks[b, 0]  # [H,W]
                    elif road_masks.dim() == 3:
                        road_mask_b = road_masks[b]     # [H,W]
                    else:
                        raise ValueError(
                            f"Unexpected road_masks shape: {road_masks.shape}"
                        )

                # подготовим road_mask как bool [H,W], если есть
                road_mask_bool = None
                if road_mask_b is not None:
                    if road_mask_b.dim() == 3:
                        road_mask_b = road_mask_b[0]
                    road_mask_bool = (road_mask_b > 0.5)

                found_good_pos = False

                for attempt in range(max_tries):
                    # выбор центра патча
                    H_half = int(H * 0.7)

                    if self.random_location:
                        if road_mask_bool is not None:
                            ys_valid, xs_valid = torch.where(road_mask_bool)

                            # фильтруем по нижней половине
                            mask_lower = ys_valid >= H_half
                            ys_valid = ys_valid[mask_lower]
                            xs_valid = xs_valid[mask_lower]

                            # --- фильтруем по горизонтальной полосе по центру ---
                            mask_center = (xs_valid >= int(W*0.45)) & (xs_valid <= int(W*0.55))
                            ys_valid = ys_valid[mask_center]
                            xs_valid = xs_valid[mask_center]

                            if len(xs_valid) == 0:
                                # fallback: случайная точка в нижней половине и центральной полосе
                                y_c = torch.randint(H_half, H, (1,)).item()
                                x_c = torch.randint(int(W*0.45), int(W*0.55), (1,)).item()
                            else:
                                idx = torch.randint(len(xs_valid), (1,)).item()
                                y_c = ys_valid[idx].item()
                                x_c = xs_valid[idx].item()


                        else:
                            # нет road_mask → просто выбираем в нижней половине
                            y_c = torch.randint(H_half, H, (1,)).item()
                            x_c = torch.randint(int(W*0.45), int(W*0.55), (1,)).item()

                    else:
                        # фиксированная позиция
                        y_c = self.y
                        x_c = self.x

                    uv_center = (x_c, y_c)

                    I1_p_b, M_b = self._project_single_on_plane(
                        image=I1[b],
                        patch=P_tex,
                        K_mat=K_mat_b,
                        normal=normal_b,
                        d_plane=d_b,
                        uv_center=uv_center,
                        road_mask=road_mask_b
                    )

                    uv_center_f2 = (uv_center[0], uv_center[1] + flow_shift) \
                        if flow_shift != 0 else uv_center
                    I2_p_b, _ = self._project_single_on_plane(
                        image=I2[b],
                        patch=P_tex,
                        K_mat=K_mat_b,
                        normal=normal_b,
                        d_plane=d_b,
                        uv_center=uv_center_f2,
                        road_mask=road_mask_b
                    )


                    # M_b: [1,H,W] bool/float
                    M_inside = M_b[0].bool()        # где патч присутствует

                    # 1) проверяем, что весь патч лежит внутри road_mask
                    if road_mask_bool is not None:
                        outside = M_inside & (~road_mask_bool)
                        if outside.any():
                            # часть патча вне дороги — пробуем другую позицию
                            continue

                    # 2) проверяем, что патч НЕ обрезан границами изображения
                    ys, xs = torch.where(M_inside)
                    if ys.numel() == 0:
                        # патч не попал ни в один пиксель (что-то пошло не так) — пробуем заново
                        continue

                    y_min = ys.min().item()
                    y_max = ys.max().item()
                    x_min = xs.min().item()
                    x_max = xs.max().item()

                    # если патч касается самых краёв, считаем, что он вылез за границы
                    if y_min <= 0 or x_min <= 0 or y_max >= H - 1 or x_max >= W - 1:
                        # значит, часть патча была бы за пределами изображения — пробуем другую позицию
                        continue

                    # если сюда дошли, позиция хорошая
                    found_good_pos = True
                    break


                # если не нашли идеальную позицию — используем последнюю
                # (I1_p_b, I2_p_b, M_b уже определены в последней итерации цикла)
                I1_list.append(I1_p_b)
                I2_list.append(I2_p_b)
                M_list.append(M_b)

                y_centers.append(y_c)
                x_centers.append(x_c)

            I1_p_batch = torch.stack(I1_list, dim=0)
            I2_p_batch = torch.stack(I2_list, dim=0)
            M_batch = torch.stack(M_list, dim=0)

            # возвращаем центры:
            # - скаляры, если B == 1 (как раньше),
            # - списки, если B > 1.
            if B == 1:
                y_out = y_centers[0]
                x_out = x_centers[0]
            else:
                y_out = y_centers
                x_out = x_centers
            # print(x_out/W, y_out/H)
            
            return I1_p_batch, I2_p_batch, M_batch, y_out, x_out

        # --------- старый 2D-режим (без K/planes) остаётся как был ---------
        # Use pre-initialized position if random_location is False
        if not self.random_location:
            y, x = self.y, self.x

        # Apply transformations only if random_location is True
        if self.random_location:
            scale = self.scale
            angle = self.angle
            current_patch = self.get_P(Mask=False)

            # Initialize Transformations
            scale = torch.rand((1,)).item() * (scale[1] - scale[0]) + scale[0] \
                if isinstance(scale, list) else scale

            angle = torch.rand((1,)).item() * (angle[1] - angle[0]) + angle[0] \
                if isinstance(angle, list) else angle

            # Apply Transformations
            if angle != 0 and scale != 1:
                P_rot = tvf.rotate(current_patch, angle)
                P_res = tvf.resize(
                    P_rot, (int(scale * P_rot.size(2)), int(scale * P_rot.size(3))))
                M = tvf.rotate(self.M, angle)
                M = tvf.resize(
                    M, (int(scale * M.size(2)), int(scale * M.size(3))))
            else:
                P_res = current_patch
                M = self.M
        else:
            # Skip transformations
            P_res = self.get_P(Mask=False)
            M = self.M

        # Initialize pos, patch, and patch size
        N, C, H, W = I1.size()
        n, c, h, w = P_res.size()
        assert H >= h and W >= w, "Patch size must be smaller than image size"

        if self.random_location:
            # Random location for each forward pass
            y = torch.randint(H - h, (1,)).item() + (h // 2)
            x = torch.randint(W - w, (1,)).item() + (w // 2)

        # Resize patch to size of one image
        P_glob = pad(P_res, ((x - w // 2), W - w - (x - w // 2),
                     (y - h // 2), H - h - (y - h // 2)))
        M_glob = pad(M, ((x - w // 2), W - w - (x - w // 2),
                     (y - h // 2), H - h - (y - h // 2)))
        # get_P() already applies the cov transform, so P_glob is always in [0,1]
        P_glob_cov = torch.clamp(P_glob, 0.0, 1.0)

        # Ceil patch to avoid black borders
        M_glob = torch.ceil(M_glob)

        # Construct result (replace image where patch is not transparent)
        R1 = (1 - M_glob) * I1 + P_glob_cov * M_glob

        if flow_shift != 0:
            y2 = y + int(round(flow_shift))
            y2 = max(h // 2, min(H - h // 2 - 1, y2))
            P_glob2 = pad(P_res, ((x - w // 2), W - w - (x - w // 2),
                         (y2 - h // 2), H - h - (y2 - h // 2)))
            M_glob2 = pad(M, ((x - w // 2), W - w - (x - w // 2),
                         (y2 - h // 2), H - h - (y2 - h // 2)))
            P_glob_cov2 = torch.clamp(P_glob2, 0.0, 1.0)
            M_glob2 = torch.ceil(M_glob2)
            R2 = (1 - M_glob2) * I2 + P_glob_cov2 * M_glob2
        else:
            R2 = (1 - M_glob) * I2 + P_glob_cov * M_glob

        M_out = torch.where(M_glob > 0, 1, 0).to(I1.device)

        return R1, R2, M_out, y, x


    def save_png(self, name):
        if os.path.dirname(name) and not os.path.exists(os.path.dirname(name)):
            os.makedirs(os.path.dirname(name))
        patch_rgba = self.get_P(Mask=True)
        if not torch.isfinite(patch_rgba).all():
            patch_rgba = torch.nan_to_num(patch_rgba, nan=0.0, posinf=1.0, neginf=0.0)
        tvf.to_pil_image(torch.clamp(patch_rgba, 0, 1)[0]).save(name)

@torch.no_grad()
def debug_plane_ellipse(rgb_tensor, K, p0, t1, t2, R=2.0, out_path="debug_plane_ellipse.png"):
    """
    p0, t1, t2: [3] в системе камеры
    R: радиус круга в метрах
    """
    # K -> fx,fy,cx,cy
    if isinstance(K, dict):
        fx = float(K["fx"])
        fy = float(K["fy"])
        cx = float(K["cx"])
        cy = float(K["cy"])
    elif isinstance(K, torch.Tensor):
        fx = K[0, 0].item()
        fy = K[1, 1].item()
        cx = K[0, 2].item()
        cy = K[1, 2].item()
    else:
        raise TypeError

    # берём несколько точек по окружности в плоскости
    import numpy as np
    thetas = np.linspace(0, 2*np.pi, 64, endpoint=True)
    pts_3d = []
    for th in thetas:
        th_t = torch.tensor(th, dtype=p0.dtype, device=p0.device)
        offset = R * (torch.cos(th_t)*t1 + torch.sin(th_t)*t2)  # [3]
        pts_3d.append(p0 + offset)
    pts_3d = torch.stack(pts_3d, dim=0)  # [N,3]

    X = pts_3d[:, 0]
    Y = pts_3d[:, 1]
    Z = pts_3d[:, 2].clamp(min=1e-3)
    u = fx * X / Z + cx
    v = fy * Y / Z + cy

    u = u.cpu().numpy().astype(np.int32)
    v = v.cpu().numpy().astype(np.int32)

    rgb = rgb_tensor.detach().cpu().permute(1, 2, 0).numpy()
    rgb = (rgb * 255.0).clip(0, 255).astype(np.uint8)
    import cv2
    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    pts_2d = np.stack([u, v], axis=1)
    cv2.polylines(img, [pts_2d], isClosed=True,
                    color=(0, 0, 255), thickness=2)

    cv2.imwrite(out_path, img)
