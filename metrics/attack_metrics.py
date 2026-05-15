import os
import torch
import numpy as np
import mlflow
import cv2
from typing import Literal
from utils.process_images import (
    quickvis_flow,
    save_depth,
    save_depth_pair_common_scale,
    save_flow_pair_common_scale,
    save_mask_overlay_pair,
    save_segmentation,
)
from datetime import datetime
from ptlflow.utils import flow_utils
import cv2 as cv
from scipy.stats import pearsonr, spearmanr
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

class AttackMetricsTracker:
    def __init__(self, output_dir="experiment_data", experiment_name="attack_experiment", run_name=None, args=None, train=False):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        mlflow.set_experiment(experiment_name)

        if run_name is None:
            current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            if train:
                run_name = f"train_{args.model_name}_{args.attack_type}_{args.dataset}_{current_date}"
            else:
                run_name = f"{args.model_name}_{args.attack_type}_{args.dataset}_{current_date}"
        self.run_name = run_name
        mlflow.start_run(run_name=self.run_name)
        print(
            f"Running experiment: {experiment_name}, with run name: {run_name}")

        # Log parameters
        args_dict = vars(args)
        mlflow.log_params(args_dict)
        # mlflow.log_param("model_name", args.net)
        # mlflow.log_param("attack_type", args.attack)
        # mlflow.log_param("target", args.target)
        # mlflow.log_param("steps", args.steps)
        # mlflow.log_param("dataset", args.dataset)
        # mlflow.log_param("small_run", args.small_run)
        # mlflow.log_param("output_dir", args.output_dir)
        # mlflow.log_param("no_softmax", args.no_softmax)
        # mlflow.log_param("epsilon", args.epsilon)
        # mlflow.log_param("alpha", args.alpha)
        # mlflow.log_param("saved_iterations", args.save_iterations)
        # if args.target_layer:
        #     mlflow.log_param("target_layer", args.target_layer)
        self.class_winning_frames = {}

        # Store save_iterations as a list of steps
        # Ensure it’s a list
        self.saved_iterations = args.saved_iterations if args.saved_iterations else []
        self.reset()

    @staticmethod
    def _finite_float(value, default=0.0):
        """Convert metric-like values to finite Python floats before MLflow logging."""
        if torch.is_tensor(value):
            if value.numel() == 0:
                return default
            value = torch.nan_to_num(value.detach().float(), nan=default, posinf=default, neginf=default).mean().item()
        elif isinstance(value, np.ndarray):
            if value.size == 0:
                return default
            value = np.nan_to_num(value.astype(np.float64), nan=default, posinf=default, neginf=default).mean()

        try:
            value = float(value)
        except (TypeError, ValueError):
            return default
        return value if np.isfinite(value) else default

    @staticmethod
    def _contains_nonfinite(value):
        if torch.is_tensor(value):
            if value.numel() == 0:
                return False
            return not bool(torch.isfinite(value.detach()).all().item())
        if isinstance(value, np.ndarray):
            if value.size == 0:
                return False
            return not bool(np.isfinite(value).all())
        try:
            return not bool(np.isfinite(float(value)))
        except (TypeError, ValueError):
            return True

    def log_metric(self, name, value, step):
        if self._contains_nonfinite(value):
            mlflow.log_metric(f"{name}_nonfinite", 1.0, step)
        mlflow.log_metric(name, self._finite_float(value), step)
    
    def reset(self):
        """Reset cumulative metric values and counts."""
        self.cumulative_metrics = {
            "aee_init_attack": 0.0,
            "aee_init_gt": 0.0,
            "aee_attacked_gt": 0.0,
            "gt_diff_error": 0.0,
            "reconstruction_error": 0.0,
            "aee_target_attack": 0.0,
            "aee_clean_target": 0.0,
            "mde_rmse_init_attack": 0.0,
            "mde_rmse_target_attack": 0.0,
            "mde_rmse_clean_target": 0.0,
            "ase_init_attack": 0.0,
            "ase_target_attack": 0.0, 
            "iou_init_attack": 0.0,
            "iou_target_attack": 0.0,
            # Mean of per-task target-distance ratios (lower ⇒ stronger targeted attack)
            "multitask_robustness_score": 0.0,
            "mrs_gamma_flow": 0.0,
            "mrs_gamma_depth": 0.0,
            "mrs_gamma_seg": 0.0,
        }
        self.count = 0

        # Dynamically add metric keys for custom iterations
        for step in self.saved_iterations:
            self.cumulative_metrics.update({
                f"aee_init_attack_step_{step}": 0.0,
                f"aee_init_gt_step_{step}": 0.0,
                f"aee_attacked_gt_step_{step}": 0.0,
                f"gt_diff_error_step_{step}": 0.0,
                f"reconstruction_error_step_{step}": 0.0,
                f"aee_target_attack_step_{step}": 0.0,
                f"mde_rmse_init_attack_step_{step}": 0.0,
                f"mde_rmse_target_attack_step_{step}": 0.0,
                f"ase_init_attack_step_{step}": 0.0,
                f"ase_target_attack_step_{step}": 0.0,
                f"iou_init_attack_step_{step}": 0.0,
                f"iou_target_attack_step_{step}": 0.0
            })

    def compute_aee(self, flow1, flow2, mask=None):
        """
        Compute the Average Endpoint Error (AEE) between two flow fields.
        Args:
            flow1 (torch.Tensor or np.ndarray): Flow field [B,2,H,W] or [2,H,W].
            flow2 (torch.Tensor or np.ndarray): Flow field [B,2,H,W] or [2,H,W].
        Returns:
            float: Average endpoint error.
        """
        if torch.is_tensor(flow1):
            flow1 = flow1.detach().cpu().numpy()
        if torch.is_tensor(flow2):
            flow2 = flow2.detach().cpu().numpy()
        # Compute per-pixel error, assuming flows have shape [2,H,W] or [B,2,H,W]
        diff_squared = (flow1 - flow2)**2
        if mask is not None:
            diff_squared = np.where(mask == 1, diff_squared, 0)
        if diff_squared.ndim == 3:
            # here, dim=0 is the 2-dimension (u and v direction of flow [2,M,N]) , which needs to be added BEFORE taking the square root. To get the length of a flow vector, we need to do sqrt(u_ij^2 + v_ij^2)
            epe = np.mean(np.sqrt(np.sum(diff_squared, axis=0)))
        elif diff_squared.ndim == 4:
            # here, dim=0 is the 2-dimension (u and v direction of flow [b,2,M,N]) , which needs to be added BEFORE taking the square root. To get the length of a flow vector, we need to do sqrt(u_ij^2 + v_ij^2)
            epe = np.mean(np.sqrt(np.sum(diff_squared, axis=1)))
        else:
            raise ValueError("The flow tensors for which the EPE should be computed do not have a valid number of dimensions (either [b,2,M,N] or [2,M,N]). Here: " + str(
                flow1.size()) + " and " + str(flow1.size()))
        return epe



    def compute_reconstruction_error(self, img1, img2, flow, mask=None):
        """
        Photometric reconstruction error (L1):
        Warp img2 back into img1 coordinate system using forward optical flow.

        Args:
            img1: torch.Tensor [C,H,W] or [B,C,H,W]
            img2: same shape as img1
            flow: torch.Tensor [2,H,W] or [B,2,H,W]
            mask: optional mask [H,W] or [B,1,H,W]

        Returns:
            float — mean photometric L1 error
            torch.Tensor — warped img2 in CHW or BCHW format
        """

        # ---- 1. Ensure batch dimension ----
        if img1.ndim == 3:  # CHW
            img1 = img1.unsqueeze(0)  # B=1
        if img2.ndim == 3:
            img2 = img2.unsqueeze(0)
        if flow.ndim == 3:  # 2,H,W
            flow = flow.unsqueeze(0)

        device = flow.device
        img1 = img1.to(device)
        img2 = img2.to(device)
        flow = flow.to(device)
        if mask is not None and torch.is_tensor(mask):
            mask = mask.to(device)
            
        B, C, H, W = img1.shape
        device = img1.device

        # ---- 2. Build meshgrid ----
        yy, xx = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing="ij"
        )

        # Forward flow defines sampling coords in img2:
        # x2 = x1 + flow_x; y2 = y1 + flow_y
        x_new = xx[None] + flow[:, 0]
        y_new = yy[None] + flow[:, 1]

        # ---- 3. Normalize coords for grid_sample ----
        x_norm = 2 * (x_new / (W - 1)) - 1
        y_norm = 2 * (y_new / (H - 1)) - 1
        grid = torch.stack((x_norm, y_norm), dim=-1)  # [B,H,W,2]

        # ---- 4. Warp img2 → img1 space ----
        img2_warped = F.grid_sample(
            img2,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True
        )

        # ---- 5. Compute photometric L1 error ----
        photometric_error = (img1 - img2_warped).abs().mean(dim=1)  # [B,H,W]

        # ---- 6. Apply mask if provided ----
        if mask is not None:
            if torch.is_tensor(mask):
                if mask.ndim == 2:  # H,W
                    mask = mask.unsqueeze(0)  # B=1
                if mask.ndim == 3:  # B,H,W
                    mask = mask.unsqueeze(1)  # B,1,H,W

                mask = mask.squeeze(1)  # B,H,W
                photometric_error = photometric_error[mask]

        # ---- 7. Return mean error + warped image (CHW or BCHW) ----
        error_value = photometric_error.mean().item()

        # if original was CHW, output should be CHW
        if img1.shape[0] == 1:
            img2_warped = img2_warped.squeeze(0)

        return error_value, img2_warped


        
    def compute_rmse(self, pred_depth: torch.Tensor, gt_depth: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Compute RMS Error between predicted and ground-truth depth maps.

        Args:
            pred_depth: Tensor of shape [B,1,H,W] or [1,H,W] of predicted depths.
            gt_depth:   Tensor of same shape as pred_depth of ground truth depths.
            mask:       Optional boolean mask tensor of shape [B,1,H,W] or [1,H,W]; only masked pixels contribute.

        Returns:
            Tensor of shape [B] containing per-sample RMSE.
        """
        # ensure same shape
        # print(pred_depth.max())
        if pred_depth.ndim == 3:
            pred_depth = pred_depth.unsqueeze(1)
        if gt_depth.ndim == 3:
            gt_depth = gt_depth.unsqueeze(1)
        pred_depth = torch.nan_to_num(pred_depth, nan=0.0, posinf=0.0, neginf=0.0)
        gt_depth = torch.nan_to_num(gt_depth, nan=0.0, posinf=0.0, neginf=0.0)

        B = pred_depth.size(0)

        # default mask = all valid pixels
        if mask is None:
            mask = torch.ones_like(gt_depth, dtype=torch.bool)
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        mask = mask.to(device=pred_depth.device)
        if mask.shape[0] == 1 and B > 1:
            mask = mask.expand(B, -1, -1, -1)
        if mask.shape[-2:] != pred_depth.shape[-2:]:
            mask = F.interpolate(
                mask.float(),
                size=pred_depth.shape[-2:],
                mode="nearest",
            )
        mask = mask > 0.5

        # squeeze channel dimension (B,1,H,W) -> (B,H,W)
        pred_depth = pred_depth.squeeze(1)
        gt_depth = gt_depth.squeeze(1)
        mask = mask.squeeze(1)

        # compute per-sample RMSE safely without flattening across batch
        rmse_list = []
        for b in range(B):
            valid_mask = mask[b] 
            if not valid_mask.any():
                rmse_list.append(pred_depth[b].sum() * 0.0)
                continue
            diff = pred_depth[b][valid_mask] - gt_depth[b][valid_mask]
            mse = (diff ** 2).mean()
            rmse_list.append(torch.sqrt(mse))

        return torch.stack(rmse_list)

    def compute_depth_correlation(self, pred_depth: torch.Tensor, gt_depth: torch.Tensor, mask: torch.Tensor = None):
        if pred_depth.ndim == 3:
            pred_depth = pred_depth.unsqueeze(1)
        if gt_depth.ndim == 3:
            gt_depth = gt_depth.unsqueeze(1)
        pred_depth = torch.nan_to_num(pred_depth, nan=0.0, posinf=0.0, neginf=0.0)
        gt_depth = torch.nan_to_num(gt_depth, nan=0.0, posinf=0.0, neginf=0.0)

        if mask is not None:
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)
            mask = mask.to(device=pred_depth.device)
            if mask.shape[0] == 1 and pred_depth.shape[0] > 1:
                mask = mask.expand(pred_depth.shape[0], -1, -1, -1)
            if mask.shape[-2:] != pred_depth.shape[-2:]:
                mask = F.interpolate(
                    mask.float(),
                    size=pred_depth.shape[-2:],
                    mode="nearest",
                )
            mask = mask > 0.5
        else:
            mask = torch.ones_like(pred_depth, dtype=torch.bool)

        pred = pred_depth.detach().cpu().numpy().flatten()
        gt = gt_depth.detach().cpu().numpy().flatten()
        mask_np = mask.detach().cpu().numpy().flatten().astype(bool)
        pred = pred[mask_np]
        gt = gt[mask_np]

        if pred.size < 2 or gt.size < 2:
            return 0.0, 0.0
        if np.std(pred) < 1e-12 or np.std(gt) < 1e-12:
            return 0.0, 0.0

        # print(pred.shape, gt_depth.shape)
        pearson_corr, _ = pearsonr(gt, pred)
        spearman_corr, _ = spearmanr(gt, pred)
        if not np.isfinite(pearson_corr):
            pearson_corr = 0.0
        if not np.isfinite(spearman_corr):
            spearman_corr = 0.0
        return pearson_corr, spearman_corr
        
    def compute_segmentation_attack_error(self, original_seg, attacked_seg, target_seg=None):
        """
        Compute segmentation attack error independent of GT.
        Args:
            original_seg (torch.Tensor or np.ndarray): Original predicted segmentation [B,H,W].
            attacked_seg (torch.Tensor or np.ndarray): Attacked predicted segmentation [B,H,W].
            target_seg (torch.Tensor or np.ndarray, optional): Target segmentation [B,H,W].
        Returns:
            dict: {
                'ase_init_attack': error between original and attacked prediction,
                'ase_target_attack': error between attacked prediction and target (if provided)
            }
        """
        if torch.is_tensor(original_seg):
            original_seg = original_seg.detach().cpu().numpy()
        if torch.is_tensor(attacked_seg):
            attacked_seg = attacked_seg.detach().cpu().numpy()
        if target_seg is not None and torch.is_tensor(target_seg):
            target_seg = target_seg.detach().cpu().numpy()

        # Error between original and attacked
        ase_init_attack = self._finite_float(np.mean(original_seg != attacked_seg))

        # Error between attacked and target, if target exists
        ase_target_attack = self._finite_float(
            np.mean(attacked_seg != target_seg) if target_seg is not None else 0.0
        )

        return {
            "ase_init_attack": ase_init_attack,
            "ase_target_attack": ase_target_attack
        }

    def compute_iou(self, pred_seg, gt_seg, num_classes, mask=None,  targeted=False, classes=[13, 17]):
        """
        Compute mean IoU for semantic segmentation.

        Args:
            pred_seg: predicted segmentation [B,H,W] or [H,W]
            gt_seg:   ground truth segmentation [B,H,W] or [H,W]
            num_classes: number of segmentation classes
            mask: optional mask [B,H,W] or [H,W] (boolean tensor)

        Returns:
            float — mean IoU over all classes present in GT
        """
        # Convert tensors -> numpy
        if torch.is_tensor(pred_seg):
            pred_seg = pred_seg.detach().cpu().numpy()
        if torch.is_tensor(gt_seg):
            gt_seg = gt_seg.detach().cpu().numpy()
        if mask is not None and torch.is_tensor(mask):
            mask = mask.detach().cpu().numpy().astype(bool)

        # Ensure batch dimension
        if pred_seg.ndim == 2:
            pred_seg = pred_seg[None, ...]
            gt_seg = gt_seg[None, ...]
            if mask is not None:
                mask = mask[None, ...]

        B = pred_seg.shape[0]
        ious = []

        # compute iou for semantically similar classes
        if targeted:
            max_iou = -1
            for cls in classes:
                pred_cls = (pred_seg == cls)


                if mask is not None:
                    pred_cls = pred_cls & mask

                intersection = pred_cls.sum()
                union = mask.sum()
                # print(intersection, union)
                # Ignore classes that do not appear in GT
                if union == 0:
                    continue
                iou = intersection / union if union > 0 else 0

                max_iou = max(max_iou, iou)
        
            return self._finite_float(max_iou if max_iou >= 0 else 0.0)
        
        for cls in range(num_classes):
                
            # Boolean masks for this class
            pred_cls = (pred_seg == cls)
            gt_cls = (gt_seg == cls)

            if mask is not None:
                pred_cls = pred_cls & mask
                gt_cls = gt_cls & mask

            intersection = np.logical_and(pred_cls, gt_cls).sum()
            union = np.logical_or(pred_cls, gt_cls).sum()

            # Ignore classes that do not appear in GT
            if union == 0:
                continue

            ious.append(intersection / union)

        if len(ious) == 0:
            return 0.0

        return self._finite_float(np.mean(ious))

    def compute_multitask_robustness_score(
        self,
        original_flow,
        attacked_flow,
        mask=None,
        target_flow=None,
        original_depth=None,
        attacked_depth=None,
        target_depth=None,
        original_seg=None,
        attacked_seg=None,
        target_seg=None,
    ):
        """
        Multi-task score: mean of per-task ratios «distance to attack target after attack»
        divided by «distance to the same target on the clean prediction».

        Per task (smaller ratio ⇒ prediction moved closer to the target, i.e. stronger attack):
          - Flow:  AEE(f_adv, f_tgt) / max(ε, AEE(f_clean, f_tgt))
          - Depth: RMSE(d_adv, d_tgt) / max(ε, RMSE(d_clean, d_tgt))
          - Seg:   mean(seg_adv ≠ seg_tgt) / max(ε, mean(seg_clean ≠ seg_tgt))

        Only tasks with all required tensors are included; the reported value is their mean.
        """
        eps = 1e-8
        gammas = []

        if (
            original_flow is not None
            and attacked_flow is not None
            and target_flow is not None
        ):
            num = float(self.compute_aee(attacked_flow, target_flow, mask))
            den = float(self.compute_aee(original_flow, target_flow, mask))
            gamma = self._finite_float(num / max(eps, den))
            gammas.append(("flow", gamma))

        if (
            original_depth is not None
            and attacked_depth is not None
            and target_depth is not None
        ):
            rmse_adv = float(
                self.compute_rmse(attacked_depth, target_depth, mask).mean().item()
            )
            rmse_clean = float(
                self.compute_rmse(original_depth, target_depth, mask).mean().item()
            )
            gamma = self._finite_float(rmse_adv / max(eps, rmse_clean))
            gammas.append(("depth", gamma))

        if (
            original_seg is not None
            and attacked_seg is not None
            and target_seg is not None
        ):
            o = original_seg.detach().cpu().numpy() if torch.is_tensor(original_seg) else original_seg
            a = attacked_seg.detach().cpu().numpy() if torch.is_tensor(attacked_seg) else attacked_seg
            t = target_seg.detach().cpu().numpy() if torch.is_tensor(target_seg) else target_seg
            if o.ndim == 4:
                o = o.squeeze(0)
            if a.ndim == 4:
                a = a.squeeze(0)
            if t.ndim == 4:
                t = t.squeeze(0)
            err_adv = self._finite_float(np.mean(a != t))
            err_clean = self._finite_float(np.mean(o != t))
            gamma = self._finite_float(err_adv / max(eps, err_clean))
            gammas.append(("seg", gamma))

        if not gammas:
            return None, {}

        mrs = self._finite_float(sum(g for _, g in gammas) / len(gammas))
        gamma_dict = {name: g for name, g in gammas}
        return mrs, gamma_dict

    def log_semantic_drift(self, attacked_seg, target_seg, mask=None):
        if torch.is_tensor(attacked_seg):
            attacked_seg = attacked_seg.detach().cpu().numpy()
        if torch.is_tensor(target_seg):
            target_seg = target_seg.detach().cpu().numpy()
        if mask is not None and torch.is_tensor(mask):
            mask = mask.detach().cpu().numpy().astype(bool)

        if attacked_seg.ndim == 4:
            attacked_seg = attacked_seg.squeeze()
        if target_seg.ndim == 4:
            target_seg = target_seg.squeeze()
        if mask is not None:
            if mask.ndim == 4:
                mask = mask.squeeze()
            
            # Проверка на случай, если squeeze убрал слишком много (для batch_size=1)
            # Нам нужно, чтобы форма была как минимум (H, W)
            if mask.shape != attacked_seg.shape:
                # Если формы всё еще разные, пробуем принудительно подогнать маску под сегментацию
                # (актуально, если маска была [1, H, W], а сегментация [H, W])
                mask = mask.reshape(attacked_seg.shape)

        # Список имен классов для Cityscapes
        class_names = [
            "road", "sidewalk", "building", "wall", "fence",
            "pole", "traffic light", "traffic sign", "vegetation",
            "terrain", "sky", "person", "rider", "car", "truck",
            "bus", "train", "motorcycle", "bicycle"
        ]

        # Ограничиваем область маской (патчем)
        if mask is not None:
            attacked_area = attacked_seg[mask]
            target_area = target_seg[mask]
        else:
            attacked_area = attacked_seg.flatten()
            target_area = target_seg.flatten()

        unique_classes = np.unique(attacked_area)
        
        best_iou = -1.0
        winning_class_id = None

        # Ищем класс с максимальным IoU относительно цели в этом кадре
        for cls in range(0,19):
            intersection = (attacked_area == cls).sum()
            union = attacked_area.size  # Весь патч — это наш целевой объем
            
            iou = intersection / union if union > 0 else 0
            
            if iou > best_iou:
                best_iou = iou
                winning_class_id = cls
                iou = intersection / union if union > 0 else 0
                
                if iou > best_iou:
                    best_iou = iou
                    winning_class_id = cls

        # --- Вывод в консоль ---
        if winning_class_id is not None:
            # Получаем имя класса или выводим ID, если индекс вне списка
            class_name = class_names[int(winning_class_id)] if int(winning_class_id) < len(class_names) else f"ID {winning_class_id}"
            
            # self.count обычно инкрементируется в методе update, 
            # используйте его как номер текущего кадра
            # print(f"[Frame {self.count}] Winning Attack Class: {class_name} (IoU: {best_iou:.4f}), {unique_classes}")
            
            # Сохраняем для итоговой гистограммы
            self.class_winning_frames[winning_class_id] = self.class_winning_frames.get(winning_class_id, 0) + 1
    
    def plot_attack_histogram(self):
        if not self.class_winning_frames:
            return

        class_names_map = [
            "road", "sidewalk", "building", "wall", "fence",
            "pole", "traffic light", "traffic sign", "vegetation",
            "terrain", "sky", "person", "rider", "car", "truck",
            "bus", "train", "motorcycle", "bicycle"
        ]

        # Данные: какой класс чаще всего выигрывал по IoU
        plot_data = []
        for c_id, frame_count in self.class_winning_frames.items():
            name = class_names_map[int(c_id)] if int(c_id) < len(class_names_map) else f"ID {c_id}"
            # Для линии IoU можно оставить среднее значение IoU этого класса, когда он выигрывал
            plot_data.append((name, frame_count))

        # Сортировка по количеству кадров
        plot_data.sort(key=lambda x: x[1], reverse=True)

        names = [x[0] for x in plot_data]
        counts = [x[1] for x in plot_data]

        fig, ax1 = plt.subplots(figsize=(14, 7))

        # Столбцы — количество кадров
        ax1.bar(names, counts, color='lightcoral', alpha=0.8, label='Frames where Class won IoU')
        ax1.set_xlabel('Class Name')
        ax1.set_ylabel('Number of Frames', color='darkred')
        ax1.tick_params(axis='y', labelcolor='darkred')
        plt.xticks(rotation=45, ha='right')

        plt.title('Semantic Drift Analysis: Most Successful Classes per Frame')
        
        # Добавим сетку для удобства счета кадров
        ax1.grid(axis='y', linestyle='--', alpha=0.7)
        
        fig.tight_layout()
        hist_path = os.path.join(self.output_dir, f"semantic_drift_{self.count}.png")
        plt.savefig(hist_path)
        mlflow.log_artifact(hist_path)
        plt.close()
        
    def update(self, original_flow, attacked_flow, original_img=None, second_img=None, gt_flow=None, target_flow=None, inverse_flow=None, mask=None, valid=None, tracked_flows=None, original_depth=None, attacked_depth=None, target_depth=None, tracked_depths=None, original_seg=None, attacked_seg=None, target_seg=None, tracked_segs=None):
        """Update metrics for the current batch."""
        aee_attack = self.compute_aee(original_flow, attacked_flow, mask)
        aee_attack_target = (
            self.compute_aee(attacked_flow, target_flow, mask)
            if target_flow is not None
            else 0.0
        )
        aee_clean_target = (
            self.compute_aee(original_flow, target_flow, mask)
            if target_flow is not None
            else 0.0
        )

        aee_gt = self.compute_aee(
            original_flow, gt_flow, valid) if gt_flow is not None else 0.0
        aee_attacked_gt = self.compute_aee(
            attacked_flow, gt_flow, valid) if gt_flow is not None else 0.0
        diff_error = aee_attacked_gt - aee_gt if gt_flow is not None else 0.0
        if original_img is not None and second_img is not None:
            rec_error, warped_img = self.compute_reconstruction_error(
                img1=original_img,
                img2=second_img,
                flow=attacked_flow,
                mask=mask
            )
            self.cumulative_metrics["reconstruction_error"] += rec_error
            self.log_metric("reconstruction_error", rec_error, step=self.count)
        else:
            rec_error = 0.0

        # MDE
        if original_depth is not None and attacked_depth is not None:
            mde_rmse_attack = self.compute_rmse(original_depth, attacked_depth, mask)
            self.cumulative_metrics["mde_rmse_init_attack"] += float(
                mde_rmse_attack.mean().item()
            )
            self.log_metric(
                "mde_rmse_init_attack",
                float(mde_rmse_attack.mean().item()),
                step=self.count,
            )
            if target_depth is not None:
                mde_rmse_attack_target = self.compute_rmse(
                    attacked_depth, target_depth, mask
                )
                v_at = float(mde_rmse_attack_target.mean().item())
                self.cumulative_metrics["mde_rmse_target_attack"] += v_at
                self.log_metric("mde_rmse_target_attack", v_at, step=self.count)

                mde_rmse_clean_target = self.compute_rmse(
                    original_depth, target_depth, mask
                )
                v_ct = float(mde_rmse_clean_target.mean().item())
                self.cumulative_metrics["mde_rmse_clean_target"] += v_ct
                self.log_metric("mde_rmse_clean_target", v_ct, step=self.count)

            pearson_corr, spearman_corr = self.compute_depth_correlation(original_depth, attacked_depth, mask)
            self.log_metric("mde_pearson_init_attack",
                              pearson_corr, step=self.count)
            self.log_metric("mde_spearman_init_attack",
                              spearman_corr, step=self.count)
        # Segmentation
        if original_seg is not None and attacked_seg is not None:
            seg_metrics = self.compute_segmentation_attack_error(
                original_seg, attacked_seg, target_seg
            )
            for k, v in seg_metrics.items():
                self.cumulative_metrics[k] += v
                self.log_metric(k, v, step=self.count)
                
        if original_seg is not None and attacked_seg is not None and target_seg is not None:
            num_classes = int(max(original_seg.max(), attacked_seg.max(), target_seg.max()) + 1)

            iou_init = self.compute_iou(original_seg, attacked_seg, num_classes, mask)
            iou_target = self.compute_iou(attacked_seg, target_seg, num_classes, mask, targeted=True, classes=[13])

            self.log_metric("iou_init_attack", iou_init, step=self.count)
            self.log_metric("iou_target_attack", iou_target, step=self.count)

            self.cumulative_metrics["iou_init_attack"] += iou_init
            self.cumulative_metrics["iou_target_attack"] += iou_target
            
            self.log_semantic_drift(attacked_seg, target_seg, mask)

        # Multi-task target-distance ratio (mean over tasks; lower ⇒ stronger attack)
        mrs, mrs_gammas = self.compute_multitask_robustness_score(
            original_flow,
            attacked_flow,
            mask,
            target_flow=target_flow,
            original_depth=original_depth,
            attacked_depth=attacked_depth,
            target_depth=target_depth,
            original_seg=original_seg,
            attacked_seg=attacked_seg,
            target_seg=target_seg,
        )
        if mrs is not None:
            self.cumulative_metrics["multitask_robustness_score"] += mrs
            for task_name, g in mrs_gammas.items():
                key = f"mrs_gamma_{task_name}"
                if key in self.cumulative_metrics:
                    self.cumulative_metrics[key] += self._finite_float(g)

        # Update cumulative metrics
        self.cumulative_metrics["aee_init_attack"] += aee_attack
        self.cumulative_metrics["aee_init_gt"] += aee_gt
        self.cumulative_metrics["aee_attacked_gt"] += aee_attacked_gt
        self.cumulative_metrics["gt_diff_error"] += diff_error
        self.cumulative_metrics["aee_target_attack"] += aee_attack_target
        self.cumulative_metrics["aee_clean_target"] += aee_clean_target
        self.count += 1

        # Log main metrics
        self.log_metric("aee_init_attack", aee_attack, step=self.count)
        self.log_metric("aee_target_attack",
                          aee_attack_target, step=self.count)
        if target_flow is not None:
            self.log_metric("aee_clean_target", aee_clean_target, step=self.count)

        if gt_flow is not None:
            self.log_metric("aee_init_gt", aee_gt, step=self.count)
            self.log_metric("aee_attacked_gt",
                              aee_attacked_gt, step=self.count)
            self.log_metric("gt_diff_error", diff_error, step=self.count)

        if mrs is not None:
            self.log_metric("multitask_robustness_score", mrs, step=self.count)
            for task_name, g in mrs_gammas.items():
                self.log_metric(f"mrs_gamma_{task_name}", g, step=self.count)

        # Process dynamically defined iterations
        if self.saved_iterations:
            for step in self.saved_iterations:
                attacked_flow = tracked_flows[step]
                aee_attack = self.compute_aee(original_flow, attacked_flow, mask)
                aee_attack_target = self.compute_aee(
                    attacked_flow, target_flow, mask=mask)

                aee_gt = self.compute_aee(
                    original_flow, gt_flow, valid) if gt_flow is not None else 0.0
                aee_attacked_gt = self.compute_aee(
                    attacked_flow, gt_flow, valid) if gt_flow is not None else 0.0
                diff_error = aee_attacked_gt - aee_gt if gt_flow is not None else 0.0
                if original_img is not None and second_img is not None:
                    rec_error, warped_img = self.compute_reconstruction_error(
                        img1=original_img,
                        img2=second_img,
                        flow=attacked_flow,
                        mask=mask
                    )
                else:
                    rec_error = 0.0

                self.log_metric(
                    f"aee_init_attack_step_{step}", aee_attack, step=self.count)
                self.log_metric(
                    f"aee_target_attack_step_{step}", aee_attack_target, step=self.count)
                if gt_flow is not None:
                    self.log_metric(
                        f"aee_gt_step_{step}", aee_gt, step=self.count)
                    self.log_metric(
                        f"aee_attacked_gt_step_{step}", aee_attacked_gt, step=self.count)
                    self.log_metric(
                        f"gt_diff_error_step_{step}", diff_error, step=self.count)
                if inverse_flow is not None:
                    self.log_metric(
                        f"reconstruction_error_step_{step}", rec_error, step=self.count)

                # Update cumulative metrics dynamically
                self.cumulative_metrics[f"aee_init_attack_step_{step}"] += aee_attack
                self.cumulative_metrics[f"aee_init_gt_step_{step}"] += aee_gt
                self.cumulative_metrics[f"aee_attacked_gt_step_{step}"] += aee_attacked_gt
                self.cumulative_metrics[f"gt_diff_error_step_{step}"] += diff_error
                self.cumulative_metrics[f"reconstruction_error_step_{step}"] += rec_error
                self.cumulative_metrics[f"aee_target_attack_step_{step}"] += aee_attack_target


                # mde
                if tracked_depths is not None:
                    attacked_depth = tracked_depths[step]
                    mde_rmse_attack = self.compute_rmse(original_depth, attacked_depth, mask)
                    self.log_metric( f"mde_rmse_init_attack_step_{step}", mde_rmse_attack, step=self.count)
                    self.cumulative_metrics[f"mde_rmse_init_attack_step_{step}"] += self._finite_float(mde_rmse_attack)
                    
                    if target_depth is not None:
                        mde_rmse_attack_target = self.compute_rmse(
                            attacked_depth, target_depth, mask)
                        self.cumulative_metrics[f"mde_rmse_target_attack_step_{step}"] += self._finite_float(mde_rmse_attack_target)
                        self.log_metric(f"mde_rmse_target_attack_step_{step}", mde_rmse_attack_target, step=self.count)
                    
                if tracked_segs is not None:
                    attacked_seg_step = tracked_segs[step]
                    seg_metrics = self.compute_segmentation_attack_error(
                        original_seg, attacked_seg_step, target_seg
                    )
                    for k, v in seg_metrics.items():
                        step_key = f"{k}_step_{step}"
                        self.cumulative_metrics[step_key] += v
                        self.log_metric(step_key, v, step=self.count)
                        
        self.save_mean_metrics(step=self.count)
        
    def get_mean_metrics(self):
        """
        Compute mean metrics over all batches.
        Returns:
            dict: Mean metrics.
        """
        if self.count == 0:
            return {}
        return {k: self._finite_float(v / self.count) for k, v in self.cumulative_metrics.items()}

    def save_mean_metrics(self, step):
        mean_metrics = self.get_mean_metrics()
        if not mean_metrics:
            print("No metrics to save.")

        for metric, value in mean_metrics.items():
            self.log_metric(f"mean_{metric}", value, step)

    def save_image(self, image, filename):
        if torch.is_tensor(image):
            image = image.detach().cpu().float().numpy()

            if image.ndim == 4:
                image = image.squeeze(0)
            if image.ndim == 3 and image.shape[0] in (1, 3):
                image = np.transpose(image, (1, 2, 0))

        image = image.astype(np.float32)
        # Guard against NaN/Inf coming from unstable model outputs.
        image = np.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0)

        if image.ndim == 2:
            image = image[:, :, None]
        if image.shape[2] == 1:
            image = np.repeat(image, 3, axis=2)

        min_val, max_val = image.min(), image.max()
        if (max_val - min_val) < 1e-8:
            image = np.clip(image, 0.0, 1.0)
        elif min_val < 0.0 or max_val > 1.0:
            image = (image - min_val) / (max_val - min_val + 1e-8)

        image = (image * 255).clip(0, 255).astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return cv2.imwrite(filename, image)

    def save_artifact(self, artifact, name, artifact_type: Literal["image", "flow", "tensor", "depth"] = "image"):
        """
        Save an artifact (e.g., flow tensor, image, attack noise) and log it with mlflow.
        Args:
            artifact: The artifact to save (torch.Tensor or np.ndarray).
            name (str): Name identifier for the artifact.
            artifact_type (str): 'image' or 'flow' etc.
        """
        artifact_path = os.path.join(self.output_dir, f"{name}.png")
        # Convert tensor to numpy if needed.

        if artifact_type == "image":
            success = self.save_image(artifact, artifact_path)
            if not success:
                raise RuntimeError(
                    f"Failed to write artifact image to {artifact_path}")
        elif artifact_type == "tensor":
            artifact_path = artifact_path.replace(".png", ".npy")
            np.save(artifact_path, artifact)
        elif artifact_type == 'flow':
            quickvis_flow(artifact, artifact_path)
        elif artifact_type == 'depth':
            # print(artifact.shape)
            save_depth(artifact, artifact_path)
        elif artifact_type == 'ss':
            save_segmentation(artifact, artifact_path)
        elif artifact_type == 'patch':
            artifact.save_png(artifact_path)
        mlflow.log_artifact(artifact_path)
        # print(f"Artifact saved to {artifact_path}")

    def save_diploma_artifacts(
        self,
        name,
        clean_image=None,
        attacked_image=None,
        clean_flow=None,
        attacked_flow=None,
        mask=None,
        clean_depth=None,
        attacked_depth=None,
    ):
        """Save paired before/after visualizations without running extra models."""
        artifact_dir = os.path.join(self.output_dir, "diploma_artifacts")
        os.makedirs(artifact_dir, exist_ok=True)

        saved_paths = []

        def save_raw(raw_name, value):
            if value is None:
                return
            if torch.is_tensor(value):
                arr = value.detach().cpu().numpy()
            else:
                arr = np.asarray(value)
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            path = os.path.join(artifact_dir, f"{name}_{raw_name}.npy")
            np.save(path, arr)
            saved_paths.append(path)

        if clean_image is not None and attacked_image is not None and mask is not None:
            path = os.path.join(artifact_dir, f"{name}_mask_overlay_pair.png")
            save_mask_overlay_pair(clean_image, attacked_image, mask, path)
            saved_paths.append(path)
            save_raw("clean_image", clean_image)
            save_raw("attacked_image", attacked_image)
            save_raw("patch_mask", mask)

        if clean_flow is not None and attacked_flow is not None:
            path = os.path.join(artifact_dir, f"{name}_flow_pair_common_scale.png")
            save_flow_pair_common_scale(clean_flow, attacked_flow, path)
            saved_paths.append(path)
            save_raw("clean_flow", clean_flow)
            save_raw("attacked_flow", attacked_flow)

        if clean_depth is not None and attacked_depth is not None:
            path = os.path.join(artifact_dir, f"{name}_depth_pair_common_scale.png")
            save_depth_pair_common_scale(clean_depth, attacked_depth, path)
            saved_paths.append(path)
            save_raw("clean_depth", clean_depth)
            save_raw("attacked_depth", attacked_depth)

        for path in saved_paths:
            mlflow.log_artifact(path, artifact_path="diploma_artifacts")
        return saved_paths

    def finalize(self):
        """
        Final step: Compute and log final mean metrics at the end of the run.
        """
        mean_metrics = self.get_mean_metrics()
        if not mean_metrics:
            print("No final metrics to save.")
            mlflow.end_run()
            return

        # Print final metrics once
        print("\nFinal AEE Metrics:")
        print(
            f"AEE (init vs attack): {mean_metrics.get('aee_init_attack', 'N/A'):.4f}")
        print(
            f"AEE (attacked vs target): {mean_metrics.get('aee_target_attack', 'N/A'):.4f}")
        if "multitask_robustness_score" in mean_metrics:
            print(
                f"Multi-task target ratio (mean AEE/RMSE/seg-error to target vs clean; "
                f"lower = stronger attack): {mean_metrics['multitask_robustness_score']:.4f}"
            )

        self.plot_attack_histogram()

        try:
            mlflow.end_run()
            print("MLflow run ended.")
        except Exception as e:
            print("mlflow.end_run() failed:", e)
            
    def save_flow(self, flow):
        flow = flow.permute(1, 2, 0)  # change from CHW to HWC shape
        flow = flow.detach().cpu().numpy()
        flow_viz = flow_utils.flow_to_rgb(flow)  # Represent the flow as RGB colors
        flow_viz = cv.cvtColor(flow_viz, cv.COLOR_BGR2RGB)
        cv.imwrite('test.png', flow_viz)
        # mlflow.log_artifact(artifact_path)
        
