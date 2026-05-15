import torch


class MDEModelAdapter:
    """
    Wraps project's MDEModel for use as adv-manhole's mde_model.

    adv-manhole calls: predicted_disp = mde_model(images)  # (N,3,H,W)
    MDEModel.forward only handles dict{'images': (B,2,C,H,W)} and extracts [0,0].

    Returns normalized depth as proxy disparity (N,1,H,W) with gradient flow.
    Push toward 1 = maximize apparent depth (objects appear farther).
    """

    def __init__(self, mde_model, device):
        self.mde_model = mde_model
        self.device = device

    def __call__(self, images):
        # images: (N,3,H,W) float [0,1]
        depth_list = []
        for i in range(images.shape[0]):
            img = images[i]  # (3,H,W)
            # MDEModel.forward: dict['images'][0,0] → single (3,H,W) image
            depth_i = self.mde_model({'images': img.unsqueeze(0).unsqueeze(0)})  # (1,1,H,W)
            depth_list.append(depth_i)
        depth = torch.cat(depth_list, dim=0)  # (N,1,H,W) in meters
        depth = torch.nan_to_num(depth, nan=0.0, posinf=1.0, neginf=0.0)

        # Normalize per-sample to [0,1] as proxy disparity
        B = depth.shape[0]
        d_min = depth.view(B, -1).min(dim=1)[0].view(B, 1, 1, 1)
        d_max = depth.view(B, -1).max(dim=1)[0].view(B, 1, 1, 1)
        return ((depth - d_min) / (d_max - d_min + 1e-8)).clamp(0.0, 1.0)

    def plot(self, image, prediction, save=False, save_path=None):
        return None


class SSModelAdapter:
    """
    Wraps project's SegModel for use as adv-manhole's ss_model.

    adv-manhole calls: predicted_semantic = ss_model(images)  # (N,3,H,W)
    Returns softmax probabilities (N,C,H,W) with gradient flow.

    mmseg_wrapper returns only results[0] for a batch, so we loop single images.
    """

    def __init__(self, ss_model, device):
        self.ss_model = ss_model
        self.device = device

    def __call__(self, images):
        # images: (N,3,H,W) float [0,1]
        logits_list = []
        for i in range(images.shape[0]):
            img = images[i]  # (3,H,W) — SegModel._prepare_batch unsqueezes to (1,3,H,W)
            logits_i = self.ss_model(img, return_logits=True)
            if logits_i.ndim == 3:
                logits_i = logits_i.unsqueeze(0)
            logits_list.append(logits_i)
        logits = torch.cat(logits_list, dim=0)  # (N,C,H,W)
        probs = torch.softmax(logits, dim=1)
        return torch.nan_to_num(probs, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)

    def plot(self, image, prediction, save=False, save_path=None):
        return None
