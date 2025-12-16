import torch
import yaml
import os
from argparse import Namespace
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from PIL import Image
import torchvision.transforms.functional as TF
import torch.nn.functional as F
def get_config_path(config_name="models.yaml"):
    """Returns the absolute path to the configuration file located in the configs folder."""
    # Get the root directory of the project (one level up from 'models')
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Build the path to the config folder
    config_dir = os.path.join(project_root, "configs")

    # Build the full path to the config file
    config_path = os.path.join(config_dir, config_name)

    # Check if the file exists
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file {config_path} not found.")

    return config_path


def load_config(config_path):
    """
    Load a configuration file.
    
    Args:
        config_path (str): Path to the YAML configuration file.
        
    Returns:
        dict: Parsed configuration as a Python dictionary.
    """
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def load_model_args(model_name):
    """Load model arguments (path weights e t.c.) from model name

    Args:
        dataset_name (string): Name of model
    """
    config = load_config(get_config_path("models.yaml"))
    return config['models'].get(model_name)

# def import_and_load(net='RAFT', make_unit_input=False, variable_change=False, device=torch.device("cpu"), make_scaled_input_model=False, **kwargs):
#     """import a model and load pretrained weights for it

#     Args:
#             net (str, optional):
#                     the desired network to load. Defaults to 'RAFT'.
#             make_unit_input (bool, optional):
#                     model will assume input images in range [0,1] and transform to [0,255]. Defaults to False.
#             variable_change (bool, optional):
#                     apply change of variables (COV). Defaults to False.
#             device (torch.device, optional):
#                     changes the selected device. Defaults to torch.device("cpu").
#             make_scaled_input_model (bool, optional):
#                     load a scaled input model which uses make_unit_input and variable_change as specified. Defaults to False.

#     Raises:
#             RuntimeWarning: Unknown model type

#     Returns:
#             torch.nn.Module: PyTorch optical flow model with loaded weights 
#     """

#     if make_unit_input == True or variable_change == True or make_scaled_input_model:
#         from scaledInputModel import ScaledInputModel
#         model = ScaledInputModel(net, make_unit_input=make_unit_input,
#                                  variable_change=variable_change, device=device, **kwargs)
#         print("--> transforming model to 'make_unit_input'=%s, 'variable_change'=%s\n" %
#               (str(make_unit_input), str(variable_change)))

#     else:
#         model = None
#         try:
#             model_args = load_model_args(net)
#             if not model_args:
#                 raise ValueError(
#                     f"Model configuration for {net} not found.")
#             if net == 'RAFT':
#                 from models.raft.raft import RAFT

#                 # set the path to the corresponding weights for initializing the model
#                 path_weights = 'models/_pretrained_weights/raft-sintel.pth'

#                 # possible adjustements to the config can be made in the file
#                 # found under models/_config/raft_config.json
#                 with open("models/_config/raft_config.json") as file:
#                     config = json.load(file)

#                 model = torch.nn.DataParallel(RAFT(config))
#                 # load pretrained weights
#                 model.load_state_dict(torch.load(
#                     path_weights, map_location=device))

#             elif net == 'GMA':
#                 from models.gma.network import RAFTGMA

#                 # set the path to the corresponding weights for initializing the model
#                 path_weights = 'models/_pretrained_weights/gma-sintel.pth'

#                 # possible adjustements to the config file can be made
#                 # under models/_config/gma_config.json
#                 with open("models/_config/gma_config.json") as file:
#                     config = json.load(file)
#                     # GMA accepts only a Namespace object when initializing
#                     config = Namespace(**config)

#                 model = torch.nn.DataParallel(RAFTGMA(config))

#                 model.load_state_dict(torch.load(
#                     path_weights, map_location=device))

#             elif net == 'PWCNet':
#                 from models.PWCNet.PWCNet import PWCDCNet

#                 # set path to pretrained weights:
#                 path_weights = 'models/_pretrained_weights/pwc_net_chairs.pth.tar'

#                 model = PWCDCNet()

#                 weights = torch.load(path_weights, map_location=device)
#                 if 'state_dict' in weights.keys():
#                     model.load_state_dict(weights['state_dict'])
#                 else:
#                     model.load_state_dict(weights)
#                 model.to(device)

#             elif net == 'SpyNet':
#                 from models.SpyNet.SpyNet import Network as SpyNet
#                 # weights for SpyNet are loaded during initialization
#                 model = SpyNet(nlevels=6, pretrained=True)
#                 model.to(device)

#             elif net == "FlowNet2":
#                 from models.FlowNet.FlowNet2 import FlowNet2

#                 # hard coding configuration for FlowNet2
#                 args_fn = Namespace(fp16=False, rgb_max=255.0)

#                 # set path to pretrained weights
#                 path_weights = 'models/_pretrained_weights/FlowNet2_checkpoint.pth.tar'
#                 model = FlowNet2(args_fn, div_flow=20, batchNorm=False)

#                 weights = torch.load(path_weights, map_location=device)
#                 model.load_state_dict(weights['state_dict'])

#                 model.to(device)
#             if model is None:
#                 raise RuntimeWarning(
#                     'The network %s is not a valid model option for import_and_load(network). No model was loaded.' % (net))
#         except FileNotFoundError as e:
#             print("\nLoading the model failed, because the checkpoint path was invalid. Are the checkpoints placed in models/_pretrained_weights/? If this folder is empty, consider to execute the checkpoint loading script from scripts/load_all_weights.sh. The full error that caused the loading failure is below:\n\n%s" % e)
#             exit()

#         print("--> flow network is set to %s" % net)
#     return model


def import_and_load(net='RAFT', make_unit_input=False, variable_change=False, device=torch.device("cpu"), make_scaled_input_model=False, **kwargs):
    """Import a model and load pretrained weights for it.

    Args:
        net (str): The desired network to load. Defaults to 'RAFT'.
        make_unit_input (bool): Model assumes input images in range [0,1] and transforms to [0,255]. Defaults to False.
        variable_change (bool): Apply change of variables (COV). Defaults to False.
        device (torch.device): The device to run the model on. Defaults to torch.device("cpu").
        make_scaled_input_model (bool): Load a scaled input model which uses make_unit_input and variable_change. Defaults to False.

    Raises:
        RuntimeWarning: Unknown model type.

    Returns:
        torch.nn.Module: PyTorch optical flow model with loaded weights.
    """

    # Use model args from config
    model_args = load_model_args(net)
    if not model_args:
        raise ValueError(f"Model configuration for {net} not found.")

    config = model_args.get('config', {})
    weights_path = model_args.get('weights_path')

    # Scaled input model logic
    if make_unit_input or variable_change or make_scaled_input_model:
        from models.scaledInputModel import ScaledInputModel 
        model = ScaledInputModel(net, make_unit_input=make_unit_input,
                                 variable_change=variable_change, device=device, **kwargs)
        print(
            f"--> transforming model to 'make_unit_input'={make_unit_input}, 'variable_change'={variable_change}\n")
    else:
        model = None
        try:
            # Loading model based on the configuration
            if net == 'RAFT':
                from models.models.raft.raft import RAFT
                model = torch.nn.DataParallel(RAFT(config))
                model.load_state_dict(torch.load(
                    weights_path, map_location=device, weights_only=True))

            elif net == 'GMA':
                from models.models.gma.network import RAFTGMA
                # Use Namespace to pass config as args
                config = Namespace(**config)

                model = torch.nn.DataParallel(RAFTGMA(config))
                model.load_state_dict(torch.load(
                    weights_path, map_location=device))

            elif net == 'PWCNet':
                from models.models.PWCNet.PWCNet import PWCDCNet
                model = PWCDCNet()
                weights = torch.load(weights_path, map_location=device)
                if 'state_dict' in weights.keys():
                    model.load_state_dict(weights['state_dict'])
                else:
                    model.load_state_dict(weights)
                model.to(device)

            elif net == 'SpyNet':
                from models.models.SpyNet.SpyNet import Network as SpyNet
                model = SpyNet(nlevels=6, pretrained=True)
                model.to(device)

            elif net == "FlowNet2":
                from models.models.FlowNet.FlowNet2 import FlowNet2
                model = FlowNet2(Namespace(**config), div_flow=20, batchNorm=False)
                weights = torch.load(weights_path, map_location=device)
                model.load_state_dict(weights['state_dict'])
                model.to(device)
            elif net == 'MeFlow':
                from models.models.MeFlow.meflow import build_model
                model = build_model(Namespace(**config))
                checkpoint = torch.load(weights_path, map_location=device)
                weights = checkpoint['model'] if 'model' in checkpoint else checkpoint
                model.load_state_dict(weights, strict=False)
                model.to(device)
            elif net == 'MemFlow':
                from models.models.MemFlow.core.Networks import build_network
                from models.models.MemFlow.configs.sintel_memflownet import get_cfg # CHANGE IF WE CHANGE CHECKPOINT!!!!
                # TODO
                if args.name == "MemFlowNet":
                    if args.stage == 'things':
                        from configs.things_memflownet import get_cfg
                    elif args.stage == 'sintel':
                        from configs.sintel_memflownet import get_cfg
                    elif args.stage == 'spring_only':
                        from configs.spring_memflownet import get_cfg
                    elif args.stage == 'kitti':
                        from configs.kitti_memflownet import get_cfg
                    else:
                        raise NotImplementedError
                elif args.name == "MemFlowNet_T":
                    if args.stage == 'things':
                        from configs.things_memflownet_t import get_cfg
                    elif args.stage == 'things_kitti':
                        from configs.things_memflownet_t_kitti import get_cfg
                    elif args.stage == 'sintel':
                        from configs.sintel_memflownet_t import get_cfg
                    elif args.stage == 'kitti':
                        from configs.kitti_memflownet_t import get_cfg
                    else:
                        raise NotImplementedError

                cfg = get_cfg()
                cfg.update(config)
                model = build_network(cfg).cuda()
                if cfg.restore_ckpt is not None:
                    ckpt = torch.load(cfg.restore_ckpt, map_location=device)
                    ckpt_model = ckpt['model'] if 'model' in ckpt else ckpt
                    if 'module' in list(ckpt_model.keys())[0]:
                        for key in ckpt_model.keys():
                            ckpt_model[key.replace(
                                'module.', '', 1)] = ckpt_model.pop(key)
                        model.load_state_dict(ckpt_model, strict=True)
                    else:
                        model.load_state_dict(ckpt_model, strict=True)
                model.eval()
                model.to(device)
            # TODO add other models
            if model is None:
                raise RuntimeWarning(
                    f'The network {net} is not a valid model option for import_and_load(). No model was loaded.')

        except FileNotFoundError as e:
            print(
                f"\nLoading the model failed, because the checkpoint path was invalid. The full error is:\n\n{e}")
            exit()

        print(f"--> flow network is set to {net}")

    return model


def compute_flow(model, network, images, test_mode=True, **kwargs):
    """subroutine to call the forward pass of the network

    Args:
            model (torch.nn.module):
                    instance of optical flow model
            network (str):
                    name of the network. [scaled_input_model | RAFT | GMA | FlowNet2 | SpyNet | PWCNet]
            x1 (tensor):
                    first image of a frame sequence
            x2 (tensor):
                    second image of a frame sequence
            test_mode (bool, optional):
                    applies only to RAFT and GMA such that the forward call yields only the final flow field. Defaults to True.

    Returns:
            tensor: optical flow field
    """
    if network == "scaled_input_model":
        flow = model(images, test_mode=True, **kwargs)

    elif network == 'RAFT':
        _, flow = model(images[0], images[1], test_mode=test_mode, **kwargs)

    elif network == 'GMA':
        _, flow = model(images[0], images[1], iters=6,
                        test_mode=test_mode, **kwargs)

    elif network[:7] == 'FlowNet':
        # all flow net types need image tensor of dimensions [batch, colors, image12, x, y] = [b,3,2,x,y]
        x = torch.stack((images[0], images[1]), dim=-3)
        # FlowNet2-variants: all fine now, input [0,255] is taken.

        if not network[:8] == 'FlowNet2':
            # FlowNet variants need input in [-1,1], which is achieved by substracting the mean rgb value from the image in [0,1]
            rgb_mean = x.contiguous().view(
                x.size()[:2]+(-1,)).mean(dim=-1).view(x.size()[:2] + (1, 1, 1,)).detach()
            x = x - rgb_mean

        flow = model(x)
    elif network == 'MeFlow':
        _, flow = model(images[0], images[1], test_mode=test_mode)
    elif network == 'MemFlow':
        from models.models.MemFlow.inference import inference_core_skflow as inference_core
        processor = inference_core.InferenceCore(
                    model, config=cfg)
        # print(len(images))
        images = torch.stack(images, dim = 1).cuda()
        # images = 2 * (images / 255.0) - 1.0
        flow_prev = None
        results = []
        # print(images.shape)
        for ti in range(images.shape[1] - 1):
            flow_low, flow_pre = processor.step(images[:, ti:ti + 1], end=(ti == images.shape[1] - 2),
                                                add_pe=('rope' in cfg and cfg.rope), flow_init=flow_prev)
            results.append(flow_pre)
            # print(flow_pre.shape)
            if 'warm_start' in cfg and cfg.warm_start:
                flow_prev = forward_interpolate(flow_low[0])[None].cuda()
        
        return torch.stack(results)[0]
    elif network == 'PWCNet' or network == 'SpyNet':  # works for PWCNet, SpyNet
        flow = model(images[0], images[1], **kwargs)
    return flow


def model_takes_unit_input(model):
    """Boolean check if a network needs input in range [0,1] or [0,255]

    Args:
            model (str):
                    name of the model

    Returns:
            bool: True -> [0,1], False -> [0,255]
    """
    model_takes_unit_input = False
    if model in ["PWCNet", "SpyNet"]:
        model_takes_unit_input = True
    return model_takes_unit_input



class MDEModel(torch.nn.Module):
    def __init__(self,
                 checkpoint: str = "depth-anything/Depth-Anything-V2-Large-hf",
                 device: torch.device = None):
        super().__init__()
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        print(f"Loading depth model on {self.device}")
        self.processor = AutoImageProcessor.from_pretrained(checkpoint, use_fast=True)
        self.model = AutoModelForDepthEstimation.from_pretrained(checkpoint).to(self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    def _to_pil_list(self, images):
        # Accept PIL.Image, single tensor or list/tuple
        pil_list = []
        if isinstance(images, (list, tuple)):
            for im in images:
                pil_list.append(self._to_pil(im))
        else:
            pil_list.append(self._to_pil(images))
        return pil_list

    def _to_pil(self, img):
        # Handle dict input with 'images' tensor [B,2,C,H,W]
        if isinstance(img, dict) and 'images' in img:
            tensor = img['images']
            # select first batch and first frame
            tensor = tensor[0, 0]  # shape [C,H,W]
            return TF.to_pil_image(tensor.cpu())
        if isinstance(img, Image.Image):
            return img
        if isinstance(img, torch.Tensor):
            return TF.to_pil_image(img.cpu())
        raise ValueError(f"Unsupported image type: {type(img)}")

    def forward(self, images):
        """
        images: PIL.Image, torch.Tensor (C,H,W) or (B,C,H,W), or list of those
        returns: torch.Tensor of shape (B, 1, H', W')
        """
        # Normalize input list
        # if isinstance(images, torch.Tensor) and images.ndim == 4:
        #     # batch tensor
        #     tensor_list = [img for img in images]
        #     pil_images = self._to_pil_list(tensor_list)
        # else:
        #     pil_images = self._to_pil_list(images)

        # prepare inputs
        # inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        # outputs = self.model(**inputs)
        # processed = self.processor.post_process_depth_estimation(
        #     outputs,
        #     target_sizes=[(img.height, img.width) for img in images]
        # )
        # pil_depths = []
        # for item in processed:
        #     depth_map = item["predicted_depth"] # torch.Tensor [1,H,W]
        #     print("outputs.requires_grad:", depth_map.requires_grad)
        #     depth_map = normalize_depth_tensor(depth_map.squeeze())
        #     pil_depths.append(depth_map)
        # # print(len(pil_depths))
        # return pil_depths if len(pil_depths) > 1 else pil_depths[0]
        if isinstance(images, dict) and 'images' in images:
            tensor = images['images']
            # select first batch and first frame
            tensor = tensor[0, 0]  # shape [C,H,W]

        # proc = self.processor(images=tensor, return_tensors="pt")
        # pixel_values = proc.pixel_values.to(self.device)

        # 2) Инференс без no_grad()
        mean = torch.tensor(self.processor.image_mean).view(1,-1,1,1).to(tensor.device)
        std  = torch.tensor(self.processor.image_std).view(1,-1,1,1).to(tensor.device)
        px = (tensor - mean) / std
        outputs = self.model(pixel_values = px)
        # print(outputs.predicted_depth.min(), outputs.predicted_depth.max())
        # 3) Пост‑процессинг в PyTorch
        orig_height, orig_width = tensor.shape[1], tensor.shape[2]
        depth_logits = outputs.predicted_depth.unsqueeze(1)  # [B,1,H',W']
        
        depth_map = F.interpolate(
            depth_logits, size=(orig_height, orig_width), mode="bicubic", align_corners=False
        )


        return depth_map  # GPU‑тензор с grad_fn


def normalize_depth_tensor(depth_tensor):
    # Normalize to [0,1]
    min_val = depth_tensor.min()
    max_val = depth_tensor.max()
    return (depth_tensor - min_val) / (max_val - min_val)


MDE_MODEL_CHECKPOINTS = {
    'depth-anything-v2': "depth-anything/Depth-Anything-V2-Large-hf",
    'marigold': "Intel/marigold-depth-estimation"
}

def load_mde_model(model_name: str = "depth-anything-v2", device=None) -> MDEModel:
    """Factory function to load the depth estimation wrapper."""
    if model_name not in MDE_MODEL_CHECKPOINTS:
        raise ValueError(f"Unknown model name '{model_name}'. Choose from: {list(MDE_MODEL_CHECKPOINTS.keys())}")

    checkpoint = MDE_MODEL_CHECKPOINTS[model_name]
    return MDEModel(checkpoint=checkpoint, device=device)


class SegModel(torch.nn.Module):
    """
    Generic segmentation model wrapper.
    - For torchvision models: expects forward to produce dict with 'out' logits.
    - For mmseg models (init_segmentor): we'll wrap the inference call separately.
    """

    def __init__(self, model, framework="torchvision", device=None, mmseg_wrapper=None):
        super().__init__()
        self.model = model
        self.framework = framework
        self.device = device or (torch.device(
            "cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.mmseg_wrapper = mmseg_wrapper  # function for mmseg inference if used

        # ensure it's on device and eval
        try:
            self.model.to(self.device)
        except Exception:
            pass
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        # ImageNet normalization (torchvision models)
        self.im_mean = torch.tensor(
            [0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.im_std = torch.tensor(
            [0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

    def _to_pil(self, img):
        if isinstance(img, Image.Image):
            return img
        if isinstance(img, torch.Tensor):
            return TF.to_pil_image(img.cpu())
        raise ValueError(f"Unsupported image type: {type(img)}")

    def _prepare_batch(self, images):
        # Convert to tensor if PIL
        if isinstance(images, Image.Image):
            t = TF.to_tensor(images).to(self.device)  # [C,H,W]
            t = (t - self.im_mean) / self.im_std
            batch = t#.unsqueeze(0)  # [1,C,H,W]
        elif isinstance(images, torch.Tensor):
            if images.ndim == 3:  # [C,H,W]
                batch = images.unsqueeze(0)  # [1,C,H,W]
            elif images.ndim == 4:  # [B,C,H,W]
                batch = images
            else:
                raise ValueError(f"Unexpected tensor shape: {images.shape}")
        elif isinstance(images, (list, tuple)):
            batch = torch.stack([self._prepare_batch(img) for img in images], dim=0).squeeze(1)
        else:
            raise ValueError(f"Unsupported input type: {type(images)}")
        
        orig_sizes = batch.shape[-2:]  # [H,W]
        return batch, orig_sizes

    def forward(self, images, return_logits=False, resize_to=None):
        """
        Forward pass for segmentation.

        Args:
            images: PIL.Image | torch.Tensor | list | dict
            return_logits: if True return raw logits [B,C,H,W], else class indices mask [B,H,W]
            resize_to: optional (H,W) to interpolate output to a specific size
        """
        if isinstance(images, dict) and 'images' in images:
            images = images['images']
            # select first batch and first frame
            images = images[0, 0]  # shape [C,H,W]
            
            
        if self.framework == "mmseg":
            if self.mmseg_wrapper is None:
                raise RuntimeError("mmseg wrapper not provided")
            return self.mmseg_wrapper(images, return_logits=return_logits, device=self.device)

        batch, orig_sizes = self._prepare_batch(images)

        # with torch.no_grad():
        out = self.model(batch)
        if isinstance(out, dict) and 'out' in out:
            logits = out['out']  # [B,C,H_out,W_out]
        elif isinstance(out, torch.Tensor):
            logits = out
        else:
            raise RuntimeError(
                f"Unexpected model output type: {type(out)}")

        # resize to requested size
        if resize_to is not None:
            logits = F.interpolate(
                logits, size=resize_to, mode='bilinear', align_corners=False
            )
            if return_logits:
                return logits
            preds = logits.argmax(dim=1)  # [B,H,W]
            return preds

        # resize to original image sizes
        if logits.shape[2:4] != orig_sizes:
            print(logits.shape[2:4], orig_sizes)
            logits = F.interpolate(
                logits, size=orig_sizes[0], mode='bilinear', align_corners=False
            )

        if return_logits:
            return logits
        return logits.argmax(dim=1)  # [B,H,W]


def load_seg_model(model_name: str = "pspnet_cityscapes", device=None) -> SegModel:
    """
    Loads a segmentation model (torchvision or modern mmsegmentation ≥1.2.0).
    """
    import torch
    from PIL import Image
    import numpy as np
    import torchvision.transforms.functional as TF
    from mmengine.config import Config
    from mmseg.registry import MODELS
    from mmseg.utils import register_all_modules
    from mmseg.apis import init_model
    from mmseg.apis.utils import _preprare_data
    import mmengine.logging.history_buffer as hb
    torch.serialization.add_safe_globals([hb.HistoryBuffer])
    model_args = load_model_args(model_name)
    if not model_args:
        raise ValueError(
            f"Model configuration for {model_name} not found in models.yaml")
    framework = model_args.get("framework", "torchvision")
    device = device or (torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu"))

    # ---- torchvision baseline ----
    if framework == "torchvision":
        from torchvision.models import segmentation as segmods
        try:
            model = segmods.deeplabv3_resnet101(pretrained=True)
        except TypeError:
            model = segmods.deeplabv3_resnet101(
                weights=segmods.DeepLabV3_ResNet101_Weights.DEFAULT
            )
        return SegModel(model=model, framework="torchvision", device=device)

    # ---- mmsegmentation (modern API ≥1.2.0) ----
    elif framework == "mmseg":
        config_path = model_args.get("config")
        weights_path = model_args.get("weights_path")
        if config_path is None or weights_path is None:
            raise ValueError(
                "mmseg model entry must contain 'config' and 'weights_path' keys in models.yaml")

        # # Register all modules and build model
        # register_all_modules(init_default_scope=True)
        # cfg = Config.fromfile(config_path)
        # model = MODELS.build(cfg.model)

        # # Load checkpoint
        # checkpoint = torch.load(weights_path, map_location=device)
        # state_dict = checkpoint.get("state_dict", checkpoint)
        # model.load_state_dict(state_dict, strict=False)

        # model.to(device)
        # model.eval()
        model = init_model(config_path, weights_path, device)
        # Inference wrapper


        def mmseg_wrapper(images, return_logits=False, device=None):
            """
            Compatible with mmsegmentation >=1.2.0.
            Accepts torch.Tensor or PIL.Image and returns a segmentation mask tensor (H, W).
            """
            import numpy as np
            import torch
            import torchvision.transforms.functional as TF
            from mmseg.structures import SegDataSample

            if isinstance(images, torch.Tensor):
                if images.ndim == 4:
                    batch = images
                else:
                    batch = images.unsqueeze(0)
            else:
                raise ValueError("Expected torch.Tensor input")

            batch = batch.to(device).float()
            mean = torch.tensor([123.675, 116.28, 103.53],
                                device=device).view(1, 3, 1, 1)
            std = torch.tensor([58.395, 57.12, 57.375], device=device).view(1, 3, 1, 1)
            batch = (batch * 255.0 - mean) / std

            model.eval()
            with torch.set_grad_enabled(True):
                results = model(batch, data_samples=None, mode="predict")

            output = results[0] if isinstance(results, (list, tuple)) else results

            if return_logits:
                # Tensor [1, num_classes, H, W]
                return output.seg_logits.data.unsqueeze(0).float()
            else:
                return output.pred_sem_seg.data  # Tensor [1, H, W]
            # if isinstance(images, torch.Tensor):
            #     if images.ndim == 4:  # B, C, H, W
            #         imgs = [img.permute(1, 2, 0).cpu().detach().numpy() for img in images]
            #     else:  # C, H, W
            #         imgs = [images.permute(1, 2, 0).cpu().detach().numpy()]
            # elif isinstance(images, np.ndarray):
            #     imgs = [images] if images.ndim == 3 else list(images)
            # elif hasattr(images, "convert"):  # PIL
            #     imgs = [np.array(images.convert("RGB"))]
            # elif isinstance(images, (list, tuple)):
            #     imgs = [np.array(TF.to_pil_image(img)).astype(np.uint8) for img in images]
            # else:
            #     raise ValueError(f"Unsupported input type: {type(images)}")
            
            # imgs = [np.clip(img * 255.0 if img.max() <= 1.0 else img,
            #                 0, 255).astype(np.uint8) for img in imgs]
            # # # Ensure float tensor with 3 channels
            # # if img_tensor.ndim == 2:
            # #     img_tensor = img_tensor.unsqueeze(0).repeat(3, 1, 1)
            # # elif img_tensor.shape[0] == 1:
            # #     img_tensor = img_tensor.repeat(3, 1, 1)
            # # img_tensor = img_tensor.float().to(device or "cpu")

            # data, is_batch = _preprare_data(imgs, model)

            # # Run inference
            # output = model.test_step(data)[0]
            # # Extract segmentation prediction
            # if return_logits:
            #     return output.seg_logits.data.unsqueeze(0).float()
            # if hasattr(output, "pred_sem_seg"):
            #     seg_mask = output.pred_sem_seg.data
            # else:
            #     raise RuntimeError(
            #         f"Unexpected output format from MMSeg model: {type(output)}")

            # return seg_mask
        
        return SegModel(model=model, framework="mmseg", device=device, mmseg_wrapper=mmseg_wrapper)

    else:
        raise ValueError(
            f"Unsupported segmentation framework '{framework}' in models.yaml")
