from __future__ import print_function, division
from models.MemFlow.configs.sintel_memflownet import get_cfg # CHANGE IF WE CHANGE CHECKPOINT!!!!
from models.MemFlow.core.utils.utils import InputPadder, forward_interpolate
from models.MemFlow.core.Networks import build_network
from models.MemFlow.inference import inference_core_skflow as inference_core
import argparse
from loguru import logger as loguru_logger
import random
import sys
print(sys.path)
# sys.path.append('models/MemFlow/core')
from PIL import Image
import os
import numpy as np
import torch

class MemFlow:
    def __init__(self, args):
        cfg = get_cfg()
        cfg.update(args)
        model = build_network(cfg).cuda()
        if cfg.restore_ckpt is not None:
            ckpt = torch.load(cfg.restore_ckpt, map_location='cpu')
            ckpt_model = ckpt['model'] if 'model' in ckpt else ckpt
            if 'module' in list(ckpt_model.keys())[0]:
                for key in ckpt_model.keys():
                    ckpt_model[key.replace('module.', '', 1)] = ckpt_model.pop(key)
                model.load_state_dict(ckpt_model, strict=True)
            else:
                model.load_state_dict(ckpt_model, strict=True)

        model.eval()
        self.model = model
        self.cfg = cfg
        
    def compute_flow(self, images):
        processor = inference_core.InferenceCore(self.model, config=self.cfg)
        # print(len(images))
        images = torch.stack(images, dim = 1).cuda()
        # images = 2 * (images / 255.0) - 1.0
        flow_prev = None
        results = []
        # print(images.shape)
        for ti in range(images.shape[1] - 1):
            flow_low, flow_pre = processor.step(images[:, ti:ti + 1], end=(ti == images.shape[1] - 2),
                                                add_pe=('rope' in self.cfg and self.cfg.rope), flow_init=flow_prev)
            results.append(flow_pre)
            # print(flow_pre.shape)
            if 'warm_start' in self.cfg and self.cfg.warm_start:
                flow_prev = forward_interpolate(flow_low[0])[None].cuda()
        
        return torch.stack(results)[0]

