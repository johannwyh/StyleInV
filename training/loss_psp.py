# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np
import torch
from torch_utils import training_stats
from torch_utils import misc
from torch_utils.ops import conv2d_gradfix
import torch.nn.functional as F

class pSpLoss:
    def __init__(self, device, pSp, G_synthesis, arcface, lpips, lpips_lambda, l2_lambda, id_lambda, noise_mode):
        super().__init__()
        self.device = device
        self.G_synthesis = G_synthesis
        self.pSp = pSp
        self.arcface = arcface
        self.lpips = lpips
        self.lpips_lambda = lpips_lambda
        self.l2_lambda = l2_lambda
        self.id_lambda = id_lambda
        self.noise_mode = noise_mode

    def accumulate_gradients(self, x, y):
        with misc.ddp_sync(self.pSp, True):
            w_inv = self.pSp(x)
        with misc.ddp_sync(self.G_synthesis, True):
            y_hat = self.G_synthesis(w_inv, noise_mode=noise_mode)
        
        loss, loss_dict = 0, dict()

        loss_l2 = F.mse_loss(y_hat, y)
        training_stats.report('l2', loss_l2)
        loss += self.l2_lambda * loss_l2

        if self.id_lambda > 0:
            with misc.ddp_sync(self.arcface, True):
                loss_id = self.arcface(y_hat, y)
                training_stats.report('id', loss_id)
                loss += self.id_lambda * loss_id

        if self.lpips_lambda > 0:            
            with misc.ddp_sync(self.lpips, True):
                loss_lpips = self.lpips(y_hat, y)
                training_stats.report('lpips', loss_lpips)
                loss += self.lpips_lambda * loss_lpips

        loss.backward()