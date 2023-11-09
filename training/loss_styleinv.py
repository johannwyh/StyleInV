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
from einops import rearrange
import random
import torch.nn.functional as F
import math
from sampling import BetaSampling, RandomSampling

class WNormLoss(torch.nn.Module):

	def __init__(self, start_from_latent_avg=True):
		super(WNormLoss, self).__init__()
		self.start_from_latent_avg = start_from_latent_avg

	def forward(self, latent, latent_avg=None):
		if self.start_from_latent_avg:
			latent = latent - latent_avg
		return torch.sum(latent.norm(2, dim=(1, 2))) / latent.shape[0]

class StyleInVLoss:
    def __init__(self, device, G_mapping, G_synthesis, SI, D, 
        augment_pipe=None,
        video_consistent_aug=True,
        D_type='stylegan-v', 
        g_sampling_cfg={}, 
        real_sampling_cfg={}, 
        num_ws=14, 
        mutual_recon=True, 
        first_frame_recon=False, 
        first_frame_in_D=True, 
        first_frame_use_gen=False, 
        run_g_parallel=True, 
        r1_gamma=1.0, 
        lambda_adv=0.1, 
        lambda_mutual=1.0, 
        lambda_recon=10, 
        lambda_w_reg=0, 
        ln_lambda_w_reg=False,
        noise_mode='none'
    ):

        super().__init__()

        assert D_type in ['digan', 'stylegan-v', 'ffc'], f'Discriminator type {D_type} not supported'

        self.device = device
        self.G_mapping = G_mapping
        self.G_synthesis = G_synthesis
        self.num_ws = num_ws
        self.SI = SI
        self.D = D
        self.augment_pipe = augment_pipe
        self.video_consistent_aug = video_consistent_aug
        self.D_type = D_type
        self.run_g_parallel = run_g_parallel
        self.g_sampling_cfg = g_sampling_cfg
        self.real_sampling_cfg = real_sampling_cfg
        self.first_frame_in_D = first_frame_in_D
        self.first_frame_use_gen = first_frame_use_gen
        self.noise_mode = noise_mode

        if self.first_frame_use_gen and self.g_sampling_cfg.add_0 == False:
            raise ValueError(f'When first frame in D uses generated, g_sampling should have add_0 = True')
        
        if self.first_frame_in_D and self.real_sampling_cfg.add_0 == False:
            raise ValueError(f'When first frame is used in D, real_sampling should have add_0 = True')
        
        self.n_sparse = self.real_sampling_cfg.num_frames_per_video
        if self.first_frame_in_D:
            if self.first_frame_use_gen:
                self.input_skip_to_D = 'no'
            else:
                if self.g_sampling_cfg.add_0:
                    self.input_skip_to_D = 'replace'
                else:
                    self.input_skip_to_D = 'append'
                    self.n_sparse += 1
        else:
            if self.g_sampling_cfg.add_0:
                self.input_skip_to_D = 'remove'
                self.n_sparse -= 1
            else:
                self.input_skip_to_D = 'no'
        
        self.mutual_recon = mutual_recon
        self.first_frame_recon = first_frame_recon
        self.r1_gamma = r1_gamma
        self.lambda_adv = lambda_adv
        self.lambda_mutual = lambda_mutual
        self.lambda_recon = lambda_recon

        # easy for ablation 
        if self.lambda_recon < 0:
            self.lambda_recon = -self.lambda_recon
            self.recon_level = 'latent'
        else:
            self.recon_level = 'image'
        print(f'First frame recon level - {self.recon_level}')
        
        self.lambda_w_reg = lambda_w_reg
        self.ln_lambda_w_reg = ln_lambda_w_reg

        if self.g_sampling_cfg.type == 'beta':
            self.sampler = BetaSampling(self.g_sampling_cfg)
        elif self.g_sampling_cfg.type == 'random':
            self.sampler = RandomSampling(self.g_sampling_cfg)
        else:
            raise ValueError(f'Sampling Tye {self.g_sampling_cfg.type} not supported')

    def broadcast_w(self, w):
        return w.unsqueeze(1).repeat([1, self.num_ws, 1])

    def run_G(self, zc, zm, sync):
        batch_size = zc.size(0)
        Ts = self.sampler.sample(batch_size, self.device)

        with torch.no_grad():
            wc = self.G_mapping(zc, None)
            assert wc.shape[1] == self.num_ws, f'Number of w+ styles not batch, {wc.shape[1]} and {self.num_ws}'
            x = self.G_synthesis(wc)
            wc = wc[:, 0, :]
        
        with misc.ddp_sync(self.SI, sync):
            if self.mutual_recon:
                styles, temporal_styles, zm_rec = self.SI(x, wc, zm, Ts, run_parallel=True, return_temporal_style=True)
            else:
                styles = self.SI(x, wc, zm, Ts, run_parallel=True, return_temporal_style=False) # [(t b) w]
                temporal_styles, zm_rec = None, None
        
        with misc.ddp_sync(self.G_synthesis, sync):
            styles_b = self.broadcast_w(styles)
            imgs = self.G_synthesis(styles_b, noise_mode=self.noise_mode)
        
        imgs_list = list(imgs.split(batch_size, dim=0))
        
        return wc, x, imgs_list, Ts, temporal_styles, zm_rec, styles

    def run_D(self, img, Ts, sync):
        """
        img: b x (3xn_img) x H x W
        Ts:  b x (n_img) x 1
        """

        logits = None
        batch_size, n_img = Ts.shape[0], Ts.shape[1]
        
        if self.D_type == 'digan':
            # augmentation
            if self.augment_pipe is not None:
                img = self.augment_pipe(img) # [n, f * ch, h, w]

            Ts = Ts.unsqueeze(-1)
            H, W = img.shape[2], img.shape[3]
            dTs = (Ts[:, 1:] - Ts[:, :-1]).repeat(1, 1, H, W) # [b, T-1, H, W]
            img = torch.cat([img, dTs], dim=1)
            with misc.ddp_sync(self.D, sync):
                logits = self.D(img, None)

        elif self.D_type in ['stylegan-v', 'ffc']:
            t = Ts.squeeze(-1) # (b, T)
            f = t.shape[1]
            if self.D_type == 'ffc':
                t = t[:, 1:] # do not insert first delta_t w.r.t t=0

            img = rearrange(img, 'b (t c) h w -> (b t) c h w', c=3)

            if self.augment_pipe is not None:
                if self.video_consistent_aug:
                    nf, ch, h, w = img.shape
                    n = nf // f
                    img = img.view(n, f * ch, h, w) # [n, f * ch, h, w]

                img = self.augment_pipe(img) # [n, f * ch, h, w]

                if self.video_consistent_aug:
                    img = img.view(n * f, ch, h, w) # [n * f, ch, h, w]
            
            with misc.ddp_sync(self.D, sync):
                logits = self.D(img, t)
        
        return logits

    def accumulate_gradients(self, phase, real_imgs, real_ts, gen_zc, gen_zm, sync, gain):
        batch_size = gen_zc.size(0)
        assert phase in ['SImain', 'SIboth', 'Dmain', 'Dreg', 'Dboth']
        do_Gmain = (phase in ['SImain', 'SIboth'])
        do_Dmain = (phase in ['Dmain', 'Dboth'])
        do_Dr1   = (phase in ['Dreg', 'Dboth']) and (self.r1_gamma != 0)

        # Gmain: Maximize logits for generated images.
        if do_Gmain:
            wc_in, in_0, imgs_list, Ts, temporal_styles, zm_rec, wc_gen = self.run_G(gen_zc, gen_zm, sync)
            all_ts = Ts.clone().detach().unsqueeze(-1)

            if self.input_skip_to_D == 'replace':
                adv_img_list = [in_0] + imgs_list[1:]
            elif self.input_skip_to_D == 'append':
                adv_img_list = [in_0] + imgs_list
                Ts = torch.cat([torch.zeros(batch_size, 1, 1).float().to(self.device), Ts], dim=1)
            elif self.input_skip_to_D == 'remove':
                adv_img_list = imgs_list[1:]
                Ts = Ts[:, 1:]
            elif self.input_skip_to_D == 'no':
                adv_img_list = imgs_list

            # advsersarial loss
            out = torch.cat(adv_img_list, dim=1)
            gen_logits = self.run_D(out, Ts, sync)
            training_stats.report('Loss/scores/fake', gen_logits)
            training_stats.report('Loss/signs/fake', gen_logits.sign())
            loss_Gadv = self.lambda_adv * torch.nn.functional.softplus(-gen_logits) # -log(sigmoid(gen_logits))
            training_stats.report('Loss/G/adv', loss_Gadv)
            loss_gmain = loss_Gadv.mean()
            
            # mutual information loss
            if self.mutual_recon:
                batch_size = gen_zc.shape[0]
                sparse_t = zm_rec.shape[0] // batch_size # (TxB) / B = T
                zm_raw = gen_zm.detach().repeat((sparse_t, 1))
                assert zm_raw.shape == zm_rec.shape, f'z_motion shape does not match, raw {zm_raw.shape} , rec {zm_rec.shape}'
                loss_Gmut = 1 - torch.nn.functional.cosine_similarity(zm_raw, zm_rec)
                loss_Gmut *= self.lambda_mutual
                training_stats.report('Loss/G/mut', loss_Gmut)
                loss_gmain += loss_Gmut.mean()
            
            # first_frame reconstruction
            if self.first_frame_recon:
                if self.recon_level == 'image':
                    loss_recon = self.lambda_recon * F.mse_loss(imgs_list[0], in_0, reduction='none')
                else:
                    # wc_gen: [(t b) w]
                    gen_frame0_latent = wc_gen[:batch_size, :]
                    loss_recon = self.lambda_recon * F.mse_loss(gen_frame0_latent, wc_in, reduction='none')
                
                training_stats.report('Loss/G/recon', loss_recon)
                loss_gmain += loss_recon.mean()
            
            # latent regularization loss
            if self.lambda_w_reg > 0:
                timesteps = wc_gen.shape[0] // wc_in.shape[0]
                wc_delta = wc_gen - wc_in.repeat(timesteps, 1)
                if self.ln_lambda_w_reg:
                    all_ts = rearrange(all_ts, 'b t -> (t b)')
                    lamda_tensor = torch.log((math.e - 1) * all_ts + 1)
                    lambda_tensor = torch.max(lambda_tensor, torch.ones_like(lambda_tensor) * 0.1)
                    lambda_tensor = self.lambda_w_reg / lambda_tensor
                    loss_Gwreg = lambda_tensor * wc_delta.norm(2, dim=1)
                else:
                    loss_Gwreg = self.lambda_w_reg * wc_delta.norm(2, dim=1)
                training_stats.report('Loss/G/wreg', loss_Gwreg)
                loss_gmain += loss_Gwreg.mean()

            # backward
            loss_gmain.mul(gain).backward()

        # Dmain: Minimize logits for generated images.
        loss_Dgen = 0
        if do_Dmain:
            wc_in, in_0, imgs_list, Ts, _, _, _ = self.run_G(gen_zc, gen_zm, sync=False)
            if self.input_skip_to_D == 'replace':
                adv_img_list = [in_0] + imgs_list[1:]
            elif self.input_skip_to_D == 'append':
                adv_img_list = [in_0] + imgs_list
                Ts = torch.cat([torch.zeros(batch_size, 1, 1).float().to(self.device), Ts], dim=1)
            elif self.input_skip_to_D == 'remove':
                adv_img_list = imgs_list[1:]
                Ts = Ts[:, 1:]
            elif self.input_skip_to_D == 'no':
                adv_img_list = imgs_list

            # adversarial loss
            out = torch.cat(adv_img_list, dim=1)
            gen_logits = self.run_D(out, Ts, sync=False) # Gets synced by loss_Dreal.
            training_stats.report('Loss/scores/fake', gen_logits)
            training_stats.report('Loss/signs/fake', gen_logits.sign())
            loss_Dgen = torch.nn.functional.softplus(gen_logits) # -log(1 - sigmoid(gen_logits))
            loss_Dgen.mean().mul(gain).backward()

        # Dmain: Maximize logits for real images.
        # Dr1: Apply R1 regularization.
        if do_Dmain or do_Dr1:
            name = 'Dreal_Dr1' if do_Dmain and do_Dr1 else 'Dreal' if do_Dmain else 'Dr1'
            real_img_tmp = real_imgs.detach().requires_grad_(do_Dr1)
            real_logits = self.run_D(real_img_tmp, real_ts, sync=sync)
            training_stats.report('Loss/scores/real', real_logits)
            training_stats.report('Loss/signs/real', real_logits.sign())

            loss_Dreal = 0
            if do_Dmain:
                loss_Dreal = torch.nn.functional.softplus(-real_logits) # -log(sigmoid(real_logits))
                training_stats.report('Loss/D/loss', loss_Dgen + loss_Dreal)

            loss_Dr1 = 0
            if do_Dr1:
                r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp], create_graph=True, only_inputs=True)[0]
                r1_penalty = r1_grads.square().sum([1,2,3])
                loss_Dr1 = r1_penalty * (self.r1_gamma / 2)
                training_stats.report('Loss/D/reg', loss_Dr1)
            
            (real_logits * 0 + loss_Dreal + loss_Dr1).mean().mul(gain).backward()

#----------------------------------------------------------------------------
