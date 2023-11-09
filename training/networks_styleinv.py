import math
import profile
import torch
from torch import nn
from torch_utils import misc, persistence
from training.styleinv.styleinv_encoder import MappingFuse, StyleEncoderIntoW
from torch_utils.ops import conv2d_resample, upfirdn2d, bias_act, fma
import torch.nn.functional as F
from training.networks_sgv import DiscriminatorBlock, DiscriminatorEpilogue, TemporalDifferenceEncoder, MappingNetwork
import pickle
import numpy as np
import os
from copy import deepcopy

def he_init(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)

@persistence.persistent_class
class StyleInV(torch.nn.Module):
    def __init__(self, checkpoint, mapping_opts, encoder_opts, input_residual=False, clip_len=128, fps=30, mutual_recon=False, skip_encoder=False):
        super().__init__()
        self.checkpoint = checkpoint
        self.mapping_opts = mapping_opts
        self.encoder_opts = encoder_opts
        self.mutual_recon = mutual_recon
        self.skip_encoder = skip_encoder
        self.input_residual = input_residual # If set true, model predcits the residual w.r.t input, otherwise latent average
        self.clip_len = clip_len
        self.fps = fps

        # Define architecture
        if not self.skip_encoder:
            self.encoder = self.set_encoder()
            self.psp_load_info = self.load_encoder_weights()

        self.mapping = self.set_mapping()

        #print("Output Size={}".format(self.opts.output_size))
        self.face_pool = torch.nn.AdaptiveAvgPool2d((256, 256))
        
        # Load PCA 
        if self.encoder_opts.pca.use_pca:
            self.use_pca = True
            pca_path = self.encoder_opts.pca.pca_path
            self.n_pca = self.encoder_opts.pca.n_pca
            self.pca_lambda = self.encoder_opts.pca.pca_lambda
            # PCA base
            pca_comp = np.load(os.path.join(pca_path, 'pca_comp.npy'))
            pca_stdev = np.load(os.path.join(pca_path, 'pca_stdev.npy'))
            pca_comp = torch.tensor(pca_comp[:self.n_pca], dtype=torch.float32)
            pca_stdev = torch.tensor(pca_stdev[:self.n_pca], dtype=torch.float32)
            pca_stdev = pca_stdev.view(-1, 1).repeat([1, pca_comp.shape[-1]])
            pca_mul = pca_comp * pca_stdev
            self.register_buffer('pca_mul', pca_mul)
            # Additional FC layer
            self.pca_w = nn.Parameter(torch.FloatTensor(512, self.n_pca))
            self.pca_b = nn.Parameter(torch.FloatTensor(self.n_pca))
            nn.init.normal_(self.pca_w, std=0.02)
            self.pca_b.data.fill_(0.0)
        else:
            self.use_pca = False
            
        # FC Network for temporal noise reconstruction - Mutual Info Loss
        if self.mutual_recon:
            w_dim, m_dim = self.mapping_opts.w_dim, self.mapping_opts.motion_dim
            self.H = nn.Sequential(
                nn.Linear(w_dim, w_dim),
                nn.ReLU(),
                nn.Linear(w_dim, m_dim)
            )

    def print_psp_load_info(self):
        print(self.psp_load_info)

    def set_mapping(self):
        mapping = MappingFuse(
            content_dim=self.mapping_opts.content_dim,
            w_dim=self.mapping_opts.w_dim,
            concat_content=self.mapping_opts.concat_content,
            lr_multiplier=self.mapping_opts.lr_multiplier,
            cfg=self.mapping_opts,
        )
        return mapping

    def set_encoder(self):
        encoder = StyleEncoderIntoW(
            num_layers=50, 
            mode='ir_se', 
            adain_mode=self.encoder_opts.adain_mode,
            style_dim=self.mapping_opts.w_dim,
            normalize_adain=self.encoder_opts.normalize_adain,
            lr_multiplier=self.encoder_opts.lr_multiplier,
            use_modulation=self.encoder_opts.use_modulation)
        return encoder

    def load_encoder_weights(self):
        # initialize first
        self.encoder.apply(he_init)
        # start loading
        with open(self.checkpoint, 'rb') as f:
            psp_encoder = pickle.load(f)['pSp'].encoder
        psp_encoder_ckpt = psp_encoder.state_dict()
        info = self.encoder.load_state_dict(psp_encoder_ckpt, strict=False)
        return info

    def set_latent_avg(self, stylegan_w_avg):
        self.register_buffer('latent_avg', stylegan_w_avg)
        
    def forward_parallel(self, x, wc, zm, Ts, ignore_adain=False, return_temporal_style=False):
        """
        Ts is always sampled in the range of [0, 1]
        The shape of zm:
        motion_and_pe: [batch, dim]
        acyclic_pe: None for training, [batch, n_anchor, dim] for inference
        """
        timesteps = Ts.shape[1]

        # styles - [(t b) w]
        styles = self.mapping(wc, zm, Ts, run_parallel=True)
        # rep_x - [(t b) c H W]
        if not self.skip_encoder:
            rep_x = x.repeat(timesteps, 1, 1, 1)
            codes = self.encoder(rep_x, styles, ignore_adain=ignore_adain) # [(t b) w]

            # PCA
            if self.use_pca:
                coord_pca = torch.matmul(codes, self.pca_w) + self.pca_b # [(t b) n_pca]
                #coord_pca = torch.tanh(coord_pca)
                codes = self.pca_lambda * torch.matmul(coord_pca, self.pca_mul) # [(t b) w]

            if self.input_residual:
                codes = codes + wc.repeat(timesteps, 1)
            elif self.latent_avg is not None:
                codes = codes + self.latent_avg.repeat(codes.shape[0], 1)
        else:
            codes = styles + wc.repeat(timesteps, 1)

        if return_temporal_style and self.mutual_recon:
            # codes: [(t b) w]
            # styles: [(t b) w]
            # zm_rec: [(t b) w]
            zm_rec = self.H(styles)
            return codes, styles, zm_rec
        else:
            # return one tensor of shape [(t b) w]
            return codes

    def forward(self, x, wc, zm, Ts, ignore_adain=False, run_parallel=True, return_temporal_style=False):
        """
        Ts is always sampled in the range of [0, 1]
        """
        if run_parallel:
            return self.forward_parallel(x, wc, zm, Ts, ignore_adain, return_temporal_style)

        styles = self.mapping(wc, zm, Ts)

        if not self.skip_encoder:
            result_latents = []
            for style in styles:
                codes = self.encoder(x, style, ignore_adain=ignore_adain)	
                # normalize with respect to the center of an average face

                # PCA
                if self.use_pca:
                    coord_pca = torch.matmul(codes, self.pca_w) + self.pca_b # [(t b) n_pca]
                    #coord_pca = torch.tanh(coord_pca)
                    codes = self.pca_lambda * torch.matmul(coord_pca, self.pca_mul) # [(t b) w]

                if self.input_residual:
                    codes = codes + wc
                elif self.latent_avg is not None:
                    codes = codes + self.latent_avg.repeat(codes.shape[0], 1)
                result_latents.append(codes)
        else:
            result_latents = styles

        if return_temporal_style and self.mutual_recon:
            # result_latents: [(b,w) , (b,w) , ... , (b,w)]
            # rt_styles: [(t b) w]
            # zm_rec: [(t b) w]
            rt_styles = torch.cat(styles, dim=0)
            zm_rec = self.H(rt_styles)
            return result_latents, rt_styles, zm_rec
        else:
            # return a list [(b,w) , (b,w) , ... , (b,w)]
            return result_latents

@persistence.persistent_class
class DiscriminatorFFC(torch.nn.Module):
    def __init__(self,
        c_dim,                          # Conditioning label (C) dimensionality.
        img_resolution,                 # Input resolution.
        img_channels,                   # Number of input color channels.
        architecture        = 'resnet', # Architecture: 'orig', 'skip', 'resnet'.
        channel_base        = 32768,    # Overall multiplier for the number of channels.
        channel_max         = 512,      # Maximum number of channels in any layer.
        num_fp16_res        = 0,        # Use FP16 for the N highest resolutions.
        conv_clamp          = None,     # Clamp the output of convolution layers to +-X, None = disable clamping.
        cmap_dim            = None,     # Dimensionality of mapped conditioning label, None = default.
        block_kwargs        = {},       # Arguments for DiscriminatorBlock.
        mapping_kwargs      = {},       # Arguments for MappingNetwork.
        epilogue_kwargs     = {},       # Arguments for DiscriminatorEpilogue.
        cfg                 = {},       # Additional config.
    ):
        super().__init__()

        self.cfg = cfg
        self.c_dim = c_dim
        self.img_resolution = img_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.img_channels = img_channels
        self.block_resolutions = [2 ** i for i in range(self.img_resolution_log2, 2, -1)]
        channels_dict = {res: min(channel_base // res, channel_max) for res in self.block_resolutions + [4]}
        fp16_resolution = max(2 ** (self.img_resolution_log2 + 1 - num_fp16_res), 8)

        if cmap_dim is None:
            cmap_dim = channels_dict[4]

        if self.cfg.sampling.num_frames_per_video > 1:
            cfg_t = deepcopy(self.cfg)
            cfg_t.sampling.num_frames_per_video -= 1
            self.time_encoder = TemporalDifferenceEncoder(cfg_t)
            assert self.time_encoder.get_dim() > 0
        else:
            self.time_encoder = None

        if self.c_dim == 0 and self.time_encoder is None:
            cmap_dim = 0

        common_kwargs = dict(img_channels=img_channels, architecture=architecture, conv_clamp=conv_clamp)
        total_c_dim = c_dim + (0 if self.time_encoder is None else self.time_encoder.get_dim())
        cur_layer_idx = 0

        for res in self.block_resolutions:
            in_channels = channels_dict[res] if res < img_resolution else 0
            tmp_channels = channels_dict[res]
            out_channels = channels_dict[res // 2]

            if res // 2 == self.cfg.concat_res:
                out_channels = out_channels // self.cfg.num_frames_div_factor
            if res == self.cfg.concat_res:
                in_channels = (in_channels // self.cfg.num_frames_div_factor) * (self.cfg.sampling.num_frames_per_video - 1)

            use_fp16 = (res >= fp16_resolution)
            block = DiscriminatorBlock(in_channels, tmp_channels, out_channels, resolution=res,
                first_layer_idx=cur_layer_idx, use_fp16=use_fp16, cfg=self.cfg, **block_kwargs, **common_kwargs)
            setattr(self, f'b{res}', block)
            cur_layer_idx += block.num_layers

            if res <= self.cfg.concat_res:
                in_channels_first = in_channels if res != self.cfg.concat_res else in_channels // (self.cfg.sampling.num_frames_per_video - 1)
                block_cond = DiscriminatorBlock(in_channels_first, tmp_channels, out_channels, resolution=res,
                first_layer_idx=cur_layer_idx, use_fp16=use_fp16, cfg=self.cfg, **block_kwargs, **common_kwargs)
                setattr(self, f'bcond{res}', block_cond)
                cur_layer_idx += block.num_layers

        if self.c_dim > 0 or not self.time_encoder is None:
            self.mapping = MappingNetwork(z_dim=0, c_dim=total_c_dim, w_dim=cmap_dim, num_ws=None, w_avg_beta=None, **mapping_kwargs)
        
        self.b4cond = DiscriminatorEpilogue(channels_dict[4], cmap_dim=cmap_dim, resolution=4, cfg=self.cfg, return_vec=True, **epilogue_kwargs, **common_kwargs)
        self.b4 = DiscriminatorEpilogue(channels_dict[4], cmap_dim=cmap_dim*2, resolution=4, cfg=self.cfg, **epilogue_kwargs, **common_kwargs)
    
    #def forward(self, img, c, t, **block_kwargs):
    def forward(self, img, t, **block_kwargs):
        nfpv = self.cfg.sampling.num_frames_per_video
        assert len(img) == t.shape[0] * t.shape[1] * nfpv // (nfpv -1), f"Wrong shape: {img.shape}, {t.shape}"
        assert t.ndim == 2, f"Wrong shape: {t.shape}"

        if not self.time_encoder is None:
            # Encoding the time distances
            t_embs = self.time_encoder(t.view(-1, self.cfg.sampling.num_frames_per_video - 1)) # [batch_size, t_dim]
            # Concatenate `c` and time embeddings
            c = t_embs
            #c = torch.cat([c, t_embs], dim=1) # [batch_size, c_dim + t_dim]
            c = (c * 0.0) if self.cfg.dummy_c else c # [batch_size, c_dim + t_dim]

        x = None
        for res in self.block_resolutions:
            block = getattr(self, f'b{res}')
            if res == self.cfg.concat_res:
                # Concatenating the frames
                x_all = x.view(-1, self.cfg.sampling.num_frames_per_video, *x.shape[1:]) # [batch_size, num_frames, c, h, w]
                x_cond = x_all[:, 0, :, :, :] # (batch_size, c, h, w)
                x = x_all[:, 1:, :, :, :]
                x = x.view(x.shape[0], -1, *x.shape[3:]) # [batch_size, num_frames * c, h, w]
            x, img = block(x, img, **block_kwargs)
            if res <= self.cfg.concat_res:
                block_cond = getattr(self, f'bcond{res}')
                x_cond, img_cond = block_cond(x_cond, None, **block_kwargs)

        cmap = None
        if self.c_dim > 0 or not self.time_encoder is None:
            assert c.shape[1] > 0
        cmap = self.b4cond(x_cond, img_cond, None)
        if c.shape[1] > 0:
            cmap_t = self.mapping(None, c)
            cmap = torch.cat([cmap_t, cmap], dim=1)
        
        x = self.b4(x, img, cmap)
        x = x.squeeze(1) # [batch_size]

        return x