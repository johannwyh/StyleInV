import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Linear, Conv2d, BatchNorm2d, PReLU, Sequential, Module

from training.styleinv.helpers import get_blocks, Flatten, bottleneck_IR, bottleneck_IR_SE, AdaIN, AdainResBlk, ModulationBlock
from training.styleinv.positional_encoding import PositionalEncoding, FourierPositionalEncoding
from training.styleinv.motion_fuse import MapAddMap, MapMapAdd, MapMapCat, MapRemainCat, NoTemporalNoise
from training.networks import FullyConnectedLayer
from einops import rearrange
from torch_utils import misc
from torch_utils import persistence
from training.styleinv.motion import MotionMappingNetwork

@misc.profiled_function
def normalize_2nd_moment(x, dim=1, eps=1e-8):
    return x * (x.square().mean(dim=dim, keepdim=True) + eps).rsqrt()

@persistence.persistent_class
class MappingFuse(torch.nn.Module):
    """
    TODO: Truncation of W Space (incl. wc, wm and wt)
    """
    def __init__(self,
                 content_dim,  # Content latent w_0 dimensionality, 0 = no latent.
                 w_dim,  # Intermediate latent (W) dimensionality.
                 concat_content, # Whether to concat content code in the motion generator
                 layer_features=None,  # Number of intermediate features in the mapping layers, None = same as w_dim.
                 lr_multiplier=0.01,  # Learning rate multiplier for the mapping layers.
                 cfg={},
                 ):
        super().__init__()
        self.cfg = cfg

        # Positional Encoding
        assert self.cfg.motion.type in ['acyclic_pe'], f'Motion type {self.cfg.motion.type} not recognized'
        
        self.motion_encoder = MotionMappingNetwork(self.cfg)
        self.mt_dim = self.motion_encoder.get_dim()

        # Fused latent Mapping
        self.concat_content = concat_content
        self.content_dim = content_dim if self.concat_content else 0
        self.w_dim = self.cfg.w_dim
        self.n_layers = self.cfg.layers_after_fuse
        if layer_features is None:
            layer_features = w_dim
        fused_features = [self.content_dim + self.mt_dim] + [layer_features] * (self.n_layers - 1) + [self.w_dim]
        self.set_mapping_layers(fused_features, "fcf", lr_multiplier)

    def set_mapping_layers(self, features_list, prefix, lr_multiplier):
        num_layers = len(features_list) - 1
        for idx in range(num_layers):
            in_features = features_list[idx]
            out_features = features_list[idx + 1]
            layer = FullyConnectedLayer(in_features, out_features, activation="lrelu", lr_multiplier=lr_multiplier)
            setattr(self, f'{prefix}{idx}', layer)
    
    def generate_motion_sequence(self, Ts):
        assert Ts.shape[-1] == 1, f'Ts should be of shape (*, 1)'
        Tidx = Ts * (self.cfg.sampling.max_num_frames - 1) # [batch_size, num_frames, 1]
        if Tidx.ndim == 2:
            Tidx = Tidx.unsqueeze(0)
        Tidx = Tidx.squeeze(-1)
        batch_size, num_frames = Tidx.shape
        dummy_c = torch.zeros((batch_size, 0)).float().to(Tidx.device)
        motion_info = self.motion_encoder(dummy_c, Tidx, motion_z=None)
        motion_z = motion_info['motion_z']

        return motion_z

    def forward_parallel(self, wc, zm, Ts):
        """
        When motion.type == motion_and_pe,
        zm: [batch_size, z_dim]
        When motion.type == acyclic_pe,
        zm: None (will let motion_encoder generate) or [batch_size, n_anchor_points, z_dim]
        """

        if self.cfg.motion.type == 'motion_and_pe':
            # pe
            Ts = Ts.to(torch.float32)
            Ts = self.pe(Ts)
            b, num_frames, pe_dim = Ts.shape

            # fuse
            wmwt = self.motion_temporal.forward_parallel(zm, Ts)
            temporal_style_motion = wmwt

        elif self.cfg.motion.type == 'acyclic_pe':
            Tidx = Ts * (self.cfg.sampling.max_num_frames - 1) # [batch_size, num_frames, 1]
            Tidx = Tidx.squeeze(-1)
            batch_size, num_frames = Tidx.shape
            dummy_c = torch.zeros((Tidx.shape[0], 0)).float().to(Tidx.device)
            motion_info = self.motion_encoder(dummy_c, Tidx, motion_z=zm)
            motion_v = motion_info['motion_v']
            motion_v = rearrange(motion_v, '(b t) d -> (t b) d', b=batch_size)
            temporal_style_motion = motion_v
        
        if self.concat_content:
            rep_wc = wc.repeat(num_frames, 1) # [ (t b) content_dim]
            w = torch.cat([rep_wc, temporal_style_motion], dim=1)
        else:
            w = temporal_style_motion

        # Fused Latent Mapping
        for idx in range(self.n_layers):
            layer = getattr(self, f'fcf{idx}')
            w = layer(w)

        # return a tensor of shape [(t b) w]
        return w


    def forward(self, wc, zm, Ts, run_parallel=False):
        if run_parallel:
            return self.forward_parallel(wc, zm, Ts)
        
        if self.cfg.motion.type == 'motion_and_pe':
            # pe
            Ts = Ts.to(torch.float32)
            Ts = self.pe(Ts)
            b, num_frames, pe_dim = Ts.shape

            # Motion Code Mapping
            wmwt = self.motion_temporal(zm, Ts)
        
        elif self.cfg.motion.type == 'acyclic_pe':
            Tidx = Ts * (self.cfg.sampling.max_num_frames - 1) # [batch_size, num_frames, 1]
            Tidx = Tidx.squeeze(-1)
            batch_size, num_frames = Tidx.shape
            dummy_c = torch.zeros((Tidx.shape[0], 0)).float().to(Tidx.device)
            motion_info = self.motion_encoder(dummy_c, Tidx, motion_z=zm)
            motion_v = motion_info['motion_v']
            motion_v = rearrange(motion_v, '(b t) d -> (t b) d', b=batch_size)
            wmwt = motion_v.split(batch_size, dim=0)

        inv_styles = []
        for it in range(num_frames):
            if self.concat_content:
                w = torch.cat([wc, wmwt[it]], dim=1)
            else:
                w = wmwt[it]

            # Fused Latent Mapping
            for idx in range(self.n_layers):
                layer = getattr(self, f'fcf{idx}')
                w = layer(w)
            inv_styles.append(w)
        
        # return a list of Tensors [ [b w] , ... , [b w] ]
        return inv_styles
    
@persistence.persistent_class
class StyleEncoderIntoW(Module):
    def __init__(self, 
        num_layers, 
        mode='ir', 
        input_nc=3, 
        adain_mode="block", 
        style_dim=512, 
        normalize_adain=False, 
        lr_multiplier=1, 
        use_modulation=False
    ):
        super().__init__()
        #print('Using StyleEncoderIntoW')
        assert num_layers in [50, 100, 152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        assert adain_mode in ['block', 'bottleneck'], 'adain_mode should be block or bottleneck'

        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.input_layer = Sequential(Conv2d(input_nc, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))
        self.output_pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.linear = FullyConnectedLayer(512, 512, lr_multiplier=lr_multiplier)

        modules = []
        adain_index = []
        bottlenecks = []
        for block in blocks:
            for bottleneck in block:
                bottlenecks.append(bottleneck)
                modules.append(unit_module(bottleneck.in_channel,
                                           bottleneck.depth,
                                           bottleneck.stride))
                if adain_mode == 'bottleneck':
                    adain_index.append(len(modules)-1)
            if adain_mode == 'block':
                adain_index.append(len(modules)-1)

        self.body = Sequential(*modules)

        self.adain_index = adain_index
        self.adains = nn.ModuleList()
        self.use_modulation = use_modulation
        for idx in self.adain_index:
            dim_in = bottlenecks[idx].depth
            dim_out = dim_in
            if use_modulation:
                blk = ModulationBlock(dim_in, dim_out, style_dim)
            else:
                blk = AdainResBlk(dim_in, dim_out, style_dim, w_hpf=0, upsample=False, lr_multiplier=lr_multiplier)
            self.adains.append(blk)

        self.normalize_adain = normalize_adain

    def forward(self, x, s, ignore_adain=False):
        x = self.input_layer(x)
        i_adain = 0
        for i, bottleneck in enumerate(self.body):
            x = bottleneck(x)
            if (i in self.adain_index) and (not ignore_adain):
                adain = self.adains[i_adain]
                if self.use_modulation:
                    x = adain(x, s)
                else:    
                    x = adain(x, s, normalize=self.normalize_adain)
                i_adain += 1

        x = self.output_pool(x)
        x = x.view(-1, 512)
        x = self.linear(x)
        return x

