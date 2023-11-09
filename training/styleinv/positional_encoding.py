from einops import rearrange
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Linear, Conv2d, BatchNorm2d, PReLU, Sequential, Module
from typing import Dict, Optional, Tuple
from torch_utils import misc
from torch_utils import persistence

@persistence.persistent_class
class PositionalEncoding(Module):
    """
    TODO: Other PE type implementation, E.g. [..., cos(2*pi*sigma^(j/m)*v), sin(2*pi*sigma^(j/m)*v), ...]
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.omega = nn.Parameter(torch.randn(1, self.dim, requires_grad=True))
        self.rho   = nn.Parameter(torch.randn(self.dim, requires_grad=True))
        self.alpha = nn.Parameter(torch.randn(self.dim, requires_grad=True))
    
    def forward(self, Ts):
        """
        Ts: [b, t, 1]
        """
        ot = torch.einsum("btc,cd->btd", [Ts, self.omega])
        pe = torch.sin(ot + self.rho) * self.alpha
        return pe

@persistence.persistent_class
class FourierAndConstPE(Module):
    def __init__(self, dim, max_num_frames=2048):
        """
        As here the PE module is used for the generator, 
        the input t can be any real number with in [0, 1]
        We embed this t with round(2048 * t)
        """
        super().__init__()

        self.dim = dim
        self.max_num_frames = max_num_frames
        self.const_embed = nn.Embedding(max_num_frames, dim)
        self.time_encoder = FourierEncoder(max_num_frames)
    
    def get_dim(self):
        return self.dim + self.time_encoder.get_dim()

    def forward(self, t):
        assert t.ndim == 3 and t.shape[-1] == 1
        t = t * self.max_num_frames
        B, T, _ = t.shape

        tvec = t.squeeze().view(-1)
        const_embs = self.const_embed(tvec.round().long()) # [(B T), d1]
        const_embs = rearrange(const_embs, "(b t) d -> b t d", t=T) # [B, T, d1]
        fourier_embs = self.time_encoder(t) # [B, T, d2]

        embs = torch.cat([const_embs, fourier_embs], dim=-1)
        return embs

@persistence.persistent_class
class FourierPositionalEncoding(Module):
    def __init__(self, max_num_frames=2048):
        """
        As here the PE module is used for the generator, 
        the input t can be any real number with in [0, 1]
        We embed this t with round(2048 * t)
        """
        super().__init__()
        self.max_num_frames = max_num_frames
        self.time_encoder = FourierEncoder(max_num_frames)
    
    def get_dim(self):
        return self.time_encoder.get_dim()
    
    def forward(self, t):
        assert t.ndim == 3 and t.shape[-1] == 1
        B, T, _ = t.shape

        fourier_embs = self.time_encoder(1.0 * self.max_num_frames * t) # [B, T, d2]
        return fourier_embs

@persistence.persistent_class
class FourierEncoder(Module):
    def __init__(self, max_num_frames):
        super().__init__()
        fourier_coefs = construct_log_spaced_freqs(max_num_frames, skip_small_t_freqs=0)
        self.fourier_coefs = nn.Parameter(fourier_coefs, requires_grad=False)
    
    def get_dim(self):
        return self.fourier_coefs.shape[1] * 2
    
    def forward(self, t_in):
        assert t_in.ndim == 3 and t_in.shape[-1] == 1
        t = t_in.squeeze(-1)
        B, T = t.shape

        t = t.view(-1).float() # [batch_size * num_frames]
        fourier_raw_embs = self.fourier_coefs * t.unsqueeze(1) # [(B T), num_feats]
        fourier_embeds = torch.cat([
            fourier_raw_embs.sin(),
            fourier_raw_embs.cos(),
        ], dim=1) # [(B T), num_feats * 2]

        fourier_embeds = rearrange(fourier_embeds, "(b t) d -> b t d", t=T)

        return fourier_embeds

@misc.profiled_function
def construct_log_spaced_freqs(max_num_frames: int, skip_small_t_freqs: int=0) -> Tuple[int, torch.Tensor]:
    time_resolution = 2 ** np.ceil(np.log2(max_num_frames))
    num_fourier_feats = np.ceil(np.log2(time_resolution)).astype(int)
    powers = torch.tensor([2]).repeat(num_fourier_feats).pow(torch.arange(num_fourier_feats)) # [num_fourier_feats]
    powers = powers[:len(powers) - skip_small_t_freqs] # [num_fourier_feats]
    fourier_coefs = powers.unsqueeze(0).float() * np.pi # [1, num_fourier_feats]

    return fourier_coefs / time_resolution