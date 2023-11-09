import numpy as np
import torch
from torch import Tensor
from torch_utils import misc, persistence
from torch_utils.ops import conv2d_resample, upfirdn2d, bias_act, fma
from typing import Dict, Optional, Tuple
import torch.nn.functional as F
from copy import deepcopy

@misc.profiled_function
def normalize_2nd_moment(x, dim=1, eps=1e-8):
    return x * (x.square().mean(dim=dim, keepdim=True) + eps).rsqrt()

@persistence.persistent_class
class EqLRConv1d(torch.nn.Module):
    def __init__(self,
        in_features: int,
        out_features: int,
        kernel_size: int,
        padding: int=0,
        stride: int=1,
        activation: str='linear',
        lr_multiplier: float=1.0,
        bias=True,
        bias_init=0.0,
        mask_right=False,
    ):
        super().__init__()

        self.activation = activation
        self.weight = torch.nn.Parameter(torch.randn([out_features, in_features, kernel_size]) / lr_multiplier)
        self.bias = torch.nn.Parameter(torch.full([out_features], float(bias_init))) if bias else None
        self.weight_gain = lr_multiplier / np.sqrt(in_features * kernel_size)
        self.bias_gain = lr_multiplier
        self.padding = padding
        self.stride = stride
        self.mask_right = mask_right

        assert self.activation in ['lrelu', 'linear']

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 3, f"Wrong shape: {x.shape}"

        w = self.weight.to(x.dtype) * self.weight_gain # [out_features, in_features, kernel_size]
        if self.mask_right:
            right_index = (w.shape[2] + 1) // 2
            w[:, :, right_index:] = 0.0

        b = self.bias # [out_features]
        if b is not None:
            b = b.to(x.dtype)
            if self.bias_gain != 1:
                b = b * self.bias_gain

        y = F.conv1d(input=x, weight=w, bias=b, stride=self.stride, padding=self.padding) # [batch_size, out_features, out_len]
        if self.activation == 'linear':
            pass
        elif self.activation == 'lrelu':
            y = F.leaky_relu(y, negative_slope=0.2) # [batch_size, out_features, out_len]
        else:
            raise NotImplementedError
        return y

@persistence.persistent_class
class MappingNetwork(torch.nn.Module):
    def __init__(self,
        z_dim,                      # Input latent (Z) dimensionality, 0 = no latent.
        c_dim,                      # Conditioning label (C) dimensionality, 0 = no label.
        w_dim,                      # Intermediate latent (W) dimensionality.
        num_ws,                     # Number of intermediate latents to output, None = do not broadcast.
        num_layers      = 8,        # Number of mapping layers.
        embed_features  = None,     # Label embedding dimensionality, None = same as w_dim.
        layer_features  = None,     # Number of intermediate features in the mapping layers, None = same as w_dim.
        activation      = 'lrelu',  # Activation function: 'relu', 'lrelu', etc.
        lr_multiplier   = 0.01,     # Learning rate multiplier for the mapping layers.
        w_avg_beta      = 0.995,    # Decay for tracking the moving average of W during training, None = do not track.
        cfg             = {},       # Additional config
    ):
        super().__init__()

        self.cfg = cfg
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.num_ws = num_ws
        self.num_layers = num_layers
        self.w_avg_beta = w_avg_beta

        if embed_features is None:
            embed_features = w_dim
        if c_dim == 0:
            embed_features = 0
        if layer_features is None:
            layer_features = w_dim

        features_list = [z_dim + embed_features] + [layer_features] * (num_layers - 1) + [w_dim]

        if c_dim > 0:
            self.embed = FullyConnectedLayer(c_dim, embed_features)

        for idx in range(num_layers):
            in_features = features_list[idx]
            out_features = features_list[idx + 1]
            layer = FullyConnectedLayer(in_features, out_features, activation=activation, lr_multiplier=lr_multiplier)
            setattr(self, f'fc{idx}', layer)

        if num_ws is not None and w_avg_beta is not None:
            self.register_buffer('w_avg', torch.zeros([w_dim]))

    def forward(self, z, c, truncation_psi=1, truncation_cutoff=None, skip_w_avg_update=False):
        # Embed, normalize, and concat inputs.
        x = None
        with torch.autograd.profiler.record_function('input'):
            if self.z_dim > 0:
                misc.assert_shape(z, [None, self.z_dim])
                x = normalize_2nd_moment(z.to(torch.float32))

            if self.c_dim > 0:
                misc.assert_shape(c, [None, self.c_dim])
                y = normalize_2nd_moment(self.embed(c.to(torch.float32)))
                x = torch.cat([x, y], dim=1) if x is not None else y

        # Main layers.
        for idx in range(self.num_layers):
            layer = getattr(self, f'fc{idx}')
            x = layer(x)

        # Update moving average of W.
        if self.w_avg_beta is not None and self.training and not skip_w_avg_update:
            with torch.autograd.profiler.record_function('update_w_avg'):
                self.w_avg.copy_(x.detach().mean(dim=0).lerp(self.w_avg, self.w_avg_beta))

        # Broadcast.
        if self.num_ws is not None:
            with torch.autograd.profiler.record_function('broadcast'):
                x = x.unsqueeze(1).repeat([1, self.num_ws, 1])

        # Apply truncation.
        if truncation_psi != 1:
            with torch.autograd.profiler.record_function('truncate'):
                assert self.w_avg_beta is not None
                if self.num_ws is None or truncation_cutoff is None:
                    x = self.w_avg.lerp(x, truncation_psi)
                else:
                    x[:, :truncation_cutoff] = self.w_avg.lerp(x[:, :truncation_cutoff], truncation_psi)
        return x


@persistence.persistent_class
class FullyConnectedLayer(torch.nn.Module):
    def __init__(self,
        in_features,                # Number of input features.
        out_features,               # Number of output features.
        bias            = True,     # Apply additive bias before the activation function?
        activation      = 'linear', # Activation function: 'relu', 'lrelu', etc.
        lr_multiplier   = 1,        # Learning rate multiplier.
        bias_init       = 0,        # Initial value for the additive bias.
    ):
        super().__init__()
        self.activation = activation
        self.weight = torch.nn.Parameter(torch.randn([out_features, in_features]) / lr_multiplier)
        self.bias = torch.nn.Parameter(torch.full([out_features], float(bias_init))) if bias else None
        self.weight_gain = lr_multiplier / np.sqrt(in_features)
        self.bias_gain = lr_multiplier

    def forward(self, x):
        w = self.weight.to(x.dtype) * self.weight_gain
        b = self.bias
        if b is not None:
            b = b.to(x.dtype)
            if self.bias_gain != 1:
                b = b * self.bias_gain

        if self.activation == 'linear' and b is not None:
            x = torch.addmm(b.unsqueeze(0), x, w.t())
        else:
            x = x.matmul(w.t())
            x = bias_act.bias_act(x, b, act=self.activation)
        return x

@persistence.persistent_class
class Conv2dLayer(torch.nn.Module):
    def __init__(self,
        in_channels,                    # Number of input channels.
        out_channels,                   # Number of output channels.
        kernel_size,                    # Width and height of the convolution kernel.
        bias            = True,         # Apply additive bias before the activation function?
        activation      = 'linear',     # Activation function: 'relu', 'lrelu', etc.
        up              = 1,            # Integer upsampling factor.
        down            = 1,            # Integer downsampling factor.
        resample_filter = [1,3,3,1],    # Low-pass filter to apply when resampling activations.
        conv_clamp      = None,         # Clamp the output to +-X, None = disable clamping.
        channels_last   = False,        # Expect the input to have memory_format=channels_last?
        trainable       = True,         # Update the weights of this layer during training?
        instance_norm   = False,        # Should we apply instance normalization to y?
        lr_multiplier   = 1.0,          # Learning rate multiplier.
    ):
        super().__init__()
        self.activation = activation
        self.up = up
        self.down = down
        self.conv_clamp = conv_clamp
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
        self.padding = kernel_size // 2
        self.weight_gain = 1 / np.sqrt(in_channels * (kernel_size ** 2))
        self.act_gain = bias_act.activation_funcs[activation].def_gain
        self.instance_norm = instance_norm
        self.lr_multiplier = lr_multiplier

        memory_format = torch.channels_last if channels_last else torch.contiguous_format
        weight = torch.randn([out_channels, in_channels, kernel_size, kernel_size]).to(memory_format=memory_format)
        bias = torch.zeros([out_channels]) if bias else None
        if trainable:
            self.weight = torch.nn.Parameter(weight)
            self.bias = torch.nn.Parameter(bias) if bias is not None else None
        else:
            self.register_buffer('weight', weight)
            if bias is not None:
                self.register_buffer('bias', bias)
            else:
                self.bias = None

    def forward(self, x, gain=1):
        w = self.weight * (self.weight_gain * self.lr_multiplier)
        b = (self.bias.to(x.dtype) * self.lr_multiplier) if self.bias is not None else None
        flip_weight = (self.up == 1) # slightly faster
        x = conv2d_resample.conv2d_resample(x=x, w=w.to(x.dtype), f=self.resample_filter, up=self.up, down=self.down, padding=self.padding, flip_weight=flip_weight)

        act_gain = self.act_gain * gain
        act_clamp = self.conv_clamp * gain if self.conv_clamp is not None else None
        x = bias_act.bias_act(x, b, act=self.activation, gain=act_gain, clamp=act_clamp)

        if self.instance_norm:
            x = (x - x.mean(dim=(2,3), keepdim=True)) / (x.std(dim=(2,3), keepdim=True) + 1e-8) # [batch_size, c, h, w]

        return x

@persistence.persistent_class
class TemporalDifferenceEncoder(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        self.use_const_embed = self.cfg.pe.get('use_const_embed', False)
        if self.cfg.sampling.num_frames_per_video > 1:
            self.time_encoder = FixedTimeEncoder(
                self.cfg.sampling.max_num_frames,
                skip_small_t_freqs=self.cfg.get('skip_small_t_freqs', 0))
            if self.use_const_embed:
                self.d = 256
                self.const_embed = torch.nn.Embedding(self.cfg.sampling.max_num_frames, self.d)
            else:
                self.d = 0
            
    def get_dim(self) -> int:
        if self.cfg.sampling.num_frames_per_video == 1:
            return 1
        else:
            if self.cfg.sampling.type == 'uniform':
                return self.d + self.time_encoder.get_dim()
            else:
                return (self.d + self.time_encoder.get_dim()) * (self.cfg.sampling.num_frames_per_video - 1)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        misc.assert_shape(t, [None, self.cfg.sampling.num_frames_per_video])
        
        batch_size = t.shape[0]

        if self.cfg.sampling.num_frames_per_video == 1:
            out = torch.zeros(len(t), 1, device=t.device)
        else:
            if self.cfg.sampling.type == 'uniform':
                num_diffs_to_use = 1
                t_diffs = t[:, 1] - t[:, 0] # [batch_size]
            else:
                num_diffs_to_use = self.cfg.sampling.num_frames_per_video - 1
                t_diffs = (t[:, 1:] - t[:, :-1]).view(-1) # [batch_size * (num_frames - 1)]
            # Note: float => round => long is necessary when it's originally long
            if self.cfg.pe.t_is_frame_idx == False: # t is in range [0, 1]
                t_diffs = t_diffs * (self.cfg.sampling.max_num_frames - 1)
            fourier_embs = self.time_encoder(t_diffs.unsqueeze(1)) # [batch_size * num_diffs_to_use, num_fourier_feats]
            if self.use_const_embed:
                const_embs = self.const_embed(t_diffs.float().round().long()) # [batch_size * num_diffs_to_use, d]
                out = torch.cat([const_embs, fourier_embs], dim=1) # [batch_size * num_diffs_to_use, d + num_fourier_feats]
            else:
                out = fourier_embs # [batch_size * num_diffs_to_use, num_fourier_feats]
            out = out.view(batch_size, num_diffs_to_use, -1).view(batch_size, -1) # [batch_size, num_diffs_to_use * (d + num_fourier_feats)]

        return out

#----------------------------------------------------------------------------

@persistence.persistent_class
class FixedTimeEncoder(torch.nn.Module):
    def __init__(self,
            max_num_frames: int,            # Maximum T size
            skip_small_t_freqs: int=0,      # How many high frequencies we should skip
        ):
        super().__init__()

        assert max_num_frames >= 1, f"Wrong max_num_frames: {max_num_frames}"
        fourier_coefs = construct_log_spaced_freqs(max_num_frames, skip_small_t_freqs=skip_small_t_freqs)
        self.register_buffer('fourier_coefs', fourier_coefs) # [1, num_fourier_feats]

    def get_dim(self) -> int:
        return self.fourier_coefs.shape[1] * 2

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        assert t.ndim == 2, f"Wrong shape: {t.shape}"

        t = t.view(-1).float() # [batch_size * num_frames]
        fourier_raw_embs = self.fourier_coefs * t.unsqueeze(1) # [bf, num_fourier_feats]

        fourier_embs = torch.cat([
            fourier_raw_embs.sin(),
            fourier_raw_embs.cos(),
        ], dim=1) # [bf, num_fourier_feats * 2]

        return fourier_embs

@misc.profiled_function
def construct_log_spaced_freqs(max_num_frames: int, skip_small_t_freqs: int=0) -> Tuple[int, torch.Tensor]:
    time_resolution = 2 ** np.ceil(np.log2(max_num_frames))
    num_fourier_feats = np.ceil(np.log2(time_resolution)).astype(int)
    powers = torch.tensor([2]).repeat(num_fourier_feats).pow(torch.arange(num_fourier_feats)) # [num_fourier_feats]
    powers = powers[:len(powers) - skip_small_t_freqs] # [num_fourier_feats]
    fourier_coefs = powers.unsqueeze(0).float() * np.pi # [1, num_fourier_feats]

    return fourier_coefs / time_resolution
    
@persistence.persistent_class
class DiscriminatorBlock(torch.nn.Module):
    def __init__(self,
        in_channels,                        # Number of input channels, 0 = first block.
        tmp_channels,                       # Number of intermediate channels.
        out_channels,                       # Number of output channels.
        resolution,                         # Resolution of this block.
        img_channels,                       # Number of input color channels.
        first_layer_idx,                    # Index of the first layer.
        architecture        = 'resnet',     # Architecture: 'orig', 'skip', 'resnet'.
        activation          = 'lrelu',      # Activation function: 'relu', 'lrelu', etc.
        resample_filter     = [1,3,3,1],    # Low-pass filter to apply when resampling activations.
        conv_clamp          = None,         # Clamp the output of convolution layers to +-X, None = disable clamping.
        use_fp16            = False,        # Use FP16 for this block?
        fp16_channels_last  = False,        # Use channels-last memory format with FP16?
        freeze_layers       = 0,            # Freeze-D: Number of layers to freeze.
        cfg                 = {},           # Main config.
    ):
        assert architecture in ['orig', 'skip', 'resnet']
        super().__init__()

        self.cfg = cfg
        self.in_channels = in_channels
        self.resolution = resolution
        self.img_channels = img_channels
        self.first_layer_idx = first_layer_idx
        self.architecture = architecture
        self.use_fp16 = use_fp16
        self.channels_last = (use_fp16 and fp16_channels_last)
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))

        self.num_layers = 0
        def trainable_gen():
            while True:
                layer_idx = self.first_layer_idx + self.num_layers
                trainable = (layer_idx >= freeze_layers)
                self.num_layers += 1
                yield trainable
        trainable_iter = trainable_gen()
        conv0_in_channels = in_channels if in_channels > 0 else tmp_channels

        if in_channels == 0 or architecture == 'skip':
            self.fromrgb = Conv2dLayer(img_channels, tmp_channels, kernel_size=1, activation=activation,
                trainable=next(trainable_iter), conv_clamp=conv_clamp, channels_last=self.channels_last)

        self.conv0 = Conv2dLayer(conv0_in_channels, tmp_channels, kernel_size=3, activation=activation,
                trainable=next(trainable_iter), conv_clamp=conv_clamp, channels_last=self.channels_last)

        self.conv1 = Conv2dLayer(tmp_channels, out_channels, kernel_size=3, activation=activation, down=2,
            trainable=next(trainable_iter), resample_filter=resample_filter, conv_clamp=conv_clamp, channels_last=self.channels_last)

        if architecture == 'resnet':
            self.skip = Conv2dLayer(conv0_in_channels, out_channels, kernel_size=1, bias=False, down=2,
                trainable=next(trainable_iter), resample_filter=resample_filter, channels_last=self.channels_last)

    def forward(self, x, img, force_fp32=False):
        dtype = torch.float16 if self.use_fp16 and not force_fp32 else torch.float32
        memory_format = torch.channels_last if self.channels_last and not force_fp32 else torch.contiguous_format

        # Input.
        if x is not None:
            misc.assert_shape(x, [None, self.in_channels, self.resolution, self.resolution])
            x = x.to(dtype=dtype, memory_format=memory_format)

        # FromRGB.
        if self.in_channels == 0 or self.architecture == 'skip':
            misc.assert_shape(img, [None, self.img_channels, self.resolution, self.resolution])
            img = img.to(dtype=dtype, memory_format=memory_format)
            y = self.fromrgb(img)
            x = x + y if x is not None else y
            img = upfirdn2d.downsample2d(img, self.resample_filter) if self.architecture == 'skip' else None

        # Main layers.
        if self.architecture == 'resnet':
            y = self.skip(x, gain=np.sqrt(0.5))
            x = self.conv0(x)
            x = self.conv1(x, gain=np.sqrt(0.5))
            x = y.add_(x)
        else:
            x = self.conv0(x)
            x = self.conv1(x)

        assert x.dtype == dtype
        return x, img

#----------------------------------------------------------------------------

@persistence.persistent_class
class MinibatchStdLayer(torch.nn.Module):
    def __init__(self, group_size, num_channels=1):
        super().__init__()
        self.group_size = group_size
        self.num_channels = num_channels

    def forward(self, x):
        N, C, H, W = x.shape
        with misc.suppress_tracer_warnings(): # as_tensor results are registered as constants
            G = torch.min(torch.as_tensor(self.group_size), torch.as_tensor(N)) if self.group_size is not None else N
        F = self.num_channels
        c = C // F

        y = x.reshape(G, -1, F, c, H, W)    # [GnFcHW] Split minibatch N into n groups of size G, and channels C into F groups of size c.
        y = y - y.mean(dim=0)               # [GnFcHW] Subtract mean over group.
        y = y.square().mean(dim=0)          # [nFcHW]  Calc variance over group.
        y = (y + 1e-8).sqrt()               # [nFcHW]  Calc stddev over group.
        y = y.mean(dim=[2,3,4])             # [nF]     Take average over channels and pixels.
        y = y.reshape(-1, F, 1, 1)          # [nF11]   Add missing dimensions.
        y = y.repeat(G, 1, H, W)            # [NFHW]   Replicate over group and pixels.
        x = torch.cat([x, y], dim=1)        # [N(C+1)HW]   Append to input as new channels.
        return x

#----------------------------------------------------------------------------

@persistence.persistent_class
class DiscriminatorEpilogue(torch.nn.Module):
    def __init__(self,
        in_channels,                    # Number of input channels.
        cmap_dim,                       # Dimensionality of mapped conditioning label, 0 = no label.
        resolution,                     # Resolution of this block.
        img_channels,                   # Number of input color channels.
        architecture        = 'resnet', # Architecture: 'orig', 'skip', 'resnet'.
        mbstd_group_size    = 4,        # Group size for the minibatch standard deviation layer, None = entire minibatch.
        mbstd_num_channels  = 1,        # Number of features for the minibatch standard deviation layer, 0 = disable.
        activation          = 'lrelu',  # Activation function: 'relu', 'lrelu', etc.
        conv_clamp          = None,     # Clamp the output of convolution layers to +-X, None = disable clamping.
        cfg                 = {},       # Architecture config.
        return_vec          = False,    # Whether to return a condition vector only
    ):
        assert architecture in ['orig', 'skip', 'resnet']
        super().__init__()

        self.cfg = cfg
        self.in_channels = in_channels
        self.cmap_dim = cmap_dim
        self.resolution = resolution
        self.img_channels = img_channels
        self.architecture = architecture
        self.return_vec = return_vec

        if architecture == 'skip':
            self.fromrgb = Conv2dLayer(img_channels, in_channels, kernel_size=1, activation=activation)
        self.mbstd = MinibatchStdLayer(group_size=mbstd_group_size, num_channels=mbstd_num_channels) if mbstd_num_channels > 0 else None
        self.conv = Conv2dLayer(in_channels + mbstd_num_channels, in_channels, kernel_size=3, activation=activation, conv_clamp=conv_clamp)
        self.fc = FullyConnectedLayer(in_channels * (resolution ** 2), in_channels, activation=activation)
        self.out = FullyConnectedLayer(in_channels, 1 if cmap_dim == 0 else cmap_dim)

    def forward(self, x, img, cmap, force_fp32=False):
        misc.assert_shape(x, [None, self.in_channels, self.resolution, self.resolution]) # [NCHW]
        _ = force_fp32 # unused
        dtype = torch.float32
        memory_format = torch.contiguous_format

        # FromRGB.
        x = x.to(dtype=dtype, memory_format=memory_format)
        if self.architecture == 'skip':
            misc.assert_shape(img, [None, self.img_channels, self.resolution, self.resolution])
            img = img.to(dtype=dtype, memory_format=memory_format)
            x = x + self.fromrgb(img)

        # Main layers.
        if self.mbstd is not None:
            x = self.mbstd(x)

        x = self.conv(x)
        x = self.fc(x.flatten(1))
        x = self.out(x) # [batch_size, out_dim]

        if self.return_vec:
            return x
            
        # Conditioning.
        if self.cmap_dim > 0:
            misc.assert_shape(cmap, [None, self.cmap_dim])
            x = (x * cmap).sum(dim=1, keepdim=True) * (1 / np.sqrt(self.cmap_dim)) # [batch_size, 1]

        assert x.dtype == dtype
        return x
        
@persistence.persistent_class
class Discriminator(torch.nn.Module):
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
            self.time_encoder = TemporalDifferenceEncoder(self.cfg)
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
                in_channels = (in_channels // self.cfg.num_frames_div_factor) * self.cfg.sampling.num_frames_per_video

            use_fp16 = (res >= fp16_resolution)
            block = DiscriminatorBlock(in_channels, tmp_channels, out_channels, resolution=res,
                first_layer_idx=cur_layer_idx, use_fp16=use_fp16, cfg=self.cfg, **block_kwargs, **common_kwargs)
            setattr(self, f'b{res}', block)
            cur_layer_idx += block.num_layers

        if self.c_dim > 0 or not self.time_encoder is None:
            self.mapping = MappingNetwork(z_dim=0, c_dim=total_c_dim, w_dim=cmap_dim, num_ws=None, w_avg_beta=None, **mapping_kwargs)
        self.b4 = DiscriminatorEpilogue(channels_dict[4], cmap_dim=cmap_dim, resolution=4, cfg=self.cfg, **epilogue_kwargs, **common_kwargs)
    
    #def forward(self, img, c, t, **block_kwargs):
    def forward(self, img, t, **block_kwargs):
        assert len(img) == t.shape[0] * t.shape[1], f"Wrong shape: {img.shape}, {t.shape}"
        assert t.ndim == 2, f"Wrong shape: {t.shape}"

        if not self.time_encoder is None:
            # Encoding the time distances
            t_embs = self.time_encoder(t.view(-1, self.cfg.sampling.num_frames_per_video)) # [batch_size, t_dim]
            # Concatenate `c` and time embeddings
            c = t_embs
            #c = torch.cat([c, t_embs], dim=1) # [batch_size, c_dim + t_dim]
            c = (c * 0.0) if self.cfg.dummy_c else c # [batch_size, c_dim + t_dim]

        x = None
        for res in self.block_resolutions:
            block = getattr(self, f'b{res}')
            if res == self.cfg.concat_res:
                # Concatenating the frames
                x = x.view(-1, self.cfg.sampling.num_frames_per_video, *x.shape[1:]) # [batch_size, num_frames, c, h, w]
                x = x.view(x.shape[0], -1, *x.shape[3:]) # [batch_size, num_frames * c, h, w]
            x, img = block(x, img, **block_kwargs)

        cmap = None
        if self.c_dim > 0 or not self.time_encoder is None:
            assert c.shape[1] > 0
        if c.shape[1] > 0:
            cmap = self.mapping(None, c)
        x = self.b4(x, img, cmap)
        x = x.squeeze(1) # [batch_size]

        return x