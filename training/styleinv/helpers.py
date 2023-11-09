import math
from collections import namedtuple
import torch
from torch.nn import functional as F
from training.networks import FullyConnectedLayer
from torch_utils import misc, persistence
from torch_utils.ops import conv2d_resample
from torch_utils.ops import upfirdn2d
from torch_utils.ops import bias_act
from torch_utils.ops import fma
from training.networks import modulated_conv2d
import numpy as np

#-------------------------------------------------------------------------------

@persistence.persistent_class
class Flatten(torch.nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

@misc.profiled_function
def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output

#-------------------------------------------------------------------------------

class Bottleneck(namedtuple('Block', ['in_channel', 'depth', 'stride'])):
    """ A named tuple describing a ResNet block. """

def get_block(in_channel, depth, num_units, stride=2):
    return [Bottleneck(in_channel, depth, stride)] + [Bottleneck(depth, depth, 1) for i in range(num_units - 1)]

def get_blocks(num_layers):
    if num_layers == 50:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=4),
            get_block(in_channel=128, depth=256, num_units=14),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 100:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=13),
            get_block(in_channel=128, depth=256, num_units=30),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 152:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=8),
            get_block(in_channel=128, depth=256, num_units=36),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    else:
        raise ValueError("Invalid number of layers: {}. Must be one of [50, 100, 152]".format(num_layers))
    return blocks

#-------------------------------------------------------------------------------

@persistence.persistent_class
class SEModule(torch.nn.Module):
    def __init__(self, channels, reduction):
        super().__init__()
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc1 = torch.nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0, bias=False)
        self.relu = torch.nn.ReLU(inplace=True)
        self.fc2 = torch.nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0, bias=False)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x

@persistence.persistent_class
class bottleneck_IR(torch.nn.Module):
    def __init__(self, in_channel, depth, stride):
        super().__init__()
        if in_channel == depth:
            self.shortcut_layer = torch.nn.MaxPool2d(1, stride)
        else:
            self.shortcut_layer = torch.nn.Sequential(
                torch.nn.Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                torch.nn.BatchNorm2d(depth)
            )
        self.res_layer = torch.nn.Sequential(
            torch.nn.BatchNorm2d(in_channel),
            torch.nn.Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False), torch.nn.PReLU(depth),
            torch.nn.Conv2d(depth, depth, (3, 3), stride, 1, bias=False), torch.nn.BatchNorm2d(depth)
        )

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        return res + shortcut

@persistence.persistent_class
class bottleneck_IR_SE(torch.nn.Module):
    def __init__(self, in_channel, depth, stride):
        super().__init__()
        if in_channel == depth:
            self.shortcut_layer = torch.nn.MaxPool2d(1, stride)
        else:
            self.shortcut_layer = torch.nn.Sequential(
                torch.nn.Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                torch.nn.BatchNorm2d(depth)
            )
        self.res_layer = torch.nn.Sequential(
            torch.nn.BatchNorm2d(in_channel),
            torch.nn.Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False),
            torch.nn.PReLU(depth),
            torch.nn.Conv2d(depth, depth, (3, 3), stride, 1, bias=False),
            torch.nn.BatchNorm2d(depth),
            SEModule(depth, 16)
        )

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        return res + shortcut

@persistence.persistent_class
class AdaIN(torch.nn.Module):
    def __init__(self, style_dim, num_features, lr_multiplier=1):
        super().__init__()
        self.norm = torch.nn.InstanceNorm2d(num_features, affine=False)
        #self.fc = torch.nn.Linear(style_dim, num_features*2)
        self.fc = FullyConnectedLayer(style_dim, num_features*2, activation='linear', lr_multiplier=lr_multiplier)

    def forward(self, x, s):
        h = self.fc(s)
        h = h.view(-1, h.size(1), 1, 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma) * self.norm(x) + beta

@persistence.persistent_class
class AdainResBlk(torch.nn.Module):
    def __init__(self, dim_in, dim_out, style_dim=64, w_hpf=0,
                 actv=torch.nn.LeakyReLU(0.2), upsample=False, lr_multiplier=1):
        super().__init__()
        self.w_hpf = w_hpf
        self.actv = actv
        self.upsample = upsample
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out, style_dim, lr_multiplier)

    def _build_weights(self, dim_in, dim_out, style_dim=64, lr_multiplier=1):
        self.conv1 = torch.nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        self.conv2 = torch.nn.Conv2d(dim_out, dim_out, 3, 1, 1)
        self.norm1 = AdaIN(style_dim, dim_in, lr_multiplier)
        self.norm2 = AdaIN(style_dim, dim_out, lr_multiplier)
        if self.learned_sc:
            self.conv1x1 = torch.nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        if self.learned_sc:
            x = self.conv1x1(x)
        return x

    def _residual(self, x, s):
        x = self.norm1(x, s)
        x = self.actv(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv1(x)
        x = self.norm2(x, s)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x, s, normalize=True):
        out = self._residual(x, s)
        if self.w_hpf == 0:
            out = out + self._shortcut(x)
            if normalize:
                out = out / math.sqrt(2)
        return out

@persistence.persistent_class
class ModulationLayer(torch.nn.Module):
    def __init__(self,
        in_channels,                    # Number of input channels.
        out_channels,                   # Number of output channels.
        w_dim,                          # Intermediate latent (W) dimensionality.
        kernel_size     = 3,            # Convolution kernel size.
        up              = 1,            # Integer upsampling factor.
        use_noise       = True,         # Enable noise input?
        activation      = 'lrelu',      # Activation function: 'relu', 'lrelu', etc.
        resample_filter = [1,3,3,1],    # Low-pass filter to apply when resampling activations.
        conv_clamp      = None,         # Clamp the output of convolution layers to +-X, None = disable clamping.
        channels_last   = False,        # Use channels_last format for the weights?
    ):
        super().__init__()
        self.up = up
        self.use_noise = use_noise
        self.activation = activation
        self.conv_clamp = conv_clamp
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
        self.padding = kernel_size // 2
        self.act_gain = bias_act.activation_funcs[activation].def_gain

        self.affine = FullyConnectedLayer(w_dim, in_channels, bias_init=1)
        memory_format = torch.channels_last if channels_last else torch.contiguous_format
        self.weight = torch.nn.Parameter(torch.randn([out_channels, in_channels, kernel_size, kernel_size]).to(memory_format=memory_format))
        self.bias = torch.nn.Parameter(torch.zeros([out_channels]))

    def forward(self, x, w, fused_modconv=True, gain=1):
        styles = self.affine(w)

        flip_weight = (self.up == 1) # slightly faster
        x = modulated_conv2d(x=x, weight=self.weight, styles=styles, noise=None, up=self.up,
            padding=self.padding, resample_filter=self.resample_filter, flip_weight=flip_weight, fused_modconv=fused_modconv)

        act_gain = self.act_gain * gain
        act_clamp = self.conv_clamp * gain if self.conv_clamp is not None else None
        x = bias_act.bias_act(x, self.bias.to(x.dtype), act=self.activation, gain=act_gain, clamp=act_clamp)
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
        w = self.weight * self.weight_gain
        b = self.bias.to(x.dtype) if self.bias is not None else None
        flip_weight = (self.up == 1) # slightly faster
        x = conv2d_resample.conv2d_resample(x=x, w=w.to(x.dtype), f=self.resample_filter, up=self.up, down=self.down, padding=self.padding, flip_weight=flip_weight)

        act_gain = self.act_gain * gain
        act_clamp = self.conv_clamp * gain if self.conv_clamp is not None else None
        x = bias_act.bias_act(x, b, act=self.activation, gain=act_gain, clamp=act_clamp)
        return x

@persistence.persistent_class
class ModulationBlock(torch.nn.Module):
    def __init__(self,
        in_channels,                        # Number of input channels, 0 = first block.
        out_channels,                       # Number of output channels.
        w_dim,                              # Intermediate latent (W) dimensionality.
        resample_filter     = [1,3,3,1],    # Low-pass filter to apply when resampling activations.
        conv_clamp          = None,         # Clamp the output of convolution layers to +-X, None = disable clamping.
        use_fp16            = False,        # Use FP16 for this block?
        fp16_channels_last  = False,        # Use channels-last memory format with FP16?
        **layer_kwargs,                     # Arguments for SynthesisLayer.
    ):
        super().__init__()
        self.in_channels = in_channels
        self.w_dim = w_dim
        self.use_fp16 = use_fp16
        self.channels_last = (use_fp16 and fp16_channels_last)
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
        self.num_conv = 0
        self.num_torgb = 0

        self.conv0 = ModulationLayer(in_channels, out_channels, w_dim=w_dim, up=1,
            resample_filter=resample_filter, conv_clamp=conv_clamp, channels_last=self.channels_last, **layer_kwargs)
        self.num_conv += 1

        self.conv1 = ModulationLayer(out_channels, out_channels, w_dim=w_dim,
            conv_clamp=conv_clamp, channels_last=self.channels_last, **layer_kwargs)
        self.num_conv += 1

        self.skip = Conv2dLayer(in_channels, out_channels, kernel_size=1, bias=False, up=1,
            resample_filter=resample_filter, channels_last=self.channels_last)

    def forward(self, x, ws, force_fp32=False, fused_modconv=None, **layer_kwargs):
        misc.assert_shape(ws, [None, self.w_dim])

        dtype = torch.float16 if self.use_fp16 and not force_fp32 else torch.float32
        memory_format = torch.channels_last if self.channels_last and not force_fp32 else torch.contiguous_format
        if fused_modconv is None:
            with misc.suppress_tracer_warnings(): # this value will be treated as a constant
                fused_modconv = (not self.training) and (dtype == torch.float32 or int(x.shape[0]) == 1)

        # Input.
        x = x.to(dtype=dtype, memory_format=memory_format)

        # Main layers.        
        y = self.skip(x, gain=np.sqrt(0.5))
        x = self.conv0(x, ws, fused_modconv=fused_modconv, **layer_kwargs)
        x = self.conv1(x, ws, fused_modconv=fused_modconv, gain=np.sqrt(0.5), **layer_kwargs)
        x = y.add_(x)
        
        assert x.dtype == dtype
        return x