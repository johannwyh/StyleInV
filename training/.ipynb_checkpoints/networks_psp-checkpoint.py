import matplotlib
matplotlib.use('Agg')
import math
from collections import namedtuple
import torch
from training.networks import FullyConnectedLayer
from torch_utils import misc, persistence

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
class BackboneEncoderUsingLastLayerIntoW(torch.nn.Module):
    def __init__(self, num_layers, mode='ir', opts=None):
        super().__init__()
        assert num_layers in [50, 100, 152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.input_layer = torch.nn.Sequential(torch.nn.Conv2d(opts.input_nc, 64, (3, 3), 1, 1, bias=False),
                                               torch.nn.BatchNorm2d(64),
                                               torch.nn.PReLU(64))
        self.output_pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.linear = FullyConnectedLayer(512, 512, lr_multiplier=1)
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(bottleneck.in_channel,
                                           bottleneck.depth,
                                           bottleneck.stride))
        self.body = torch.nn.Sequential(*modules)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.body(x)
        x = self.output_pool(x)
        x = x.view(-1, 512)
        x = self.linear(x)
        return x

#-------------------------------------------------------------------------------

def get_keys(d, name):
    if 'state_dict' in d:
        d = d['state_dict']
    d_filt = {k[len(name) + 1:]: v for k, v in d.items() if k[:len(name)] == name}
    return d_filt

@persistence.persistent_class
class pSp(torch.nn.Module):
    def __init__(self, opts, rank):
        super().__init__()
        self.rank = rank
        self.set_opts(opts)
        # compute number of style inputs based on the output resolution
        self.opts.n_styles = int(math.log(self.opts.output_size, 2)) * 2 - 2
        # Define architecture
        self.encoder = self.set_encoder()
        self.face_pool = torch.nn.AdaptiveAvgPool2d((256, 256))
        # Load weights if needed
        self.load_weights()

    def set_encoder(self):

        if self.opts.encoder_type == 'GradualStyleEncoder':
            raise NotImplementedError(f"{self.opts.encoder_type} has not been supported by this repo.")
            #encoder = psp_encoders.GradualStyleEncoder(50, 'ir_se', self.opts)
        elif self.opts.encoder_type == 'BackboneEncoderUsingLastLayerIntoW':
            encoder = BackboneEncoderUsingLastLayerIntoW(50, 'ir_se', self.opts)
        elif self.opts.encoder_type == 'BackboneEncoderUsingLastLayerIntoWPlus':
            raise NotImplementedError(f"{self.opts.encoder_type} has not been supported by this repo.")
            #encoder = psp_encoders.BackboneEncoderUsingLastLayerIntoWPlus(50, 'ir_se', self.opts)
        else:
            raise Exception('{} is not a valid encoders'.format(self.opts.encoder_type))

        if self.rank == 0:
            print(f'Using {self.opts.encoder_type}')
        return encoder

    def load_weights(self):
        if self.rank == 0:
            print('Loading encoders weights from irse50!')
        encoder_ckpt = torch.load(self.opts.irse50, map_location=torch.device('cpu'))
        self.encoder.load_state_dict(encoder_ckpt, strict=False)
    
    def set_latent_avg(self, stylegan_w_avg):
        self.register_buffer('latent_avg', stylegan_w_avg)

    def forward(self, x, latent_mask=None, inject_latent=None, alpha=None):
        
        codes = self.encoder(x)
        # normalize with respect to the center of an average face
        if self.opts.start_from_latent_avg:
            if self.opts.learn_in_w:
                codes = codes + self.latent_avg.repeat(codes.shape[0], 1)
            else:
                codes = codes + self.latent_avg.repeat(codes.shape[0], 1, 1)

        if latent_mask is not None:
            for i in latent_mask:
                if inject_latent is not None:
                    if alpha is not None:
                        codes[:, i] = alpha * inject_latent[:, i] + (1 - alpha) * codes[:, i]
                    else:
                        codes[:, i] = inject_latent[:, i]
                else:
                    codes[:, i] = 0

        if self.opts.learn_in_w:
            codes = codes.unsqueeze(1).repeat([1, self.opts.n_styles, 1])
        
        return codes

    def set_opts(self, opts):
        self.opts = opts
