import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm


# -----------------------------------------------
#            Weights Initialization
# -----------------------------------------------

def weights_init(net, init_type: str='xavier', init_gain: float=0.02):
    """ Initialize network weights.
    :param net: is the network to be initialized
    :param init_type: is the name of an initialization method [normal | xavier | kaiming | orthogonal]
    :param init_gain: is scaling factor for [normal | xavier | orthogonal].
    """
    def init_func(m):
        classname = m.__class__.__name__

        if classname.find('Norm3d') != -1:
            # Weight
            if hasattr(m, 'weight') and m.weight is not None:
                nn.init.normal_(m.weight.data, 1.0, init_gain)
            # Bias
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)

        elif hasattr(m, 'weight') and classname.find('Conv') != -1 or classname.find('Linear') != -1:
            # Weight
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                m.reset_parameters()
            # Bias
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)

    # Apply the initialization function
    net.apply(init_func)



# -----------------------------------------------
#            Spectral Normalization
# -----------------------------------------------

def SN_Conv3d(*args, **kwargs):
    return spectral_norm(nn.Conv3d(*args, **kwargs), eps=1e-04)


def SN_Linear(*args, **kwargs):
    return spectral_norm(nn.Linear(*args, **kwargs), eps=1e-04)


# -----------------------------------------------
#                 Basic Modules
# -----------------------------------------------

# Vanilla ResNet block
class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = None, use_spectral_norm: bool = True):
        """
        :param in_channels: is the number of channels in the input
        :param out_channels: is the number of channels in the output
        """
        super().__init__()
        if out_channels is None:
            out_channels = in_channels

        # Two successive convolutional layers
        if use_spectral_norm:
            conv_layers = [
                SN_Conv3d(in_channels, out_channels, 3, padding=1),
                nn.InstanceNorm3d(out_channels, eps=1e-6, affine=False),
                nn.LeakyReLU(0.2),
                SN_Conv3d(out_channels, out_channels, 3, padding=1),
                nn.InstanceNorm3d(out_channels, eps=1e-6, affine=False),
                nn.LeakyReLU(0.2),
            ]
        else:
            conv_layers = [
                nn.Conv3d(in_channels, out_channels, 3, padding=1),
                nn.InstanceNorm3d(out_channels, eps=1e-6, affine=False),
                nn.LeakyReLU(0.2),
                nn.Conv3d(out_channels, out_channels, 3, padding=1),
                nn.InstanceNorm3d(out_channels, eps=1e-6, affine=False),
                nn.LeakyReLU(0.2),
            ]
        self.conv_layers = nn.Sequential(*conv_layers)

        # `in_channels` to `out_channels` mapping layer for residual connection
        if in_channels != out_channels:
            if use_spectral_norm:
                self.shortcut = SN_Conv3d(in_channels, out_channels, 1, stride=1, padding=0)
            else:
                self.shortcut = nn.Conv3d(in_channels, out_channels, 1, stride=1, padding=0)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor):
        """
        :param x: is the input feature map with shape `[batch_size, in_channels, depth, height, width]`
        """
        return self.shortcut(x) + self.conv_layers(x)

# Upsampling layer
class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_spectral_norm=False):
        super().__init__()

        if use_spectral_norm:
            self.main = nn.Sequential(
                nn.LeakyReLU(0.2),
                nn.Upsample(scale_factor=2, mode='trilinear'),
                SN_Conv3d(in_channels, out_channels, 3, 1, 1, bias=False),
            )
        else:
            self.main = nn.Sequential(
                nn.LeakyReLU(0.2),
                nn.Upsample(scale_factor=2, mode='trilinear'),
                nn.Conv3d(in_channels, out_channels, 3, 1, 1, bias=False),
            )

    def forward(self, feat):
        return self.main(feat)


# Downsampling layer
class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_spectral_norm=False):
        super().__init__()

        if use_spectral_norm:
            self.main = nn.Sequential(
                SN_Conv3d(in_channels, out_channels, 4, 2, 1, bias=False),
                nn.InstanceNorm3d(out_channels, eps=1e-6, affine=False),
                nn.LeakyReLU(0.2),
            )
        else:
            self.main = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, 4, 2, 1, bias=False),
                nn.InstanceNorm3d(out_channels, eps=1e-6, affine=False),
                nn.LeakyReLU(0.2),
            )

    def forward(self, feat):
        return self.main(feat)
