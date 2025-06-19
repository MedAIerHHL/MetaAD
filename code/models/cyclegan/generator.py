import random
import torch
import torch.nn as nn
from typing import List

from .modules import ResBlock, UpBlock, DownBlock, SN_Conv3d

class Generator(nn.Module):
    def __init__(
            self,
            ngf: int,
            n_res_blocks: int,
            channel_multipliers: List[int],
            use_spectral_norm: bool=False,
            mask = False,
            mask_k = None,
            mask_prob = 1.0,
            mask_size = 1,
    ):
        super().__init__()
        self.mask = mask
        self.mask_k = mask_k
        self.mask_prob = mask_prob
        self.mask_size = mask_size
        # Number of channels at each resolution level
        channels_list = [ngf * m for m in channel_multipliers]
        levels = len(channel_multipliers)  # number of levels

        # First 3×3 convolution
        if use_spectral_norm:
            self.in_proj = SN_Conv3d(1, channels_list[0], 3, padding=1)
        else:
            self.in_proj = nn.Conv3d(1, channels_list[0], 3, padding=1)

        # Encoder of the U-Net
        encoder_block_channels = []
        channels = channels_list[0]

        self.encoder_blocks = nn.ModuleList()
        for i in range(levels):
            for j in range(n_res_blocks):
                enc_layers = []
                # Before residual block, downsample at all scales except for the first one
                if i != 0 and j == 0:
                    enc_layers.append(DownBlock(channels, channels, use_spectral_norm=use_spectral_norm))
                # Add residual block: [previous channels --> current channels]
                enc_layers.append(ResBlock(channels, channels_list[i], use_spectral_norm=use_spectral_norm))
                channels = channels_list[i]
                # Add them to the encoder of the U-Net
                self.encoder_blocks.append(nn.Sequential(*enc_layers))
                # Keep track of the channel number of the output
                encoder_block_channels.append(channels)

        # The middle of the U-Net
        self.middle_block = nn.Sequential(
            ResBlock(channels, channels, use_spectral_norm=True),
            ResBlock(channels, channels, use_spectral_norm=True),
        )

        # Decoder of the U-Net
        self.decoder_blocks = nn.ModuleList()
        for i in reversed(range(levels)):  # levels in reverse order
            for j in range(n_res_blocks):
                # Add residual block: [previous channels + skip connections --> current channels]
                dec_layers = [ResBlock(channels + encoder_block_channels.pop(), channels_list[i], use_spectral_norm=use_spectral_norm)]
                channels = channels_list[i]
                # After last residual block, up-sample at all levels except for the first one
                if i != 0 and j == n_res_blocks - 1:
                    dec_layers.append(UpBlock(channels, channels, use_spectral_norm=use_spectral_norm))
                # Add them to the decoder of the U-Net
                self.decoder_blocks.append(nn.Sequential(*dec_layers))

        # Final 3×3 convolution
        if use_spectral_norm:
            self.out_proj = nn.Sequential(
                SN_Conv3d(channels, 1, 3, padding=1),
                nn.Tanh(),
            )
        else:
            self.out_proj = nn.Sequential(
                nn.Conv3d(channels, 1, 3, padding=1),
                nn.Tanh(),
            )
            
    def gen_random_mask(self, x, mask_k, mask_size, mask_prob):
        if random.uniform(0, 1) <= mask_prob:
            b, c, d, h, w = x.shape
            L = (d // mask_size) * (h // mask_size) * (w // mask_size)
            len_keep = int(L * (1 - mask_k))

            noise = torch.randn(b, L, device=x.device)

            # sort noise for each sample
            ids_shuffle = torch.argsort(noise, dim=1)
            ids_restore = torch.argsort(ids_shuffle, dim=1)

            # generate the binary mask: 0 is keep 1 is remove
            mask = torch.ones([b, L], device=x.device)
            mask[:, :len_keep] = 0
            # unshuffle to get the binary mask
            mask = torch.gather(mask, dim=1, index=ids_restore)
            mask = mask.reshape(-1, d // mask_size, h // mask_size, w // mask_size).\
                repeat_interleave(mask_size, dim=1).\
                repeat_interleave(mask_size, dim=2).\
                repeat_interleave(mask_size, dim=3)
            mask = mask.unsqueeze(1).type_as(x)
            x = x * (1. - mask) - mask
        return x

    def add_mask(self, images, mask_k, mask_size, mask_prob):
        if random.uniform(0, 1) <= mask_prob:
            b, c, d, h, w = images.shape
            mask = torch.ones(b, 1, d, h, w)
            # Generate k random indices for each batch, channel, depth, height, width
            for i in range(b):
                for j in range(mask_k):
                    # Randomly choose indices for depth, height, and width
                    d_idx = random.randint(0, d - 1)
                    h_idx = random.randint(0, h - 1)
                    w_idx = random.randint(0, w - 1)
                    # Set the chosen pixel to 0
                    mask[i, 0, d_idx, h_idx, w_idx] = 0
            # Apply the mask
            mask = mask.cuda()
            masked_images = images * mask
            return masked_images


    def forward(
            self,
            x: torch.Tensor,
            test_mask = False,
    ):
        """
        :param x: is the input feature map with shape `[batch_size, in_channels, depth, height, width]`
        """
        self.mask = False
        if test_mask == True and self.mask:
            print('================================================================================')
            x = self.gen_random_mask(x, self.mask_k, self.mask_size, self.mask_prob)
        elif test_mask == False and self.mask:
            print('================================================================================')
            if self.training:
                x = self.gen_random_mask(x, self.mask_k, self.mask_size, self.mask_prob)

        # First 3×3 convolution
        x = self.in_proj(x)

        # Encoder of the U-Net
        x_input_block = []  # To store the encoder outputs for skip connections
        for module in self.encoder_blocks:
            x = module(x)
            x_input_block.append(x)

        # Middle of the U-Net
        x = self.middle_block(x)

        # Decoder of the U-Net
        for module in self.decoder_blocks:
            x = torch.cat([x, x_input_block.pop()], dim=1)
            x = module(x)

        # Final normalization and 3×3 convolution
        return self.out_proj(x)
