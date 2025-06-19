import torch
import torch.nn as nn

from .modules import DownBlock, SN_Conv3d


class Discriminator(nn.Module):
    def __init__(self, ndf: int = 16, use_spectral_norm: bool = True):
        super().__init__()
        
        # Feature channels at each spatial resolution
        nfc_multi = {8: 8, 16: 4, 32: 2, 64: 1}
        nfc = {}
        for k, v in nfc_multi.items():
            nfc[k] = int(v * ndf)

        # Network structure
        self.to_map = nn.Sequential(
            SN_Conv3d(1, nfc[64], 3, 1, 1, bias=False),
            DownBlock(nfc[64], nfc[32], use_spectral_norm=use_spectral_norm),
            DownBlock(nfc[32], nfc[16], use_spectral_norm=use_spectral_norm),
            DownBlock(nfc[16], nfc[8], use_spectral_norm=use_spectral_norm),
        )
        if use_spectral_norm:
            self.to_logits = SN_Conv3d(nfc[8], 1, 1, 1, 0, bias=False)
        else:
            self.to_logits = nn.Conv3d(nfc[8], 1, 1, 1, 0, bias=False)

    def forward(self, input_img: torch.Tensor):
        feat_out = self.to_map(input_img)
        logits = self.to_logits(feat_out)

        return logits
