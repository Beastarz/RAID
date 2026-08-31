"""
Module: frontend_bayar
Project: AI Image Detector

M3 fallback frontend: Bayar constrained convolution + fixed SRM filters,
replacing NPR's single fixed nearest-neighbour residual operator.

npr_stream_guide.md SS7 predicted NPR would fail the resize/downscale stress
test, and training/evaluate_npr.py confirmed it: clean AUC 0.9194 collapsed
to 0.48-0.62 (near chance) under every resize severity tested. This module
is the guide's prescribed fix -- swap the frontend, keep everything else
(backbone, fusion, data pipeline) identical.

Two branches, both operating on raw [0, 1] pixels:

- `BayarConv2d`: a *learnable* constrained convolution (Bayar & Stamm,
  2018). Each depthwise 5x5 filter's center tap is hard-fixed at -1 and the
  other 24 taps always renormalize to sum to +1 (reparameterized fresh every
  forward call, so the constraint holds exactly and gradients still flow) --
  it's forced to always compute *some* local prediction-error residual, but
  which predictor it learns is adaptive, unlike NPR's one fixed
  nearest-neighbour rule. That adaptivity is the point: a learned filter can
  settle on a predictor that survives a resize round trip in a way a fixed
  one cannot.
- `SRMFilterBank`: fixed (non-learnable), classic Spatial Rich Model
  high-pass kernels from steganalysis, widely reused in forensic CNNs
  (e.g. Zhou et al., CVPR 2018) as a complementary, non-adaptive residual
  signal.

`BayarSRMFrontend` runs both, concatenates, and fuses back down to
[B, 3, H, W] via a learnable 1x1 conv -- matching `NPR`'s exact interface
([B, 3, H, W] in [0, 1] -> [B, 3, H, W]), so it drops into
`NPRStream(frontend=BayarSRMFrontend())` with no changes to NPRStream,
BaseFeatureStream, or anything downstream. NPR itself is untouched and
remains the default -- this is an additional option, not a replacement.
"""

from typing import Final

import torch
import torch.nn as nn
import torch.nn.functional as F

_KERNEL_SIZE: Final[int] = 5
_NUM_SRM_KERNELS: Final[int] = 3

# Three classic fixed SRM (Spatial Rich Model) high-pass kernels, widely
# reused in forensic/manipulation-detection CNNs. Normalized so each sums to
# ~0 (true high-pass) with unit-ish scale.
_SRM_KERNELS = [
    # First-order horizontal edge (3x3, zero-padded to 5x5)
    [
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, -1, 1, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ],
    # Second-order (5x5 "SQUARE5x5" family), /4
    [
        [-1, 2, -2, 2, -1],
        [2, -6, 8, -6, 2],
        [-2, 8, -12, 8, -2],
        [2, -6, 8, -6, 2],
        [-1, 2, -2, 2, -1],
    ],
    # Third-order horizontal (5x5), /3
    [
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 1, -3, 3, -1],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ],
]
_SRM_NORMALIZERS: Final[list] = [1.0, 4.0, 3.0]


class BayarConv2d(nn.Module):
    """Depthwise, learnable constrained convolution (Bayar & Stamm, 2018).

    One 5x5 filter per input channel. The center tap is always -1 and the
    remaining 24 taps always sum to +1 -- reparameterized from raw learnable
    weights on every forward call, so the constraint is exact and
    differentiable rather than enforced by a post-optimizer-step hook.
    """

    def __init__(self, channels: int = 3, kernel_size: int = _KERNEL_SIZE) -> None:
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.center = (kernel_size * kernel_size) // 2
        # One filter per channel, kernel_size**2 - 1 free (non-center) taps.
        self.raw_weight = nn.Parameter(torch.randn(channels, 1, kernel_size * kernel_size - 1) * 0.01)

    def _constrained_kernel(self) -> torch.Tensor:
        denom = self.raw_weight.sum(dim=-1, keepdim=True)
        # sum(raw_weight / denom) == 1 exactly whenever denom == sum(raw_weight),
        # for ANY nonzero denom -- positive or negative, since raw_weight can be
        # either. Flooring the magnitude with clamp_min alone would silently
        # flip a negative denom positive and break that identity; guard only
        # the degenerate near-zero case instead, preserving sign otherwise.
        safe_denom = torch.where(denom.abs() < 1e-8, torch.full_like(denom, 1e-8), denom)
        normalized = self.raw_weight / safe_denom
        center = torch.full(
            (self.channels, 1, 1), -1.0, device=self.raw_weight.device, dtype=self.raw_weight.dtype
        )
        flat_kernel = torch.cat([normalized[:, :, : self.center], center, normalized[:, :, self.center :]], dim=-1)
        return flat_kernel.view(self.channels, 1, self.kernel_size, self.kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        kernel = self._constrained_kernel()
        return F.conv2d(x, kernel, groups=self.channels, padding=self.kernel_size // 2)


class SRMFilterBank(nn.Module):
    """Fixed (non-learnable) SRM high-pass filters, applied per channel.

    Each of the 3 input channels is convolved with all 3 fixed SRM kernels
    independently, producing 9 output channels total (3 channels x 3
    kernels). No learnable parameters.
    """

    def __init__(self, channels: int = 3) -> None:
        super().__init__()
        self.channels = channels
        self.num_kernels = _NUM_SRM_KERNELS
        kernels = torch.tensor(_SRM_KERNELS, dtype=torch.float32)
        normalizers = torch.tensor(_SRM_NORMALIZERS, dtype=torch.float32).view(-1, 1, 1)
        kernels = kernels / normalizers  # [num_kernels, 5, 5]
        # Repeat the same kernel set for every input channel: weight shape
        # [channels * num_kernels, 1, k, k], grouped so each input channel
        # only ever sees its own copy of the 3 kernels.
        weight = kernels.unsqueeze(0).repeat(channels, 1, 1, 1).reshape(channels * self.num_kernels, 1, 5, 5)
        self.register_buffer("weight", weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, self.weight, groups=self.channels, padding=2)


class BayarSRMFrontend(nn.Module):
    """Bayar + SRM residual frontend, drop-in for NPRStream's `frontend=`.

    Interface: [B, 3, H, W] in [0, 1] -> [B, 3, H, W], identical to `NPR`.
    Runs the Bayar (3 channels) and SRM (9 channels) branches on raw pixels,
    concatenates to 12 channels, and fuses back to 3 via a learnable 1x1
    conv. Forced to fp32 like NPR -- residual-scale signals are small enough
    that fp16 autocast can underflow them into noise.
    """

    def __init__(self) -> None:
        super().__init__()
        self.bayar = BayarConv2d(channels=3)
        self.srm = SRMFilterBank(channels=3)
        self.fuse = nn.Conv2d(3 + 3 * _NUM_SRM_KERNELS, 3, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.float()
            bayar_out = self.bayar(x)  # [B, 3, H, W]
            srm_out = self.srm(x)  # [B, 9, H, W]
            combined = torch.cat([bayar_out, srm_out], dim=1)  # [B, 12, H, W]
            return self.fuse(combined)  # [B, 3, H, W]
