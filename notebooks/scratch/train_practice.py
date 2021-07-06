from math import log2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from narrow import narrow_by
from resample import Resampler

import h5py
import pickle
import pathlib
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

class G(nn.Module):
    def __init__(self, in_chan, out_chan, scale_factor=16,
                 chan_base=512, chan_min=64, chan_max=512, cat_noise=False,
                 **kwargs):
        super().__init__()

        self.scale_factor = scale_factor
        num_blocks = round(log2(self.scale_factor))

        assert chan_min <= chan_max

        def chan(b):
            c = chan_base >> b
            c = max(c, chan_min)
            c = min(c, chan_max)
            return c

        self.block0 = nn.Sequential(
            nn.Conv3d(in_chan, chan(0), 1),
            nn.LeakyReLU(0.2, True),
        )

        self.blocks = nn.ModuleList()
        for b in range(num_blocks):
            prev_chan, next_chan = chan(b), chan(b+1)
            self.blocks.append(
                HBlock(prev_chan, next_chan, out_chan, cat_noise))

    def forward(self, x):
        y = x  # direct upsampling from the input
        x = self.block0(x)

        #y = None  # no direct upsampling from the input
        for block in self.blocks:
            x, y = block(x, y)

        return y


class HBlock(nn.Module):
    """The "H" block of the StyleGAN2 generator.

        x_p                     y_p
         |                       |
    convolution           linear upsample
         |                       |
          >--- projection ------>+
         |                       |
         v                       v
        x_n                     y_n

    See Fig. 7 (b) upper in https://arxiv.org/abs/1912.04958
    Upsampling are all linear, not transposed convolution.

    Parameters
    ----------
    prev_chan : number of channels of x_p
    next_chan : number of channels of x_n
    out_chan : number of channels of y_p and y_n
    cat_noise: concatenate noise if True, otherwise add noise

    Notes
    -----
    next_size = 2 * prev_size - 6
    """
    def __init__(self, prev_chan, next_chan, out_chan, cat_noise):
        super().__init__()

        self.upsample = Resampler(3, 2)

        self.conv = nn.Sequential(
            AddNoise(cat_noise, chan=prev_chan),
            self.upsample,
            nn.Conv3d(prev_chan + int(cat_noise), next_chan, 3),
            nn.LeakyReLU(0.2, True),

            AddNoise(cat_noise, chan=next_chan),
            nn.Conv3d(next_chan + int(cat_noise), next_chan, 3),
            nn.LeakyReLU(0.2, True),
        )

        self.proj = nn.Sequential(
            nn.Conv3d(next_chan, out_chan, 1),
            nn.LeakyReLU(0.2, True),
        )

    def forward(self, x, y):
        x = self.conv(x)  # narrow by 3

        if y is None:
            y = self.proj(x)
        else:
            y = self.upsample(y)  # narrow by 1

            y = narrow_by(y, 2)

            y = y + self.proj(x)

        return x, y


class AddNoise(nn.Module):
    """Add or concatenate noise.

    Add noise if `cat=False`.
    The number of channels `chan` should be 1 (StyleGAN2)
    or that of the input (StyleGAN).
    """
    def __init__(self, cat, chan=1):
        super().__init__()

        self.cat = cat

        if not self.cat:
            self.std = nn.Parameter(torch.zeros([chan]))

    def forward(self, x):
        noise = torch.randn_like(x[:, :1])

        if self.cat:
            x = torch.cat([x, noise], dim=1)
        else:
            std_shape = (-1,) + (1,) * (x.dim() - 2)
            noise = self.std.view(std_shape) * noise

            x = x + noise

        return x
    
class D(nn.Module):
    def __init__(self, in_chan, out_chan, scale_factor=16,
                 chan_base=512, chan_min=64, chan_max=512,
                 **kwargs):
        super().__init__()

        self.scale_factor = scale_factor
        num_blocks = round(log2(self.scale_factor))

        assert chan_min <= chan_max

        def chan(b):
            if b >= 0:
                c = chan_base >> b
            else:
                c = chan_base << -b
            c = max(c, chan_min)
            c = min(c, chan_max)
            return c

        self.block0 = nn.Sequential(
            nn.Conv3d(in_chan, chan(num_blocks), 1),
            nn.LeakyReLU(0.2, True),
        )

        self.blocks = nn.ModuleList()
        for b in reversed(range(num_blocks)):
            prev_chan, next_chan = chan(b+1), chan(b)
            self.blocks.append(ResBlock(prev_chan, next_chan))

        self.block9 = nn.Sequential(
            nn.Conv3d(chan(0), chan(-1), 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv3d(chan(-1), 1, 1),
        )

    def forward(self, x):
        x = self.block0(x)

        for block in self.blocks:
            x = block(x)

        x = self.block9(x)

        return x


class ResBlock(nn.Module):
    """The residual block of the StyleGAN2 discriminator.

    Downsampling are all linear, not strided convolution.

    Notes
    -----
    next_size = (prev_size - 4) // 2
    """
    def __init__(self, prev_chan, next_chan):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv3d(prev_chan, prev_chan, 3),
            nn.LeakyReLU(0.2, True),

            nn.Conv3d(prev_chan, next_chan, 3),
            nn.LeakyReLU(0.2, True),
        )

        self.skip = nn.Conv3d(prev_chan, next_chan, 1)

        self.downsample = Resampler(3, 0.5)

    def forward(self, x):
        y = self.conv(x)

        x = self.skip(x)
        x = narrow_by(x, 2)

        x = x + y

        x = self.downsample(x)

        return x

def init_weights(m, init_weight_std=0.02):
    if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d,
        nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
        m.weight.data.normal_(0.0, init_weight_std)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
        nn.SyncBatchNorm, nn.LayerNorm, nn.GroupNorm,
        nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)):
        if m.affine:
            # NOTE: dispersion from DCGAN, why?
            m.weight.data.normal_(1.0, init_weight_std)
            m.bias.data.fill_(0)
            
def pad_data(x, pad_top_bot=5, pad_sides=5):

    if pad_sides > 0:
        x = torch.cat((x[:,:,:,:,-pad_sides:], x, x[:,:,:,:,:pad_sides]), axis=-1)
    
    if pad_top_bot > 0:
        x = torch.cat((torch.flip(x, [3])[:,:,:,-pad_top_bot-1:-1,:], # mirror array and select top rows
                         x, 
                         torch.flip(x, [3])[:,:,:,1:pad_top_bot+1,:]), # mirror array and select bottom rows
                      axis=-2) # append along longitudinal (left-right) axis
        
    return x

def crop_x_data(x, crop_height=1, crop_width=1,):
    "symetrically crop tensor"
    w = crop_width
    h = crop_height
    
    if crop_width == 0 and crop_height == 0:
        return x
    elif crop_width == 0:
        return x[:,:,:,h:-h,:]
    elif crop_height == 0:
        return x[:,:,:,:,w:-w]
    else:
        return x[:,:,:,h:-h,w:-w]


gen = G(in_chan=4, out_chan=4, scale_factor=8, chan_base=4, chan_min=64, chan_max=128, cat_noise=True).to(device)
critic = D(in_chan=4, out_chan=4, scale_factor=8, chan_base=4, chan_min=64, chan_max=128).to(device)

# initialize weights
gen.apply(init_weights)
critic.apply(init_weights)

# initializate optimizer
opt_gen = optim.Adam(gen.parameters(), lr=1e-4, betas=(0.0, 0.9))
opt_critic = optim.Adam(critic.parameters(), lr=1e-4, betas=(0.0, 0.9))


# load data
# load and prep data
with h5py.File("x_truth_01.hdf5", "r") as f:
    x_truth = torch.tensor(f["x_truth"][:])
print('shape.x_truth:', x_truth.shape)

with h5py.File("x_01.hdf5", "r") as f:
    x = torch.tensor(f["x"][:])
print('shape.x:', x.shape)

# training loop
NUM_EPOCHS = 1
CRITIC_ITERATIONS = 5

# x = x.to(device)
# x_truth = x_truth.to(device)
torch.cuda.empty_cache()

# radial indices that we will keep for the input (to make a depth of 30 radial layers)
index_keep = np.round(np.arange(0,x.shape[2], x.shape[2]/30.0)).astype(int)

for epoch in range(NUM_EPOCHS):
    # PREP DATA
    # input to the generator will be of shape (1, 4, 198, 18, 28)
    # therefore, pad the downscaled x to the appropriate size
    x_input = torch.roll(x, np.random.randint(0,14), 4)
    x_input = pad_data(x_input, pad_top_bot=3, pad_sides=2)
    
    # need an upsampled version of the x input
    # this will be concatenated onto both the generator output
    # and the true value as input to the descriminator
    x_up = F.interpolate(x_input, scale_factor=(1,8,8),
                          mode='trilinear', align_corners=False)
    x_up = crop_x_data(x_up, crop_height=21, crop_width=21,)
#     print('x_up shape:',x_up.shape)
    x_input = x_input[:,:,index_keep,:,:]
#     print('x_input shape:',x_input.shape)

    # Train Critic: max E[critic(real)] - E[critic(fake)]
    # equivalent to minimizing the negative of that
    for _ in range(CRITIC_ITERATIONS):
        fake = gen(x_input.to(device))
        