import torch
import torch.nn as nn
import torch.nn.functional as F
from models.unet_blocks import *
import numpy as np


class Encoder(nn.Module):
    def __init__(self, in_channels, config, out_channels=2, features=32, device = None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.features = features
        self.device = device
        self.config = config
        
        self.downblocks = nn.ModuleList([
            ConvBlock(feats_in=in_channels, feats_out=features, feats_start=features),
            DownSample(feats_in=features, feats_out=features*2,config=config),
            DownSample(feats_in=features*2, feats_out=features*4, config=config),
            DownSample(feats_in=features*4, feats_out=features*8,config=config),
            DownSample(feats_in=features*8, feats_out=features*16,config=config)
        ])

        
    
    def forward(self,x):
        
            
        levels = []
        for downblock in self.downblocks:

            x = downblock(x)
            levels.append(x)

        return x, levels
        