import torch
import torch.nn as nn
import torch.nn.functional as F
from models.unet_blocks import *
import numpy as np

import torch.nn as nn

class Fuser(nn.Module):
    def __init__(self, modalities, primary_encoder = None, out_channels=256, features=16, device=None,kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.out = features*16
        self.features = features
        self.device = device
        
        if primary_encoder in ["rgb","rgb+material","rgb+depth"]:
           self.in_channels = out_channels*(len(modalities)+1)
        else:
            self.in_channels = out_channels *(len(modalities))
        




        self.conv = nn.Conv2d(self.in_channels, out_channels, kernel_size, stride, padding)


    def forward(self,inputs):

        fused = torch.concat([input for input in inputs], dim=1)
        
        formatted = self.conv(fused)
            

        return formatted

