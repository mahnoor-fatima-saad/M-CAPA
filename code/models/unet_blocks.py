import torch 
import torch.nn as nn 
import torch.nn.functional as F




        
class ConvBlock(nn.Module):
  
    def __init__(self, feats_in=3, feats_out=16, feats_start=16):
        super().__init__()
        
        self.conv_block = nn.Sequential(
            nn.Conv2d(feats_in, feats_start, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(feats_start), 
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(feats_start, feats_out, kernel_size=3, padding=1 ,bias=False),
            nn.BatchNorm2d(feats_out),
            nn.LeakyReLU(0.2,inplace=True),
        )
    def forward(self, x):
        return self.conv_block(x)
    
class Bottleneck(nn.Module):
    
    def __init__(self, feats_in=3, feats_out=16, feats_start=16):
        super().__init__()
        
        self.bottleneck = nn.Sequential(
            nn.Conv2d(feats_in, feats_out, kernel_size=3, padding=0, bias=False),
            nn.BatchNorm2d(feats_out), 
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.bottleneck(x)

class DownSample(nn.Module):
    def __init__(self, feats_in, feats_out, config):
        super().__init__()
        self.maxpool = nn.Sequential( 
            nn.MaxPool2d(kernel_size=2, stride =2), 
            ConvBlock(feats_in, feats_out),
            nn.Dropout(p=config["dropout"]),
            
        )
    def forward(self, x):
        return self.maxpool(x)



class UpSample(nn.Module):
    def __init__(self, feats_in, feats_out, config):
        super().__init__()
        self.config = config
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(feats_in, feats_out, kernel_size = 2, stride =2)
        )
        if config["use_skipcons"]:
            self.conv_block = ConvBlock(feats_in, feats_out)
        else: 
            self.conv_block = ConvBlock(feats_out, feats_out)
        
    def forward(self, x, skipcon, use_skipcon):
        
        up = self.upsample(x)
        if(use_skipcon):  
            to_pad_y, to_pad_x = self.padding_for_upsample(up, skipcon)
            x_dir = to_pad_x//2, to_pad_x-to_pad_x//2
            y_dir = to_pad_y//2, to_pad_y-to_pad_y//2
            up = F.pad(up, [x_dir[0], x_dir[1], y_dir[0], y_dir[1]])
            x = torch.cat([skipcon, up], dim=1)
        else:
            x=up
        return self.conv_block(x)
    
   
    def padding_for_upsample(self,x1,x2):
        in_y = x2.size()[2] - x1.size()[2]
        in_x = x2.size()[3] - x1.size()[3]
        return in_y, in_x
    

class OutputLayer(nn.Module):
    def __init__(self, feats_in, feats_out):
        super().__init__()
        
        self.output = nn.Conv2d(feats_in, feats_out, kernel_size=1)
        

    def forward(self,x):
        return self.output(x)
     
