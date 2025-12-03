import torch
import torch.nn as nn
import torch.nn.functional as F
from models.unet_blocks import *
import numpy as np
from models.encoder import *
from models.fuser import *



def get_model(config, in_channels=2, out_channels=2, features=32, device=None, primary_encoder = None):
    if config["use_bias_term"]:
           out_channels = 4
    if "audio" not in config['modalities']:
        if primary_encoder == "rgb":
            in_channels = 3
        elif primary_encoder == "rgb+material" or primary_encoder == "rgb+depth":
           in_channels = 4
        else: 
           raise ValueError("Incorrect primary encoder")
        print(f"returning visual unet with primary encoder as {primary_encoder}", flush=True)
        return UNet_Visual(in_channels=in_channels, out_channels=out_channels, features=features, device=device, config = config, primary_encoder=primary_encoder)
    else: 
       print("returning Unet")
       return UNet(in_channels=in_channels, out_channels=out_channels, features=features, device=device,config=config)  
    
           


class UNet(nn.Module):
    def __init__(self, config, in_channels=2, out_channels=2, features=16, device=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.features = features
        self.device = device
        print("Device for Model: ", self.device, flush=True)
        self.modalities = config["modalities"]
        self.config = config
        self.use_skipcons = self.config["use_skipcons"]
        self.encoders = self.get_encoders(self.modalities, self.config)
        print("Encoders", self.encoders.keys())

      
        self.downblocks = nn.ModuleList([
            ConvBlock(feats_in=in_channels, feats_out=features, feats_start=features),
            DownSample(feats_in=features, feats_out=features*2,config=config),
            DownSample(feats_in=features*2, feats_out=features*4,config=config),
            DownSample(feats_in=features*4, feats_out=features*8,config=config),
            DownSample(feats_in=features*8, feats_out=features*16,config=config)
            
        ])
        self.fuser = Fuser(modalities=self.modalities, out_channels=self.features*16, features=self.features)
        

   
        self.upblocks = nn.ModuleList([
            UpSample(feats_in=features*16, feats_out=features*8,config=config),
            UpSample(feats_in=features*8, feats_out=features*4,config=config),
            UpSample(feats_in=features*4, feats_out=features*2,config=config),
            UpSample(feats_in=features*2, feats_out=features,config=config)
        ])

      
        self.output = OutputLayer(feats_in=features, feats_out=out_channels)

    
    


    def forward(self, x):
       
        embeddings = []
        levels = dict()
       
        for modality in self.modalities:
            if modality == "rgb":
                final, level_mod = self.encoders["rgb"](x["rgb"])
                levels["rgb"] = level_mod
             
                embeddings.append(final)
            elif modality == "semantic" or modality == "pretrained_semantic":
                final, level_mod = self.encoders["semantic"](x["semantic"])
                levels["semantic"] = level_mod
                embeddings.append(final)
            elif modality == "target_material":
            

                final, level_mod = self.encoders["material"](x["target_material"])
                levels["target_material"] = level_mod
                embeddings.append(final)
            elif modality =="audio":
                continue
            else:
                raise ValueError("Incorrect modalities. Try with proper modalities")

        x_down, skipcons = self.downsample(x=x["source_mag"])
       
        
        fuser_inputs = []
        
        fuser_inputs.append(x_down)
        for modality in levels.keys():
            fuser_inputs.append(levels[modality][-1])
        x_down = self.fuser(inputs = fuser_inputs)
            
        x_up = self.upsample(x_down, skipcons)
        x_out = self.output(x_up)
        
        del(x_up)
        del(embeddings)
        del(x_down)
        del(skipcons)
        del(x)

        
        return x_out
    def get_encoders(self, modalities, config):
        encoders = {}

        for modality in modalities:
            if modality == "rgb":
                print("getting rgb encoder", flush=True)

                encoder = encoder = Encoder(in_channels=3, config=config).to(self.device)
                encoders["rgb"] = encoder
            elif modality == "semantic" or modality == "pretrained_semantic":
                print("getting sem encoder", flush=True)

                    
                encoder = Encoder(in_channels=1, config=config).to(self.device)
                encoders["semantic"] = encoder
            elif modality == "target_material" or modality=="source_material":
                print("getting material encoder", flush=True)

                encoder = Encoder(in_channels=1, config=config).to(self.device)
                encoders["material"] = encoder

            elif modality == "audio":
                continue
            else:
                raise ValueError("Incorrect modalities. Try with proper modalities")
        return encoders

        
    def downsample(self, x):
        down_samples = []
          
        for i in range(len(self.downblocks)):
            down_samples.append(x)
            x = self.downblocks[i](x)

        return x, down_samples

    def upsample(self, x, skipcons):
      
        for upblock in self.upblocks:
            x = upblock(x, skipcons.pop(), use_skipcon = self.use_skipcons)

        return x
       

    




class UNet_Visual(nn.Module):
    def __init__(self, config, primary_encoder, in_channels=2, out_channels=2, features=16, device=None):
        super().__init__()
        self.in_channels = in_channels
        self.config = config
        self.out_channels = out_channels
        self.features = features
        self.device = device
        self.modalities = config["modalities"]
        self.primary_encoder = primary_encoder
        print("primary encoder in unet", self.primary_encoder, flush=True)
        if self.primary_encoder is not None:
           
            if self.primary_encoder == "rgb":
                self.modalities.remove("rgb")
            elif self.primary_encoder == "rgb+material":
                self.modalities.remove("rgb")
                self.modalities.remove("target_material")
            elif self.primary_encoder == "rgb+depth":
                self.modalities.remove("rgb")
                self.modalities.remove("depth")
            
            self.encoders = self.get_encoders()
            print("Encoders", self.encoders.keys())
        
        else:
             raise ValueError("Incorrect configuration, please check config file again")





        # Define down-sampling blocks
        
        self.downblocks = nn.ModuleList([
            ConvBlock(feats_in=in_channels, feats_out=features, feats_start=features),
            DownSample(feats_in=features, feats_out=features*2,config=config),
            DownSample(feats_in=features*2, feats_out=features*4,config=config),
            DownSample(feats_in=features*4, feats_out=features*8,config=config),
            DownSample(feats_in=features*8, feats_out=features*16,config=config)
        ])
        
        self.fuser = Fuser(modalities=self.modalities, out_channels=self.features*16, primary_encoder=self.primary_encoder)

        # Define up-sampling blocks
        self.upblocks = nn.ModuleList([
            UpSample(feats_in=features*16, feats_out=features*8, config=config),
            UpSample(feats_in=features*8, feats_out=features*4,config=config),
            UpSample(feats_in=features*4, feats_out=features*2, config=config),
            UpSample(feats_in=features*2, feats_out=features, config=config)
        ])  

        self.output = OutputLayer(feats_in=features, feats_out=out_channels)

    
    
    def forward(self, x):
        embeddings = []
        levels = dict()
        
        for modality in self.modalities:
            if modality == "rgb":
                if "rgb" in self.primary_encoder:
                    continue
                else:
                    final, level_mod = self.encoders["rgb"](x["rgb"].float())
                    levels["rgb"] = level_mod
                    embeddings.append(final)
            elif modality == "semantic" or modality == "pretrained_semantic":
                final, level_mod = self.encoders["semantic"](x["semantic"].float())
                levels["semantic"] = level_mod
                embeddings.append(final)
            elif modality == "target_material":
                if "material" in self.primary_encoder:
                    continue
                else:

                    final, level_mod = self.encoders["material"](x["target_material"].float())
                    levels["target_material"] = level_mod
                    embeddings.append(final)

            else:
                raise ValueError("Incorrect modalities. Try with proper modalities")

        if(self.primary_encoder == "rgb+material"):
            x_down, skipcons = self.downsample(x=torch.cat((x['rgb'], x['target_material']), dim=1), levels=levels)
        elif(self.primary_encoder == "rgb+depth"):
            x_down, skipcons = self.downsample(x=torch.cat((x['rgb'], x['depth']), dim=1), levels=levels)
        else:
            x_down, skipcons = self.downsample(x=x["rgb"].float(), levels=levels)
           

         
        fuser_inputs = []

        fuser_inputs.append(x_down)
        for modality in levels.keys():
            fuser_inputs.append(levels[modality][-1])
        x_down = self.fuser(inputs = fuser_inputs)
            
        

    
        


    


        x_up = self.upsample(x_down, skipcons)
        
        x_out = self.output(x_up)
        del(x_up)
        del(embeddings)
        del(x_down)
        del(skipcons)
        del(x)
       
        return x_out

        



            
    def get_encoders(self):
        encoders = {}


       
        for modality in self.modalities:
            if modality == "rgb":
              
                encoder = Encoder(in_channels=3, config=self.config).to(self.device)
                encoders["rgb"] = encoder
            elif modality == "semantic" or modality=="pretrained_semantic":
              
                encoder = Encoder(in_channels=1, config=self.config).to(self.device)
                encoders["semantic"] = encoder
            elif modality == "target_material": 

                encoder = Encoder(in_channels=1, config=self.config).to(self.device)
                encoders["material"] = encoder
           
            elif modality == "audio":
                encoder = Encoder(in_channels=2, config=self.config).to(self.device)
                encoders["audio"] = encoder
            else:
                raise ValueError("Incorrect modalities. Try with proper modalities")
        return encoders



    def downsample(self, x, levels):
       
        down_samples = []
        inputs = []
        for i in range(len(self.downblocks)):
          down_samples.append(x)
          x = self.downblocks[i](x)
       
        return x, down_samples

    def upsample(self, x, skipcons):
   
        for upblock in self.upblocks:
            x = upblock(x, skipcons.pop(), use_skipcon = True)
    

        return x


