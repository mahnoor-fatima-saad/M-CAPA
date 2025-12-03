from PIL import Image
from torch.utils.data import Dataset
import os 
import numpy as np
import torch
import random
from torch.utils.data.dataloader import default_collate
from torchvision import transforms
import torchvision.transforms.functional as TF
from utils import *
import time
import cv2
from material_mapper import *
import pandas as pd
from scipy.signal import fftconvolve
from PIL import Image



class CustomDataset(Dataset):
    def __init__(self, config, dataset,material_mapper, unseen_pairs, train=False, seed = 42):
        print("GETTING DATASET READY", flush=True)
        self.train = train
        self.config = config
        self.img_size = self.config["image_size"]
        self.seed = seed
        self.k_val = self.config['k_threshold']
        self.data = dataset
        self.data_path = self.config['DataPath']
        self.iterations_cap = self.config['iterations_for_matching']
 
        self.modalities = self.config['modalities']
        self.num_materials = self.config['num_materials']
   

        if("rgb" in self.modalities):
            self.use_rgb = True
        else: self.use_rgb = False


        if("semantic" in self.modalities or "pretrained_semantic" in self.modalities ):
            self.use_semantic = True
        else: self.use_semantic = False


        if("target_material" in self.modalities or "source_material" in self.modalities):
            self.use_material = True
        else: self.use_material = False

        self.hop_length = config["hop_length"]
        self.window_size = config["win_length"]
        self.nfft = config["nfft"]
        self.transform = config["transform"]
        if self.config["transform"]:
            self.crop = transforms.RandomResizedCrop((self.config["image_size"],self.config["image_size"]))
            self.color_jitter = transforms.ColorJitter(brightness=0.25, contrast=0.1, saturation=0.1, hue=0.1)

        if config["sampling_rate"]!=48000:
            self.resampling_kernel = torchaudio.transforms.Resample(orig_freq=48000, new_freq=config["sampling_rate"], resampling_method="sinc_interp_kaiser", rolloff = 0.9475937167399596, lowpass_filter_width=64,beta=14.769656459379492)            
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
        self.unseen_pairs = unseen_pairs
        self.material_mapper = material_mapper
        self.material_mapping_dict = self.material_mapper.material_map_dict
                
            
        
        
    @property
    def dataset(self):
        return self.data 
     
    def __len__(self):
        return self.data.shape[0]
    
    def is_in_unseen_pairs(self, source_mat, target_mat):
        if self.unseen_pairs is None:
            raise ValueError("Unseen Pairs List must be passed into the custom dataset")
        elif([source_mat, target_mat] in self.unseen_pairs):
            return True
    
    def audio_l2_diffs(self,source_audio, target_audio):
        l2_difference = torch.sqrt(torch.sum((source_audio - target_audio) ** 2))
        return l2_difference

    def k_thresholding(self, source_mat, target_mat, k=0.20):
        average_difference = np.mean(np.where(np.abs(source_mat - target_mat) >0,1,0))
        return average_difference>=k

    def fetch_audios(self,dir, is_source = False):
        
        waveform,sr = torchaudio.load(dir)

        return waveform



    
    def transformations(self,input):
        if len(self.modalities) == 1 and self.modalities[0] == "audio":
            return input
        else:
            if self.use_material:
                i,j,h,w = self.crop.get_params(input['target_material'], scale=(0.7, 1.0), ratio=(1.0, 1.0))
            else: 
                i,j,h,w = self.crop.get_params(input['rgb'], scale=(0.7, 1.0), ratio=(1.0, 1.0))
            if "rgb" in self.modalities:
                input["rgb"] = self.color_jitter(TF.resized_crop(input["rgb"], i, j, h, w, size=(256, 256)))
            if "semantic" in self.modalities:
                input["semantic"] = TF.resized_crop(input['semantic'], i, j, h,w,size=(self.config["image_size"], self.config["image_size"]), interpolation=TF.InterpolationMode.NEAREST) 
            if "target_material" in self.modalities:
                input["source_material"] = TF.resized_crop(input['source_material'], i, j, h,w,size=(self.config["image_size"], self.config["image_size"]), interpolation=TF.InterpolationMode.NEAREST)
                input["target_material"] = TF.resized_crop(input['target_material'], i, j, h,w,size=(self.config["image_size"], self.config["image_size"]), interpolation=TF.InterpolationMode.NEAREST) 
            if "depth" in self.modalities:
                input["depth"] = TF.resized_crop(input['depth'], i, j, h,w,size=(self.config["image_size"], self.config["image_size"]), interpolation=TF.InterpolationMode.NEAREST)
        return input


    def __getitem__(self, idx):  
       
        sample = {}
        while True:
           
            source_sample = self.data.iloc[idx]
            
            scene = source_sample['scene_name']
            source_mat_idx = source_sample['material_idx']
            location_idx = source_sample['location_idx']
                    
            for _ in range(self.iterations_cap):
                target_df = self.data[
                    (self.data['scene_name']==scene) & 
                    (self.data['location_idx']==location_idx) & 
                    (self.data['material_idx'] != source_mat_idx)
                    ]
                
                target_sample = target_df.sample()
                target_mat_idx = target_sample.iloc[0]['material_idx']
                if(self.is_in_unseen_pairs(str(source_mat_idx), str(target_mat_idx))):
                    continue
                else: 
                    source_material_dir = os.path.join(self.data_path, scene, "material_config", str(source_mat_idx), "material", f"{scene}_{str(source_mat_idx)}_material_{location_idx}.npz")
                    target_material_dir = os.path.join(self.data_path, scene, "material_config", str(target_mat_idx), "material", f"{scene}_{str(target_mat_idx)}_material_{location_idx}.npz")
                    if not os.path.exists(source_material_dir) or not os.path.exists(target_material_dir):
                        raise FileNotFoundError(f"Material files missing: {source_material_dir} or {target_material_dir}")
                    try:
                        target_material = read_npz(target_material_dir)
                    except Exception as e:
                        print(f"Exception raised for target material at {target_material_dir}: {e}")
                    
                    try: 
                        source_material = read_npz(source_material_dir)
                    except Exception as e:
                        print(f"Exception raised for source material at {source_material_dir}: {e}")

                    if(not self.k_thresholding(k=self.k_val, source_mat=source_material, target_mat=target_material)):
                        continue
                    else: 
                      
                        target_ir_dir = os.path.join(self.data_path, scene, "material_config", str(target_mat_idx), "waveform", f"{scene}_{str(target_mat_idx)}_ir_{location_idx}.wav")
                        source_ir_dir = os.path.join(self.data_path, scene, "material_config", str(source_mat_idx), "waveform", f"{scene}_{str(source_mat_idx)}_ir_{location_idx}.wav")
                        sample['target_audio'] = self.fetch_audios(target_ir_dir)
                        sample['source_audio'] = self.fetch_audios(source_ir_dir, is_source=True)
                        if self.use_material:  
                            sample['source_material'] = to_tensor(normalize_images(np.expand_dims(cv2.resize(source_material, dsize=(256, 256), interpolation=cv2.INTER_NEAREST), axis=0), max_val = self.num_materials))
                            sample['target_material'] = to_tensor(normalize_images(np.expand_dims(cv2.resize(target_material, dsize=(256, 256), interpolation=cv2.INTER_NEAREST), axis=0), max_val = self.num_materials))
                           

                        if self.use_semantic:
                            if "pretrained_semantic" in self.modalities:
                                semantic_dir = os.path.join(self.config["pretrained_semantic_data_path"], scene, f"{scene}_category_{location_idx}_0.npz")
                                sem_data = read_npz(semantic_dir)
                                sample['semantic']= to_tensor(normalize_images(np.expand_dims(cv2.resize(sem_data, dsize=(256, 256), interpolation=cv2.INTER_NEAREST), axis=0), max_val=149))
                                source_mat_inferred = self.material_mapper.map_material_to_semantic_image(material_index=source_mat_idx, semantic_data=sem_data)
                                sample['source_material'] = to_tensor(normalize_images(np.expand_dims(cv2.resize(source_mat_inferred, dsize=(256, 256), interpolation=cv2.INTER_NEAREST), axis=0), max_val = self.num_materials))
                                target_mat_inferred = self.material_mapper.map_material_to_semantic_image(material_index=target_mat_idx, semantic_data=sem_data)
                                sample['target_material'] =  to_tensor(normalize_images(np.expand_dims(cv2.resize(target_mat_inferred, dsize=(256, 256), interpolation=cv2.INTER_NEAREST), axis=0), max_val = self.num_materials))
                            else:
                                semantic_dir = os.path.join(self.data_path, scene, "category", f"{scene}_category_{location_idx}_0.npz")
                                sample['semantic']= to_tensor(normalize_images(np.expand_dims(cv2.resize(read_npz(semantic_dir), dsize=(256, 256), interpolation=cv2.INTER_NEAREST), axis=0), max_val=40))
                        if self.use_rgb:
                            rgb_dir = os.path.join(self.data_path, scene, "rgb", f"{scene}_rgb_{location_idx}_0.png")
                            sample['rgb']  = to_tensor(normalize_images(np.transpose(cv2.resize(cv2.cvtColor(cv2.imread(rgb_dir), cv2.COLOR_BGR2RGB), dsize=(256, 256), interpolation=cv2.INTER_AREA)[..., :3], (2, 0, 1)), max_val=255))
                                
                            
                        
                        if self.transform and random.random()<0.5 and self.train:
                            sample = self.transformations(sample)
                        return sample
            idx = random.randint(0,len(self.data)-1)
                                     
                      


class CustomPairDataset(Dataset):
    def __init__(self, dataset, config,material_mapper, seed = 42):
        print("GETTING VAL SETS READY", flush=True)
        self.config = config
        self.data_path = self.config['DataPath']
        self.img_size = self.config["image_size"]
        self.seed = seed
        self.k_val = self.config['k_threshold']
        self.data = dataset
        self.modalities = self.config['modalities']
    
        if("rgb" in self.modalities):
            self.use_rgb = True
        else: self.use_rgb = False

        if("semantic" in self.modalities or "pretrained_semantic" in self.modalities):
            self.use_semantic = True

        else: self.use_semantic = False
        if("target_material" in self.modalities or "source_material" in self.modalities):
            self.use_material = True
        else: self.use_material = False


        self.hop_length = config["hop_length"]
        self.window_size = config["win_length"]
        self.nfft = config["nfft"]
        if config["sampling_rate"]!=48000:       
            self.resampling_kernel = torchaudio.transforms.Resample(orig_freq=48000, new_freq=config["sampling_rate"], resampling_method="sinc_interp_kaiser", rolloff = 0.9475937167399596, lowpass_filter_width=64,beta=14.769656459379492)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
        self.material_mapper = material_mapper
        self.material_mapping_dict = self.material_mapper.material_map_dict
        self.num_materials = self.config['num_materials']
        # self.recovery_path = self.config["recovered_ir_dir"]
        

    @property
    def dataset(self):
        return self.data
    
      
    def __len__(self):
        return len(self.data)
    
    def k_thresholding(self, source_mat, target_mat, k=0.20):
        average_difference = np.mean(np.where(np.abs(source_mat - target_mat) >0,1,0))
        return average_difference
    
    def audio_l2_diffs(self,source_audio, target_audio):
        l2_difference = torch.sqrt(torch.sum((source_audio - target_audio) ** 2))
        return l2_difference
    
    def fetch_audios(self,dir, is_source=False):
       
        waveform,sr = torchaudio.load(dir)
        return waveform
    
    def add_gaussian_noise(self, waveform, snr):
       
        signal_power = torch.mean(waveform ** 2, dim=-1, keepdim=True)
        noise_power = signal_power / snr
        noise = torch.randn_like(waveform) * torch.sqrt(noise_power)
        return waveform + noise

    def fetch_audios_noise(self,dir):
        waveform,sr = torchaudio.load(dir)
        if (self.config["sampling_rate"] != 48000):
            waveform = self.resampling_kernel(waveform)
        
        
        waveform = self.add_gaussian_noise(waveform, snr=self.config['snr'])
            
            
        waveform_length = waveform.shape[1]
        
        
        target_shape = (self.config["image_size"],self.config["image_size"])
        waveform_length = (target_shape[1] -1) * self.config["hop_length"]
        

        waveform = normalize_audio(waveform, norm='peak')
        if waveform_length >= waveform.shape[1]:
            new_waveform = torch.nn.functional.pad(waveform, (0, max(0, waveform_length - waveform.shape[1]+1)), 'constant', 0)
        else: 
            new_waveform = waveform[:, :waveform_length+1]
            

        return new_waveform
    
    
    def __getitem__(self,idx):
        
        sample = {}
       
        source_ir, target_ir = self.data[idx]
        scene, source_mat_index, location_idx = filename_parser(source_ir)
        _, target_mat_index, _ = filename_parser(target_ir)
        source_material_dir = os.path.join(self.data_path, scene, "material_config", source_mat_index, "material", f"{scene}_{source_mat_index}_material_{location_idx}.npz")
        source_material = read_npz(source_material_dir)
        target_material_dir = os.path.join(self.data_path, scene, "material_config", str(target_mat_index), "material", f"{scene}_{str(target_mat_index)}_material_{location_idx}.npz")
        target_material = read_npz(target_material_dir)
      
        target_ir_dir = os.path.join(self.data_path, scene, "material_config", str(target_mat_index), "waveform", target_ir)
        source_ir_dir = os.path.join(self.data_path, scene, "material_config", source_mat_index, "waveform", source_ir)
        sample['target_audio'] = self.fetch_audios(target_ir_dir)
        sample['source_audio'] = self.fetch_audios(source_ir_dir, is_source=True)
            
        if self.use_material:     
            sample['source_material'] = to_tensor(normalize_images(np.expand_dims(cv2.resize(source_material, dsize=(256, 256), interpolation=cv2.INTER_NEAREST), axis=0), max_val = self.num_materials))
            sample['target_material'] = to_tensor(normalize_images(np.expand_dims(cv2.resize(target_material, dsize=(256, 256), interpolation=cv2.INTER_NEAREST), axis=0), max_val = self.num_materials))
        if self.use_semantic:
            if "pretrained_semantic" in self.modalities:
                semantic_dir = os.path.join(self.config["pretrained_semantic_data_path"], scene, f"{scene}_category_{location_idx}_0.npz")
                sem_data = read_npz(semantic_dir)
                sample['semantic']= to_tensor(normalize_images(np.expand_dims(cv2.resize(sem_data, dsize=(256, 256), interpolation=cv2.INTER_NEAREST), axis=0), max_val=149))
                source_mat_inferred = self.material_mapper.map_material_to_semantic_image(material_index=source_mat_index, semantic_data=sem_data)
                sample['source_material'] = to_tensor(normalize_images(np.expand_dims(cv2.resize(source_mat_inferred, dsize=(256, 256), interpolation=cv2.INTER_NEAREST), axis=0), max_val = self.num_materials))
                target_mat_inferred = self.material_mapper.map_material_to_semantic_image(material_index=target_mat_index, semantic_data=sem_data)
                sample['target_material'] =  to_tensor(normalize_images(np.expand_dims(cv2.resize(target_mat_inferred, dsize=(256, 256), interpolation=cv2.INTER_NEAREST), axis=0), max_val = self.num_materials))
            
            else:
                semantic_dir = os.path.join(self.data_path, scene, "category", f"{scene}_category_{location_idx}_0.npz")
                sample['semantic']= to_tensor(normalize_images(np.expand_dims(cv2.resize(read_npz(semantic_dir), dsize=(256, 256), interpolation=cv2.INTER_NEAREST), axis=0), max_val=40))
        if self.use_rgb:
            rgb_dir = os.path.join(self.data_path, scene, "rgb", f"{scene}_rgb_{location_idx}_0.png")
            sample['rgb']  = to_tensor(normalize_images(np.transpose(cv2.resize(cv2.cvtColor(cv2.imread(rgb_dir), cv2.COLOR_BGR2RGB), dsize=(256, 256), interpolation=cv2.INTER_AREA)[..., :3], (2, 0, 1)), max_val=255)) 
        return sample


