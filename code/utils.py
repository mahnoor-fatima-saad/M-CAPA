import os
import glob 
import random
from itertools import combinations
import matplotlib.pyplot as plt
import torch
import numpy as np
import concurrent.futures
import torchaudio
import torchaudio.transforms
from torch.utils.data import DataLoader, random_split, DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from scipy import signal as si  
from scipy.io import wavfile
from scipy.signal import convolve, resample, lfilter, fftconvolve, butter, filtfilt
import pandas as pd
from scipy.spatial.distance import cdist
from torch.nn.parallel import DistributedDataParallel as DDP





def normalize_audio(audio, norm='peak'):
    if norm == 'peak':
        peak = abs(audio).max()
        if peak != 0:
            return audio.div_(peak)
        else:
            return audio
    elif norm == 'rms':
        if torch.is_tensor(audio):
            audio = audio.numpy()
        audio_without_padding = np.trim_zeros(audio, trim='b')
        rms = np.sqrt(np.mean(np.square(audio_without_padding))) * 100
        if rms != 0:
            return audio / rms
        else:
            return audio
    else:
        raise NotImplementedError


    
def to_tensor(v):
    if torch.is_tensor(v):
        return v.float()
    elif isinstance(v, np.ndarray):
        return torch.from_numpy(v).float()
    else:
        return torch.tensor(v).float()


def convert_wav_to_stft_torch(waveforms, config, use_phase = True):
    magnitudes = []
    phases = []
    NFFT=config["nfft"]
    HOP=config["hop_length"]
    WIN=config["win_length"]
    for waveform in waveforms:
        stft = torch.stft(waveform, n_fft=NFFT, hop_length=HOP, win_length=WIN, window=torch.hann_window(WIN, device=waveform.device), return_complex=True)
        magnitude = torch.abs(stft)
        magnitudes.append(magnitude)
        phases.append(torch.angle(stft))

    magnitudes = torch.stack(magnitudes)
    phases = torch.stack(phases)

    if use_phase:
        return magnitudes,phases
    else:
        return magnitudes
    


def make_audio_transforms(config, max_width = 25):
    maskings = {}
    if(config["apply_masking"]):
        if(config["audio_masking_type"]=="frequency"):
            maskings["frequency"]=torchaudio.transforms.FrequencyMasking(freq_mask_param=max_width)
        elif(config["audio_masking_type"]=="time"):
            maskings["time"] = torchaudio.transforms.TimeMasking(time_mask_param=max_width)
        elif(config["audio_masking_type"]=="both"):
            maskings["frequency"]=torchaudio.transforms.FrequencyMasking(freq_mask_param=max_width)
            maskings["time"] = torchaudio.transforms.TimeMasking(time_mask_param=max_width)
        else:
            raise RuntimeError ("incorrect audio masking type. check config file")
        return maskings
    else: return None
    
def apply_audio_transforms(stfts, maskings, train=True):
    masked_stfts = []
    if maskings is None:
        return stfts
    else:
        if train:
            apply_audio_transforms_prob = random.random()
            if(apply_audio_transforms_prob<0.5):
                if len(maskings.keys())>1:
                    double_masking_prob = random.random()
                    if double_masking_prob <0.3:
                        stfts = [maskings['frequency'](stft) for stft in stfts]
                    elif double_masking_prob<0.6:
                        stfts = [maskings['time'](stft) for stft in stfts]
                    else:
                        stfts = [maskings['frequency'](stft) for stft in stfts]
                        stfts = [maskings['time'](stft) for stft in stfts]
                    
                    return torch.stack(stfts)
                elif "frequency" in maskings.keys():
                    stfts = [maskings['frequency'](stft) for stft in stfts]
                    return torch.stack(stfts)
                elif "time" in maskings.keys():
                    stfts = [maskings['time'](stft) for stft in stfts]
                    return torch.stack(stfts)
            return stfts
        
        
def k_thresholding(source_mat, target_mat, k=0.10):
    diff = np.abs(source_mat - target_mat)
    thresholded_difference = np.where(diff > 0, 1, 0)
    return thresholded_difference


def additive_output(source, predicted_mask):
    return source+predicted_mask    

def multiplicative_output(source, predicted_mask):
    normalized = torch.sigmoid(predicted_mask)
    return source*normalized

def multiplicative_output_with_bias(source, predicted_mask):
    normalized = torch.sigmoid(predicted_mask[:,:2,:,:])
    output = (source*normalized) + predicted_mask[:,2:,:,:]
    return output

def filename_parser(file_path):
    scene_name, material_index, _ ,  location_index = os.path.basename(file_path).split('_')[0:4]
    #location_index = location_index.split(".")[0]
    return scene_name, material_index,location_index.split(".")[0]


    


def get_gpu():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device, flush=True)
    print(torch.__version__, flush=True)
    print(torch.cuda.is_available(), flush=True)
    print(torch.backends.cudnn.enabled, flush=True)
    print(torch.cuda.device_count(), flush=True)
    if torch.cuda.is_available():
        print("CUDA is available", flush=True)
    else:
        print("CUDA is not available", flush=True)
    print("CUDA version:", torch.version.cuda, flush=True)
    properties = torch.cuda.get_device_properties(device)
    total_memory = properties.total_memory / (1024 ** 3) 
    print(f"Total GPU memory: {total_memory:.2f} GB")
    return device


def get_seen_unseen_configs(total_configs,seen_ratio=0.90, random_seed = 42):
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
    
    all_configs = list(range(1,total_configs+1))
    num_seen = int(total_configs * seen_ratio)
    
    seen_configs = random.sample(all_configs, num_seen)
    unseen_configs = [config for config in all_configs if config not in seen_configs]
    
    return seen_configs, unseen_configs



def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def read_npz(file):
    with np.load(file) as data:
        data_array = data['data'] 
        return data_array
       
    
def worker_init_fn(worker_id):
    seed = 42 + worker_id                                                                                                                                  
    torch.cuda.manual_seed(seed)                                                                                                                              
    torch.cuda.manual_seed_all(seed)                                                                                          
    np.random.seed(seed)                                                                                                             
    random.seed(seed)                                                                                                       
    torch.manual_seed(seed)                                                                                                                                   
    return
        

def make_validation_loaders(datasets, config):
    
    sscene_smaterial_loader = DataLoader(
        datasets['sscene_smaterial'], 
        batch_size=config["val_batch_size"], 
        shuffle=False,
        num_workers=int(config["num_workers"]),
        drop_last=False, 
        #persistent_workers=True, 
        pin_memory=False
    )
    
    sscene_umaterial_loader = DataLoader(
        datasets['sscene_umaterial'], 
        batch_size=config["val_batch_size"], 
        shuffle=False,
        num_workers=int(config["num_workers"]),
        drop_last=False, 
        #persistent_workers=True, 
        pin_memory=False
    )
    
    uscene_smaterial_loader = DataLoader(
        datasets['uscene_smaterial'], 
        batch_size=config["val_batch_size"], 
        shuffle=False,
        num_workers=int(config["num_workers"]),
        drop_last=False, 
        #persistent_workers=True, 
        pin_memory=False
    )
    
    uscene_umaterial_loader = DataLoader(
        datasets['uscene_umaterial'], 
        batch_size=config["val_batch_size"], 
        shuffle=False,
        num_workers=int(config["num_workers"]),
        drop_last=False, 
        #persistent_workers=True, 
        pin_memory=False,
    )

    unseen_pairs_loader = DataLoader(
        datasets['unseen_pairs'], 
        batch_size=config["val_batch_size"], 
        shuffle=False,
        num_workers=int(config["num_workers"]),
        drop_last=False, 
        #persistent_workers=True, 
        pin_memory=False
    )
    
    loaders = {
    
    'uscene_umaterial_loader': uscene_umaterial_loader,
    'unseen_pairs_loader':unseen_pairs_loader,  
    'uscene_smaterial_loader': uscene_smaterial_loader,
    'sscene_smaterial_loader': sscene_smaterial_loader,
    'sscene_umaterial_loader': sscene_umaterial_loader    
    }

    return loaders

def normalize_images(arr, max_val):
    
    if max_val != 0:
        return arr/max_val
    else:
        return arr 
    
def setup_ddp(rank):
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(rank)

def cleanup_ddp():
    dist.destroy_process_group()



def change_location_index_waveforms(waveform, location_index):
    
    ir_paths = waveform.split('_')
    ir_paths[-1] = f"{location_index}.wav"
    new_ir_dir = '_'.join(ir_paths)
    
    return new_ir_dir



def resampling_kernel_util(config):
    if config["sampling_rate"] != 48000:
        resampling_kernel = torchaudio.transforms.Resample(
                orig_freq=48000, new_freq=config["sampling_rate"],
                  resampling_method="sinc_interp_kaiser", 
                  rolloff = 0.9475937167399596, 
                  lowpass_filter_width=64,
                  beta=14.769656459379492)
        return resampling_kernel
    else: 
        return None



def fetch_all_ready_models(dir):
    ready_to_evaluate = []
    for root, dirs, files in os.walk(dir):
        if "config_training.json" in files:
            ready_to_evaluate.append(root)
    return ready_to_evaluate

def collect_test_results(dir):
    output_csv = os.path.join(dir, "combined_test_results.csv")
    all_results = []
    for model_name in os.listdir(dir):
        model_path = os.path.join(dir, model_name)
        test_results_dir = os.path.join(model_path, 'test_results')
        if not os.path.isdir(test_results_dir):
            continue
        for test_name in os.listdir(test_results_dir):
            test_dir = os.path.join(test_results_dir, test_name)
            overall_results_file = os.path.join(test_dir, 'overall_test_results.csv')
            if os.path.exists(overall_results_file):
                df = pd.read_csv(overall_results_file)
                df['model_name'] = model_name
                df['test_name'] = test_name
                df['diff_cte'] = (df['diff_cte']).apply(lambda x: f"{x:.2f}")
                df['diff_rt60'] = (df['diff_rt60'] * 1000).apply(lambda x: f"{x:.2f}")
                df['stft_l1_distance'] = (df['stft_l1_distance'] * 100).apply(lambda x: f"{x:.2f}")
                df['stft_l2_distance'] = (df['stft_l2_distance'] * 100).apply(lambda x: f"{x:.2f}")

                all_results.append(df)
    final_df = pd.concat(all_results, ignore_index=True)
    final_df = final_df[['model_name', 'test_name', 'checkpoint', 'loader_name', 'test_loss', 
                         'diff_rt60', 'stft_l1_distance', 'stft_l2_distance', 'diff_cte']]
    final_df.to_csv(output_csv, index=False)


    
def sync_scalar(tensor: torch.Tensor, world_size: int, op=dist.ReduceOp.SUM):
    """All‐reduce a single‐element tensor and return the result on each rank."""
    dist.all_reduce(tensor, op=op)
    return tensor / world_size