import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms
import wandb
import numpy as np
import matplotlib.pyplot as plt
import os
from models.unet import *
import random
from eval_torch import evaluate
from utils import *
from tqdm import tqdm
from torchsummary import summary
from loss import compute_spect_losses
import gc




def get_trainer(model, optimizer, criterion, config, val_loader, train_loader, eval_save_dir, device, scheduler, material_distribution_net, rank=None, world_size=None):   
    trainer = Trainer(model=model, rank=rank, world_size = world_size,  optimizer=optimizer, criterion=criterion, config=config, val_loader=val_loader, train_loader=train_loader, seed=config["seed"], device=device, qual_results_dir= eval_save_dir, scheduler=scheduler, material_distribution_net=material_distribution_net)
    return trainer
        
class Trainer():
    def __init__(self, world_size, train_loader, val_loader, criterion, optimizer, config, seed, device, qual_results_dir, scheduler, material_distribution_net, model, rank):
        
        self.model = model
        self.train_loader = train_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.seed = seed
        self.config = config
        self.val_loaders = val_loader
        self.device = device
        self.save_dir = qual_results_dir
        self.scheduler = scheduler
        self.material_distribution_net = material_distribution_net
        self.audio_transforms = make_audio_transforms(self.config)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        self.rank = rank
        self.world_size = world_size
        
                
    
    def log_metrics(self, mode, iteration_counter, running_counter, total_iteration_train_loss=None,metrics_to_log=None, lr=None, individual_losses=None):

        if mode == "val":
            wandb.log(metrics_to_log)
                 
        

        if mode == "train":
            log_dict = {"iteration": iteration_counter + 1}
            average_train_loss_iterations = total_iteration_train_loss / running_counter
            log_dict["train_loss"] = average_train_loss_iterations
            if individual_losses is not None:
                for k, v in individual_losses.items():
                    log_dict[f"train_{k}"] = v / running_counter
            if lr is not None:
                log_dict["learning_rate"] = lr
            wandb.log(log_dict)

            
                     
    def train(self, iteration_counter, epoch):
        iteration_loss = 0.0
        total_train_loss=0
        self.model.train()
        running_iteration_counter =1
        current_lr = self.config["lr"]
        individual_loss_accumulator = {} 
        iteration_loss_accumulator = {}
        for loss_type in self.config['losses']:
            individual_loss_accumulator[loss_type] = 0.0
            iteration_loss_accumulator[loss_type] = 0.0
        
        for sample in tqdm(self.train_loader):

            for key in sample:   
                sample[key] = sample[key].to(self.device)
                
           
            sample["original_source_mag"] = convert_wav_to_stft_torch(sample['source_audio'],config=self.config,use_phase = False)
            
            sample['source_mag']= apply_audio_transforms(sample["original_source_mag"], maskings=self.audio_transforms, train=True)
            sample['target_mag'] = convert_wav_to_stft_torch(sample['target_audio'], config=self.config,use_phase = False)
            outputs,encoded_source, levels = self.model(sample, training_mode=True, epoch=epoch)
            
            
            if(self.config["target_type"]=="difference"):
                if(self.config["output_method"]=="additive"):
                    outputs = additive_output(source=sample['original_source_mag'], predicted_mask=outputs)
                   
                elif(self.config["output_method"]=="multiplicative"):
                    if self.config["use_bias_term"]:
                        outputs = multiplicative_output_with_bias(source=sample['original_source_mag'], predicted_mask=outputs)
                    else:
                   
                        outputs = multiplicative_output(source=sample['original_source_mag'], predicted_mask=outputs)
                else:
                    raise RuntimeError("Incorrect output method")
          
            self.optimizer.zero_grad()
            if(self.criterion is not None):
                loss = self.criterion(outputs, sample["target_mag"])
                
            elif(self.config["distance_metric"] == "multiple_losses"):
               
                
                loss, individual_losses_dict = compute_spect_losses(
                            loss_types=self.config["losses"],
                            loss_weights=self.config["loss_weights"],
                            gt_spect=sample["target_mag"],
                            pred_spect=outputs,
                            mask=None,
                            config = self.config,
                            logspace=False,
                            log1p_gt=False,
                            slice_till_direct_signal = self.config["slice_till_direct_signal"],
                            log_instead_of_log1p_in_logspace=False,
                            log_gt=False,
                            )
                for k, v in individual_losses_dict.items():
                    individual_loss_accumulator[k] += v
                    iteration_loss_accumulator[k]+=v
            else:
                raise RuntimeError("incorrect loss type")
            
            loss.backward()
            self.optimizer.step()  

            if self.scheduler is not None:
                self.scheduler.step()
                current_lr = self.scheduler.get_last_lr()[0]
           
            iteration_loss += loss.item()
            total_train_loss += loss.item()

            del loss
            del outputs
            del sample
            gc.collect()
            
            
            if (iteration_counter+1) % self.config["logging_n_iterations"] == 0:  
                self.log_metrics(mode="train", iteration_counter=iteration_counter,
                                 running_counter=running_iteration_counter,total_iteration_train_loss=iteration_loss, lr=current_lr, individual_losses = iteration_loss_accumulator)
                iteration_loss = 0.0
                running_iteration_counter =1
                for k in iteration_loss_accumulator.keys():
                    iteration_loss_accumulator[k]=0
            iteration_counter += 1
            running_iteration_counter +=1
        
        

        average_train_loss = total_train_loss / len(self.train_loader)
        return average_train_loss, iteration_counter
    

    def validate(self, epoch):   
        self.model.eval()
        eval_splits_to_save = ["uscene_smaterial_loader", "uscene_umaterial_loader", "unseen_pairs_loader"]
        total_val_loss_across_loaders = 0.0
        total_val_metrics_across_loaders = {metric: 0 for metric in self.config["evaluation_metrics"]}
        total_samples_across_loaders = 0
        with torch.no_grad():
            counter = 0
            
            for val_loader_name, val_loader in self.val_loaders.items():
                print(val_loader_name, flush=True)
                total_val_loss = 0.0
                total_eval_metrics = {}
                num_samples = 0
                for metric in self.config["evaluation_metrics"]:
                    total_eval_metrics[metric] = 0
                for sample in tqdm(val_loader):      
                             
                    for key in sample:
                        sample[key] = sample[key].to(self.device)

                    
                    counter +=1
                   
                    sample['source_mag'], sample['source_phase'] = convert_wav_to_stft_torch(sample['source_audio'],config=self.config)
                    
                    sample['target_mag'], sample['target_phase'] = convert_wav_to_stft_torch(sample['target_audio'],config=self.config)
                    outputs, encoded_source, levels = self.model(sample, training_mode=False)
                    
                    if(self.config["target_type"]=="difference"):
                        if(self.config["output_method"]=="additive"):
                            outputs = additive_output(source=sample['source_mag'], predicted_mask=outputs)
                        elif(self.config["output_method"]=="multiplicative"):
                            if self.config["use_bias_term"]:
                                outputs = multiplicative_output_with_bias(source=sample['source_mag'], predicted_mask=outputs)
                            else:
                                outputs = multiplicative_output(source=sample['source_mag'], predicted_mask=outputs)
                    
                    if self.criterion is not None:
                        val_loss = self.criterion(outputs, sample["target_mag"]).item()
                    elif(self.config["distance_metric"] == "multiple_losses"):
                       
                        val_loss, _   = compute_spect_losses(
                                    loss_types=self.config["losses"],
                                    loss_weights=self.config["loss_weights"],
                                    gt_spect=sample["target_mag"],
                                    pred_spect=outputs,
                                   
                                    mask=None,
                                    config = self.config,
                                    logspace=False,
                                    log1p_gt=False,
                                    log_instead_of_log1p_in_logspace=False,
                                    slice_till_direct_signal = self.config["slice_till_direct_signal"],
                                    log_gt=False,
                                    
                                    )
                    else:
                        raise RuntimeError("incorrect loss type")

                    evaluation_dict = evaluate(config=self.config, predicted_mag_ir=outputs, target_mag_ir=sample["target_mag"], 
                                            target_phase_ir=sample["target_phase"], source_mag_ir=sample["source_mag"], source_phase_ir=sample["source_phase"], 
                                            save_dir=self.save_dir, dump_audios=False)
                    
                    total_val_loss += val_loss.item() * sample["source_mag"].size(0)
                    for metric in evaluation_dict:
                        total_eval_metrics[metric]+=(evaluation_dict[metric]*sample["source_mag"].size(0))

                    
                    num_samples += sample["source_mag"].size(0)

                average_val_loss = total_val_loss / num_samples
                average_eval_metrics ={}
                for metric, values in total_eval_metrics.items():
                    average_eval_metrics[metric] = values/num_samples

                metrics_to_log = {"epoch":epoch+1,f"{val_loader_name}/val_loss": average_val_loss}
                metrics_to_log.update({f"{val_loader_name}/{metric}": value for metric, value in average_eval_metrics.items()})
                wandb.log(metrics_to_log)
                if val_loader_name in eval_splits_to_save:
                    total_val_loss_across_loaders += total_val_loss
                    total_samples_across_loaders += num_samples
                    for metric in total_eval_metrics:
                        total_val_metrics_across_loaders[metric] += total_eval_metrics[metric]
                    
                del total_eval_metrics
                del outputs
                del val_loss
                del sample
                del average_val_loss
                del metrics_to_log

            average_baseline_loss_all = total_val_loss_across_loaders / total_samples_across_loaders
            average_eval_baseline_metrics_all = {metric: total / total_samples_across_loaders for metric, total in total_val_metrics_across_loaders.items()}

            overall_metrics_to_log = {
                "overall/val_loss": average_baseline_loss_all
            }
            overall_metrics_to_log.update({f"overall/{metric}": value for metric, value in average_eval_baseline_metrics_all.items()})
    
            
        
        return average_baseline_loss_all, overall_metrics_to_log


    def save_checkpoint(self,checkpoint_path, epoch, best_loss, best_metrics):
        checkpoint_path = os.path.join(checkpoint_path, f'{epoch+1}.pt')
        torch.save({
            'model_state_dict':self.model.state_dict(), 
            'loss': best_loss, 
            'metrics': best_metrics,
            'optimizer_state_dict': self.optimizer.state_dict(), 
            }, checkpoint_path)


    