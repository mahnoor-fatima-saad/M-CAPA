import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms
import wandb
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import utils
from models.unet import *
import argparse
import time
import random
from eval_torch import evaluate
import itertools 
from trainer_v3 import *
#from trainer_orientations import *
from baseline import BaselineTrainer
from retrieval_baseline_2 import RetrievalBaseline
from torchsummary import summary
from evaluator import *
import models.distribution_net as distribution_net
import distribution_trainer
import psutil
from val_splits_utils import *
from data_3 import *
from models.material_change_detector import *
import material_difference_trainer
import dry_runner


def main_worker(device, config, dataset, validation_sets, material_mapper):
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    np.random.seed(config["seed"])
    random.seed(config["seed"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if config["distance_metric"] == "Huber":
        criterion = nn.HuberLoss().to(device)
    elif config["distance_metric"] == "MAE":
        criterion = nn.L1Loss().to(device)
    elif config["distance_metric"] == "MSE":
        criterion = nn.MSELoss().to(device)
    else:
        criterion = None
        print("loss to be used", config["distance_metric"])

    print("Creating Dataloaders")
    
    
    train_loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True,num_workers = config["num_workers"],drop_last = True, persistent_workers=True)
    val_loaders = make_validation_loaders(validation_sets, config=config)

   
    if config["create_baseline"]:
        if config["baseline_type"] == "copy_baseline":
            trainer = BaselineTrainer(criterion=criterion, config=config, seed=config["seed"], device=device, material_mapper_instance=material_mapper, qual_results_dir=None)
            trainer.create_naive_baseline(loss=criterion)
        elif config["baseline_type"] == "material_agnostic_baseline" or config["baseline_type"] == "material_aware_baseline" or config["baseline_type"] == "av-rir":
            print("Running baseline: ", flush=True)
            trainer = RetrievalBaseline(train_set = dataset, criterion=criterion, config=config, seed=config["seed"], device=device, qual_results_dir=None, material_mapper=material_mapper)
            trainer.create_baseline(loss=criterion)
        else: 
            raise ValueError ("Incorrect baseline type, check config file")
    
    else:
        save_dir = os.path.join(config["save_dir"], config["task"])
        if not (os.path.exists(save_dir)):
            os.mkdir(save_dir)
       
        output_dir = os.path.join(save_dir,config["run_name"])
        if not(os.path.exists(output_dir)):
            os.mkdir(output_dir)
        else:
            print("Cannot create directory with the same name as a previously made directory.")
            return None
        checkpoint_dir = os.path.join(output_dir, "checkpoints")
        if not(os.path.exists(checkpoint_dir)):
            os.mkdir(checkpoint_dir)
        eval_save_dir = os.path.join(output_dir, "evaluation_results_train")
        config['training_output_folder'] = eval_save_dir
        if not(os.path.exists(eval_save_dir)):
            os.mkdir(eval_save_dir)
        else:
            print("Creating Model")
            if config['primary_encoder'] is not None:
                model = get_model(device=device, config=config, primary_encoder=config['primary_encoder'])
            else:
                model = get_model(device=device, config=config)
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

                print(f"Total parameters: {total_params}")
                print(f"Trainable parameters: {trainable_params}")
            if device is not None: 
                print(device)
                print("CUDA version:", torch.version.cuda)
                model = model.to(device)
 
                
            if config['optimizer'] == 'adam':
                optimizer = optim.Adam(model.parameters(), lr=config["lr"])
            elif config['optimizer'] == 'adamw':
                optimizer = optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config['weight_decay'])
            else: 
                raise ValueError ("Incorrect optimizer type, check config file")
            
            scheduler=None




        wandb.init(project="Target_IR_Prediction", name=config["run_name"], config=config)
        
        
        trainer = get_trainer(model=model, optimizer=optimizer, criterion=criterion, config=config, val_loader=val_loaders, train_loader=train_loader, eval_save_dir=eval_save_dir, device=device, scheduler=scheduler, material_distribution_net=material_distribution_model)
        config['cp_dir'] = checkpoint_dir
        config_filename = os.path.join(output_dir,"config_training.json")
        with open(config_filename, "w") as f:
            json.dump(config, f, indent=2)
        best_loss = float('inf') 
        iteration_counter = 0
    




        for epoch in range(config["epochs"]):
            print("Epoch", epoch+1)
            
            train_loss, iteration_counter = trainer.train(iteration_counter, epoch)
            
            print("Validation", flush=True)
            val_loss, metrics = trainer.validate(epoch)
            
                       
            
            if(val_loss<best_loss) :
                best_loss = val_loss
                trainer.save_checkpoint(checkpoint_dir, epoch, best_loss, metrics)
            
            

            metrics_to_log = {"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss}
            metrics_to_log.update(metrics) 
            wandb.log(metrics_to_log)
            del metrics_to_log
            del train_loss
            del val_loss
            del metrics          
                
        wandb.finish()
        





            
