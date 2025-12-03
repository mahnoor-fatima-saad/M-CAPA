import wandb
from models.unet import *
from eval_torch import evaluate
from utils import *
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import time
from loss import *
from data_3 import * 
#from data_orientations_3 import *
from val_splits_utils import * 
from inference import *
import csv
from calflops import calculate_flops

class Evaluator():
    def __init__(self, testing_path, device, ckpt, test_name, noise=None, per_sample_eval_mode = False):
        self.testing_path = testing_path
        self.checkpoint = ckpt
        self.test_name = test_name
        self.train_config_path =  os.path.join(self.testing_path, 'config.json')
        with open(self.train_config_path, 'r') as f:
            self.config = json.load(f)
            print("modalities in config", self.config["modalities"])
        self.config['run_type'] = "test"
        self.config['run_name'] = self.test_name
        self.config['per_sample_eval_mode'] = False
        if per_sample_eval_mode:
            self.config['per_sample_eval_mode'] = True
        if noise is not None: 
            self.config['ambient_noise'] = True
            self.config['snr'] = int(noise)
        self.test_results_path = os.path.join(self.testing_path, "test_results", self.test_name)
        os.makedirs(self.test_results_path, exist_ok=True)

        self.checkpoint_dir = os.path.join(testing_path, 'checkpoints')
        
        
        #self.device = 'cpu'
        self.device = device
        self.material_mapper_instance = MaterialMapper(configs_dir = self.config["material_combos_dir"], config=self.config)
        self.material_mapper_instance.create_mapping_dict_for_all_configs()
           


        set_seed(self.config["seed"])
        self.sscene_smaterial = read_val_splits_json(json_file_path="/uufs/chpc.utah.edu/common/home/alhalah-group1/users/mahnoor/UNet/code/mat_unet/version_4/dataset/test_caches/sscene_smaterial.json")
        self.sscene_umaterial = read_val_splits_json(json_file_path="/uufs/chpc.utah.edu/common/home/alhalah-group1/users/mahnoor/UNet/code/mat_unet/version_4/dataset/test_caches/sscene_umaterial.json")
        self.uscene_smaterial = read_val_splits_json(json_file_path="/uufs/chpc.utah.edu/common/home/alhalah-group1/users/mahnoor/UNet/code/mat_unet/version_4/dataset/test_caches/uscene_smaterial.json")
        self.uscene_umaterial = read_val_splits_json(json_file_path="/uufs/chpc.utah.edu/common/home/alhalah-group1/users/mahnoor/UNet/code/mat_unet/version_4/dataset/test_caches/uscene_umaterial.json")
        unseen_pairs_indices, self.unseen_pairs = read_val_splits_json(json_file_path="/uufs/chpc.utah.edu/common/home/alhalah-group1/users/mahnoor/UNet/code/mat_unet/version_4/dataset/test_caches/unseen_pairs.json", split="unseen_pairs")
        
       
        self.test_datasets = dict()
        self.test_datasets['sscene_smaterial'] = CustomPairDataset(self.sscene_smaterial, config=self.config, material_mapper=self.material_mapper_instance)
        self.test_datasets['sscene_umaterial'] = CustomPairDataset(self.sscene_umaterial, config=self.config, material_mapper=self.material_mapper_instance)
        self.test_datasets['uscene_umaterial'] = CustomPairDataset(self.uscene_umaterial, config=self.config, material_mapper=self.material_mapper_instance)
        self.test_datasets['uscene_smaterial'] = CustomPairDataset(self.uscene_smaterial, config=self.config, material_mapper=self.material_mapper_instance)
        self.test_datasets['unseen_pairs'] = CustomPairDataset(self.unseen_pairs, config=self.config, material_mapper=self.material_mapper_instance)

       

            
        self.test_loaders = make_validation_loaders(self.test_datasets, config=self.config)        
                
        
        if self.config["distance_metric"] == "Huber":
            self.criterion = nn.HuberLoss().to(device)
        elif self.config["distance_metric"] == "MAE":
            self.criterion = nn.L1Loss().to(device)
        elif self.config["distance_metric"] == "MSE":
            self.criterion = nn.MSELoss().to(device)
        else:
            self.criterion = None
        print("loss to be used", self.config["distance_metric"])
        self.material_distribution_net = None
        if(self.checkpoint is None):
            
            self.best_checkpoint = self.find_best_checkpoint(self.checkpoint_dir)
            print(self.best_checkpoint, flush=True)

        elif (self.checkpoint is not None):
            
            self.best_checkpoint = os.path.join(self.checkpoint_dir, f'{self.checkpoint}.pt')
            print(self.best_checkpoint, flush=True)

        else:
            raise RuntimeError("incorrect checkpoint dir, check config file.")
        self.config['cp_dir'] = self.best_checkpoint

        print("primary_encoder", self.config["primary_encoder"])
        checkpoint = torch.load(self.best_checkpoint)
        self.model = get_model(device=self.device, config=self.config, primary_encoder=self.config["primary_encoder"])
        
        if self.config['optimizer'] == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.config["lr"])
        elif self.config['optimizer'] == 'adamw':
            self.optimizer = optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.config["lr"], weight_decay=self.config['weight_decay'])
        else: 
            raise ValueError ("Incorrect optimizer type, check config file")
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])


        self.model.to(self.device)
        torch.manual_seed(self.config["seed"])
        torch.cuda.manual_seed_all(self.config["seed"])
        np.random.seed(self.config["seed"])
        random.seed(self.config["seed"])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        evaluation_config_path = os.path.join(self.test_results_path, 'config_evaluation.json')
        with open(evaluation_config_path, 'w') as f:
            json.dump(self.config, f, indent=4)
    
    def flops(self):
        print("Evaluating flops", flush=True)
        wrapped_model = WrappedModel(self.model)

      
        rgb = torch.randn(1, 3, 256, 256).to(self.device)
        semantic = torch.randn(1, 1, 256, 256).to(self.device)
        target_material = torch.randn(1, 1, 256, 256).to(self.device)
        source_mag = torch.randn(1, 2, 256, 256).to(self.device)
        flops, macs, params = calculate_flops(model=wrapped_model, 
                                      kwargs={"rgb": rgb, 
                                              "semantic": semantic, 
                                              "target_material":target_material, 
                                              "source_mag":source_mag},
                                      output_as_string=True,
                                      output_precision=4)

        print("Model FLOPs:%s   MACs:%s   Params:%s \n" %(flops, macs, params), flush=True)

    def time_forward_pass(self):
        self.model.eval()  
        with torch.no_grad():
            # Create dummy inputs as before
            
            rgb = torch.randn(1, 3, 256, 256).to(self.device)
            semantic = torch.randn(1, 1, 256, 256).to(self.device)
            target_material = torch.randn(1, 1, 256, 256).to(self.device)
            source_mag = torch.randn(1, 2, 256, 256).to(self.device)
            dict_mods = { "rgb":rgb,
                         "target_material":target_material, 
                        }
            

            start = time.time()
            output = self.model(dict_mods)
            end = time.time()

        elapsed_time = end - start
        print(f"Forward pass time: {elapsed_time * 1000:.3f} ms")


    
    def evaluate(self):
        
        csv_filename = os.path.join(self.test_results_path,"overall_test_results.csv")
        
        wandb.init(project="Target_IR_Prediction", name=self.test_name, 
                   config=self.config)
        

        self.model.eval()
        
        total_val_loss_across_loaders = 0.0
        total_val_metrics_across_loaders = {metric: 0 for metric in self.config["evaluation_metrics"]}
        total_samples_across_loaders = 0
        with torch.no_grad():
            counter = 0
            for val_loader_name, val_loader in self.test_loaders.items():  
                print(val_loader_name, flush=True)
                total_val_loss = 0.0
                total_eval_metrics = {}
                num_samples = 0
                for metric in self.config["evaluation_metrics"]:
                    total_eval_metrics[metric] = 0
                for sample in tqdm(val_loader):      
                  
                    for key in sample:
                        if key =='pair':
                            continue
                        sample[key] = sample[key].to(self.device)
                        
                    
                    counter +=1
                    sample['source_mag'], sample['source_phase'] = convert_wav_to_stft_torch(sample['source_audio'],config=self.config)
                    sample['target_mag'], sample['target_phase'] = convert_wav_to_stft_torch(sample['target_audio'],config=self.config)
                
                    outputs= self.model(sample)
               
         
                    if(self.config["target_type"]=="difference"):
                        if(self.config["output_method"]=="additive"):
                            outputs = additive_output(source=sample['source_mag'], predicted_mask=outputs)
                        elif(self.config["output_method"]=="multiplicative"):
                            if self.config["use_bias_term"]:
                                outputs = multiplicative_output_with_bias(source=sample['source_mag'], predicted_mask=outputs)
                            else:
                                outputs = multiplicative_output(source=sample['source_mag'], predicted_mask=outputs)
                        else: raise ValueError("need a output method in config")


                    if self.criterion is not None:
                        val_loss = self.criterion(outputs, sample["target_mag"]).item()
                    elif(self.config["distance_metric"] == "multiple_losses"):
                        val_loss,_ = compute_spect_losses(
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
                                    log_gt=False,)
                    
                    evaluation_dict = evaluate(config=self.config, predicted_mag_ir=outputs, target_mag_ir=sample["target_mag"], 
                                            target_phase_ir=sample["target_phase"], source_mag_ir=sample["source_mag"], source_phase_ir=sample["source_phase"], 
                                            save_dir=self.test_results_path, dump_audios=False)
                    
                    total_val_loss += val_loss * sample["source_mag"].size(0)
                    
                    for metric in evaluation_dict:
                        total_eval_metrics[metric]+=(evaluation_dict[metric]*sample["source_mag"].size(0))

                    num_samples += sample["source_mag"].size(0)
                print("num_samples: ", num_samples)
                average_val_loss = total_val_loss / num_samples
                average_eval_metrics ={}
                
                for metric, values in total_eval_metrics.items():
                    average_eval_metrics[metric] = values/num_samples
    

                metrics_to_log = {"epoch":1,f"{val_loader_name}/test_loss": average_val_loss}
                metrics_to_log.update({f"{val_loader_name}/{metric}": value for metric, value in average_eval_metrics.items()})
                
                formatted_metrics = {"checkpoint": os.path.basename(self.best_checkpoint),"loader_name": val_loader_name, "test_loss": average_val_loss}
                for metric, values in average_eval_metrics.items():
                    formatted_metrics[metric] = values

                df = pd.DataFrame([formatted_metrics])
                if os.path.exists(csv_filename):
                    existing_df = pd.read_csv(csv_filename)
                    df = pd.concat([existing_df, df], ignore_index=True)
                df.to_csv(csv_filename, index=False)

                
           
                
                
                wandb.log(metrics_to_log)

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
                "overall/test_loss": average_baseline_loss_all
            }
            overall_metrics_to_log.update({f"overall/{metric}": value for metric, value in average_eval_baseline_metrics_all.items()})
            wandb.log(overall_metrics_to_log)
        wandb.finish()
        
       
        
    def save_to_csv(file_path, data, columns):
        with open(file_path, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(columns)  # Write headers
            writer.writerows(data)    # Write data rows
    
    def find_best_checkpoint(self, checkpoints_dir):
        best_loss = float('inf') 
        best_checkpoint = None
        checkpoint_files = [f for f in os.listdir(checkpoints_dir) if f.endswith('.pt')]

        for checkpoint_file in checkpoint_files:
            checkpoint_path = os.path.join(checkpoints_dir, checkpoint_file)
            
            checkpoint = torch.load(checkpoint_path)
            loss_value = checkpoint.get('loss', float('inf'))
            if loss_value < best_loss:
                best_loss = loss_value
                best_checkpoint = checkpoint_path
        return best_checkpoint
    
    

class WrappedModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, **args):
        
        return self.model(args, training_mode = False)
    