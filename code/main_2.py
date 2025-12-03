import torch
import numpy as np
import json
from utils import *
from evaluator import Evaluator
from main_worker_2 import main_worker
import argparse
from material_mapper import *
from monitor_memory import *
import pandas as pd
from val_splits_utils import *
from data_3 import *

parser = argparse.ArgumentParser()
parser.add_argument("--run_type", default="train")
parser.add_argument("--test_path", default="",help="path to model parent dir")
parser.add_argument("--config", help="configuration for run")
parser.add_argument("--local_rank", type=int, default=0, help="Local rank. Necessary for using the torch.distributed.launch utility.")
parser.add_argument("--checkpoint", default=None)
parser.add_argument("--test_name", default="mcapa")
parser.add_argument("--noise", default=None)
parser.add_argument("--per_sample_eval_mode", action='store_true', help="Set evaluation mode to per sample.")
parser.add_argument("--all", action='store_true', help="set true if you want all models evaluated")

args = parser.parse_args()
device = get_gpu()
if args.run_type == "collect":
  collect_test_results("/uufs/chpc.utah.edu/common/home/alhalah-group1/users/mahnoor/UNet/run_outputs/target-material-conditioning-short")
elif args.run_type == "test":
  if args.all:
    ready_to_evaluate_models = fetch_all_ready_models("/uufs/chpc.utah.edu/common/home/alhalah-group1/users/mahnoor/UNet/run_outputs/target-material-conditioning-short")
    for model in ready_to_evaluate_models:
      print("testing: ", model, flush=True)
      args.test_path = model
      args.test_name = f"{os.path.basename(model)}_test"
      evaluator = Evaluator(device=device, testing_path=args.test_path, ckpt=args.checkpoint, test_name=args.test_name, noise=args.noise, per_sample_eval_mode=args.per_sample_eval_mode)
      
  else:
    print("Testing path: ", args.test_path, flush=True)
    evaluator = Evaluator(device=device, testing_path=args.test_path, ckpt=args.checkpoint, test_name=args.test_name, noise=args.noise, per_sample_eval_mode=args.per_sample_eval_mode)
    evaluator.evaluate()

else: 
  config_path = args.config
  with open(config_path, 'r') as f:
    config = json.load(f)
    
  print("Staring Experiment", flush=True)
  print("Training scenes: ", len(config['scenes']), flush=True)
  print("Validation scenes: ", len(config['eval_scenes']), flush=True)
  print("Test scenes: ", len(config['test_scenes']), flush=True)



  seen_configs, unseen_configs = read_seen_unseen_configs_json(json_file_path="/uufs/chpc.utah.edu/common/home/alhalah-group1/users/mahnoor/UNet/code/mat_unet/version_4/dataset/caches/seen_unseen_configs.json")
  
  material_mapper_instance = MaterialMapper(configs_dir = config["material_combos_dir"], config=config)
  material_mapper_instance.create_mapping_dict_for_all_configs()
  
  print("Seen Configs: ", len(seen_configs), flush=True)
  print("Unseen Configs: ", len(unseen_configs), flush=True)

  print("\nValidation Splits:", flush=True)

  sscene_smaterial = read_val_splits_json(json_file_path="/uufs/chpc.utah.edu/common/home/alhalah-group1/users/mahnoor/UNet/code/mat_unet/version_4/dataset/caches/sscene_smaterial.json")
  sscene_umaterial = read_val_splits_json(json_file_path="/uufs/chpc.utah.edu/common/home/alhalah-group1/users/mahnoor/UNet/code/mat_unet/version_4/dataset/caches/sscene_umaterial.json")
  uscene_umaterial = read_val_splits_json(json_file_path="/uufs/chpc.utah.edu/common/home/alhalah-group1/users/mahnoor/UNet/code/mat_unet/version_4/dataset/caches/uscene_umaterial.json")
  uscene_smaterial = read_val_splits_json(json_file_path="/uufs/chpc.utah.edu/common/home/alhalah-group1/users/mahnoor/UNet/code/mat_unet/version_4/dataset/caches/uscene_smaterial.json")
  unseen_pairs_indices, unseen_pairs_data = read_val_splits_json(json_file_path="/uufs/chpc.utah.edu/common/home/alhalah-group1/users/mahnoor/UNet/code/mat_unet/version_4/dataset/caches/unseen_pairs.json", split="unseen_pairs")
  
  val_datasets = dict()
  val_datasets['sscene_smaterial'] = CustomPairDataset(sscene_smaterial, config=config, material_mapper=material_mapper_instance)
  val_datasets['sscene_umaterial'] = CustomPairDataset(sscene_umaterial, config=config, material_mapper=material_mapper_instance)
  val_datasets['uscene_umaterial'] = CustomPairDataset(uscene_umaterial, config=config, material_mapper=material_mapper_instance)
  val_datasets['uscene_smaterial'] = CustomPairDataset(uscene_smaterial, config=config, material_mapper=material_mapper_instance)
  val_datasets['unseen_pairs'] = CustomPairDataset(unseen_pairs_data, config=config, material_mapper=material_mapper_instance)
    
  print("sscene_smaterial size: ", len(val_datasets['sscene_smaterial']))
  print("sscene_umaterial size: ", len(val_datasets['sscene_umaterial']))
  print("uscene_umaterial size: ", len(val_datasets['uscene_umaterial']))
  print("uscene_smaterial size: ", len(val_datasets['uscene_smaterial']))
  print("unseen pairs size: ", len(val_datasets['unseen_pairs']))

  print("Training Split:")
  train_set_df = pd.read_json('/uufs/chpc.utah.edu/common/home/alhalah-group1/users/mahnoor/UNet/code/mat_unet/version_4/dataset/caches/train_df_2.json')
  train_dataset = CustomDataset(config, dataset=train_set_df, material_mapper=material_mapper_instance, unseen_pairs=unseen_pairs_indices, train=True)

  print("training split size: ", len(train_dataset))


  main_worker(config=config, dataset=train_dataset, validation_sets=val_datasets, device=device, material_mapper = material_mapper_instance)

