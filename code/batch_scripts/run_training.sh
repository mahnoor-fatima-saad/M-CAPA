#!/bin/bash
#SBATCH --account=
#SBATCH --partition=
#SBATCH --time=6-23:00:00
#SBATCH -o slurm-%j.out-%N.rgbsemantic # name of the stdout, using the job number (%j) and the first node (%N)
#SBATCH -e slurm-%j.err-%N.rgbsemantic # name of the stderr, using job and first node values
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50000 
#SBATCH --gres=gpu:1

    




NUM_GPU=1
export WORKDIR="{parent_directory}" 

source ~/miniconda3/etc/profile.d/conda.sh

conda activate MASynth
echo "Environment Activated"

python $WORKDIR/main_2.py --config "$WORKDIR/configs/config_training.json" 
