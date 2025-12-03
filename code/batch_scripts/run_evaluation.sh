#!/bin/bash
#SBATCH --account=
#SBATCH --partition=
#SBATCH --time=12:00:00
#SBATCH -o slurm-%j.out-%N.eval # name of the stdout, using the job number (%j) and the first node (%N)
#SBATCH -e slurm-%j.err-%N.eval # name of the stderr, using job and first node values
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



python $WORKDIR/main_2.py --run_type "test" --test_path "{model_path}" --test_name "MCAPA-AudioVisual-release" --checkpoint best_checkpoint


python $WORKDIR/main_2.py --run_type "test" --test_path "{model_path}" --test_name "MCAPA-Visual-release" --checkpoint best_checkpoint

python $WORKDIR/main_2.py --run_type "test" --test_path "{model_path}" --test_name "MCAPA-Audio-release" --checkpoint best_checkpoint

