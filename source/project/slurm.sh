#!/bin/bash
#SBATCH --job-name=marttave-VRRT   # create a short name for your job
#SBATCH --nodes=1                           # node count
#SBATCH --ntasks=1                          # total number of tasks across all nodes
#SBATCH --cpus-per-task=16                   # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=2G                    # memory per cpu-core (4G per cpu-core is default)
#SBATCH --time=02:00:00                     # total run time limit (HH:MM:SS)
#SBATCH --gres=gpu:1                        # number of gpus per node
#SBATCH --output=output/output_headless.txt # Standard output file
#SBATCH --error=output/error_headless.txt   # Standard error file

# You can change this line to target either ChaCha or Calypso
#SBATCH --partition=Disco

uv run main.py
