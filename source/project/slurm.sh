#!/bin/bash
#SBATCH --job-name=VRRT   # create a short name for your job
#SBATCH --nodes=1                           # node count
#SBATCH --ntasks=1                          # total number of tasks across all nodes
#SBATCH --cpus-per-task=8                   # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=2G                    # memory per cpu-core (4G per cpu-core is default)
#SBATCH --time=10:00:00                     # total run time limit (HH:MM:SS)
#SBATCH --gres=gpu:1                        # number of gpus per node
#SBATCH --gres=shard:10
#SBATCH --output=output/output_%j.txt # Standard output file
#SBATCH --error=output/error_%j.txt   # Standard error file

#SBATCH --partition=Dance
#SBATCH --nodelist=disco
uv run main.py
