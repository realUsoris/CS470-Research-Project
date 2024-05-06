#!/bin/bash

#SBATCH --job-name=TCMMFP16
#SBATCH --output=TCMMFP16_DP.txt
#SBATCH --time=1:00:00

srun --gres=gpu ./TCMM