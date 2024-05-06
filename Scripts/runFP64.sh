#!/bin/bash

#SBATCH --job-name=MM_DPFP64
#SBATCH --output=MM_DPFP64.txt
#SBATCH --time=1:00:00

for i in 5000 10000 15000 20000 25000;
do
    ./MMFP64 $i
done