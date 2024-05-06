#!/bin/bash

#SBATCH --job-name=MM_DPFP16
#SBATCH --output=MM_DPFP16.txt
#SBATCH --time=1:00:00

for i in 5000 10000 15000 20000 25000;
do
    ./MMFP16 $i
done