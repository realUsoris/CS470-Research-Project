#!/bin/bash

#SBATCH --job-name=MM_DPBF16
#SBATCH --output=MM_DPBF16.txt
#SBATCH --time=1:00:00

for i in 5000 10000 15000 20000 25000;
do
    ./MMBF16 $i
done
