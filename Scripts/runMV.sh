#!/bin/bash

#SBATCH --job-name=MV_DP
#SBATCH --output=MV_DP.txt
#SBATCH --time=1:00:00

for i in 10000 20000 30000 40000;
do
    ./MV $i
done