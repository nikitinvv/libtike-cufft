#!/bin/bash
#SBATCH -t 4-00:00:00
#SBATCH -J ptychotomo_rec
#SBATCH -p v100
#SBATCH --exclude gn1
##SBATCH --nodelist gn2
#SBATCH --mem 165G

#SBATCH -o rec.out
#SBATCH -e rec.err

module add GCC/8.2.0-2.31.1 CUDA/10.1.105

source ~/.bashrc-2019-04-27
source activate ptychotomo

python -u test.py 0 >rec_output &

wait
