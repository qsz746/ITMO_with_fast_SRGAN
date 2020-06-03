#!/bin/bash
#SBATCH --gres=gpu:1  # Request GPU "generic resources"
#SBATCH --cpus-per-task=10 # Cores proportional to GPUs: 6 on Cedar, 10 on Béluga, 16 on Graham.
#SBATCH --mem=47000M       # Memory proportional to GPUs: 32000 Cedar, 47000 Béluga, 64000 Graham.
#SBATCH --time=0-01:00:00     # DD-HH:MM:SS
#SBATCH --account=def-panos
#SBATCH --job-name=SRGAN

module load nixpkgs/16.09 gcc/7.3.0 opencv/4.2.0 python/3.7 cuda cudnn arch/avx512 StdEnv/2018.3

echo "Job start at $(date)"

nvidia-smi

SOURCEDIR=$(pwd)
echo "source directory: $SOURCEDIR"

# Prepare virtualenv
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index -r $SOURCEDIR/requirements.txt

# Prepare data
mkdir $SLURM_TMPDIR/data
# cp -r $SOURCEDIR/data/Set5_x2.h5 $SLURM_TMPDIR/data/
tar xf ~/projects/def-panos/shared_itmo/bridge.tar -C $SLURM_TMPDIR/data


python main.py \
	--image_dir $SLURM_TMPDIR/data/bridge/HDR \
	--hr_size 384 \
	--lr 1e-4 \
	--save_iter 200 \
	--epochs 10 \
	--batch_size 8

echo "Job finish at $(date)"
