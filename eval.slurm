#!/bin/bash
#SBATCH --gres=gpu:p100:1  # Request GPU "generic resources"
#SBATCH --cpus-per-task=6  # Cores proportional to GPUs: 6 on Cedar, 10 on Béluga, 16 on Graham.
#SBATCH --mem=32000M       # Memory proportional to GPUs: 32000 Cedar, 47000 Béluga, 64000 Graham.
#SBATCH --time=0-01:00:00     # DD-HH:MM:SS
#SBATCH --account=def-panos
#SBATCH --job-name=DRRN

module load python/3.7 cuda cudnn arch/avx512 StdEnv/2018.3

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
cp -r $SOURCEDIR/data/Set5_x2.h5 $SLURM_TMPDIR/data/
# tar xf ~/projects/def-xxxx/data.tar -C $SLURM_TMPDIR/data

# Start Evaluation
python $SOURCEDIR/eval.py \
	--weights-file "$SOURCEDIR/pretrained/drrn_x234.pth" \
	--eval-file "$SLURM_TMPDIR/data/Set5_x2.h5" \
	--eval-scale 2 \
	--B 1 \
	--U 9 \
	--num-features 128
echo "Job finish at $(date)"
