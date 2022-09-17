#!/bin/bash
#SBATCH --account=def-hamilton
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=1  # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=32000M        # memory per node
#SBATCH --output=%N-%j.out  # %N for node name, %j for jobID

rm logs/loss/*
rm logs/acc/train/*
rm logs/acc/validation/*

source ./bin/activate
module load cuda cudnn 
tensorboard --logdir ./logs --bind_all  --port 6006 --load_fast=false &
python ./lung_cancer_cnn.py --variable_update=parameter_server --local_parameter_device=gpu 