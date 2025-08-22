#!/bin/bash

# standard CPU directives, tip: use cpus-per-task to allocate one or more cores per GPU  
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=1  
#SBATCH --threads-per-core=1
# plus the GPU line, you can request one or more (now up to 3) GPUs
#SBATCH --gres=gpu:1
 
# partition 'gpu' or 'gpu_limited', or paying project partition
#SBATCH --partition=gpuscavenger
# account 'shared' is default, there are also paying accounts 
#SBATCH --account=shared
 
#SBATCH --job-name=
#SBATCH --output=gpu_out.txt
 
#SBATCH --time=24:00:00 #(24 hours is the default in the partition 'gpu_limited')
#SBATCH --mem= 
# in the 'gpu' partition rather use  --mem=0 plus --exclusive 
##SBATCH --mem=0 --exclusive 

#optional for diagnostics, shows the hostname and available GPUs  
hostname
echo 
nvidia-smi
echo
echo CUDA_VISIBLE_DEVICES $CUDA_VISIBLE_DEVICES
echo

#and the actual job
