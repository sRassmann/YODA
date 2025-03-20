#! /bin/bash
#SBATCH --job-name="training_yoda"
#SBATCH --partition="HPC-8GPUs"
#SBATCH --mem="200G"
#SBATCH --cpus-per-task="64"
#SBATCH --gres gpu:8
#SBATCH --time="4-10:00:00"
#SBATCH --nodes=1

module load singularity
cd $HOME/YODA
NUM_GPUS=$(nvidia-smi -L | wc -l)


singularity exec --nv -B $HPCWORK "$HPCWORK/$USER"_dagobah.sif torchrun \
   --nproc_per_node $NUM_GPUS train/2d_yoda_ddp.py \
   --data.batch_size 12 --data.num_workers 8 \  	# adapt to your system 
   --data.cache processes \  		# cache in RAM, share chaches between workers
   -n new_master_yoda configs/datasets/ixi_train.ylm    # specify config		
   -r output/partial_train/ckpt/epoch_0599.pth    	# resume from existing checkpoint
