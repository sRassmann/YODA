#! /bin/bash
#SBATCH --job-name="predict_job"
#SBATCH --partition="HPC-8GPUs"
#SBATCH --mem="8G"
#SBATCH --cpus-per-task="4"
#SBATCH --gres=gpu:1
#SBATCH --time="09:00:00"
#SBATCH --array=0-42   # number of jobs

# assume you load singularity
module load singularity
cd $HOME/diffusion/flairsyn

singularity exec --nv -B $HPCWORK \ 	# potentially other data binds
    "$HPCWORK/$USER"_dagobah.sif python  \  # change name and path of sing. image
     predict/25d_yoda_predict.py --start $SLURM_ARRAY_TASK_ID --end $((SLURM_ARRAY_TASK_ID + 1)) -nv -ce \  # select job
    --run_name rs_FLAIR_from_T1T2  --coronal_run_name rs_FLAIR_from_T1T2 --sagittal_run_name rs_FLAIR_from_T1T2 \  # same weights for all views
    configs/datasets/bmb_test.yml  -lazy 250 -nex 10 -mexds 250 -o inference_bmb_MEX10
