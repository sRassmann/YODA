#!/bin/bash
#SBATCH --partition="HPC-4GPUs"
#SBATCH --mem="8G"
#SBATCH --cpus-per-task="4"
#SBATCH --gres=gpu:1
#SBATCH --time="24:00:00"
#SBATCH --output=/home/rassmanns/diffusion/flairsyn/output/logs/slurm/%j_%x.out
#SBATCH --job-name="reg_view_agg"

module load singularity
cd /home/$USER/diffusion/flairsyn

model_name="sr3/multi_slice_large_t1pred_highres"
cfg="configs/datasets/rs_t1_MEX_highres_noT1guid.yml" #"configs/datasets/test.yml"          # run on validation data per default
outname="inference_t1MEX_DDIM_1_linsp"
target="pred_t1_tar.nii.gz"

START=0
END=34

args="-guid t1"

singularity exec --nv -B /groups/ag-reuter/projects/flair_synthesis $HOME/rassmanns_dif.sif \
  bash -c "
   python predict/2d_sr3_predict.py -nv -ce --start $START --end $END \
   --run_name $model_name configs/inference_schedulers/DDIM.yml $cfg $args --steps 1 -ts linspace \
   -o ${outname} && \
   python predict/2d_sr3_predict.py -nv -ce --start $START --end $END \
   --run_name $model_name configs/inference_schedulers/DDIM.yml $cfg $args --steps 1 -ts linspace \
   -o ${outname}_cor --slicing_direction coronal && \
   python predict/2d_sr3_predict.py -nv -ce --start $START --end $END \
   --run_name $model_name configs/inference_schedulers/DDIM.yml $cfg $args --steps 1 -ts linspace \
   -o ${outname}_sag --slicing_direction sagittal && \
    python scripts/postprocessing/average_echos.py \
    output/$model_name/${outname}  \
    output/$model_name/${outname}_cor  \
    output/$model_name/${outname}_sag  \
    --o output/$model_name/${outname}_rms \
    -s $target"

sh scripts/postprocessing/symlink_subjects.sh  $model_name/$outname $model_name/${outname}_rms ${target/pred_/}