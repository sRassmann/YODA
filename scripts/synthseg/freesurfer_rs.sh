#! /bin/bash
#SBATCH --job-name="freesurf_RS"
#SBATCH --partition="HPC-CPUs"
#SBATCH --mem="30G"
#SBATCH --cpus-per-task="8"
#SBATCH --time="1-00:00:00"
#SBATCH --output=/home/rassmanns/diffusion/flairsyn/output/logs/slurm/%j_%x.out
#SBATCH --nodes=1
#SBATCH --array=1-600

export FREESURFER_HOME=/groups/ag-reuter/software-centos/fs741;source /groups/ag-reuter/software-centos/fs741/SetUpFreeSurfer.sh  # fs74
export SUBJECTS_DIR=$HPCWORK/freesurfer/RS_test  # make dir
INPUT=$HPCWORK/data/RS/conformed_mask_reg_test_dataset

subj=$(ls $INPUT | sed -n ${SLURM_ARRAY_TASK_ID}p)
echo $subj
echo $SLURM_ARRAY_TASK_ID
recon-all -i $INPUT/$subj/T1*.nii.gz -s $subj -all