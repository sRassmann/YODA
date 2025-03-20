#!/bin/bash
# Full brain segmentation from IXI T2 images using SynthSeg
# usage after fs74 and activate conda evn
# (dif) rassmanns@moebius:~/diffusion/flairsyn$ bash scripts/synthseg/ixi_synthseg_predict.sh <folder>

folder=$1
file=${2:-pred_t2}

echo "Running SynthSeg"

# Create file list
PID=$$
inlist=/tmp/files_synthseg_$PID.txt
outlist=/tmp/files_synthseg_output_$PID.txt
find $folder/*/$file* | grep "pred_t2.nii.gz" > $inlist
sed "s|.nii.gz|_synthseg_resample.nii.gz|g" $inlist > $outlist

echo $inlist
echo $outlist

mri_synthseg --i $inlist --o $outlist --threads 64 --cpu

rm $inlist $outlist

/home/rassmanns/miniconda3/envs/dif/bin/python scripts/synthseg/synthseg_eval_fastsurfer_res.py \
 -f pred_t2_synthseg_resample "$folder" --calc_hd # &

# ref on acquired: copy to inference_ixi then:
# rassmanns@euler:/groups/ag-reuter-2/users/rassmanns/paper_temp/flairsyn/output/original/inference_ixi$ sh ~/diffusion/flairsyn/scripts/postprocessing/symlink_subjects.sh . . t2.nii.gz pred_t2.nii.gz
# bash scripts/synthseg/ixi_synthseg_predict.sh /groups/ag-reuter-2/users/rassmanns/paper_temp/flairsyn/output/original/inference_ixi
# else:
# bash scripts/synthseg/ixi_synthseg_predict.sh /groups/ag-reuter-2/users/rassmanns/paper_temp/flairsyn/output/sr3/multi_slice_mv_ixi/inference_ixit_25D_lazy_250