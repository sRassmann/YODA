# predict all predicted BraTS images for a given output folder

reference=/groups/ag-reuter/projects/flair_synthesis/public_test/BraTS23/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData
preproc_flair=/home/rassmanns/diffusion/flairsyn/output/sr3/multi_slice_mv_brats/inference_DDIM_1_linsp
target=/home/rassmanns/diffusion/flairsyn/output/original/inference_brats


# note that the container/image requires pretty old CUDA versions, so this only runs on ada, lovelace, moebius

# TOOD:

# instead of softlinking the files, they should be resampled to the size of original files (take seg)
# see Desktop utils



# for each file in the folder
folder=$(realpath $preproc_flair)
for sfile in $folder/*
do
  subj=$(basename $sfile)

  mkdir -p $target/$subj

  ln -sf $reference/$subj/$subj-seg.nii.gz $target/$subj/seg.nii.gz
  ln -sf $reference/$subj/$subj-t1c.nii.gz $target/$subj/_t1ce.nii.gz
  ln -sf $reference/$subj/$subj-t1n.nii.gz $target/$subj/_t1.nii.gz
  ln -sf $reference/$subj/$subj-t2w.nii.gz $target/$subj/_t2.nii.gz

  mri_vol2vol --mov $folder/$subj/flair.nii.gz --targ $target/$subj/seg.nii.gz \
   --o $target/$subj/_flair.nii.gz --regheader --no-save-reg

  docker run --gpus 0 --rm --user "$(id -u):$(id -g)" --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -it \
    -v $target/$subj:/app/data -v $sfile:$sfile -v $reference/$subj:$reference/$subj brats/isen-20 python runner.py

done

# singularity
# singularity exec --pwd workspace --no-home --nv -B /home/rassmanns/Desktop/brats/BraTS-GLI-01666-000_sing:/app/data nnunet_brats2020.sif python runner.py

# docker run --gpus all --rm --user "$(id -u):$(id -g)" --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -it -v /home/rassmanns/Desktop/brats/BraTS-GLI-01666-000_lovelace/:/app/data  brats/isen-20 python runner.py