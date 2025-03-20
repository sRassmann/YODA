# predict all predicted BraTS images for a given output folder
# note that the container/image requires pretty old CUDA versions, so this only runs on ada, lovelace, moebius

folder=$1
device=${2:-0}
flair_file=${3:-pred_flair.nii.gz}

reference=/groups/ag-reuter/projects/flair_synthesis/public_test/BraTS23/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData

# for each file in the folder
folder=$(realpath $folder)
for sfile in $folder/*
do
  if [ -f $sfile/results/tumor_isen2020_class.nii.gz ]; then
    echo "skipping $sfile"
    continue
  fi

  subj=$(basename $sfile)
  echo $sfile
  ln -sf $reference/$subj/$subj-seg.nii.gz $sfile/seg.nii.gz
  ln -sf $reference/$subj/$subj-t1c.nii.gz $sfile/_t1ce.nii.gz
  ln -sf $reference/$subj/$subj-t1n.nii.gz $sfile/_t1.nii.gz
  ln -sf $reference/$subj/$subj-t2w.nii.gz $sfile/_t2.nii.gz

  mri_vol2vol --mov $folder/$subj/$flair_file --targ $sfile/seg.nii.gz \
   --o $sfile/_flair.nii.gz --regheader --no-save-reg

  docker run --gpus $device --rm --user "$(id -u):$(id -g)" --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -it \
    -v $sfile:/app/data -v $sfile:$sfile -v $reference/$subj:$reference/$subj brats/isen-20 python runner.py

done

# singularity
# singularity exec --pwd workspace --no-home --nv -B /home/rassmanns/Desktop/brats/BraTS-GLI-01666-000_sing:/app/data nnunet_brats2020.sif python runner.py

# docker run --gpus all --rm --user "$(id -u):$(id -g)" --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -it -v /home/rassmanns/Desktop/brats/BraTS-GLI-01666-000_lovelace/:/app/data  brats/isen-20 python runner.py
