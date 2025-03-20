#!/bin/bash
# organ and bone segmentation on CT images in Gold Atlas using TotalSegmentor
# wrapper to manual command:
# # docker run --gpus 'device=0' --ipc=host -v /localmount/volume-ssd/users/rassmanns/tmp/inference_DDIM_1_linsp/1_03_P:/tmp wasserth/totalsegmentator:2.2.1 TotalSegmentator -i /tmp/ct.nii.gz -o /tmp/segmentations

folder=$1
file=${2:-pred_ct.nii.gz}
tmpdir=$(mktemp -d)  # force on localmount to avoid issues with NFS
echo "Using $tmpdir as temporary directory"

for subj in $folder/*; do
  subjname=$(basename $subj)
  echo "predicting subject $subjname"
  mkdir -p $tmpdir/$subjname
  cp $subj/$file $tmpdir/$subjname

  docker run --rm --gpus 'device=0' --ipc=host -v $tmpdir/$subjname:/tmp \
   wasserth/totalsegmentator:2.2.1 TotalSegmentator -i /tmp/$file -o /tmp/segmentations
done

echo "cleaning up tmp"
rsync -av $tmpdir/ $folder
docker run --rm --ipc=host -v $tmpdir:/tmp wasserth/totalsegmentator:2.2.1 /bin/bash -c "rm -rf /tmp/*"
rm $tmpdir -r

# evaluate
python scripts/totalsegmentor/totalsegmentor_eval.py $folder --calc_hd &