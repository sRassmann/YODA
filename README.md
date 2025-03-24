# YODA (You Only Denoise once - or Average)

"Hello there", welcome to the implementation of YODA described in the paper ["Regression is all you need for medical image denoising"]()


Abstract:

TBA

So, basically, YODA is a diffusion model (DM), which, however, also allows for single-step sampling just like a regression model (RM).
In fact, training RM is just as powerful.
Turns out, unless for whichever reason, realistic noise is required, regression sampling is not only faster but also more accurate than DM sampling including in several tested downstream tasks
(unless drawing and averaging $N_\text{Ex} \gg 1$ samples, which of course further exacerbates the required computational force). 

<p align="center">
  <img src="https://i.imgflip.com/9nwe3z.jpg" alt = "Star Wars meme" style="width:400px;"/>  
</p>

## What to Expect
Some example results demonstrating YODA's performance.  
TBA


# Code Instructions 
Here are some instructions to run our code and replicate some of our results:

## Code dependencies

This code is based on PyTorch and makes heavy use of the force of `MONAI` and the (by now deprecated) `MONAI generative` frameworks.  
The exact dependencies can be found in the `requirements.txt` file, yet, we recommend using docker/singularity:

### Docker 🐋
The dependencies are available from `dockerhub` and can be pulled using the following command:
```bash
docker pull srassmann/dif
```

### Singularity
Alternatively, the docker image can be converted to a singularity image using the following command:
```bash
SING_FILE=$HOME/singularity/${USER}_dagobah.sif
singularity build $SING_FILE docker-daemon://rassmanns/dif:latest
```

We will for now assume that `python` is from the correct environment, e.g. by using `singularity exec $SING_FILE python` or `docker exec -v <binds> -it $USER/dif python`.
This could be done via setting in your bash session:
```
alias python="singularity exec --nv -B <potential binds of symlinked data etc> $SING_FILE python"
```

## Preprocessing
Our preprocessing pipeline consists of registration & resampling followed by segmentation. 
We used the following tools for this purpose:

### FreeSurfer 
The registration of source and target modalities is performed using _FreeSurfer_ (v7.4). 
This can be [installed natively](https://surfer.nmr.mgh.harvard.edu/fswiki/DownloadAndInstall) or via 
[docker/singularity](https://hub.docker.com/r/freesurfer/freesurfer) (Note, that _FreeSurfer_ requires a license).

Yet, other tools can likely also be used to perform the registration.

### Full-brain segmentation
The full-brain segmentation is performed using [_FastSurfer_](https://github.com/Deep-MI/FastSurfer) (v2.2), 
which is also available as [docker/singularity image](https://hub.docker.com/r/deepmi/fastsurfer).

Yet, except for obtaining precise label-wise brain metrics like the noise-level of the WM, 
segmentation (including _FreeSurfer_'s `mri_synthstrip`) can be used just as well. 
The mask is only used to constraint the synthesis ROI and, optionally, for skull-stripping / background masking.
If you want, you can also omit it all together, however, than precious computation time is wasted on translating the background, 
which is rather bothersome for diffusion sampling (again, not really a need for that ...).

## Inference

### Weights 

Model weights will be released on Zenodo (link tba).

We expect the model weights to be placed in `output/<run_name>/ckpt` where `<run_name>` is the name of the run and model's base config to be in `output/<run_name>/config.yml`.

### Data organization

For simplicity, we assume the data to be stored in `../data/<dataset_name>` where `<dataset_name>` is the name of the dataset.
Within is directory, we expect one folder per subject, each containing the modalities as `.nii.gz` files.

E.g. to reproduce FLAIR synthesis in the Rhineland study using the [released example images](https://zenodo.org/records/11186582) (as shown above), the data should be organized as follows:

```bash
RAW_DATA=../data/rs_example_raw
mkdir -p $RAW_DATA
wget https://zenodo.org/records/11186582/files/sub_rs_mri_raw.zip -o ../data/rs_example
unzip -j $RAW_DATA/sub_rs_mri_raw.zip sub_rs_mri_raw/T1_RMS.nii.gz sub_rs_mri_raw/T2_caipi.nii.gz sub_rs_mri_raw/FLAIR.nii.gz -d $RAW_DATA && trash $RAW_DATA/sub_rs_mri_raw.zip
tree $RAW_DATA
../data/rs_example/
├── subj_0000
│   ├── FLAIR.nii.gz
│   ├── T1_RMS.nii.gz
│   └── T2_caipi.nii.gz
└── subj_0001  # in case you had more subjects
    ├── [...]
```


#### Registration and resampling
<details>
  <summary> see here for details </summary>
In the case (like here) that the data is not already registered and resampled, do that with your tool of choice, e.g. (assuming sourced _FreeSurfer_):

```bash
REGISTERED_DATA=../data/rs_example_registered
SOURCE_MODS=("T1_RMS T2_caipi")
TARGET_MOD="FLAIR"
mkdir -p $REGISTERED_DATA
for subj in $RAW_DATA/*; do
  subj_name=$(basename $subj)
  mkdir -p $REGISTERED_DATA/$subj_name
  ln -s  $(realpath $RAW_DATA/$subj_name/FLAIR.nii.gz) $REGISTERED_DATA/$subj_name/FLAIR.nii.gz
  for mod in $SOURCE_MODS; do
    mri_synthstrip -i $RAW_DATA/$subj_name/${mod}.nii.gz -m $REGISTERED_DATA/$subj_name/${mod}_brainmask.nii.gz --gpu
    mri_coreg --mov $RAW_DATA/$subj_name/${mod}.nii.gz --ref $REGISTERED_DATA/$subj_name/$TARGET_MOD.nii.gz --reg $REGISTERED_DATA/$subj_name/${mod}_to_${TARGET_MOD}.lta \
     --mov-mask $REGISTERED_DATA/$subj_name/${mod}_brainmask.nii.gz --ref-mask $REGISTERED_DATA/$subj_name/FLAIR.nii.gz --threads 16
    mri_vol2vol --cubic --mov $RAW_DATA/$subj_name/${mod}.nii.gz  --targ $REGISTERED_DATA/$subj_name/FLAIR.nii.gz \
     --reg $REGISTERED_DATA/$subj_name/${mod}_to_${TARGET_MOD}.lta --o $REGISTERED_DATA/$subj_name/${mod}.nii.gz
  done
done
```
This might take a couple of minutes / subject.

<p align="center">
<img src="https://media1.tenor.com/m/5AwAZOY-F94AAAAd/star-wars-yoda.gif" alt = "Patience to learn you must have" style="width:500px;"/>  
</p>

Note that here we register to the target modality (FLAIR) to assert that the images are aligned. 
If the target modality is not available (e.g. IXI or HCP), we recommend registering to the T2w images (resampling to ~1mm iso.).

</details>

### Intensity normalization

We rely on the [_FastSurfer_ script to robustly normalize the intensities](https://github.com/Deep-MI/FastSurfer/blob/dev/FastSurferCNN/data_loader/conform.py) of the registered images to 8 bit.
To do so, we use the following command (assuming appropriate python env, see above, e.g. replace with
`singularity --nv exec $SING_FILE python` and don't forget to mount the data via `-B` or, in docker via `-v`):
```bash
INPUT=$REGISTERED_DATA  # change if registered otherwise
CONFORMED_DATA=../data/rs_example_conformed
python scripts/preprocessing/conform.py -i $INPUT -o $CONFORMED_DATA --seqs $SOURCE_MODS $TARGET_MOD
```
Note that conformed/normalized/other pre-processed datasets (e.g. BraTS) might not require this step.

Furthermore, both inference and training requires a tissue mask to define the translation ROI.
Here, we simply use the `mri_synthstrip` masks, which are already in the original space:
```bash
for subj in $RAW_DATA/*; do ln -s $subj/${TARGET_MOD}_brainmask.nii.gz $CONFORMED_DATA/$(basename $subj)/mask.nii.gz ; done
```

In case you were to use _Fast/FreeSurfer_ for brain masking, you also want to map the brain mask (`aseg.auto_noCCseg.mgz`) back to the original space.
See the [respective script](scripts/preprocessing/conform_brain_mask.py) to this end.

In the lazy case, you can, however, omit the mask and simply symlink e.g. one of the input modalities. 
Then, the whole image (cropped to the max size of the model) will be translated.

### Dataset JSON definition
To inform YODA about the data, define a dataset JSON file we need.

```bash
JASON=../data/rs_example.json
```

<details>
    <summary> This file looks like smth like so: </summary>
    
```bash
JASON=../data/rs_example.json
touch $JASON
echo $'''
{
  "training": [
    {
      "subject_ID": "subj_0001",
      "_comment": "theoretically, multiple scans per subject are possible for each sequence",
      "flair": ["subj_0001/FLAIR.nii.gz"],
      "t1": ["subj_0001/T1.nii.gz", "subj_0001/T1_RMS.nii.gz"],
      "t2": ["subj_0001/T2_caipi.nii.gz"],
      "mask": "subj_0001/mask.nii.gz"
    }
  ], "validation": [
    {
      "subject_ID": "subj_0000",
      "_comment" : "same structure as training, however only one modality per subject!",
      "flair": "subj_0000/FLAIR.nii.gz",
      "t1": "subj_0000/T1_RMS.nii.gz",
      "t2": "subj_0000/T2_caipi.nii.gz",
      "mask": "subj_0000/mask.nii.gz"
    }
  ]
} ''' > $JASON

JASONwM=../data/rs_example_noMask.json
touch $JASONwM
echo $'''                               
{
  "training": [], 
  "validation": [
    {
      "subject_ID": "subj_0000_noMask",
      "_comment" : "same as before, but using as dummy as mask",
      "flair": "subj_0000/FLAIR.nii.gz",
      "t1": "subj_0000/T1_RMS.nii.gz",
      "t2": "subj_0000/T2_caipi.nii.gz",
      "mask": "subj_0000/T2_caipi.nii.gz"
     }
  ]
} ''' > $JASONwM
```
</details>

### Prediction

Here you can find the basic usage of the prediction scripts.
See the respective `--help` options for further options and ways to customize such as e.g. using different guidance/target sequence names.

#### Regression sampling
To predict the FLAIR image of `subj_0000` using the model weights and regression single-step sampling, run the following command 
(assuming python to be in the correct environment, don't forget to mount the data via `-B` or, in docker via `-v` and enable docker via `--nv`!):
```bash
RUN=rs_FLAIR_from_T1T2  # name of the run, the main configs are taken from output/<run_name>/config.yml
OUTNAME=predict_RS_example
CONF=configs/inference_schedulers/Regression.yml  # config we just created
SHARED_ARGS=" -r $RUN -dj $JASON -dd $CONFORMED_DATA"  # shared arguments
python predict/2d_yoda_predict.py $SHARED_ARGS $CONF -o $OUTNAME
```
Congrats, you have just used the force of YODA to predict a noise-free FLAIR image from T1w and T2w.

If you now want to also predict the other views for view aggregation, you can additionally run the following commands:
```bash
python predict/2d_yoda_predict.py $SHARED_ARGS $CONF -o ${OUTNAME}_cor -sd coronal
python predict/2d_yoda_predict.py $SHARED_ARGS $CONF -o ${OUTNAME}_sag -sd sagittal
python scripts/postprocessing/average_echos.py  output/$RUN/${OUTNAME}* --o output/$RUN/${OUTNAME}_rms -s "pred_flair.nii.gz"  # average the views
``` 
The view-aggregation results are in `output/$RUN/${OUTNAME}_rms/subj_0000/pred_flair.nii.gz`.

Note: experts use the `--force` flag to maximize YODA's capabilities.

Sampling without a mask (as specified in `$JASONwM`), can be done as:
```bash
SHARED_ARGS=" -r $RUN -dj $JASONwM -dd $CONFORMED_DATA"  # shared arguments
python predict/2d_yoda_predict.py $SHARED_ARGS $CONF -o $OUTNAME -om
python predict/2d_yoda_predict.py $SHARED_ARGS $CONF -o ${OUTNAME}_cor -sd coronal -om
python predict/2d_yoda_predict.py $SHARED_ARGS $CONF -o ${OUTNAME}_sag -sd sagittal -om
rm -r output/$RUN/${OUTNAME}_rms
python scripts/postprocessing/average_echos.py  output/$RUN/${OUTNAME}* --o output/$RUN/${OUTNAME}_rms -s "pred_flair.nii.gz"  # average the views
````
However, note this will simply center-crop the image, which might chop some important parts off.

#### Diffusion sampling
Alternatively, diffusion sampling can be conducted as follows:

```bash
NEX=4  # how many images to average, can also be one
LAZY=250  # truncation, i.e. step to which to skip --> here the diffusion will skip from step 999 -> 250 sparing 1/4 of compute
MEXds=250  # multi-excitation sampling diversion step --> step from which on to diverge into individual sampling trajectories 
OUTNAME=predict_RS_example_diffusion_mex$NEX
python predict/25d_yoda_predict.py $SHARED_ARGS -o $OUTNAME -cor $RUN -sag $RUN \
  -nex $NEX -lazy $LAZY -mexds $MEXds
```
Here, `-cor` and `-sag` could be distinct, view-specific models. Yet, we don't usually do that as we found no benefit for the extra training effort.
Note that we use a differen script `25d_` rather than `2d_`.  
Furthermore, note that diffusion sampling is inherently very time-consuming.
Thus, if the computational force is strong in your lab,
you can go for subject-wise parralelization [on multi-GPU systems](batch/parallel_predict_25D.sh) and on a [SLURM cluster](batch/example_array_predict_job.sh) for which we provide the scipts in the `batch` folder,

#### Dataset configs
You can also use configs for predefinecd combinations such as data sets.
E.g. to the test the RS YODA on other datasets, you'd had to always set the `-ds` and `-dj` flags.
For e.g. the IXI (which does not have a FLAIR) sequence you'd also to need specify the src and trg sequences.

To simplify we can alternatively merge the [corresponding config](configs/datasets/ixi_test.yml) like so:
```bash
python predict/2d_yoda_predict.py $SHARED_ARGS -o $OUTNAME configs/inference_schedulers/Regression.yml configs/datasets/ixi_test.yml
```
Note that when using multiple configs, they overwrite each other (from right to left),
i.e. the model config `output/$RUN/config.yml` is overwritten by `**/Regression.yml`, which is again overwritten by `**ixi_test.yml`.

Furthermore, note that some options (e.g. setting the 'target_sequences=null' or the `skullstripping`) are not supperted via the flags. 
Just create simple (`tmp`) configs instead as shown above.

## Training
To train your own YODA model preprocess the data, i.e. register and create tissue masks.
For brain MRI translation, we recommend the same processing as described above for the inference.

<p align="center">
<img src="https://i.giphy.com/3ohuAxV0DfcLTxVh6w.webp" alt="Much to learn you still have" style="width:500px;"/>  
</p>

### Dataset JSON
You will need to create a JSON file specifying your data, similar to the inference cases explained above.
Some examples (for IXI, BraTS, and the Gold Altas) for creating these JSONs can be found at [`nb/config_creation`](nb/config_creation).

### Start training
The models can be trained using
```bash
python train/train_yoda_ddp.py -n a_new_hope output/rs_FLAIR_from_T1T2/config.yml
```

### Training options
The options and their default values are defined in the [`configs/defaults.yml`](configs/defaults.yml) file.
You can either add configs or cmd-line flags to the train script. 
Child nodes (`c`) of parents nodes (`p`) can be specifiec in the dot notation (`--p.c <value>`), so e.g. the batch size can be set using `--data.batch_size <value>`. 
Note that, again, the configs are overwritten from left to right, and cmd flags overwrite the respective configs, e.g. assume we want to train the BraTS model on the RS with an effective batch size of 96 (12*8):
```bash
python train/train_yoda_ddp.py -n empire_strikes_back \
  output/brats_FLAIR_from_T1T2/config.yml configs/datasets/rs_train.yml \
  --data.batch_size 12 --data.num_workers 8 --trainer.gradient_accumulation_steps 8  
```

### Distributed Data Parallel
YODA can be easily trained on multiple GPUs (on a single node) with DDP
(again assuming that torchrun refers to the correct env, e.g. by 
`alias torchrun="singularity exec --nv -B <binds> $SING_FILE torchrun"`):

```bash
NUM_GPUS=$(nvidia-smi -L | wc -l)  # use all GPUs, assume 8, i.e. 12 * 8 = 96 effective batch size
torchrun --nproc_per_node $NUM_GPUS train/train_yoda_ddp.py -n return_of_jedi \
  output/brats_FLAIR_from_T1T2/config.yml configs/datasets/rs_train.yml \
  --data.batch_size 12 --data.num_workers 8
```

We also provide a template for SLURM jobs ([`batch/example_train_job_slurm.sh`](batch/example_train_job_slurm.sh)).

<p align="center">
    <img src="https://i.giphy.com/26tn8zNgVmit475RK.webp" alt = "No more training do you require" style="width:500px;"/>
</p>

Congrats, you have now trained your very own first YODA model! 
"I feel the force is strong with you."
