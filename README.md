# Skull Stripping

Please follow the following steps to setup the repository. 

## Download and prepare the dataset

1. The following command witll help you in downloading the dataset from the official NFBS website. 
```
wget https://fcp-indi.s3.amazonaws.com/data/Projects/RocklandSample/NFBS_Dataset.tar.gz
```

2. Once done extract the files using

```
tar -xzvf <path to the downlaoded zip file>
```
Please check the number of files are 125. Each folder would have three files namely:

```
sub-<identifier>_ses-NFB3_T1w.nii.gz -> Input T1W image 
sub-<identifier>_ses-NFB3_T1w_brain.nii.gz -> skull stripped image 
sub-<identifier>_ses-NFB3_T1w_brainmask.nii.gz -> brain mask image 

```

## Prepare the virtual environment 

Please get the latest installation of miniconda which can be easily downloaded from here https://docs.conda.io/en/latest/miniconda.html

Once you have verified anaconda/miniconda installed, please create a virtual environment using the `conda.yml` in the repo using the following command. 

```
conda env create -f conda.yml
```

## Running training for skull stripping

Run the following command:

```
python train.py --path <path to the extracted data folder> 
```

Other key arguments that we can set are:

`--gpus` : By default the code uses all the gpus on the machine 
`--max_epochs`: te number of epochs we want the training to run 

PS: With some very simple changes this code can support multi node, multi gpu training and various other possibilities like resuming from a previous checkpoint etc. I would add those changes later as and when time permits! 

By default the code uses and sets up a tensorboard as a logger. One can view the training curves obtained by the following command in a different terminal: 

```
tensorboard --logdir ./logs/lightning_logs/ --bind_all
```

The above will return a URL, please use that to view the logs on any browser (checked and tested on google chrome)

## Inference (documentation to be added)


# Canonical ICA

## Preparing the dataset 

Please download and transfer the filtered data from [here](https://drive.google.com/file/d/1B6Zy_3FXp5eCa--RMXEnojEW3Trhm-7K/view?usp=sharing) 

The following was the reference I sued for writing the code. [NiLearns]( https://nilearn.github.io/modules/generated/nilearn.decomposition.CanICA.html)

## Running the code

```
python canICA.py --path <path to the filtered data file> --num_com <number of components to run ICA on>
```

This will save the results at `untracked/canica_resting_state.nii.gz`

If you are working on a jupyter notebook, or machine that has supprt for GUI it will also show some plots and results. 


