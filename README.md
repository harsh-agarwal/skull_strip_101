# Steps to setup the repo

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
