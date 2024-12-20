# MV-eye
repo for MultiVariate EYEs closed/open

## introduction
The repo contains code for features extraction from resting state with eyes open/closed. The input data (not on github) comes in a fieldtrip-like format in source space based on an anatomical parcellation in 360 brain areas. The workflow is in MATLAB, for features extraction, and python for classification.  

## basic repos structure
The division of the subfolders is coarsely made in 3 categories: storage (STRG), stable code (STABLE) and sandbox(es), where to try things out.

### input/output storage (STRG)
Serving for storage of input/output data. (Hopefully) absent in the online github repo due to gitignore: github cannot store large datasets.
Three different sub folders:

**STRG_** * 
 - data 
 - computed_features
 - decoding_accuracy
       
### stable code
the tested code to be used with parallel computing. This is run preferably, but not exclusively, on the High Performance Cluster (HPC). The scripts made for the HPC have informative filenames. They are contained in the folder:

**STABLE_multithread**

### supporting functions

everything that gets called from the main codes. Distinction between:

**helper_functions**
The sub-folder containing the functions that _perform computation_ and, even more importantly, _of the function that generates the cfg file!!!_

**plotting_functions**
tools for visualization

