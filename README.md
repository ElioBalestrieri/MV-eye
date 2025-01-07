# MV-eye
repo for MultiVariate EYEs closed/open. The manuscript is currently under revision, and as the reader knows, the analysis extended to the evaluation of another pait of brain states, namely baseline/Visual stimulation. Here the [link](https://doi.org/10.1101/2024.04.17.589917) to the preprint.

## introduction
The repo contains code for features extraction from M-EEG data in different states. The input data (not on github) comes in a fieldtrip-like format. The workflow is in MATLAB, for features extraction, and python for classification.
The analysis contained in the manuscript rely on the code contained in:

**STABLE_multithread**
**reply_reviewers**

### supporting functions

everything that gets called from the main codes. Distinction between:

**helper_functions**
The sub-folder containing the functions that _perform computation_ and, even more importantly, _of the function that generates the cfg file!!!_

**plotting_functions**
tools for visualization

