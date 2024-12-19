# local functions

# imports
import numpy as np
from scipy.stats import pearsonr

import matplotlib.pyplot as plt
import pandas as pd

# signal processing
from scipy import signal



# correlation
def get_parcel_corr(dat, embedding):

    nparcs = dat.shape[1]; ndims = embedding.shape[1]
    RvalsMat = np.empty((nparcs, ndims))

    for iparc in range(nparcs):
        
        for idim in range(ndims):
            
            Res = pearsonr(dat[:, iparc], embedding[:, idim])
            RvalsMat[iparc, idim] = Res.statistic
            
    return RvalsMat


    
# surface plotting    
def read_mesh(surf_file):
    """get vertices and triangles from mesh
    Returns : (vertices, triangles)
    """
    gii = nib.load(surf_file)
    return gii.darrays[0].data, gii.darrays[1].data
    
    
def remap2mesh(dat):

    map2mesh = pd.read_csv('../../Resources/map_parcels2mesh.csv')
    data_mapped = np.zeros((32492, 2))

    acc = 0
    for iHem in map2mesh.columns:

        # remap activation into the mesh
        tmp = map2mesh[iHem] 

        for iParcel in range(len(dat)):

            mask_parcel = tmp==(iParcel+1) # convert to MATLAB-like indexing system
            data_mapped[mask_parcel, acc] = dat[iParcel]    

        acc+=1
    
    return data_mapped
    
    
def butter_highpass(cutoff, fs, order=5):
    b, a = signal.butter(order, cutoff, btype='high', analog=False, fs=fs)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=5, axis=0):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data, axis=axis)
    return y

def butter_lowpass_filter(data, cutoff, fs, order=5, axis=0):
    b, a = signal.butter(order, cutoff, btype='low', analog=False, fs=fs)
    y = signal.filtfilt(b, a, data, axis=axis)
    return y
