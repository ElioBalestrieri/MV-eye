#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 11:18:24 2024

@author: balestrieri
"""


# setting path for mv_python_utils
import sys
sys.path.append('../helper_functions')
from mv_python_utils import loadmat_struct

# basic
import numpy as np

# prep
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.metrics import balanced_accuracy_score

# NN
import torch
from torch import nn

#input folder 
infold = '../STRG_computed_features/NNprestim/'


#%% first part: load features, standardize within participants and concatenate
# them across participants

cnt_list, Y_list, prev_resp_list = [], [], []
brain_dict = {}

for isubj in range(30):

    # single subj part
    fname = infold + f'{isubj+1}.mat'
    mat_content = loadmat_struct(fname)
    F = mat_content['F'] # 'VariableName' if saved in parfor!!!
    single_feats = F['single_feats']
    
    # data extraction
    for key_feat, brain_X in single_feats.items():
        
        if isubj == 0:
            brain_dict.update({key_feat : []})

        brain_X = RobustScaler().fit_transform(brain_X)
        brain_dict[key_feat].append(brain_X)

    # behavioral predictors
    prev_resp = np.insert(np.sign(F['Y']-.5)[:-1], 0, 0, axis=0)
    contrast_X = F['x'][:, np.newaxis]    
    cnt_list.append(contrast_X)
    Y_list.append(F['Y'][:, np.newaxis])
    prev_resp_list.append(prev_resp[:, np.newaxis])
        
    print(isubj)
    

#%% create dataset (train/test)

strt_subj_test = 25

TRAIN_set, TEST_set = {'X' : [], 'Y' : []}, {'X' : [], 'Y' : []}
for ifeat in brain_dict.keys():
    
    # train
    tmp_brain = np.concatenate(brain_dict[ifeat][:strt_subj_test], axis=0)
    mdl_pca = PCA(n_components=.9).fit(tmp_brain)
    red_X = mdl_pca.transform(tmp_brain)
    TRAIN_set['X'].append(red_X)

    tmp_brain_test = np.concatenate(brain_dict[ifeat][strt_subj_test:], axis=0)
    TEST_set['X'].append(mdl_pca.transform(tmp_brain_test))
    
    print(ifeat)

TRAIN_set['Y'] = np.concatenate(Y_list[:strt_subj_test], axis=0)[:, 0]
TEST_set['Y'] = np.concatenate(Y_list[strt_subj_test:], axis=0)[:, 0]

TRAIN_set['X'].append(np.concatenate(cnt_list[:strt_subj_test], axis=0))
TEST_set['X'].append(np.concatenate(cnt_list[strt_subj_test:], axis=0))

TRAIN_set['X'].append(np.concatenate(prev_resp_list[:strt_subj_test], axis=0))
TEST_set['X'].append(np.concatenate(prev_resp_list[strt_subj_test:], axis=0))


#%%

TRAIN_set['X'] = np.concatenate(TRAIN_set['X'], axis=1)
TEST_set['X'] = np.concatenate(TEST_set['X'], axis=1)

#%% NN part

train_X = torch.from_numpy(TRAIN_set['X']).type(torch.float)
train_Y = torch.from_numpy(TRAIN_set['Y']).type(torch.float)

test_X = torch.from_numpy(TEST_set['X']).type(torch.float)
test_Y = torch.from_numpy(TEST_set['Y']).type(torch.float)

# Make device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"

#%% model definition
class NN_DecisionMaking(nn.Module):    
    
    def __init__(self):
        super().__init__()
        
        self.layer_1 = nn.LazyLinear(out_features=250)
        self.layer_2 = nn.Linear(in_features=250, out_features=50)
        self.layer_3 = nn.Linear(in_features=50, out_features=10)
        self.layer_4 = nn.Linear(in_features=10, out_features=1)
        self.relu = nn.ReLU()
#        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Intersperse the ReLU activation function between layers
        return self.layer_4(self.relu(self.layer_3(self.layer_2(self.relu(self.layer_1(x))))))
        # return self.layer_4(self.relu(self.layer_2(self.relu(self.layer_1(x)))))

    
test_mdl = NN_DecisionMaking().to(device)
print(test_mdl)    
    
# Setup loss and optimizer 
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(test_mdl.parameters(), lr=0.001)
    
torch.manual_seed(42)
epochs = int(4e4)


for epoch in range(epochs):

    # 1. Forward pass
    y_logits = test_mdl(train_X).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits)) 
    
    # 2. Calculate loss and accuracy
    loss = loss_fn(y_logits, train_Y) # BCEWithLogitsLoss calculates loss using logits
    acc = balanced_accuracy_score(y_true=train_Y, y_pred=y_pred.detach().numpy())
    
    # 3. Optimizer zero grad
    optimizer.zero_grad()

    # 4. Loss backward
    loss.backward()
    
    # 5. Optimizer step
    optimizer.step()
    
    ### testing
    test_mdl.eval()

    with torch.inference_mode():

        # 1. Forward pass
        test_logits = test_mdl(test_X).squeeze()
        test_pred = torch.round(torch.sigmoid(test_logits)) # logits -> prediction probabilities -> prediction labels
        # 2. Calcuate loss and accuracy
        test_loss = loss_fn(test_logits, test_Y)
        test_acc = balanced_accuracy_score(y_true=test_Y,
                               y_pred=test_pred.detach().numpy())
        
    if epoch % 1000 == 0:
        print(f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test Loss: {test_loss:.5f}, Test Accuracy: {test_acc:.2f}%" )
    
#%%
    
    
    
cnt = train_X.detach().numpy()[:, -1]
y_cnt = train_Y.detach().numpy()

arr_cnt = np.array([cnt, y_cnt]).T
srtd_cnt = arr_cnt[arr_cnt[:, 0].argsort()]

from scipy.signal.windows import gaussian
import matplotlib.pyplot as plt

L = 151
gwin = gaussian(L, 1)
out = np.convolve(srtd_cnt[:, 1], gwin, mode='same')/L

plt.plot(srtd_cnt[:, 0], out)





#%%


# tmp_cnt = np.concatenate(cnt_list[:strt_subj_test], axis=0)

# # shuffler 
# # idx_len = np.arange(len(tmp_brain))
# # np.random.shuffle(idx_len)
# # tmp_brain = tmp_brain[idx_len, :]

# mdl_pca = PCA(n_components=.99).fit(tmp_brain)
# red_X = mdl_pca.transform(tmp_brain)
# tmp_X = np.concatenate((red_X, tmp_cnt), axis=1)

# mdl_sclr = RobustScaler()
# mdl_sclr.fit(tmp_X)

# train_X = mdl_sclr.transform(tmp_X)
# train_Y = np.concatenate(Y_list[:strt_subj_test], axis=0)[:, 0]

# # test
# tmp_brain = np.concatenate(brain_list[strt_subj_test:], axis=0)
# tmp_cnt = np.concatenate(cnt_list[strt_subj_test:], axis=0)

# red_X = mdl_pca.transform(tmp_brain)
# tmp_X = np.concatenate((red_X, tmp_cnt), axis=1)

# test_X = mdl_sclr.transform(tmp_X)
# test_Y = np.concatenate(Y_list[strt_subj_test:], axis=0)[:, 0]
