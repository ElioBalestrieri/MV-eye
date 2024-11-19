#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 09:27:20 2024

@author: balestrieri
"""



root_ntd = "/home/balestrieri/Projects/neural_timeseries_diffusion/"

import sys, os, logging
sys.path.append(root_ntd + "ntd")

from ntd.train_diffusion_model import (set_seed, 
                                       init_diffusion_model,
                                       run_experiment)
from ntd.diffusion_model import Diffusion, Trainer
from ntd.utils.utils import standardize_array, count_parameters
from ntd.networks import SinusoidalPosEmb

from hydra import compose, initialize
from omegaconf import OmegaConf as OC
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader, Subset, random_split
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
import wandb

from scipy.io.matlab import loadmat, savemat
from scipy.fft import fft, fftfreq
import numpy as np
import matplotlib.pyplot as plt

data_path = "/home/balestrieri/TBraiC/MV-eye/STRG_data/reconstruct_signal"

log = logging.getLogger('test')

#%%

with initialize(version_base=None, config_path= "./conf"):
    cfg = compose(
        config_name="config",
        overrides=[
            "base.experiment=mv_eye",
            "base.tag=unconditional_wn",
            "base.wandb_mode=disabled",
            f"dataset.filepath={data_path}",
            "base.save_path=null",
            "optimizer.num_epochs=100",
            "optimizer.lr=0.0004",
            "network.signal_channel=52",
            "+experiments/generate_samples=generate_samples",
            "network.cond_channel=0",  # L 177 networks.py?
            "network.time_dim=0"
        ],
    )
    print(OC.to_yaml(cfg))

#%%

if cfg.base.use_cuda_if_available and torch.cuda.is_available():
    device = torch.device("cuda")
    environ_kwargs = {"num_workers": 0, "pin_memory": True}
#    log.info("Using CUDA")
else:
    device = torch.device("cpu")
    environ_kwargs = {}
#    log.info("Using CPU")

#%% define dataset class & split train set (init_dataset function)

class MV_EYE(Dataset):
    
    def __init__(
            self,
            subj_id,
            exp_compare, # VS/ECEO
            with_time_emb=False,
            cond_time_dim = 32, # ?
            filepath=None,
        ):
            super().__init__
            
            self.with_time_emb = with_time_emb
            self.cond_time_dim = cond_time_dim
            self.signal_length = 256
            self.num_channels = 52

            
            tmp_array = loadmat(os.path.join(filepath, f"{subj_id}{exp_compare}.mat"))
            tmp_array = tmp_array['tmp_out_dat']
            self.data_array = standardize_array(tmp_array, (0, 2)) # IMPORTANT! THIS NORMALIZATION DOES NOT DISTINGUISHES TRAIN TEST!!!

            temp_emb = SinusoidalPosEmb(cond_time_dim).forward(
                                        torch.arange(self.signal_length)
                                        )
            self.emb = torch.transpose(temp_emb, 0, 1)

    def __getitem__(self, index, cond_channel=None):
        return_dict = {}
        return_dict["signal"] = torch.from_numpy(np.float32(self.data_array[index]))
        cond = self.get_cond()
        if cond is not None:
            return_dict["cond"] = cond
        return return_dict

    def get_cond(self):
        cond = None
        if self.with_time_emb:
            cond = self.emb
        return cond
    
    def __len__(self):
        return len(self.data_array)

         
#%% 
data_set = MV_EYE(subj_id="01",
                  exp_compare="VS",
                  filepath=data_path)

train_size = int(len(data_set) * cfg.dataset.train_test_split)
test_size = len(data_set) - train_size
train_data_set, test_data_set = random_split(
    data_set,
    [train_size, test_size],
    generator=torch.Generator().manual_seed(cfg.dataset.split_seed),
)

train_loader = DataLoader(
    train_data_set,
    batch_size=cfg.optimizer.train_batch_size,
    shuffle=True,
    **environ_kwargs,
)

diffusion, network = init_diffusion_model(cfg)

optimizer = optim.AdamW(
    network.parameters(),
    lr=cfg.optimizer.lr,
    weight_decay=cfg.optimizer.weight_decay,
)

trainer = Trainer(diffusion, train_loader, optimizer, device)
wandb_conf = OC.to_container(cfg, resolve=True, throw_on_missing=True)
run = wandb.init(
    mode=cfg.base.wandb_mode,
    project=cfg.base.wandb_project,
    entity=cfg.base.wandb_entity,
    config=wandb_conf,
    dir=cfg.base.home_path,
    group=cfg.base.experiment,
    name=cfg.base.tag,
)

run.summary["num_network_parameters"] = count_parameters(network)

scheduler = MultiStepLR(
        optimizer,
        milestones=cfg.optimizer.scheduler_milestones,
        gamma=cfg.optimizer.scheduler_gamma,
    )

#%%

for i in range(cfg.optimizer.num_epochs):
    batchwise_losses = trainer.train_epoch()

    epoch_loss = 0
    for batch_size, batch_loss in batchwise_losses:
        epoch_loss += batch_size * batch_loss
    epoch_loss /= len(train_data_set)

    wandb.log({"Train loss": epoch_loss})
    log.info(f"Epoch {i} loss: {epoch_loss}")

    print(f"Epoch {i} loss: {epoch_loss}")

    scheduler.step()

run.finish()
log.info("Training done!")

samples = run_experiment(cfg, diffusion, test_data_set, train_data_set)

savemat(data_path + "/gen_sample_100epochs.mat", {"samples":samples})

#%%
trl = 22
plt.figure()
plt.plot(train_data_set.dataset.data_array[trl, 42, :])
plt.plot(samples[trl, 42, :])

#%% compare spectra

ntrl, nchan, ntp = samples.shape
T = 1/ntp # sfreq == ntp, 256 Hz
xf = fftfreq(ntp, T)[:ntp//2]

out = np.array(samples)

orig = fft(train_data_set.dataset.data_array, axis=2)
gener = fft(out, axis=2)

avg_or = np.mean(2/ntp * np.abs(orig[:, :, 0:ntp//2]), axis=0)
avg_gen = np.mean(2/ntp * np.abs(gener[:, :, 0:ntp//2]), axis=0)

plt.figure()
plt.plot(np.log10(xf[1:]), np.log10(avg_or[42, 1:]))
plt.plot(np.log10(xf[1:]), np.log10(avg_gen[42, 1:]))





