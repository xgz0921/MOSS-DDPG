import torch
import Networks
import DSAO_env
from DSAO_env import MOSSDDPG_Env
import os
import numpy as np
import matplotlib.pyplot as plt
from functions import *


os.environ['KMP_DUPLICATE_LIB_OK']='True'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
env = MOSSDDPG_Env()
n_modes = DSAO_env.n_modes
input_dim_obs = 13
hidden_dim = 128
actor_class = Networks.Actor(input_dim_obs,hidden_dim,n_modes,1)
critic_class = Networks.Critic(input_dim_obs,hidden_dim,n_modes)
file_path = 'Training History/2024_07_04/02-09-27_12_m_0.15_bound_40000_stp_128_unts_0.00025_lrA_0.0025_LrC_8192_bs_0.06_noise_0.001_nsmin/'
exp_per_ep = 15
warm_up = 4000

reward_record = np.load(file_path+'rewards_record.npy')
reward_record_ep = reward_record.reshape((int(reward_record.shape[0]/exp_per_ep),exp_per_ep))

reward_avg = np.mean(reward_record_ep,axis = 1)
#np.save(file_path+'averaged_reward1.npy',reward_avg)
#reward_avg = np.load(file_path+'averaged_reward.npy')
where = np.where(reward_avg<0.95)
ref = [int(exp_per_ep*i) for i in range(int(reward_record.shape[0]/exp_per_ep))]
WFE_record = np.load(file_path+'WFE_record.npy')
print(WFE_record[-1])
x = np.linspace(1,int(reward_record.shape[0]/exp_per_ep),int(reward_record.shape[0]/exp_per_ep))
plt.figure(figsize = (35,20))
plt.scatter(x,reward_avg,s = 0.3,color = 'green',marker = 'o')
plt.xlabel('Training Episode',fontsize =64)
plt.ylabel('Average Reward',fontsize = 64)
plt.axvline(x = warm_up, color='blue', linestyle='--', linewidth =5,label=f'Warm-up Phase: {warm_up} Episodes')
plt.axvline(x = 1, color='blue', linestyle='--', linewidth =5)
plt.ylim([0,1.005])
plt.tick_params(axis='both', which='major', labelsize=64)  # Adjust the font size as needed
plt.tight_layout()
plt.legend(fontsize = 64)
plt.savefig(file_path+'rwd_plot.pdf',dpi = 100)
#%%
wfes = np.sqrt(np.sum(WFE_record**2,axis = 1))
xx = np.linspace(1,wfes.shape[0],wfes.shape[0])
plt.figure(figsize = (8,4))
plt.scatter(xx,wfes,s = 0.001)
plt.xlabel('Training Episode',fontsize = 14)
plt.ylabel('RMS Wavefront Error / $\mu$m',fontsize = 14)
plt.tick_params(axis='both', which='major', labelsize=12)  # Adjust the font size as needed
plt.tight_layout()
plt.axvline(x = 3000, color='green', linestyle='--', label=f'Warm-up Episodes: {3000}')
plt.ylim([0,0.4])
plt.legend()
# %%
fig, ax = plt.subplots(2,1,figsize = (8,6))
ax[0].scatter(x,reward_avg,s = 0.005,color = 'g')
#ax[0].set_xlabel('Training Episode',fontsize = 14)
ax[0].set_ylabel('Reward',fontsize = 14)
ax[0].tick_params(axis='both', which='major', labelsize=12)  # Adjust the font size as needed
ax[0].axvline(x = warm_up, color='blue', linestyle='--', linewidth = 2,label=f'Warm-up Phase: {warm_up} Episodes')
ax[1].scatter(xx,wfes,s = 0.005,color = 'r')
ax[1].set_xlabel('Training Episode',fontsize = 14)
ax[1].set_ylabel('RMS Wavefront Error / $\mu$m',fontsize = 14)
ax[1].tick_params(axis='both', which='major', labelsize=12)  # Adjust the font size as needed
ax[1].axvline(x = warm_up, color='blue', linestyle='--',linewidth =2, label=f'Warm-up Phase: {warm_up} Episodes')
ax[1].axhline(y = 0.0349, color='g', linestyle='--',linewidth = 1, label='Diffraction-limited: 0.0349 $\mu$m')
ax[1].set_ylim([0,0.4])
plt.tight_layout()
ax[0].legend()
ax[1].legend()
plt.savefig(file_path+'combined_training_vis.png',dpi = 300)
# %%

