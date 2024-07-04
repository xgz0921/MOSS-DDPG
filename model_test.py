#%%
#This code is to examine the performance of trained models with different random aberrations.
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
actor = Networks.load_model(file_path+'actor',actor_class)
critic = Networks.load_model(file_path+'critic',critic_class)
critic.to(device)
actor.to(device)

num_tests = 1000 
cf_bias = 0.2

ab_mode = 'decreasing'
training = False

def random_aberration_test(num_tests,actor,cf_bias,ab_mode,file_path,return_result = False,saved = True):
    wfe_record = [] # Wavefront error record for each random test, MOSSDDPG
    wfe_record_cf = [] #Wavefront error record for each random test,parabola fitting
    sim_wfes = [] #Simulated wavefront errors, MOSSDDPG
    sim_wfes_cf = [] #Simulated wavefront errors, parabola fitting
    actions = [] # Network predictions for each random test, MOSSDDPG
    cf_predictions = [] # Curve fitting results for each random test, parabola fitting 
    rewards = [] # Image metric after correction, relative to the max, MOSSDDPG
    rewards_cf = [] # Image metric after correction, relative to the max, parabola fitting
    
    for i in range(num_tests):
        obs = env.reset(training=training, ab_mode= ab_mode)
        obs_hd = env.expand_dims(obs)
        obs_hd = torch.Tensor(obs_hd).to(device)
        #print(obs_hd.shape)
        action = actor(obs_hd)
        if action.requires_grad:
            action_cpu = action.detach()  # Detach from the computation graph if needed
        action_cpu = action_cpu.cpu()
        action_numpy = action_cpu.numpy().reshape((DSAO_env.n_modes,))
        next_obs, reward, done, info = env.step(action_numpy)
        rewards.append(reward)
        sim_wfes.append(env.ab.c[0:n_modes])
        actions.append(action_numpy)
        wfe_co = action_numpy+env.ab.c[0:n_modes].copy()
        wfe_record.append(wfe_co)
        
        
        obs_cf = env.reset(training=training, ab_mode=ab_mode,obs_bias=cf_bias,new_ab = False)
        metrics = obs_cf[:,0]
        sim_wfes_cf.append(env.ab.c[0:n_modes])
        prediction_cf = np.clip(curve_fitting_SAO(metrics,cf_bias),-0.2,0.2)
        cf_predictions.append(prediction_cf)
        wfe_co_cf = prediction_cf+env.ab.c[0:n_modes].copy()
        wfe_record_cf.append(wfe_co_cf)
        rewards_cf.append(reward)
        
    
    wfe_record = np.array(wfe_record)
    wfe_record_cf = np.array(wfe_record_cf)
    
    rewards = np.array(rewards)
    rewards_cf = np.array(rewards_cf)
    
    actions = np.array(actions)
    cf_predictions = np.array(cf_predictions)
    
    sim_wfes = np.array(sim_wfes)
    sim_wfes_cf = np.array(sim_wfes_cf)
    if saved == True:
        np.save(file_path+'WFE_record_'+ab_mode+'_ra.npy',wfe_record)
        np.save(file_path+'WFE_record_cf_'+ab_mode+'_ra.npy',wfe_record_cf)
    if return_result:
        return [wfe_record,wfe_record_cf],[sim_wfes,sim_wfes_cf],[actions,cf_predictions],[rewards,rewards_cf]
    return

random_aberration_test(num_tests,actor, cf_bias, 'uniform', file_path,saved=True)
random_aberration_test(num_tests,actor, cf_bias, 'decreasing', file_path,saved=True)
random_aberration_test(num_tests,actor, cf_bias, 'normal', file_path,saved=True)

