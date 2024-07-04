#%%
import torch
from torch import optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import DSAO_env as E
from memory import Memory
from noise import GaussianWhiteNoiseProcess
from Networks import Actor, Critic
from critic_dataset import CriticDataset
from train import train_actor, train_critic
import os
from functions import create_folder_if_not_exists as cf
from functions import write_txt as wt
import time
os.environ['KMP_DUPLICATE_LIB_OK']='True'

#%%
n_modes = E.n_modes
co_range = E.co_range
env = E.MOSSDDPG_Env()
num_episodes = 40000 #Number of episodes in total
warm_up_critic = 4000 #Number of warm up steps, critic network
warm_up_actor = 4000 #Number of warm up steps, actor network
batch_size = 8192 #Batch size of sampled data from Memory used for training
lr_critic = 25e-4 #Critic network learning rate
lr_actor = 25e-5 #Actor network learning rate
disp_gap = 400 # Number of steps per display of current training infos in text file.
batch_split = 16
batch_size_train = int(batch_size/batch_split) #Split sampled data to perform multiple gradient updates per step.

#%% Initialization of networks
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
input_dim_obs = 13 #Input dimension for LSTM network.
hidden_dim = 128 #Number of hidden layers in actor LSTM network.
critic = Critic(input_dim_obs,hidden_dim,n_modes) #Critic network
actor = Actor(input_dim_obs,hidden_dim,n_modes,1) #Actor network
critic.to(device)
actor.to(device)
loss_model_critic = torch.nn.MSELoss()
optimizer_critic = optim.Adam(critic.parameters(),lr = lr_critic)
optimizer_actor = optim.Adam(actor.parameters(), lr=lr_actor)

#%% Initialization of Memory
expl_rpt = 15 #Number of random noise explorations repeated per step, enhanced exploration that is unique with single-step DDPG.
max_memo_size = num_episodes *expl_rpt 
mem = Memory(max_size= max_memo_size)

#%% Initialization of Noise
sigma = 0.06
sigma_min = 0.001
random_process = GaussianWhiteNoiseProcess(sigma=sigma,sigma_min=sigma_min,n_steps_annealing=num_episodes,size = n_modes,env = env)

#%% Create a txt file to record training.
import datetime
records_path = 'Training History/'
current_time = datetime.datetime.now()
time_string = current_time.strftime("%Y_%m_%d") +'/'
training_para_str = current_time.strftime("%H-%M-%S")+'_'+ f'{n_modes}_m_{co_range}_bound_{num_episodes}_stp_{hidden_dim}_unts_{lr_actor}_lrA_{lr_critic}_LrC_{batch_size}_bs_{sigma}_noise_{sigma_min}_nsmin/'
file_path = records_path+time_string+training_para_str
print(file_path)
cf(file_path)
try:
    file = open(file_path+'records.txt', 'w')
except FileNotFoundError as e:
    print(f'An error occurred: {e}')
comment = f'Critic training mini batch size is 1/{batch_split} of sampled batch'+'\n'
wt(file_path+'records.txt',comment)
#%% Training Loop
loss_c = []
loss_a = []
rewards_storage = []
start_ep = time.perf_counter()
for ep in range(num_episodes):
    obs = env.reset()
    obs_hd = env.expand_dims(obs) 
    obs_hd = torch.Tensor(obs_hd).to(device)
    action = actor(obs_hd)
    if action.requires_grad:
        action_cpu = action.detach()  
    action_cpu = action_cpu.cpu()
    action_numpy = action_cpu.numpy().reshape((n_modes,)) #Get action from actor
    
    #Record the residual wavefront error in coefficients
    wfe_co = action_numpy+env.ab.c[0:n_modes] 
    env.wfe_record.append(wfe_co)
    
    #Within each episode, explore the environment with 15 random noise profiles, save all observation-action-reward to memory.
    for i in range(expl_rpt):
        noise_random = random_process.sample()
        action_noisy = action_numpy  + noise_random 
        action_noisy_clipped = np.clip(action_noisy, -0.2, 0.2)
        next_obs, reward, done, info = env.step(action_noisy_clipped)
        rewards_storage.append(reward)
        mem.push([obs,action_noisy_clipped,reward])
        
    #Code for displaying training infos:
    if ep % disp_gap==0 and ep!=0:
        wfes = np.array(env.wfe_record[-disp_gap:])
        RMS_WFE_mean = np.mean(np.sqrt(np.sum(wfes**2,axis = 1)))
        RMS_WFE_min = np.min(np.sqrt(np.sum(wfes**2,axis = 1)))
        RMS_WFE_max = np.max(np.sqrt(np.sum(wfes**2,axis = 1)))
        rewards_mean = np.mean(np.array(rewards_storage[-disp_gap*expl_rpt:]))
        rewards_min = np.min(np.array(rewards_storage[-disp_gap*expl_rpt:]))
        rewards_max = np.max(np.array(rewards_storage[-disp_gap*expl_rpt:]))
        if ep != num_episodes-1:
            end_ep = time.perf_counter()
            ep_rcd_str = f'Episodes: {ep-disp_gap} - {ep}, Average RMS WFE: {RMS_WFE_mean:.4f} [{RMS_WFE_min:.4f},{RMS_WFE_max:.4f}], Average Reward: {rewards_mean:.4f} [{rewards_min:.4f},{rewards_max:.4f}], Time: {end_ep-start_ep:0.2f} s'
            start_ep = time.perf_counter()
            print(ep_rcd_str)
            wt(file_path+'records.txt',ep_rcd_str)
            wt(file_path+'records.txt','\n')
            
    # After warm-up phase, start updating the networks.        
    if ep > warm_up_critic:
        states,actions,rewards = mem.sample(batch_size)
        states_train = torch.Tensor(states)
        actions_train = torch.Tensor(actions)
        rewards_train = torch.Tensor(rewards)
        critic_dataset = CriticDataset(states_train, actions_train, rewards_train)
        critic_loader = DataLoader(critic_dataset, batch_size=batch_size_train, shuffle=True)
        loss_critic = train_critic(critic,critic_loader,loss_model_critic,optimizer_critic,device)
        loss_c.append(loss_critic)
        if ep > warm_up_actor:
            actor_dataset = TensorDataset(states_train, actions_train)
            actor_loader = DataLoader(actor_dataset,batch_size=batch_size_train,shuffle = True)
            loss_actor = train_actor(actor,critic,actor_loader,optimizer_actor,device)
            loss_a.append(loss_actor)
            
    #Code for displaying training infos:        
    if ep % disp_gap==0 and ep!=0 and len(loss_c)>=disp_gap:
        critic_loss_avg_current = np.mean(np.array(loss_c[-disp_gap*expl_rpt:]).flatten())
        critic_loss_max_current = np.max(np.array(loss_c[-disp_gap*expl_rpt:]).flatten())
        critic_loss_min_current = np.min(np.array(loss_c[-disp_gap*expl_rpt:]).flatten())
        actor_loss_current = None if len(loss_a)< disp_gap else np.array(loss_a[-disp_gap*expl_rpt:]).flatten()
        actor_loss_avg_current = np.mean(actor_loss_current) if actor_loss_current is not None else 0
        actor_loss_max_current = np.max(actor_loss_current) if actor_loss_current is not None else 0
        actor_loss_min_current = np.min(actor_loss_current) if actor_loss_current is not None else 0
        ep_rcd_str_ac = f'Critic Loss: {critic_loss_avg_current:.6f} [{critic_loss_min_current:.6f},{critic_loss_max_current:.6f}], Actor Loss: {actor_loss_avg_current:.4f} [{actor_loss_min_current:.4f},{actor_loss_max_current:.4f}]'
        print(ep_rcd_str_ac)
        print('\n')
        wt(file_path+'records.txt',ep_rcd_str_ac)
        wt(file_path+'records.txt','\n\n')

#%% Save all records after training.
wfe_record = np.array(env.wfe_record)
np.save(file_path+'WFE_record.npy',wfe_record)
rewards_storage = np.array(rewards_storage)
np.save(file_path+'rewards_record.npy',rewards_storage)
#%% Save Models
from Networks import save_model, load_model
save_model(actor,file_path+'actor')
save_model(critic,file_path+'critic')