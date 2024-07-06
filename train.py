### Script for network updating based on sampled data.
### @author : Guozheng Xu
### @date   : 2024-07-06
############################################################################
import torch
def train_actor(actor,critic,train_loader,optimizer,device):
    actor.train()
    loss_sum = 0
    for states,actions in train_loader:
        states,actions = states.to(device),actions.to(device)
        outputs = actor(states)
        loss = -critic(states,outputs).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()
    return loss_sum/len(train_loader)

def train_critic(model,train_loader,loss_fn,optimizer,device):
    model.train()
    loss_sum = 0
    for obs,actions,rewards in train_loader:
        obs,actions,rewards = obs.to(device),actions.to(device),rewards.to(device)
        outputs = model(obs,actions)
        loss = loss_fn(outputs,rewards)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()
    return loss_sum/len(train_loader)