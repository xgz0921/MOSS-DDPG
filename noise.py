### Script for generating annealing gaussian exploration noise.
### This code is from Keras-RL: "https://github.com/keras-rl/keras-rl.git"
############################################################################

import numpy as np
'''
This code is from Keras-RL: "https://github.com/keras-rl/keras-rl.git"
'''

class RandomProcess(object):
    def reset_states(self):
        pass


class AnnealedGaussianProcess(RandomProcess):
    def __init__(self, mu, sigma, sigma_min, n_steps_annealing):
        self.mu = mu
        self.sigma = sigma
        self.n_steps = 0
        ##Guozheng Added
        self.n_steps_annealing = n_steps_annealing
        
        if sigma_min is not None:
            #self.m = -float(sigma - sigma_min) / float(n_steps_annealing) #This is original
            full_step = np.power(n_steps_annealing,1)
            self.m = (sigma - sigma_min) / full_step
            #print(self.m)
            self.c = sigma
            self.sigma_min = sigma_min
        else:
            self.m = 0.
            self.c = sigma
            self.sigma_min = sigma

    @property
    def current_sigma(self):
        #sigma = max(self.sigma_min, self.m * float(self.n_steps) + self.c)#Original
        current_step_power = np.power(abs(self.n_steps-self.n_steps_annealing),1)
        #print(current_step_power)
        sigma = self.m*current_step_power+self.sigma_min
        #print(sigma)
        return sigma


class GaussianWhiteNoiseProcess(AnnealedGaussianProcess):
    def __init__(self, mu=0., sigma=1., sigma_min=None, n_steps_annealing=1000, size=1,env = None):
        super(GaussianWhiteNoiseProcess, self).__init__(mu=mu, sigma=sigma, sigma_min=sigma_min, n_steps_annealing=n_steps_annealing)
        self.size = size
        self.env = env
    def sample(self):
        sample = np.random.normal(self.mu, self.current_sigma, self.size)
        #self.n_steps += 1
        self.n_steps = self.env.episode_count
        return sample