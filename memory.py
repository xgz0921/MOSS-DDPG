import torch
import numpy as np
class Memory():
    '''
    Code based on:
    https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
    Expects tuples of (state, next_state, action, reward, done)
    '''
    def __init__(self, max_size=1000):
        """Create Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def push(self, data):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self, batch_size):
        """Sample a batch of experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        Returns
        -------
        state: np.array
            batch of state or observations
        action: np.array
            batch of actions executed given a state
        reward: np.array
            rewards received as results of executing action
        """
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        state, action, reward = [], [], []

        for i in ind:
            st, act, rew = self.storage[i]
            state.append(np.array(st, copy=False))
            action.append(np.array(act, copy=False))
            reward.append(np.array(rew, copy=False))


        return np.array(state), np.array(action), np.array(reward).reshape(-1, 1)
