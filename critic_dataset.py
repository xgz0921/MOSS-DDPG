from torch.utils.data import Dataset

class CriticDataset(Dataset):
    def __init__(self, observations, actions, rewards):
        self.observations = observations
        self.actions = actions
        self.rewards = rewards

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (self.observations[idx], self.actions[idx], self.rewards[idx])