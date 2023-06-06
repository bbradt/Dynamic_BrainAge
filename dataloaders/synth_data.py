import torch
import numpy as np
from torch.utils.data import Dataset


def static_spatial_patterns(samples, t_steps=100, features=50, p_steps=10, classes = 2, seed=None):
    if seed != None:
        np.random.seed(seed)
    X = np.zeros([samples, t_steps, features])
    L = np.zeros([samples])
    patterns = []
    patterns.append(np.arange(0,int(features/2)))
    patterns.append(np.arange(int(features/2),features))

    D = np.zeros([samples, t_steps, features])
    labels = np.random.randint(0, 2, samples)
    starts = []
    for i in range(samples):
        start = np.random.randint(0,t_steps-p_steps)
        x = np.random.normal(0, 1, [1, t_steps, features])
        lift = np.random.normal(1, 1,[p_steps, features])
        mask = np.zeros([p_steps, features])
        label = np.random.randint(0, classes)

        mask[:,patterns[label]] = 1
        lift = mask * lift
        X[i,:,:] = x
        X[i,start:start+p_steps, :] += lift
        L[i] = label
        starts.append(start)
    return np.array(X), np.array(L), np.array(starts), patterns

class SynthTCs(Dataset):
    def __init__(self,ret_starts=False, samples = 10000, features=50, seqlen = 100):
        self.N_subjects = samples
        self.dim = features
        self.seqlen = seqlen
        self.ret_starts = ret_starts

        X, L, starts, patterns = static_spatial_patterns(self.N_subjects, self.seqlen, self.dim)
        self.starts = starts
        self.data = X
        self.age = L

    def __len__(self):
        return self.N_subjects

    def __getitem__(self, k):
        if self.ret_starts:
            return torch.from_numpy(self.data[k, ...]).float(), torch.as_tensor(self.age[k]).long(), torch.as_tensor(self.starts[k]).long()
        else:
            return torch.from_numpy(self.data[k, ...]).float(), torch.as_tensor(self.age[k]).long()
