import os
from torch.utils.data import DataLoader, Subset
from dataloaders.synth_data import SynthTCs
from sklearn.model_selection import train_test_split
import numpy as np
import torch

from captum.attr import (
    GradientShap,
    DeepLift,
    IntegratedGradients,
    NeuronConductance,
    Saliency
)

cuda = torch.device('cuda')

def get_saliency_synth(model, loaderFull):
    sal = IntegratedGradients(model)
    saliencies = []
    samples = []
    labels = []
    starts = []
    model.train()
    for i, data in enumerate(loaderFull):
        model.zero_grad()
        if i%1000==0:
            print(i)
        x, y, st = data
        x = x.to(cuda)
        y = y.to(cuda)
        b1 = torch.zeros(x.shape).to(cuda)
        S = sal.attribute(x,baselines=b1, target=y)
        #S = sal.attribute(x, target=y)
        S = S.detach().cpu().numpy()
        y = y.detach().cpu().numpy()
        starts.append(st.detach().cpu().numpy())

        saliencies.append(S)
        labels.append(y)
        samples.append(x.detach().cpu().numpy())
    samples = np.concatenate(samples,0)
    saliencies = np.concatenate(saliencies,0)
    starts = np.concatenate(starts,0)
    print(saliencies.shape)


    print(saliencies.shape)
    return saliencies, samples, starts


def get_saliency_continuous(model, loaderFull):
    sal = IntegratedGradients(model)

    saliencies = []
    samples = []
    labels = []
    model.train()
    for i, data in enumerate(loaderFull):
        model.zero_grad()
        if i%1000==0:
            print(i)
        x, y, st = data
        x = x.to(cuda)
        y = y.to(cuda)
        b1 = torch.zeros(x.shape).to(cuda)
        S = sal.attribute(x,baselines=b1)
        #S = sal.attribute(x)
        S = S.detach().cpu().numpy()
        y = y.detach().cpu().numpy()

        saliencies.append(S)
        labels.append(y)
        samples.append(x.detach().cpu().numpy())
    samples = np.concatenate(samples,0)
    saliencies = np.concatenate(saliencies,0)
    print(saliencies.shape)


    print(saliencies.shape)
    return saliencies, samples