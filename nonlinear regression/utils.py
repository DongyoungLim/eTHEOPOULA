import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pandas as pd
from torch import Tensor
import torch.nn.functional as F
eps = 1e-8
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class FS_Dataset(Dataset):
    def __init__(self, cov):
        self.cov = cov

    def __len__(self):
        return len(self.cov)

    def __getitem__(self, idx):
        x = torch.FloatTensor(self.cov.loc[idx])
        return x

def get_n_alpha(f_dist):
    if f_dist == 'POI':
        n_alpha = 0
    elif f_dist == 'ZIP':
        n_alpha = 1
    elif f_dist == 'NB':
        n_alpha = 1
    elif f_dist == 'ZNB':
        n_alpha = 2
    return n_alpha

def get_nll_F(dist, n, output):

    if dist == 'POI':
        lam = np.squeeze(output[0])
        loglik = -lam + n * torch.log(lam + eps)
        loglik = -torch.mean(loglik)
    elif dist == 'ZIP':
        lam = np.squeeze(output[0])
        pi = output[1]

        flag = torch.tensor(n == 0, dtype=torch.bool, device=device)
        loglik = flag * torch.log(pi + (1 - pi) * torch.exp(-lam))
        loglik += (~flag) * (torch.log(1 - pi) - lam + n * torch.log(lam))
        loglik = -torch.mean(loglik)

    return loglik

def get_nll_S(dist, y, output):
    if dist == 'Gamma':
        mu = np.squeeze(output[0])
        phi = output[1] + eps

        loglik = - torch.log(y) - torch.lgamma(1/phi) + 1/phi * (torch.log(y/phi) - torch.log(mu)) - y/(mu*phi)
        loglik = -torch.mean(loglik)

    return loglik


def convert_age(x):
    return int(x[1:-1].split(',')[0])


def encode_and_bind(df, features, drop_first=False):
    dummies = pd.get_dummies(df[features], drop_first=drop_first)
    res = pd.concat([df, dummies], axis=1)
    res = res.drop(features, axis=1)
    return (res)


def silu(input):
    return input * torch.sigmoid(input)

class SiLU(nn.Module):

    def __init__(self):
        super(SiLU, self).__init__()

    def forward(self, input: Tensor) -> Tensor:
        return silu(input)

    def extra_repr(self) -> str:
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str