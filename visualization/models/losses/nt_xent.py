import os, torch, time, argparse
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import yaml
import numpy as np


class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1, z2):
        # Normalize
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        N = z1.size(0)


        # All similarities
        reps = torch.cat([z1, z2], dim=0)
        sim = torch.mm(reps, reps.t()) / self.temperature

        # Mask out self‑similarity
        mask = torch.eye(2 * N, device=z1.device).bool()
        sim = sim.masked_fill(mask, -9e15)

        # Labels: positives are offset by N
        labels = torch.cat([torch.arange(N, 2 * N), torch.arange(0, N)]).to(z1.device)

        loss = F.cross_entropy(sim, labels)
        return loss
