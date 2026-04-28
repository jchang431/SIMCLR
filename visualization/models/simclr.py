import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
from pdb import set_trace


class SimCLRModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.cfg = config
        
        # Flag to toggle the projector during forward pass
        self.linear_eval = False
        # Encoder
        self.encoder = resnet18(weights=None)
        self.encoder.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.encoder.maxpool = nn.Identity()
        self.encoder.fc = nn.Identity() 
        # Projector
        self.projector = nn.Sequential(
           nn.Linear(512, 2048),
           nn.ReLU(),
           nn.Linear(2048, self.cfg.network.proj_dim)
        )
        # Classifier for linear eval
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        # Encoder 
        h = self.encoder(x)
         
        if self.linear_eval:
            return h, self.classifier(h)
        
        z = self.projector(h)
        z = F.normalize(z, dim=1)
        return h, z
