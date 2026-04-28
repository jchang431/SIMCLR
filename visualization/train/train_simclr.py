from models.simclr import SimCLRModel
from models.losses.nt_xent import NTXentLoss
from utils.train_utils import Trainer
from utils.data_utils import AverageMeter

import os, json, torch, time, argparse
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import yaml
import numpy as np
from pdb import set_trace


class SimCLRTrainer(Trainer):
    def __init__(self, config, checkpoint_dir=None, device=None):
        super().__init__(config, output_dir=checkpoint_dir, device=device, pretrain=True)
        self.net = self._init_model()
        self.optimizer = self._init_optimizer()
        self.scheduler = self._init_scheduler()
        self.criterion = NTXentLoss(config.simclr.temperature)

    @staticmethod
    def _build_model(cfg):
        return SimCLRModel(cfg)

    def _init_model(self):
        net = self._build_model(self.config).to(self.device)
        net.linear_eval = False 
        return net

    def _init_optimizer(self):
        return torch.optim.Adam(
            self.net.parameters(),
            lr=self.config.train.lr,
            weight_decay=self.config.optimizer.weight_decay #or 1e-6
        )

    def _init_scheduler(self):
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.config.train.n_epochs
        )

    def train(self):
        start = time.time()
        epoch_losses = []

        for epoch in range(self.config.train.n_epochs):
            self.net.train()
            loss_meter = AverageMeter()
            iter_meter = AverageMeter()
            for i, (x1, x2) in enumerate(self.train_loader):
                x1 = x1.to(self.device)
                x2 = x2.to(self.device)
                images = torch.cat([x1, x2], dim=0)
                _, z = self.net(images)
                # Split them back for the loss function
                z1, z2 = torch.split(z, x1.size(0))
                loss = self.criterion(z1, z2)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                loss_meter.update(loss.item())

            self.scheduler.step()
            iter_meter.update(time.time() - start)
            epoch_losses.append(loss_meter.avg)
            print(
                f'Epoch: [{epoch}][{i}/{len(self.train_loader)}]\t'
                f'Loss {loss_meter.val:.3f} ({loss_meter.avg:.3f})\t'
                f'Time {iter_meter.val:.3f} ({iter_meter.avg:.3f})\t'
            )

        total_time = time.time() - start
        print(f"Completed in {total_time:.3f}")

        torch.save(
            {"encoder": self.net.encoder.state_dict()},
            f'{self.output_dir}/simclr_pretrain2_{self.dataset.lower()}.pth',
        )

        metrics = {
            "epoch_losses": epoch_losses,
            "total_time_s": round(total_time, 3),
            "n_epochs": self.config.train.n_epochs,
            "dataset": self.dataset,
        }
        metrics_path = f'{self.output_dir}/pretrain_metrics.json'
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to {metrics_path}")
