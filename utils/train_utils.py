from utils.data_utils import set_seed, get_device, AverageMeter
import os, torch, shutil
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from utils.simclr_utils import SimCLRDataset
from torch.utils.data import Subset
from pdb import set_trace


# ── Imbalanced split registry ──────────────────────────────────────────────────
# Keys are the values accepted by config.data.imbalanced_split / --imbalanced-split
_IMB_DIR = "data/imbalanced"
IMBALANCED_SPLIT_MAP = {
    # Difficulty-weighted: hard classes (bird/cat/dog) get more labels
    "difficulty":        f"{_IMB_DIR}/cifar10_imb_difficulty_seed42.npz",
    "difficulty_1pct":   f"{_IMB_DIR}/cifar10_imb_difficulty_1pct_seed42.npz",
    # Long-tail IF=10: airplane has 10× more labels than truck
    "lt_if10":           f"{_IMB_DIR}/cifar10_imb_lt_if10_seed42.npz",
    "lt_if10_1pct":      f"{_IMB_DIR}/cifar10_imb_lt_if10_1pct_seed42.npz",
    "lt_if10_25pct":     f"{_IMB_DIR}/cifar10_imb_lt_if10_25pct_seed42.npz",
    # Long-tail IF=50: airplane has 50× more labels than truck
    "lt_if50":           f"{_IMB_DIR}/cifar10_imb_lt_if50_seed42.npz",
}

# ── Balanced (FixMatch-style) fractional splits ────────────────────────────────
_BAL_DIR = "data/balanced"
_BALANCED_SPLIT_MAP = {
    0.01: f"{_BAL_DIR}/cifar10_split_0p01_seed42.npz",
    0.05: f"{_BAL_DIR}/cifar10_split_0p05_seed42.npz",
    0.10: f"{_BAL_DIR}/cifar10_split_0p1_seed42.npz",
    0.25: f"{_BAL_DIR}/cifar10_split_0p25_seed42.npz",
    0.50: f"{_BAL_DIR}/cifar10_split_0p5_seed42.npz",
    1.0:  f"{_BAL_DIR}/cifar10_split_1p0_seed42.npz",
}


def _pick_split(full_train: datasets.CIFAR10,
                data_dir: str,
                imbalanced_split: str | None,
                train_fraction: float,
                seed: int = 42):
    """
    Return (train_subset, val_subset) for linear evaluation.

    Priority order
    ──────────────
    1. imbalanced_split is set  → load the named imbalanced .npz
    2. train_fraction < 1.0     → load the closest balanced FixMatch .npz
    3. train_fraction == 1.0    → 90/10 random split of full training set
    """

    # ── Path 1: explicit imbalanced split ─────────────────────────────────────
    if imbalanced_split:
        npz_path = IMBALANCED_SPLIT_MAP.get(imbalanced_split)
        if npz_path is None:
            valid = ", ".join(IMBALANCED_SPLIT_MAP.keys())
            raise ValueError(
                f"Unknown imbalanced_split '{imbalanced_split}'. "
                f"Valid choices: {valid}"
            )
        if not os.path.isfile(npz_path):
            raise FileNotFoundError(
                f"Imbalanced split file not found: {npz_path}\n"
                f"Make sure the file exists under data/imbalance_splits_km/splits/."
            )
        s = np.load(npz_path)
        print(f"[data] Loading imbalanced split '{imbalanced_split}' "
              f"→ {len(s['labeled_idx'])} train / {len(s['val_idx'])} val")
        return Subset(full_train, s["labeled_idx"]), Subset(full_train, s["val_idx"])

    # ── Path 2: balanced fractional .npz ──────────────────────────────────────
    for frac, path in _BALANCED_SPLIT_MAP.items():
        if abs(float(train_fraction) - frac) < 1e-4:
            if os.path.isfile(path):
                s = np.load(path)
                print(f"[data] Loading balanced split ({int(train_fraction*100)}%) "
                      f"→ {len(s['labeled_idx'])} train / {len(s['val_indices'])} val")
                return (Subset(full_train, s["labeled_idx"]),
                        Subset(full_train, s["val_indices"]))

    # ── Path 3: fallback for non-standard fractions (e.g. 0.33, 0.75) ───────────
    # Only reached when train_fraction has no matching entry in _BALANCED_SPLIT_MAP.
    # The six standard values (0.01, 0.05, 0.10, 0.25, 0.50, 1.0) never reach here
    # because they all have pre-computed .npz files and are caught by Path 2 above.
    
    n = len(full_train)
    train_size = int(0.9 * n)
    val_size = n - train_size
    print(f"[data] 90/10 random split → {train_size} train / {val_size} val")
    return random_split(
        full_train,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )


class Trainer:
    """
    Base trainer class.
    Specific training loops handled by subclasses.
    """

    def __init__(self, config, output_dir=None, device=None, pretrain=False):
        self.config = config
        self.pretrain = pretrain
        self.num_workers = self.config.train.num_workers
        self.batch_size = self.config.train.batch_size
        self.lr = self.config.train.lr
        self.n_epochs = self.config.train.n_epochs
        self.dataset = self.config.data.dataset
        # percent is only used during linear eval; default to 1.0 for pretrain
        self.train_percent = getattr(self.config.data, "percent", 1.0)
        # optional name of an imbalanced split (only used during linear eval)
        self.imbalanced_split = getattr(self.config.data, "imbalanced_split", None)

        set_seed(seed=42)  # do not change seed for reproducibility

        if device is None:
            self.device = get_device()
        else:
            self.device = device

        self.data_dir = "./data"
        os.makedirs(self.data_dir, exist_ok=True)

        if output_dir is None:
            self.output_dir = f"./outputs/{self.config.network.model.lower()}"
        else:
            self.output_dir = output_dir

        os.makedirs(self.output_dir, exist_ok=True)

        # ── Transforms ────────────────────────────────────────────────────────
        train_transform = None
        test_transform = None
        if not self.pretrain:
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
            test_transform = transforms.Compose([
                transforms.ToTensor(),
            ])

        # ── Load base CIFAR-10 ─────────────────────────────────────────────────
        if self.dataset.lower() == "cifar":
            self.trainset = datasets.CIFAR10(
                root=self.data_dir, train=True, download=True,
                transform=train_transform,
            )
            self.testset = datasets.CIFAR10(
                root=self.data_dir, train=False, download=True,
                transform=test_transform,
            )

            # ── Split for linear eval ──────────────────────────────────────────
            if not self.pretrain:
                self.trainset, self.val_set = _pick_split(
                    full_train=self.trainset,
                    data_dir=self.data_dir,
                    imbalanced_split=self.imbalanced_split,
                    train_fraction=float(self.train_percent),
                    seed=42,
                )

        # For pretrain we need two augmented views per image
        if self.pretrain:
            self.trainset = SimCLRDataset(self.trainset)
            self.testset = SimCLRDataset(self.testset)

        # ── Data loaders ───────────────────────────────────────────────────────
        self.train_loader = DataLoader(
            self.trainset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
        if not self.pretrain:
            self.val_loader = DataLoader(
                self.val_set,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
            )
        self.test_loader = DataLoader(
            self.testset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

        dummy_iterator = iter(self.train_loader)
        sample_input, _ = next(dummy_iterator)
        assert sample_input.dim() == 4, (
            "data shape not expected. You are doing something wrong wrt setup or dataset"
        )
        _, _, self.height, self.width = sample_input.size()
        self.input_dim = int(self.height * self.width)
        self.input_shape_dim = sample_input.dim()

        if self.pretrain:
            self.fixed_eval_batch, _ = self.get_fixed_samples(self.testset, n_samples=8)
            self.fixed_eval_batch.to(self.device)
        else:
            self.fixed_eval_batch, _ = self.get_fixed_testset_samples(self.testset, n_samples=8)
            self.fixed_eval_batch.to(self.device)

    def _init_optimizer(self, net):
        if self.config.optimizer.type.lower() == "sgd":
            optimizer = torch.optim.SGD(
                net.parameters(), lr=self.lr,
                momentum=0.9, weight_decay=self.config.optimizer.weight_decay,
            )
        elif self.config.optimizer.type.lower() == "adamw":
            optimizer = torch.optim.AdamW(
                net.parameters(), lr=self.lr,
                betas=(0.9, 0.999), weight_decay=self.config.optimizer.weight_decay,
            )
        else:
            raise ValueError("unsupported optimizer. use sgd or adamw")
        return optimizer

    @staticmethod
    def get_fixed_samples(dataset, n_samples=8, start_idx=100):
        xs1, xs2 = [], []
        for i in range(start_idx, start_idx + n_samples):
            x1, x2 = dataset[i]
            xs1.append(x1.clone())
            xs2.append(x2.clone())
        xs1 = torch.stack(xs1, dim=0)
        xs2 = torch.stack(xs2, dim=0)
        return xs1, xs2

    @staticmethod
    def get_fixed_testset_samples(dataset, n_samples=8, start_idx=100):
        images, labels = [], []
        for i in range(start_idx, start_idx + n_samples):
            image, label = dataset[i]
            images.append(image.clone())
            labels.append(label)
        images = torch.stack(images, dim=0)
        labels = torch.tensor(labels)
        return images, labels

    @staticmethod
    def save_model(model, model_path):
        model_s = torch.jit.script(model)
        model_s.save(model_path)
        print(f"Model saved to {model_path}")

    @staticmethod
    def load_model(model_path, map_location="cpu"):
        model = torch.jit.load(model_path, map_location=map_location)
        return model

    def train(self):
        raise NotImplementedError

    def evaluate(self, epoch):
        raise NotImplementedError
