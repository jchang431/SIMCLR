from torch.utils.data import Dataset
from torchvision import transforms
from pdb import set_trace

class SimCLRDataset(Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
        self.transform = self._get_simclr_transform()

    def _get_simclr_transform(self, size=32):
        color_jitter = transforms.ColorJitter(
        brightness=0.8,
        contrast=0.8,
        saturation=0.8,
        hue=0.2,
    )

        return transforms.Compose([
        transforms.RandomResizedCrop(size=size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([color_jitter], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=3),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2470, 0.2435, 0.2616],
        ),
    ])

    def __getitem__(self, idx):
        img, _ = self.base_dataset[idx]   # ignore label
        x1 = self.transform(img)
        x2 = self.transform(img)
        return (x1, x2)

    def __len__(self):
        return len(self.base_dataset)