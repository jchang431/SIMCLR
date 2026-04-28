from models.simclr import SimCLRModel
from utils.train_utils import Trainer
from utils.data_utils import AverageMeter

import json, torch, time
import torch.nn as nn


def evaluate_confusion_matrix(model, loader, device, save_path):
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    model.eval()

    class_names = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"
    ]

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            _, logits = model(x)
            preds = logits.argmax(dim=1)

            all_labels.extend(y.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    cm_norm = cm.astype("float") / (cm.sum(axis=1, keepdims=True) + 1e-12)

    plt.figure(figsize=(8, 6))
    plt.imshow(cm_norm, interpolation="nearest")
    plt.title("SimCLR Partial Fine-tune Normalized Confusion Matrix")
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)

    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            plt.text(
                j, i, f"{cm_norm[i, j]:.2f}",
                ha="center",
                va="center",
                color="white" if cm_norm[i, j] > 0.5 else "black"
            )

    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    return cm, cm_norm


class PartialFineTuneTrainer(Trainer):
    """
    SimCLR partial fine-tuning trainer.
    - Loads pretrained SimCLR encoder
    - Freezes the full encoder, then unfreezes ONLY layer4
    - Trains layer4 + classifier
    - Uses validation set + best checkpoint + cosine LR scheduler
    - Saves files with label_pct in the name to avoid overwriting
    """

    def __init__(self, config, checkpoint_dir=None, device=None):
        super().__init__(
            config, output_dir=checkpoint_dir, device=device, pretrain=False
        )
        self.net = self._init_model()
        self.optimizer = self._init_optimizer()
        self.scheduler = self._init_scheduler()
        self.criterion = nn.CrossEntropyLoss()

    @staticmethod
    def _build_model(cfg):
        return SimCLRModel(cfg)

    def _label_pct_name(self):
        return int(self.config.data.percent * 100)

    def _init_model(self):
        net = self._build_model(self.config).to(self.device)

        ckpt_path = "./checkpoints/simclr_pretrain3_cifar.pth"
        ckpt = torch.load(
            ckpt_path,
            map_location=self.device,
            weights_only=True,
        )

        net.encoder.load_state_dict(ckpt["encoder"])

        for p in net.encoder.parameters():
            p.requires_grad = False

        for p in net.encoder.layer4.parameters():
            p.requires_grad = True

        net.linear_eval = True
        return net

    def _init_optimizer(self):
        encoder_lr = self.config.train.lr * 0.1
        classifier_lr = self.config.train.lr

        trainable_encoder_params = [
            p for p in self.net.encoder.parameters() if p.requires_grad
        ]

        return torch.optim.SGD(
            [
                {"params": trainable_encoder_params, "lr": encoder_lr},
                {"params": self.net.classifier.parameters(), "lr": classifier_lr},
            ],
            momentum=self.config.optimizer.momentum,
            weight_decay=self.config.optimizer.weight_decay,
        )

    def _init_scheduler(self):
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.train.n_epochs,
        )

    @torch.no_grad()
    def _run_val(self):
        self.net.eval()

        loss_meter = AverageMeter()
        correct, total = 0, 0

        for x, y in self.val_loader:
            x = x.to(self.device)
            y = y.to(self.device)

            _, logits = self.net(x)
            loss = self.criterion(logits, y)

            loss_meter.update(loss.item())
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)

        return 100.0 * correct / total, loss_meter.avg

    def train(self):
        start = time.time()

        label_pct = self._label_pct_name()
        dataset_name = self.dataset.lower()

        epoch_losses = []
        epoch_accs = []
        epoch_val_accs = []
        epoch_val_losses = []
        epoch_lrs = []

        best_val_acc = 0.0
        best_epoch = -1

        best_ckpt_path = (
            f"{self.output_dir}/simclr_partialft_{dataset_name}_{label_pct}pct_best.pth"
        )

        for epoch in range(self.config.train.n_epochs):
            self.net.train()

            for name, m in self.net.encoder.named_modules():
                if not name.startswith("layer4") and isinstance(
                    m, (nn.BatchNorm2d, nn.BatchNorm1d)
                ):
                    m.eval()

            loss_meter = AverageMeter()
            acc_meter = AverageMeter()

            for images, labels in self.train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                _, logits = self.net(images)
                loss = self.criterion(logits, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                loss_meter.update(loss.item())
                acc_meter.update(
                    (logits.argmax(1) == labels).float().mean().item() * 100
                )

            train_loss = float(loss_meter.avg)
            train_acc = float(acc_meter.avg)

            val_acc, val_loss = self._run_val()

            epoch_losses.append(train_loss)
            epoch_accs.append(train_acc)
            epoch_val_accs.append(float(val_acc))
            epoch_val_losses.append(float(val_loss))
            epoch_lrs.append(self.optimizer.param_groups[0]["lr"])

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                torch.save(self.net.state_dict(), best_ckpt_path)

            print(
                f"Epoch [{epoch:3d}/{self.config.train.n_epochs}]\t"
                f"Loss {train_loss:.3f}\tACC {train_acc:.2f}\t"
                f"val_loss {val_loss:.3f}\tval_acc {val_acc:.2f}\t"
                f"lr_enc {self.optimizer.param_groups[0]['lr']:.5f}\t"
                f"time {time.time() - start:.1f}s"
            )

            self.scheduler.step()

        total_time = time.time() - start

        print(
            f"Completed in {total_time:.3f}s | "
            f"best val_acc {best_val_acc:.2f} @ epoch {best_epoch}"
        )

        last_path = (
            f"{self.output_dir}/simclr_partialft_{dataset_name}_{label_pct}pct_last.pth"
        )
        torch.save(self.net.state_dict(), last_path)

        print(f"Last model    → {last_path}")
        print(f"Best val ckpt → {best_ckpt_path}")

        metrics = {
            "epoch_losses": epoch_losses,
            "epoch_accs": epoch_accs,
            "epoch_val_losses": epoch_val_losses,
            "epoch_val_accs": epoch_val_accs,
            "epoch_lrs": epoch_lrs,
            "best_val_acc": best_val_acc,
            "best_epoch": best_epoch,
            "total_time_s": round(total_time, 3),
            "n_epochs": self.config.train.n_epochs,
            "dataset": self.dataset,
            "label_pct": label_pct,
            "experiment": "simclr_partial_finetune_layer4",
            "encoder_lr": self.optimizer.param_groups[0]["lr"],
            "classifier_lr": self.optimizer.param_groups[1]["lr"],
            "scheduler": "CosineAnnealingLR",
            "trainable_encoder": "layer4_only",
        }

        metrics_path = (
            f"{self.output_dir}/partialft_metrics_{dataset_name}_{label_pct}pct.json"
        )

        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        print(f"Metrics saved to {metrics_path}")

    def test(self):
        CIFAR10_CLASSES = [
            "airplane", "automobile", "bird", "cat", "deer",
            "dog", "frog", "horse", "ship", "truck",
        ]

        label_pct = self._label_pct_name()
        dataset_name = self.dataset.lower()

        best_ckpt_path = (
            f"{self.output_dir}/simclr_partialft_{dataset_name}_{label_pct}pct_best.pth"
        )

        state = torch.load(
            best_ckpt_path,
            map_location=self.device,
            weights_only=True,
        )
        self.net.load_state_dict(state)

        print(f"[test] Loaded best checkpoint: {best_ckpt_path}")

        self.net.eval()

        loss_meter = AverageMeter()
        correct = 0
        total = 0

        num_classes = 10
        correct_per_class = [0] * num_classes
        total_per_class = [0] * num_classes

        with torch.no_grad():
            for images, labels in self.test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                _, logits = self.net(images)
                loss = self.criterion(logits, labels)

                loss_meter.update(loss.item())

                preds = logits.argmax(1)

                correct += (preds == labels).sum().item()
                total += labels.size(0)

                for c in range(num_classes):
                    mask = labels == c
                    total_per_class[c] += mask.sum().item()
                    correct_per_class[c] += (preds[mask] == labels[mask]).sum().item()

        acc = 100.0 * correct / total

        print(
            f"FINAL TEST RESULTS\t"
            f"Loss {loss_meter.avg:.3f}\t"
            f"Average Accuracy {acc:.2f}"
        )

        per_class_acc = {}

        print("\nPer-class accuracy:")
        for c in range(num_classes):
            if total_per_class[c] > 0:
                class_acc = 100.0 * correct_per_class[c] / total_per_class[c]
                per_class_acc[CIFAR10_CLASSES[c]] = round(class_acc, 2)
                print(f"{CIFAR10_CLASSES[c]}: {class_acc:.2f}%")
            else:
                per_class_acc[CIFAR10_CLASSES[c]] = None
                print(f"{CIFAR10_CLASSES[c]}: No samples")

        cm_path = (
            f"{self.output_dir}/cm_simclr_partialft_{dataset_name}_{label_pct}pct.png"
        )

        cm, cm_norm = evaluate_confusion_matrix(
            self.net,
            self.test_loader,
            self.device,
            save_path=cm_path,
        )

        print(f"Confusion matrix saved to {cm_path}")

        test_results = {
            "avg_accuracy": round(acc, 2),
            "avg_loss": round(loss_meter.avg, 4),
            "per_class_accuracy": per_class_acc,
            "experiment": "simclr_partial_finetune_layer4",
            "label_pct": label_pct,
            "trainable_encoder": "layer4_only",
            "evaluated_checkpoint": best_ckpt_path,
            "confusion_matrix_path": cm_path,
        }

        results_path = (
            f"{self.output_dir}/partialft_test_results_{dataset_name}_{label_pct}pct.json"
        )

        with open(results_path, "w") as f:
            json.dump(test_results, f, indent=2)

        print(f"Partial fine-tune test results saved to {results_path}")