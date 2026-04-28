import argparse
import os
import yaml
from pdb import set_trace
import torch
from train.train_simclr import SimCLRTrainer
from train.train_linear import LinearEvalTrainer
from train.train_finetune import FineTuneTrainer
from train.train_finetune_partial import PartialFineTuneTrainer
from utils.data_utils import Config


def _run_visualization(checkpoint_dir: str, out_dir: str, modes: list) -> None:
    """
    Call the relevant plot_* functions from visualization/plot_metrics.py.
    `modes` is a subset of ["pretrain", "linear", "test", "combined"].
    Silently skips any plot whose source JSON is missing.
    """
    from visualization.plot_metrics import (
        plot_pretrain,
        plot_linear,
        plot_test_accuracy,
        plot_combined_loss,
    )

    pretrain_json = os.path.join(checkpoint_dir, "pretrain_metrics.json")
    linear_json   = os.path.join(checkpoint_dir, "linear_metrics.json")
    test_json     = os.path.join(checkpoint_dir, "test_results.json")
    full_finetune_json = os.path.join(checkpoint_dir, "full_finetune_metrics.json")
    full_finetune_test_json = os.path.join(checkpoint_dir, "full_finetune_test_results.json")

    partial_finetune_json = os.path.join(checkpoint_dir, "partial_finetune_metrics.json")
    partial_finetune_test_json = os.path.join(checkpoint_dir, "partial_finetune_test_results.json")
    
    print("\n── Generating visualizations ──")
    generated = []

    if "pretrain" in modes and os.path.exists(pretrain_json):
        generated.append(plot_pretrain(pretrain_json, out_dir))

    if "linear" in modes and os.path.exists(linear_json):
        generated.append(plot_linear(linear_json, out_dir))

    if "test" in modes and os.path.exists(test_json):
        generated.append(plot_test_accuracy(test_json, out_dir))

    if "combined" in modes and os.path.exists(pretrain_json) and os.path.exists(linear_json):
        generated.append(plot_combined_loss(pretrain_json, linear_json, out_dir))
    
    if "finetune_full" in modes and os.path.exists(full_finetune_json):
        generated.append(plot_linear(full_finetune_json, out_dir))

    if "finetune_full_test" in modes and os.path.exists(full_finetune_test_json):
        generated.append(plot_test_accuracy(full_finetune_test_json, out_dir))

    if "finetune_partial" in modes and os.path.exists(partial_finetune_json):
        generated.append(plot_linear(partial_finetune_json, out_dir))

    if "finetune_partial_test" in modes and os.path.exists(partial_finetune_test_json):
        generated.append(plot_test_accuracy(partial_finetune_test_json, out_dir))

    if generated:
        print(f"Plots saved to: {os.path.abspath(out_dir)}/")
    else:
        print("No metric files found — nothing plotted.")


def _resolve_device(requested: str) -> torch.device:
    """
    Convert CLI device choice to an available torch.device.
    Falls back safely on macOS/CPU-only installs.
    """
    requested = (requested or "auto").lower()

    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if requested == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        print("CUDA requested but not available; falling back to CPU.")
        return torch.device("cpu")

    if requested == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        print("MPS requested but not available; falling back to CPU.")
        return torch.device("cpu")

    return torch.device("cpu")


class SIMCLRRunner(object):
    def __init__(self, args):
        self.mode = args.mode
        self.device = _resolve_device(args.device)
        self.checkpoint_dir = args.checkpoint_dir
        self.test = args.test_data
        self.generate_viz = args.generate_visualization
        self.viz_out = args.viz_out

        if args.config:
            with open(args.config, "r") as f:
                self.config = yaml.safe_load(f)
                self.config = Config(self.config)
        else:
            self.config = {}

        # CLI --imbalanced-split overrides whatever is in the config file
        if args.imbalanced_split is not None:
            self.config.data.imbalanced_split = args.imbalanced_split

    def run(self):
        if self.mode == "pretrain":
            return self._train_simclr()
        elif self.mode == "linear":
            return self._run_linear_eval()
        elif self.mode == "finetune_full":
            return self._run_finetune_full()
        elif self.mode == "finetune_partial":
            return self._run_finetune_partial()

    def _train_simclr(self):
        print("Running SimCLR pretraining...")
        trainer = SimCLRTrainer(
            self.config, checkpoint_dir=self.checkpoint_dir, device=self.device
        )
        trainer.train()
        if self.generate_viz:
            _run_visualization(
                checkpoint_dir=self.checkpoint_dir,
                out_dir=self.viz_out,
                modes=["pretrain"],
            )

    def _run_linear_eval(self):
        print("Running linear evaluation...")
        trainer = LinearEvalTrainer(
            config=self.config,
            checkpoint_dir=self.checkpoint_dir,
            device=self.device,
        )
        trainer.train()
        if self.test:
            trainer.test()
        if self.generate_viz:
            _run_visualization(
                checkpoint_dir=self.checkpoint_dir,
                out_dir=self.viz_out,
                modes=["linear", "test", "combined"],
            )
    def _run_finetune_full(self):
        print("Running FULL fine-tuning...")

        trainer = FineTuneTrainer(
            config=self.config,
            checkpoint_dir=self.checkpoint_dir,
            device=self.device,
        )

        trainer.train()

        if self.test:
            trainer.test()

        if self.generate_viz:
            _run_visualization(
                checkpoint_dir=self.checkpoint_dir,
                out_dir=self.viz_out,
                modes=["finetune_full", "finetune_full_test"],
            )
    def _run_finetune_partial(self):
        print("Running PARTIAL fine-tuning...")

        trainer = PartialFineTuneTrainer(
            config=self.config,
            checkpoint_dir=self.checkpoint_dir,
            device=self.device,
        )

        trainer.train()

        if self.test:
            trainer.test()

        if self.generate_viz:
            _run_visualization(
                checkpoint_dir=self.checkpoint_dir,
                out_dir=self.viz_out,
                modes=["finetune_partial", "finetune_partial_test"],
            )

def get_args():
    parser = argparse.ArgumentParser(description="SimCLR Training Script")
    parser.add_argument(
        "--mode",
        "-m",
        type=str,
        required=True,
        choices=["pretrain", "linear", "finetune_full", "finetune_partial"],
        help="Which pipeline to run",
    )

    parser.add_argument(
        "--config", "-c", type=str, default=None, help="Path to YAML config file"
    )

    parser.add_argument(
        "--device",
        "-d",
        type=str,
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Device to use",
    )

    parser.add_argument(
        "--checkpoint-dir",
        "-x",
        type=str,
        default="./checkpoints",
        help="Where to save model checkpoints",
    )
    
    parser.add_argument(
        "--test-data",
        "-t",
        action="store_true",
        default=False,
        help="Specify whether to use test data",
    )

    parser.add_argument(
        "--generate-visualization",
        "--viz",
        action="store_true",
        default=False,
        dest="generate_visualization",
        help="Generate training curve plots after training completes",
    )

    parser.add_argument(
        "--viz-out",
        type=str,
        default="plots",
        help="Output directory for visualization PNGs (default: plots/)",
    )

    parser.add_argument(
        "--imbalanced-split",
        "--imb",
        type=str,
        default=None,
        dest="imbalanced_split",
        choices=["difficulty", "difficulty_1pct", "lt_if10", "lt_if10_1pct", "lt_if10_25pct", "lt_if50"],
        help=(
            "Use a pre-computed imbalanced split for linear eval. "
            "Choices: difficulty | difficulty_1pct | lt_if10 | lt_if10_1pct. "
            "Overrides config data.imbalanced_split."
        ),
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    Runner = SIMCLRRunner(args)
    Runner.run()
