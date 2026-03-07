"""
DEIMv2: Real-Time Object Detection Meets DINOv3
Copyright (c) 2025 The DEIMv2 Authors. All Rights Reserved.
---------------------------------------------------------------------------------
DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright (c) 2023 lyuwenyu. All Rights Reserved.
"""

import os

os.environ.update(
    {
        "NCCL_DEBUG": "INFO",
        "MKL_THREADING_LAYER": "INTEL",
        "MKL_SERVICE_FORCE_INTEL": "1",
        "CUDA_VISIBLE_DEVICES": "0,1",
        # "PYTORCH_NVML_BASED_CUDA_CHECK": "1",
        "CUDA_LAUNCH_BLOCKING": "1",
    }
)

import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import argparse

from engine.misc import dist_utils
from engine.core import YAMLConfig, yaml_utils
from engine.solver import TASKS

debug = True

if debug:
    import torch

    print(f"PyTorch CUDA version: {torch.__version__}")
    print(f"GPU available: {torch.cuda.is_available()}")
    print(f"Device count: {torch.cuda.device_count()}")

    def custom_repr(self):
        return f"{{Tensor:{tuple(self.shape)}}} {original_repr(self)}"

    original_repr = torch.Tensor.__repr__
    torch.Tensor.__repr__ = custom_repr


def print_training_config(cfg: YAMLConfig) -> None:
    """Print structured training configuration"""
    print("\n" + "="*60)
    print("Training Configuration")
    print("="*60)

    # Dataset info
    print(f"\n📊 Dataset:")
    print(f"  Classes: {cfg.yaml_cfg.get('num_classes', 'N/A')}")
    train_ds = cfg.yaml_cfg.get('train_dataloader', {}).get('dataset', {})
    val_ds = cfg.yaml_cfg.get('val_dataloader', {}).get('dataset', {})
    if isinstance(train_ds, dict):
        print(f"  Train images: {train_ds.get('img_folder', 'N/A')}")
        print(f"  Val images: {val_ds.get('img_folder', 'N/A')}")

    # Batch size
    print(f"\n📦 Batch Size:")
    train_bs = cfg.yaml_cfg.get('train_dataloader', {}).get('total_batch_size', 'N/A')
    val_bs = cfg.yaml_cfg.get('val_dataloader', {}).get('total_batch_size', 'N/A')
    print(f"  Train: {train_bs}")
    print(f"  Val: {val_bs}")

    # Model architecture
    print(f"\n🏗️ Model:")
    if 'DINOv3STAs' in cfg.yaml_cfg:
        print(f"  Backbone: DINOv3 ({cfg.yaml_cfg['DINOv3STAs'].get('name', 'N/A')})")
    elif 'HGNetv2' in cfg.yaml_cfg:
        print(f"  Backbone: HGNetv2 ({cfg.yaml_cfg['HGNetv2'].get('name', 'N/A')})")
    print(f"  Decoder: {cfg.yaml_cfg.get('DEIMTransformer', {}).get('num_layers', 'N/A')} layers, {cfg.yaml_cfg.get('DEIMTransformer', {}).get('num_queries', 'N/A')} queries")

    # Training schedule
    print(f"\n📅 Schedule:")
    print(f"  Epochs: {cfg.epoches}")
    print(f"  Warmup iters: {cfg.warmup_iter}")
    print(f"  Flat epochs: {cfg.flat_epoch}")
    print(f"  No-aug epochs: {cfg.no_aug_epoch}")

    # Optimizer
    print(f"\n⚙️ Optimizer:")
    opt_cfg = cfg.yaml_cfg.get('optimizer', {})
    print(f"  Type: {opt_cfg.get('type', 'N/A')}")
    print(f"  Base LR: {opt_cfg.get('lr', 'N/A')}")
    print(f"  Weight decay: {opt_cfg.get('weight_decay', 'N/A')}")

    # EMA & AMP
    print(f"\n🔧 Techniques:")
    print(f"  EMA: {cfg.use_ema} (decay={cfg.ema_decay})")
    print(f"  AMP: {cfg.use_amp}")
    print(f"  SyncBN: {cfg.sync_bn}")

    # Output
    print(f"\n💾 Output:")
    print(f"  Directory: {cfg.output_dir}")
    print(f"  Checkpoint freq: {cfg.checkpoint_freq}")
    print(f"  Device: {cfg.device}")

    print("="*60 + "\n")


def main(
    args,
) -> None:
    """main"""
    dist_utils.setup_distributed(args.print_rank, args.print_method, seed=args.seed)

    assert not all(
        [args.tuning, args.resume]
    ), "Only support from_scrach or resume or tuning at one time"

    update_dict = yaml_utils.parse_cli(args.update)
    update_dict.update(
        {
            k: v
            for k, v in args.__dict__.items()
            if k
            not in [
                "update",
            ]
            and v is not None
        }
    )

    cfg = YAMLConfig(args.config, **update_dict)

    if args.resume or args.tuning:
        if "HGNetv2" in cfg.yaml_cfg:
            cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    print_training_config(cfg)

    solver = TASKS[cfg.yaml_cfg["task"]](cfg)

    if args.test_only:
        solver.val()
    else:
        solver.fit()

    dist_utils.cleanup()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # priority 0
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="configs/deimv2/deimv2_dinov3_x_custom.yml",
    )
    parser.add_argument("-r", "--resume", type=str, help="resume from checkpoint")
    parser.add_argument("-t", "--tuning", type=str, help="tuning from checkpoint")
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default="cuda:0",
        help="device",
    )
    parser.add_argument("--seed", type=int, default=0, help="exp reproducibility")
    parser.add_argument(
        "--use-amp",
        action="store_true",
        default=False,
        help="auto mixed precision training",
    )
    parser.add_argument("--output-dir", type=str, help="output directoy")
    parser.add_argument("--summary-dir", type=str, help="tensorboard summry")
    parser.add_argument(
        "--test-only",
        action="store_true",
        default=False,
    )

    # priority 1
    parser.add_argument("-u", "--update", nargs="+", help="update yaml config")

    # env
    parser.add_argument(
        "--print-method", type=str, default="builtin", help="print method"
    )
    parser.add_argument("--print-rank", type=int, default=0, help="print rank id")

    parser.add_argument("--local-rank", type=int, help="local rank id")
    args = parser.parse_args()

    # model test part
    args.config = "./configs/custom/deimv2_dinov3_b_custom.yml"
    args.test_only = False
    # args.resume = (
    #     "./outputs/deimv2_dinov3_x_custom/best_stg2_freeze_1109_e186_mAP67.pth"
    # )
    args.device = "cuda:0"

    main(args)
