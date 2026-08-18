#!/usr/bin/env python3
"""DEIMv2-OBB 模型对比研究核心工具类。

加载 DEIMv2-OBB checkpoint，对图像目录批量推理，将检测结果保存为
per-image DOTA 格式（8 坐标 + 类别名 + 置信度），供后续模型比对和
可视化分析使用。原先散落在 ``test/tool_deimv2_obb_infer.py`` 等脚本中
的公共管线收拢于此，各分析脚本作为消费者复用。

Each image gets a .txt file with one detection per line::

    x1 y1 x2 y2 x3 y3 x4 y4 class_name confidence

Programmatic single model::

    from tools.compare.core import infer_obb_and_export
    infer_obb_and_export(img_dir, ckpt, config, output_dir, classes_txt)

Batch over model specs::

    from tools.compare.core import OBBModelSpec, run_model_specs
    specs = [OBBModelSpec(name="rep0", config="...", ckpt="...",
                          output_dir="./test/data/outputs/.../rep0")]
    run_model_specs(specs, img_dir=img_dir, classes_txt=classes_txt)

CLI 单模型入口见 ``tools/compare/run_infer.py``。
"""

import os
import sys
from dataclasses import dataclass, field

# tools/compare/ → two dirnames up is the repo root (the engine package root).
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
from torchvision import transforms

from engine.backbone import DINOv3STAsResAtten
from engine.deim import HybridEncoder, DEIMTransformer
from engine.deim.postprocessor import PostProcessor
from engine.data.transforms import ConvertPILImage
from tools.model_compare.obb_utils import deimv2_obb_outputs_to_dota, visualize_dota_predictions
from tools.model_compare.obb_inference_geometry import rescale_obb_to_original

#: training YAML 中可选的 backbone 段名（按此顺序探测）。
_BACKBONE_SECTIONS = ("DINOv3STAsResAtten", "DINOv3STAs", "HGNetv2")


class DEIMv2OBB(nn.Module):
    """DEIMv2-OBB model wrapper for inference."""

    def __init__(self, config: dict, device: str = "cpu"):
        super().__init__()
        backbone_kind = next(k for k in _BACKBONE_SECTIONS if k in config)
        backbone_cls = {
            "DINOv3STAsResAtten": DINOv3STAsResAtten,
        }.get(backbone_kind)
        # HGNetv2/DINOv3STAs 变体按需补充；当前 OBB 研究矩阵仅 ResAtten。
        if backbone_cls is None:
            raise ValueError(
                f"unsupported backbone section {backbone_kind!r} for OBB compare; "
                f"supported: {_BACKBONE_SECTIONS[:1]}"
            )
        self.backbone = backbone_cls(**config[backbone_kind]).to(device)
        self.encoder = HybridEncoder(**config["HybridEncoder"]).to(device)
        self.decoder = DEIMTransformer(**config["DEIMTransformer"]).to(device)
        self.postprocessor = PostProcessor(**config["PostProcessor"]).to(device)

    def forward(self, x, orig_target_sizes):
        x1 = self.backbone(x)
        x2 = self.encoder(x1)
        x3 = self.decoder(x2)
        x4 = self.postprocessor(x3, orig_target_sizes)
        return x4


def load_checkpoint(model: nn.Module, ckpt_path: str, map_location: str = "cpu"):
    """Load checkpoint with standard key remapping (ema.module → model)."""
    state = torch.load(ckpt_path, weights_only=True, map_location=map_location)
    if "ema" in state:
        state = state["ema"]
        if "module" in state:
            state = state["module"]
    elif "model" in state:
        state = state["model"]
    new_state = {}
    for k, v in state.items():
        new_state[k.replace("module.", "")] = v
    model.load_state_dict(new_state, strict=False)
    print(f"Loaded checkpoint from {ckpt_path}")
    return model


def _peek_checkpoint(ckpt_path: str) -> dict:
    """Load a checkpoint on CPU for head-shape probing (patchable seam)."""
    return torch.load(ckpt_path, weights_only=True, map_location="cpu")


def resolve_num_classes(ckpt: dict, num_classes_from_txt: int) -> int:
    """Head size comes from the checkpoint when it carries one.

    Legacy configs may declare a different class count than the dataset
    (synthetic-ellipse: 3 classes vs config's 15) — the checkpoint is the
    trained artifact, so its ``dec_score_head`` shape is ground truth and
    the model is built to match instead of erroring in load_state_dict.
    """
    model_state = ckpt.get("model", ckpt)
    if "ema" in ckpt and isinstance(ckpt["ema"], dict):
        model_state = ckpt["ema"].get("module", ckpt["ema"])
    for probe in (
        "decoder.dec_score_head.0.weight",
        "dec_score_head.0.weight",
    ):
        layer = model_state.get(probe)
        if layer is not None and layer.ndim == 2:
            return int(layer.shape[0])
    return num_classes_from_txt


def build_model_cfg(config: dict, ckpt: str, num_classes: int, max_det: int) -> dict:
    """Assemble the inference model config from a resolved training-YAML dict.

    ``num_classes``/``max_det`` are injected into the decoder and
    postprocessor; the backbone ``weights_path`` is overridden by ``ckpt``.
    """
    backbone_kind = next(k for k in _BACKBONE_SECTIONS if k in config)
    return {
        backbone_kind: {**config[backbone_kind], "weights_path": ckpt},
        "HybridEncoder": config["HybridEncoder"],
        "DEIMTransformer": {
            **config["DEIMTransformer"],
            "num_classes": num_classes,
            "num_queries": max_det,
        },
        "PostProcessor": {
            **config["PostProcessor"],
            "num_classes": num_classes,
            "num_top_queries": max_det,
        },
    }


def infer_obb_and_export(
    img_dir: str,
    ckpt: str,
    config: str,
    output_dir: str,
    classes_txt: str,
    imgsz: tuple = (640, 640),
    max_det: int = 300,
    score_threshold: float = 0.0,
    device: str = "cuda:0",
    vis_dir: str = None,
):
    """Run OBB inference on all images and export to DOTA per-image format.

    num_classes is auto-detected from classes_txt. ``score_threshold``
    filters only the visualization; the DOTA export retains all detections.
    """
    os.makedirs(output_dir, exist_ok=True)

    with open(classes_txt, "r") as f:
        class_names = [line.strip() for line in f if line.strip()]
    num_classes = len(class_names)
    labels_map = {i: name for i, name in enumerate(class_names)}
    print(f"Classes ({num_classes}): {labels_map}")

    from engine.core.yaml_utils import load_config

    ckpt_state = _peek_checkpoint(ckpt)
    model_classes = resolve_num_classes(ckpt_state, num_classes)
    if model_classes != num_classes:
        print(
            f"[warn] checkpoint heads carry {model_classes} classes while "
            f"{classes_txt} declares {num_classes} — building to the "
            f"checkpoint; detection labels beyond the txt list will be "
            f"unresolvable."
        )
    model_cfg = build_model_cfg(
        load_config(config), ckpt, model_classes, max_det
    )

    model = DEIMv2OBB(model_cfg, device)
    # map straight to the target device — CPU staging of large checkpoints
    # can hit ENOMEM on memory-tight hosts.
    load_checkpoint(model, ckpt, map_location=device)
    model.eval()

    transform = transforms.Compose(
        [
            transforms.Resize(imgsz),
            ConvertPILImage(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    img_list = [
        f
        for f in os.listdir(img_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    ]
    print(f"Found {len(img_list)} images")

    outputs_dict = {}

    for img_name in tqdm(img_list, desc="Inference"):
        img_path = os.path.join(img_dir, img_name)
        try:
            image = Image.open(img_path).convert("RGB")
            orig_w, orig_h = image.size
            input_tensor = transform(image).unsqueeze(0).to(device)
            dst_sz = torch.tensor([imgsz[1], imgsz[0]], device=device)[None, :]
            src_sz = torch.tensor([orig_h, orig_w], device=device)

            with torch.no_grad():
                results = model(input_tensor, orig_target_sizes=dst_sz)

            for output in results:
                labels = output["labels"].cpu().numpy()
                scores = output["scores"].cpu().numpy()

                if len(labels) == 0:
                    continue

                boxes = rescale_obb_to_original(
                    output["boxes"].cpu(),
                    original_size=(orig_h, orig_w),
                    inference_size=imgsz,
                ).numpy()

                outputs_dict[img_name] = {
                    "labels": labels.tolist(),
                    "boxes": boxes.tolist(),
                    "scores": scores.tolist(),
                }

        except Exception as e:
            print(f"Error processing {img_name}: {e}")
            continue

    print(f"\nInference completed: {len(outputs_dict)} images with detections")
    print(f"\nExporting to DOTA format → {output_dir}")
    deimv2_obb_outputs_to_dota(outputs_dict, output_dir, labels_map)

    if vis_dir is not None:
        print(f"\nVisualizing → {vis_dir}")
        visualize_dota_predictions(
            img_dir=img_dir,
            dota_dir=output_dir,
            vis_dir=vis_dir,
            score_threshold=score_threshold,
        )

    print(f"\nDone. Predictions saved to {output_dir}/")


@dataclass
class OBBModelSpec:
    """一个待对比的模型变体：config + checkpoint + 输出目录。"""

    name: str
    config: str
    ckpt: str
    output_dir: str
    vis_dir: str | None = None
    infer_flag: bool = True
    extra: dict = field(default_factory=dict)


def run_model_specs(
    specs: list[OBBModelSpec],
    img_dir: str,
    classes_txt: str,
    imgsz: tuple = (640, 640),
    max_det: int = 300,
    score_threshold: float = 0.0,
    device: str = "cuda:0",
) -> list[str]:
    """按 spec 列表批量推理导出，返回实际执行的 spec 名称。"""
    done: list[str] = []
    for spec in specs:
        if not spec.infer_flag:
            print(f"[skip] {spec.name} (infer_flag=False)")
            continue
        print(f"\n===== {spec.name} =====")
        infer_obb_and_export(
            img_dir=img_dir,
            ckpt=spec.ckpt,
            config=spec.config,
            output_dir=spec.output_dir,
            classes_txt=classes_txt,
            imgsz=spec.extra.get("imgsz", imgsz),
            max_det=spec.extra.get("max_det", max_det),
            score_threshold=spec.extra.get("score_threshold", score_threshold),
            device=spec.extra.get("device", device),
            vis_dir=spec.vis_dir,
        )
        done.append(spec.name)
    return done
