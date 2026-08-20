#!/usr/bin/env python3
"""DEIMv2-OBB 模型对比研究工具包。

``core`` 提供可复用的推理-导出管线；各分析脚本（差异分析、分布对比、
decoder 逐层调试等）消费其产出的 per-image DOTA 结果目录。
"""

from tools.compare.core import (  # noqa: F401
    DEIMv2OBB,
    OBBModelSpec,
    build_model_cfg,
    infer_obb_and_export,
    load_checkpoint,
    run_model_specs,
)

__all__ = [
    "DEIMv2OBB",
    "OBBModelSpec",
    "build_model_cfg",
    "infer_obb_and_export",
    "load_checkpoint",
    "run_model_specs",
]
