"""OBB 角度域纪律审计测试（spec §9.1, §9.6, §11.5）。

grep 断言：
  1. physical_rad_to_norm / norm_to_physical_rad 只在 decoder 私有边界使用
     （deim_decoder, denoising, dfine_decoder, obb_angle_contract 定义处）。
     criterion 的非周期消融路径获豁免（用户决定保留）。
  2. 其他公开模块（matcher, postprocessor, obb_geometry, yolo_obb_loss, obb_ops,
     dfine_utils, chamfer_cost, obb_transforms）禁止引用上述 decoder 私有函数。
  3. seam 变换字面量 pi/4（作为角度平移）只出现在 obb_angle_contract 的
     physical_rad_to_loss_rad 定义中，不渗入 decoder/denoising/geometry 等公开边界。

Run:
    python -m pytest test/test_obb_domain_audit.py -q
"""

import os
import re
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

DEIM_DIR = os.path.join(ROOT, "engine", "deim")
TRANSFORMS_DIR = os.path.join(ROOT, "engine", "data", "transforms")

PRIVATE_FN_RE = re.compile(r"\b(physical_rad_to_norm|norm_to_physical_rad)\b")


def _read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


# criterion 的非周期消融路径获豁免（用户决定保留 physical_rad_to_norm）
ALLOWED_PRIVATE_USERS = {"deim_decoder.py", "denoising.py", "dfine_decoder.py", "obb_angle_contract.py", "deim_criterion.py"}

AUDITED_PUBLIC_MODULES = [
    "matcher.py",
    "postprocessor.py",
    "obb_geometry.py",
    "yolo_obb_loss.py",
    "obb_ops.py",
    "dfine_utils.py",
    "chamfer_cost.py",
]


@pytest.mark.parametrize("modname", AUDITED_PUBLIC_MODULES)
def test_public_modules_do_not_import_decoder_private_angle_fns(modname):
    """公开模块禁止引用 physical_rad_to_norm / norm_to_physical_rad。"""
    path = os.path.join(DEIM_DIR, modname)
    src = _read(path)
    hits = PRIVATE_FN_RE.findall(src)
    assert not hits, (
        f"{modname} 引用了 decoder 私有角度函数 {set(hits)}；"
        f"公开模块只允许物理域与 loss 规范域辅助函数（spec §9.1）"
    )


def test_transforms_do_not_import_decoder_private_angle_fns():
    """obb_transforms 禁止引用 decoder 私有角度函数。"""
    path = os.path.join(TRANSFORMS_DIR, "obb_transforms.py")
    if not os.path.exists(path):
        pytest.skip("obb_transforms.py 不存在")
    src = _read(path)
    hits = PRIVATE_FN_RE.findall(src)
    assert not hits, f"obb_transforms 引用了 decoder 私有角度函数 {set(hits)}"


# ---------------------------------------------------------------------------
# seam 变换字面量纪律：pi/4 平移只允许在 physical_rad_to_loss_rad 定义中
# ---------------------------------------------------------------------------

# 匹配「pi/4 作为角度平移变换」的模式：remainder(... + pi/4 ...) 或 (... + pi/4)/pi 等
SEAM_TRANSFORM_RE = re.compile(
    r"(remainder\s*\([^)]*?\+\s*(?:torch\.|math\.)?pi\s*/\s*4"
    r"|\+\s*(?:torch\.|math\.)?pi\s*/\s*4\s*\)\s*/\s*(?:torch\.|math\.)?pi)"
)


@pytest.mark.parametrize(
    "modname",
    ["deim_decoder.py", "denoising.py", "dfine_decoder.py", "postprocessor.py",
     "matcher.py", "obb_geometry.py", "yolo_obb_loss.py", "obb_ops.py",
     "dfine_utils.py", "chamfer_cost.py"],
)
def test_no_seam_transform_in_boundary_modules(modname):
    """decoder/denoising/geometry/postprocessor/matcher 等边界模块不含 pi/4 seam 平移变换。

    obb_angle_contract.physical_rad_to_loss_rad 是唯一允许的位置（loss 内部规范域）。
    """
    path = os.path.join(DEIM_DIR, modname)
    src = _read(path)
    hits = SEAM_TRANSFORM_RE.findall(src)
    assert not hits, (
        f"{modname} 含 pi/4 seam 平移变换 {hits}；"
        f"seam 只允许在 obb_angle_contract.physical_rad_to_loss_rad（spec §9.6）"
    )


def test_criterion_nonperiodic_uses_proportional_norm_not_seam():
    """criterion 非周期路径用 physical_rad_to_norm (等比 θ/π)，不含 (θ+π/4)/π seam。"""
    src = _read(os.path.join(DEIM_DIR, "deim_criterion.py"))
    # 禁止 (θ + π/4)/π seam 公式残留
    assert not re.search(r"\+\s*(?:torch\.|math\.)?pi\s*/\s*4\s*\)\s*/\s*(?:torch\.|math\.)?pi", src), (
        "criterion 仍含 (θ+π/4)/π seam 公式；非周期路径应改用 physical_rad_to_norm（等比）"
    )
