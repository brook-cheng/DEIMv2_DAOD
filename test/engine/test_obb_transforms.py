"""逐个测试每个 OBB 数据增强方法，生成 before/after 对比图并验证框不丢失。

用法：
    python test/test_obb_transforms.py                    # 全部测试
    python test/test_obb_transforms.py --only OBBFlip     # 单个测试
    python test/test_obb_transforms.py --only MixUp        # MixUp 增强
    python test/test_obb_transforms.py --only CopyBlend    # CopyBlend 增强
"""

import os, sys, argparse, math, random, copy
import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from engine.deim.obb_geometry import xywhr_to_xyxyxyxy

COLORS = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (255, 128, 0),
    (128, 255, 0),
]
OUTPUT_DIR = os.path.join(ROOT, "test", "outputs", "obb_aug_test")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def make_test_sample():
    """构造测试用假图 + 4 个像素坐标 OBB。"""
    W, H = 800, 600
    img = Image.new("RGB", (W, H), (60, 60, 60))
    draw = ImageDraw.Draw(img)
    for _ in range(50):
        x, y = random.randint(0, W), random.randint(0, H)
        r = random.randint(5, 30)
        c = random.randint(100, 200)
        draw.ellipse([x - r, y - r, x + r, y + r], fill=(c, c + 20, c - 20))
    draw.line([(W // 2, 0), (W // 2, H)], fill=(255, 255, 255), width=1)
    draw.line([(0, H // 2), (W, H // 2)], fill=(255, 255, 255), width=1)
    boxes = torch.tensor(
        [
            [200.0, 150.0, 120.0, 50.0, math.radians(30.0)],
            [500.0, 200.0, 80.0, 80.0, 0.0],
            [650.0, 400.0, 60.0, 100.0, math.radians(-15.0) % math.pi],
            [150.0, 450.0, 90.0, 40.0, math.radians(75.0)],
        ]
    )
    labels = torch.tensor([0, 1, 2, 3])
    return img, {"boxes": boxes, "labels": labels}, W, H


def annotate_image(img, tgt, title=""):
    """在图像副本上绘制 OBB 框并返回标注后的图像。"""
    result = img.copy()
    draw = ImageDraw.Draw(result)
    if title:
        draw.text((5, 2), title, fill=(255, 255, 0))
    boxes = tgt["boxes"]
    labels = tgt["labels"]
    if boxes.numel() == 0:
        draw.text((10, 20), "NO BOXES", fill=(255, 0, 0))
        return result
    verts = xywhr_to_xyxyxyxy(boxes)
    for i in range(len(boxes)):
        c = COLORS[labels[i].item() % len(COLORS)]
        pts = [(float(verts[i, j, 0]), float(verts[i, j, 1])) for j in range(4)]
        draw.polygon(pts, outline=c, width=2)
        cx, cy = float(boxes[i, 0]), float(boxes[i, 1])
        label_str = str(labels[i].item())
        bbox = draw.textbbox((cx + 4, cy - 14), label_str)
        draw.rectangle(
            [bbox[0] - 2, bbox[1] - 1, bbox[2] + 2, bbox[3] + 1], fill=(30, 30, 30)
        )
        draw.text((cx + 4, cy - 14), label_str, fill=c)
    return result


def box_diag(tgt, label=""):
    """打印框诊断信息：数量、坐标范围。"""
    b = tgt["boxes"]
    n = len(b)
    if n == 0:
        print(f"  {label}: 0 boxes")
        return False
    print(
        f"  {label}: {n} boxes  "
        f"cx=[{b[:,0].min():.0f},{b[:,0].max():.0f}]  "
        f"cy=[{b[:,1].min():.0f},{b[:,1].max():.0f}]  "
        f"w=[{b[:,2].min():.0f},{b[:,2].max():.0f}]  "
        f"h=[{b[:,3].min():.0f},{b[:,3].max():.0f}]  "
        f"theta=[{b[:,4].min():.3f},{b[:,4].max():.3f}]"
    )
    return True


def save_comparison(name, img_bef, tgt_bef, img_aft, tgt_aft, info=""):
    """左 before / 右 after 对比图。框直接画在图像副本上，保证坐标对齐。"""
    before_anno = annotate_image(img_bef, tgt_bef, "BEFORE")
    after_anno = annotate_image(img_aft, tgt_aft, "AFTER")
    Wb, Hb = before_anno.size
    Wa, Ha = after_anno.size
    tw = Wb + Wa + 10
    th = max(Hb, Ha)
    canvas = Image.new("RGB", (tw, th), (30, 30, 30))
    canvas.paste(before_anno, (0, 0))
    canvas.paste(after_anno, (Wb + 10, 0))
    if info:
        ImageDraw.Draw(canvas).text((5, th - 18), info, fill=(150, 150, 150))
    path = os.path.join(OUTPUT_DIR, f"{name}.jpg")
    canvas.save(path)
    n_bef = len(tgt_bef["boxes"])
    n_aft = len(tgt_aft["boxes"])
    status = "boxes OK" if n_aft == n_bef else f"LOST {n_bef - n_aft} boxes"
    print(f"  saved: {path}  [{status}]")


def assert_no_loss(name, tgt_bef, tgt_aft, allow_crop=False):
    """验证变换后框未丢失（IoUCrop 允许裁剪掉部分框）。"""
    n_b = len(tgt_bef["boxes"])
    n_a = len(tgt_aft["boxes"])
    if allow_crop:
        assert n_a <= n_b, f"{name}: boxes grew ({n_b}->{n_a})"
    else:
        assert n_a == n_b, f"{name}: lost {n_b - n_a} boxes ({n_b}->{n_a})"
    print(
        f"  assert: {n_b}->{n_a} boxes (no loss)"
        if not allow_crop
        else f"  assert: {n_b}->{n_a} boxes (crop allowed)"
    )


def denorm(boxes, W, H):
    b = boxes.clone()
    b[:, 0] *= W
    b[:, 1] *= H
    b[:, 2] *= W
    b[:, 3] *= H
    return b


# ═══════════════════════════════════════════
# 1. OBBFlip
# ═══════════════════════════════════════════
def test_obb_flip():
    from engine.data.transforms.obb_transforms import OBBFlip

    print("\n=== OBBFlip ===")
    img, tgt, W, H = make_test_sample()
    img_b, tgt_b = img.copy(), copy.deepcopy(tgt)
    flip = OBBFlip(p=1.0)
    img_a, tgt_a, _ = flip((img.copy(), copy.deepcopy(tgt), None))
    box_diag(tgt_b, "before")
    box_diag(tgt_a, "after")
    save_comparison(
        "01_OBBFlip", img_b, tgt_b, img_a, tgt_a, "Flip: cx'=W-cx, theta'=(pi-theta)%pi"
    )
    b = tgt_a["boxes"]
    assert abs(b[0, 0].item() - (W - 200)) < 1
    assert_no_loss("OBBFlip", tgt_b, tgt_a)


# ═══════════════════════════════════════════
# 2. OBBZoomOut
# ═══════════════════════════════════════════
def test_obb_zoomout():
    from engine.data.transforms.obb_transforms import OBBZoomOut

    print("\n=== OBBZoomOut ===")
    random.seed(42)
    img, tgt, W, H = make_test_sample()
    img_b, tgt_b = img.copy(), copy.deepcopy(tgt)
    zoom = OBBZoomOut(fill=50, pad_level=0.3)
    img_a, tgt_a, _ = zoom((img.copy(), copy.deepcopy(tgt), None))
    box_diag(tgt_b, "before")
    box_diag(tgt_a, "after")
    save_comparison(
        "02_OBBZoomOut",
        img_b,
        tgt_b,
        img_a,
        tgt_a,
        "pure translation: only cx,cy changed",
    )
    assert_no_loss("OBBZoomOut", tgt_b, tgt_a)


# ═══════════════════════════════════════════
# 3. OBBResize
# ═══════════════════════════════════════════
def test_obb_resize():
    from engine.data.transforms.obb_transforms import OBBResize

    print("\n=== OBBResize ===")
    img, tgt, W, H = make_test_sample()
    img_b, tgt_b = img.copy(), copy.deepcopy(tgt)
    resize = OBBResize(size=[480, 640])
    img_a, tgt_a, _ = resize((img.copy(), copy.deepcopy(tgt), None))
    b = tgt_a["boxes"]
    box_diag(tgt_b, "before")
    box_diag(tgt_a, "after")
    sx, sy = 640 / W, 480 / H
    assert abs(b[0, 0].item() - 200 * sx) < 3, f"resize cx fail"
    save_comparison(
        "03_OBBResize",
        img_b,
        tgt_b,
        img_a,
        tgt_a,
        f"800x600->640x480, scale=({sx:.3f},{sy:.3f})",
    )
    assert_no_loss("OBBResize", tgt_b, tgt_a)


# ═══════════════════════════════════════════
# 4. OBBIoUCrop
# ═══════════════════════════════════════════
def test_obb_ioucrop():
    from engine.data.transforms.obb_transforms import OBBIoUCrop

    print("\n=== OBBIoUCrop ===")
    random.seed(123)
    img, tgt, W, H = make_test_sample()
    img_b, tgt_b = img.copy(), copy.deepcopy(tgt)
    crop = OBBIoUCrop(p=1.0, scale=(0.4, 0.7), ratio=(0.8, 1.2), trials=40)
    img_a, tgt_a, _ = crop((img.copy(), copy.deepcopy(tgt), None))
    Wa, Ha = img_a.size
    box_diag(tgt_b, "before")
    box_diag(tgt_a, "after")
    save_comparison(
        "04_OBBIoUCrop", img_b, tgt_b, img_a, tgt_a, f"IoUCrop -> {Wa}x{Ha}"
    )
    assert_no_loss("OBBIoUCrop", tgt_b, tgt_a, allow_crop=True)


# ═══════════════════════════════════════════
# 5. OBBConvertBoxes
# ═══════════════════════════════════════════
def test_obb_convert():
    from engine.data.transforms.obb_transforms import OBBConvertBoxes, OBBResize

    print("\n=== OBBConvertBoxes ===")
    img, tgt, W, H = make_test_sample()
    img_r, tgt_r, _ = OBBResize(size=[640, 640])((img.copy(), copy.deepcopy(tgt), None))
    img_b, tgt_b = img_r.copy(), copy.deepcopy(tgt_r)
    cnv = OBBConvertBoxes(normalize=True)
    _, tgt_a, _ = cnv((img_r.copy(), copy.deepcopy(tgt_r), None))
    b = tgt_a["boxes"]
    box_diag(tgt_b, "before (px)")
    box_diag(tgt_a, "after (norm)")
    # 反归一化绘制
    dv = annotate_image(
        img_b, {"boxes": denorm(b, *img_b.size), "labels": tgt_a["labels"]}
    )
    d = ImageDraw.Draw(dv)
    for i, bb in enumerate(b):
        cx, cy, w, h, th = bb.tolist()
        d.text(
            (5, 12 + i * 14),
            f"[{i}] cx={cx:.3f} cy={cy:.3f} w={w:.3f} h={h:.3f} th={th:.3f}",
            fill=COLORS[i % len(COLORS)],
        )
    save_comparison(
        "05_OBBConvertBoxes",
        img_b,
        tgt_b,
        dv,
        tgt_a,
        "normalize: cx/=W, cy/=H, w/=W, h/=H",
    )
    assert_no_loss("OBBConvertBoxes", tgt_b, tgt_a)


# ═══════════════════════════════════════════
# 6. OBBSanitize
# ═══════════════════════════════════════════
def test_obb_sanitize():
    from engine.data.transforms.obb_transforms import OBBSanitize

    print("\n=== OBBSanitize ===")
    img, _, W, H = make_test_sample()
    small = torch.tensor(
        [
            [200.0, 150.0, 3.0, 50.0, 0.5],
            [500.0, 200.0, 80.0, 2.0, 0.0],
            [650.0, 400.0, 60.0, 100.0, 1.2],
            [150.0, 450.0, 90.0, 40.0, 2.0],
        ]
    )
    tgt = {"boxes": small, "labels": torch.tensor([0, 1, 2, 3])}
    tgt_b = copy.deepcopy(tgt)
    sani = OBBSanitize(min_size=4)
    _, tgt_a, _ = sani((img.copy(), copy.deepcopy(tgt), None))
    box_diag(tgt_b, "before")
    box_diag(tgt_a, "after")
    iv = annotate_image(img, tgt_a)
    save_comparison(
        "06_OBBSanitize",
        img,
        tgt_b,
        iv,
        tgt_a,
        f"filter w<=4 or h<=4: {len(tgt_b['boxes'])}->{len(tgt_a['boxes'])}",
    )
    assert len(tgt_a["boxes"]) == 2, f"expected 2, got {len(tgt_a['boxes'])}"
    print("  verified: 4->2")


# ═══════════════════════════════════════════
# 8. Full Pipeline (no Mosaic)
# ═══════════════════════════════════════════
def test_full_pipeline():
    from engine.data.transforms.obb_transforms import (
        OBBZoomOut,
        OBBIoUCrop,
        OBBSanitize,
        OBBFlip,
        OBBResize,
    )

    print("\n=== Full Pipeline ===")
    random.seed(99)
    img, tgt, W, H = make_test_sample()
    img_b, tgt_b = img.copy(), copy.deepcopy(tgt)
    pipeline = [
        OBBZoomOut(fill=30, pad_level=0.1),
        OBBIoUCrop(p=1.0, scale=(0.5, 0.8)),
        OBBSanitize(min_size=3),
        OBBFlip(p=0.5),
        OBBResize(size=[640, 640]),
        OBBSanitize(min_size=1),
    ]
    s = (img.copy(), copy.deepcopy(tgt), None)
    for tr in pipeline:
        s = tr(s)
        box_diag(s[1], f"  after {tr.__class__.__name__}")
    iv = annotate_image(s[0], s[1])
    n_aft = len(s[1]["boxes"])
    save_comparison(
        "08_FullPipeline",
        img_b,
        tgt_b,
        iv,
        s[1],
        f"Pipeline -> 640x640, {n_aft}/{len(tgt_b['boxes'])} boxes",
    )
    print(f"  done: {n_aft} boxes final")


# ═══════════════════════════════════════════
# 9. Mosaic OBB (with affine)
# ═══════════════════════════════════════════
def test_mosaic_obb():
    print("\n=== Mosaic OBB (with affine) ===")
    random.seed(7)
    torch.manual_seed(7)
    from engine.data.transforms.mosaic import Mosaic

    class FakeDS:
        def __len__(self):
            return 100

        def load_item(self, idx):
            iw, ih = 400, 300
            im = Image.new(
                "RGB",
                (iw, ih),
                (
                    random.randint(30, 80),
                    random.randint(60, 120),
                    random.randint(50, 100),
                ),
            )
            d = ImageDraw.Draw(im)
            for _ in range(10):
                x, y = random.randint(0, iw), random.randint(0, ih)
                r = random.randint(3, 20)
                d.ellipse(
                    [x - r, y - r, x + r, y + r],
                    fill=(
                        random.randint(50, 150),
                        random.randint(80, 180),
                        random.randint(40, 140),
                    ),
                )
            n = random.randint(1, 3)
            bx = torch.tensor(
                [
                    [
                        random.uniform(80, iw - 80),
                        random.uniform(60, ih - 60),
                        random.uniform(30, 80),
                        random.uniform(20, 60),
                        random.random() * math.pi,
                    ]
                    for _ in range(n)
                ]
            )
            lb = torch.randint(0, 4, (n,))
            return im, {"boxes": bx, "labels": lb, "orig_size": torch.tensor([iw, ih])}

    ds = FakeDS()
    main_img = Image.new("RGB", (400, 300), (80, 80, 80))
    md = ImageDraw.Draw(main_img)
    for _ in range(30):
        x, y = random.randint(0, 400), random.randint(0, 300)
        r = random.randint(5, 25)
        md.ellipse([x - r, y - r, x + r, y + r], fill=(random.randint(60, 160),) * 3)
    md.line([(200, 0), (200, 300)], fill=(100, 100, 100))
    md.line([(0, 150), (400, 150)], fill=(100, 100, 100))
    main_boxes = torch.tensor(
        [
            [200.0, 150.0, 100.0, 40.0, math.radians(30.0)],
            [300.0, 200.0, 60.0, 70.0, math.radians(-20.0) % math.pi],
        ]
    )
    main_tgt = {
        "boxes": main_boxes,
        "labels": torch.tensor([0, 1]),
        "orig_size": torch.tensor([400, 300]),
    }
    img_b, tgt_b = main_img.copy(), copy.deepcopy(main_tgt)

    mosaic = Mosaic(
        output_size=320,
        rotation_range=10,
        translation_range=(0.1, 0.1),
        scaling_range=(0.8, 1.2),
        probability=1.0,
        fill_value=30,
        use_cache=False,
    )
    mi, mt, _ = mosaic(main_img.copy(), copy.deepcopy(main_tgt), ds)
    Wm, Hm = mi.size
    box_diag(tgt_b, "main before")
    box_diag(mt, "mosaic after")
    iv = annotate_image(mi, mt, "MOSAIC+affine")
    save_comparison(
        "09_MosaicOBB",
        img_b,
        tgt_b,
        iv,
        mt,
        f"Mosaic+affine: 400x300->{Wm}x{Hm}, rot+-10deg",
    )
    print(f"  done: {len(mt['boxes'])} boxes, {Wm}x{Hm}")


# ═══════════════════════════════════════════
# 10. Mosaic + MixUp（合成数据）
# ═══════════════════════════════════════════
def test_mosaic_mixup():
    print("\n=== Mosaic + MixUp (synthetic) ===")
    random.seed(123)
    torch.manual_seed(123)

    # 用 make_test_sample 生成 4 张变体图来模拟 batch
    W, H = 800, 600
    all_imgs, all_tgts = [], []
    for ci in range(4):
        random.seed(42 + ci)
        torch.manual_seed(42 + ci)
        img, tgt, _, _ = make_test_sample()
        all_imgs.append(img)
        tgt["area"] = tgt["boxes"][:, 2] * tgt["boxes"][:, 3]
        all_tgts.append(tgt)

    img_bef = all_imgs[0].copy()
    tgt_bef = copy.deepcopy(all_tgts[0])

    # 转为 batch tensor
    import torchvision.transforms.v2.functional as TF

    images = torch.stack([TF.to_image(img).float() / 255.0 for img in all_imgs])

    from engine.data.dataloader import BatchImageCollateFunction

    collate = BatchImageCollateFunction(
        mixup_prob=1.0,
        mixup_epochs=[0, 100],
        copyblend_prob=0.0,
        copyblend_epochs=[0, 0],
        stop_epoch=200,
    )
    collate.set_epoch(5)
    mixed_images, mixed_targets = collate.apply_mixup(images, copy.deepcopy(all_tgts))

    for i in range(4):
        img_v = (
            (mixed_images[i].cpu().clamp(0, 1) * 255).byte().permute(1, 2, 0).numpy()
        )
        img_pil = Image.fromarray(img_v)
        t = mixed_targets[i]
        boxes_px = denorm(t["boxes"].cpu(), W, H)
        has_m = "mixup" in t
        title = f"MixUp[{i}]: {len(t['boxes'])} boxes" + (
            f" beta={t['mixup'][0]:.3f}" if has_m else ""
        )
        anno = annotate_image(
            img_pil, {"boxes": boxes_px, "labels": t["labels"].cpu()}, title
        )
        anno.save(os.path.join(OUTPUT_DIR, f"10a_MosaicMixUp_i{i}.jpg"))
        print(f"  img[{i}]: {len(t['boxes'])} boxes, mixup={'YES' if has_m else 'NO'}")

    img0_v = (mixed_images[0].cpu().clamp(0, 1) * 255).byte().permute(1, 2, 0).numpy()
    t0 = mixed_targets[0]
    boxes_px = denorm(t0["boxes"].cpu(), W, H)
    aft_anno = annotate_image(
        Image.fromarray(img0_v), {"boxes": boxes_px, "labels": t0["labels"].cpu()}
    )
    save_comparison(
        "10a_MosaicMixUp_cmp",
        img_bef,
        tgt_bef,
        aft_anno,
        t0,
        f"MixUp: {len(tgt_bef['boxes'])}->{len(t0['boxes'])} boxes",
    )
    print("  done")


# ═══════════════════════════════════════════
# 11. Mosaic + CopyBlend（合成数据）
# ═══════════════════════════════════════════
def test_mosaic_copyblend():
    print("\n=== Mosaic + CopyBlend (synthetic) ===")
    random.seed(456)
    torch.manual_seed(456)

    W, H = 800, 600
    all_imgs, all_tgts = [], []
    for ci in range(4):
        random.seed(42 + ci)
        torch.manual_seed(42 + ci)
        img, tgt, _, _ = make_test_sample()
        all_imgs.append(img)
        tgt["area"] = tgt["boxes"][:, 2] * tgt["boxes"][:, 3]
        all_tgts.append(tgt)

    img_bef = all_imgs[0].copy()
    tgt_bef = copy.deepcopy(all_tgts[0])

    import torchvision.transforms.v2.functional as TF

    images = torch.stack([TF.to_image(img).float() / 255.0 for img in all_imgs])

    from engine.data.dataloader import BatchImageCollateFunction

    collate = BatchImageCollateFunction(
        mixup_prob=0.0,
        mixup_epochs=[0, 0],
        copyblend_prob=1.0,
        copyblend_epochs=[0, 100],
        area_threshold=500,
        num_objects=3,
        stop_epoch=200,
    )
    collate.set_epoch(5)
    blended_images, blended_targets = collate.apply_mixup(
        images, copy.deepcopy(all_tgts)
    )

    for i in range(4):
        img_v = (
            (blended_images[i].cpu().clamp(0, 1) * 255).byte().permute(1, 2, 0).numpy()
        )
        img_pil = Image.fromarray(img_v)
        t = blended_targets[i]
        boxes_px = denorm(t["boxes"].cpu(), W, H)
        anno = annotate_image(
            img_pil,
            {"boxes": boxes_px, "labels": t["labels"].cpu()},
            f"CopyBlend[{i}]: {len(t['boxes'])} boxes",
        )
        anno.save(os.path.join(OUTPUT_DIR, f"11a_MosaicCopyBlend_i{i}.jpg"))
        print(f"  img[{i}]: {len(t['boxes'])} boxes (was {len(all_tgts[i]['boxes'])})")

    img0_v = (blended_images[0].cpu().clamp(0, 1) * 255).byte().permute(1, 2, 0).numpy()
    t0 = blended_targets[0]
    boxes_px = denorm(t0["boxes"].cpu(), W, H)
    aft_anno = annotate_image(
        Image.fromarray(img0_v), {"boxes": boxes_px, "labels": t0["labels"].cpu()}
    )
    save_comparison(
        "11a_MosaicCopyBlend_cmp",
        img_bef,
        tgt_bef,
        aft_anno,
        t0,
        f"CopyBlend: {len(tgt_bef['boxes'])}->{len(t0['boxes'])} boxes",
    )
    print("  done")


# ═══════════════════════════════════════════
def test_mixup():
    print("\n=== MixUp (collate function) ===")
    from engine.core import YAMLConfig
    from engine.solver import TASKS

    cfg = YAMLConfig(os.path.join(ROOT, "configs/custom_obb/dlzdt/ablation/abl_rep3.yml"))
    cfg.yaml_cfg["epoches"] = 2
    cfg.yaml_cfg["train_dataloader"]["total_batch_size"] = 4
    cfg.yaml_cfg["train_dataloader"]["num_workers"] = 0
    cfg.yaml_cfg["val_dataloader"]["num_workers"] = 0
    cfg.yaml_cfg["checkpoint_freq"] = 100

    # 关闭 CopyBlend + Mosaic，只留 MixUp + epoch=5 触发
    cfg.yaml_cfg["train_dataloader"]["collate_fn"]["copyblend_prob"] = 0.0
    cfg.yaml_cfg["train_dataloader"]["collate_fn"]["mixup_prob"] = 1.0
    cfg.yaml_cfg["train_dataloader"]["collate_fn"]["mixup_epochs"] = [0, 100]
    cfg.yaml_cfg["train_dataloader"]["collate_fn"]["stop_epoch"] = 200
    for op in cfg.yaml_cfg["train_dataloader"]["dataset"]["transforms"]["ops"]:
        if op.get("type") == "Mosaic":
            op["probability"] = 0.0

    solver = TASKS[cfg.yaml_cfg["task"]](cfg)
    solver.train()
    loader = solver.train_dataloader
    # 设置 epoch=5 触发 MixUp（默认 epoch=-1 不会触发）
    loader.collate_fn.set_epoch(5)
    print(f"  collate epoch: {loader.collate_fn.epoch}")

    for batch_idx, (images, targets) in enumerate(loader):
        if batch_idx >= 1:
            break
        B = images.shape[0]
        print(f"  batch {batch_idx}: {images.shape}")
        for i in range(min(B, 3)):
            t = targets[i]
            n = len(t["boxes"])
            has_mixup = "mixup" in t
            print(f"  img[{i}]: {n} boxes, mixup={'YES' if has_mixup else 'NO'}")
            if has_mixup:
                ratios = t["mixup"]
                print(f"    mixup ratios ({len(ratios)}): {ratios.tolist()[:8]}...")

            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img_v = images[i].cpu() * std + mean
            img_v = (img_v.clamp(0, 1) * 255).byte().permute(1, 2, 0).numpy()
            img_pil = Image.fromarray(img_v)
            Wv, Hv = img_pil.size
            boxes_px = denorm(t["boxes"].cpu(), Wv, Hv)
            title = f"MixUp img[{i}]: {n} boxes"
            if has_mixup:
                title += f" (mixup)"
            anno = annotate_image(
                img_pil, {"boxes": boxes_px, "labels": t["labels"].cpu()}, title
            )
            anno.save(os.path.join(OUTPUT_DIR, f"10_MixUp_b{batch_idx}_i{i}.jpg"))
            print(f"    saved: 10_MixUp_b{batch_idx}_i{i}.jpg")
    print("  done")


# ═══════════════════════════════════════════
# 11. CopyBlend (batch-level, via dataloader)
# ═══════════════════════════════════════════
def test_copyblend():
    print("\n=== CopyBlend (collate function) ===")
    from engine.core import YAMLConfig
    from engine.solver import TASKS

    cfg = YAMLConfig(os.path.join(ROOT, "configs/custom_obb/dlzdt/ablation/abl_rep3.yml"))
    cfg.yaml_cfg["epoches"] = 2
    cfg.yaml_cfg["train_dataloader"]["total_batch_size"] = 4
    cfg.yaml_cfg["train_dataloader"]["num_workers"] = 0
    cfg.yaml_cfg["val_dataloader"]["num_workers"] = 0
    cfg.yaml_cfg["checkpoint_freq"] = 100

    # 关闭 MixUp + Mosaic，只留 CopyBlend + epoch=5 触发
    cfg.yaml_cfg["train_dataloader"]["collate_fn"]["mixup_prob"] = 0.0
    cfg.yaml_cfg["train_dataloader"]["collate_fn"]["copyblend_prob"] = 1.0
    cfg.yaml_cfg["train_dataloader"]["collate_fn"]["copyblend_epochs"] = [0, 100]
    cfg.yaml_cfg["train_dataloader"]["collate_fn"]["stop_epoch"] = 200
    for op in cfg.yaml_cfg["train_dataloader"]["dataset"]["transforms"]["ops"]:
        if op.get("type") == "Mosaic":
            op["probability"] = 0.0

    solver = TASKS[cfg.yaml_cfg["task"]](cfg)
    solver.train()
    loader = solver.train_dataloader
    loader.collate_fn.set_epoch(5)
    print(f"  collate epoch: {loader.collate_fn.epoch}")

    for batch_idx, (images, targets) in enumerate(loader):
        if batch_idx >= 1:
            break
        B = images.shape[0]
        print(f"  batch {batch_idx}: {images.shape}")
        for i in range(min(B, 3)):
            t = targets[i]
            n = len(t["boxes"])
            has_mixup = "mixup" in t
            print(f"  img[{i}]: {n} boxes, mixup={'YES' if has_mixup else 'NO'}")

            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img_v = images[i].cpu() * std + mean
            img_v = (img_v.clamp(0, 1) * 255).byte().permute(1, 2, 0).numpy()
            img_pil = Image.fromarray(img_v)
            Wv, Hv = img_pil.size
            boxes_px = denorm(t["boxes"].cpu(), Wv, Hv)
            title = f"CopyBlend img[{i}]: {n} boxes"
            if has_mixup:
                title += f" (blend)"
            anno = annotate_image(
                img_pil, {"boxes": boxes_px, "labels": t["labels"].cpu()}, title
            )
            anno.save(os.path.join(OUTPUT_DIR, f"11_CopyBlend_b{batch_idx}_i{i}.jpg"))
            print(f"    saved: 11_CopyBlend_b{batch_idx}_i{i}.jpg  ({n} boxes)")
    print("  done")


# ═══════════════════════════════════════════
# 12. Mosaic affine IoU alignment
# ═══════════════════════════════════════════
def test_mosaic_alignment():
    print("\n=== Mosaic Alignment IoU ===")
    from engine.deim.obb_geometry import affine_obb_matrix

    W, H = 400, 300
    img = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(img)
    boxes = torch.tensor(
        [
            [150.0, 120.0, 80.0, 30.0, math.radians(25.0)],
            [300.0, 180.0, 50.0, 70.0, math.radians(-10.0) % math.pi],
        ]
    )
    verts = xywhr_to_xyxyxyxy(boxes)
    for i in range(len(boxes)):
        d.polygon(
            [(float(verts[i, j, 0]), float(verts[i, j, 1])) for j in range(4)], fill=255
        )
    img_bef = img.copy()

    phi, s_val, tx, ty = math.radians(15.0), 0.9, 20.0, -15.0
    cx, cy = W / 2, H / 2
    cp, sp = math.cos(phi), math.sin(phi)
    A = torch.tensor(
        [[s_val * cp, -s_val * sp], [s_val * sp, s_val * cp]], dtype=torch.float32
    )
    c = torch.tensor([cx, cy], dtype=torch.float32)
    t = torch.tensor([tx, ty], dtype=torch.float32)
    b_vec = c + t - A @ c
    A_inv = torch.inverse(A)
    data = (
        A_inv[0, 0].item(),
        A_inv[0, 1].item(),
        (-A_inv @ b_vec)[0].item(),
        A_inv[1, 0].item(),
        A_inv[1, 1].item(),
        (-A_inv @ b_vec)[1].item(),
    )
    img_warped = img.transform(
        (W, H), Image.AFFINE, data, resample=Image.BILINEAR, fillcolor=0
    )

    mat_fwd = torch.cat([A, b_vec[:, None]], dim=1)
    boxes_w = affine_obb_matrix(boxes, mat_fwd)
    img_obb = Image.new("L", (W, H), 0)
    d2 = ImageDraw.Draw(img_obb)
    verts_w = xywhr_to_xyxyxyxy(boxes_w)
    for i in range(len(boxes_w)):
        d2.polygon(
            [(float(verts_w[i, j, 0]), float(verts_w[i, j, 1])) for j in range(4)],
            fill=255,
        )

    mask_i = np.array(img_warped) > 128
    mask_o = np.array(img_obb) > 128
    inter = (mask_i & mask_o).sum()
    union = (mask_i | mask_o).sum()
    iou = inter / union if union > 0 else 0.0

    canvas = Image.new("RGB", (W * 2 + 20, H + 40), (30, 30, 30))
    canvas.paste(img_bef.convert("RGB"), (0, 0))
    canvas.paste(img_warped.convert("RGB"), (W + 10, 0))
    canvas.paste(img_obb.convert("RGB"), (0, H + 20))
    overlay = Image.blend(img_warped.convert("RGB"), img_obb.convert("RGB"), alpha=0.5)
    canvas.paste(overlay, (W + 10, H + 20))
    cd = ImageDraw.Draw(canvas)
    cd.text((5, 0), "Before", fill=(200, 200, 200))
    cd.text((W + 15, 0), "Image warp", fill=(200, 200, 200))
    cd.text((5, H + 20), "Box transform", fill=(200, 200, 200))
    cd.text(
        (W + 15, H + 20),
        f"Overlay (IoU={iou:.4f})",
        fill=(0, 255, 0) if iou > 0.9 else (255, 0, 0),
    )
    path = os.path.join(OUTPUT_DIR, "12_MosaicAlignment.jpg")
    canvas.save(path)
    print(f"  saved: {path}")
    print(f"  IoU = {iou:.4f} {'>0.9' if iou > 0.9 else '<0.9'}")
    assert iou > 0.9
    print("  verified")


# ═══════════════════════════════════════════
# 13. DataLoader — Mosaic 多组增强
# ═══════════════════════════════════════════
def test_dataloader_mosaic():
    print("\n=== DataLoader Mosaic (multi-batch) ===")
    from engine.core import YAMLConfig
    from engine.solver import TASKS

    cfg = YAMLConfig(os.path.join(ROOT, "configs/custom_obb/dlzdt/ablation/abl_rep3.yml"))
    cfg.yaml_cfg["train_dataloader"]["total_batch_size"] = 4
    cfg.yaml_cfg["train_dataloader"]["num_workers"] = 0
    cfg.yaml_cfg["val_dataloader"]["num_workers"] = 0
    cfg.yaml_cfg["checkpoint_freq"] = 100
    cfg.yaml_cfg["epoches"] = 2
    for op in cfg.yaml_cfg["train_dataloader"]["dataset"]["transforms"]["ops"]:
        if op.get("type") == "Mosaic":
            op["probability"] = 1.0
    cfg.yaml_cfg["train_dataloader"]["collate_fn"]["mixup_prob"] = 0.0
    cfg.yaml_cfg["train_dataloader"]["collate_fn"]["copyblend_prob"] = 0.0
    cfg.yaml_cfg["train_dataloader"]["collate_fn"]["stop_epoch"] = 200

    solver = TASKS[cfg.yaml_cfg["task"]](cfg)
    solver.train()
    loader = solver.train_dataloader
    print(f"  dataset size: {len(loader.dataset)}")

    for batch_idx, (images, targets) in enumerate(loader):
        if batch_idx >= 3:
            break
        B = images.shape[0]
        print(f"  --- batch {batch_idx}: {images.shape} ---")
        for i in range(min(B, 3)):
            t = targets[i]
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img_v = images[i].cpu() * std + mean
            img_v = (img_v.clamp(0, 1) * 255).byte().permute(1, 2, 0).numpy()
            img_pil = Image.fromarray(img_v)
            Wv, Hv = img_pil.size
            boxes_px = denorm(t["boxes"].cpu(), Wv, Hv)
            title = f"Mosaic b{batch_idx} i{i}: {len(t['boxes'])} boxes ({Wv}x{Hv})"
            anno = annotate_image(img_pil, {"boxes": boxes_px, "labels": t["labels"].cpu()}, title)
            path = os.path.join(OUTPUT_DIR, f"13_Mosaic_b{batch_idx}_i{i}.jpg")
            anno.save(path)
            print(f"    saved: {os.path.basename(path)}  ({len(t['boxes'])} boxes)")
    print("  done")


# ═══════════════════════════════════════════
# 14. DataLoader — 完整 Pipeline 多组增强
# ═══════════════════════════════════════════
def test_dataloader_pipeline():
    print("\n=== DataLoader Full Pipeline (multi-batch) ===")
    from engine.core import YAMLConfig
    from engine.solver import TASKS

    cfg = YAMLConfig(os.path.join(ROOT, "configs/custom_obb/dlzdt/ablation/abl_rep3.yml"))
    cfg.yaml_cfg["train_dataloader"]["total_batch_size"] = 4
    cfg.yaml_cfg["train_dataloader"]["num_workers"] = 0
    cfg.yaml_cfg["val_dataloader"]["num_workers"] = 0
    cfg.yaml_cfg["checkpoint_freq"] = 100
    cfg.yaml_cfg["epoches"] = 2
    for op in cfg.yaml_cfg["train_dataloader"]["dataset"]["transforms"]["ops"]:
        if op.get("type") == "Mosaic":
            op["probability"] = 1.0
    cfg.yaml_cfg["train_dataloader"]["collate_fn"]["mixup_prob"] = 1.0
    cfg.yaml_cfg["train_dataloader"]["collate_fn"]["mixup_epochs"] = [0, 100]
    cfg.yaml_cfg["train_dataloader"]["collate_fn"]["copyblend_prob"] = 0.5
    cfg.yaml_cfg["train_dataloader"]["collate_fn"]["copyblend_epochs"] = [0, 100]
    cfg.yaml_cfg["train_dataloader"]["collate_fn"]["stop_epoch"] = 200

    solver = TASKS[cfg.yaml_cfg["task"]](cfg)
    solver.train()
    loader = solver.train_dataloader
    loader.collate_fn.set_epoch(5)
    print(f"  dataset size: {len(loader.dataset)}, collate epoch: {loader.collate_fn.epoch}")

    for batch_idx, (images, targets) in enumerate(loader):
        if batch_idx >= 3:
            break
        B = images.shape[0]
        print(f"  --- batch {batch_idx}: {images.shape} ---")
        for i in range(min(B, 3)):
            t = targets[i]
            n = len(t["boxes"])
            has_m = "mixup" in t
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img_v = images[i].cpu() * std + mean
            img_v = (img_v.clamp(0, 1) * 255).byte().permute(1, 2, 0).numpy()
            img_pil = Image.fromarray(img_v)
            Wv, Hv = img_pil.size
            boxes_px = denorm(t["boxes"].cpu(), Wv, Hv)
            aug_type = "mixup" if has_m else "none"
            title = f"Full b{batch_idx} i{i}: {n} boxes ({aug_type})"
            anno = annotate_image(img_pil, {"boxes": boxes_px, "labels": t["labels"].cpu()}, title)
            path = os.path.join(OUTPUT_DIR, f"14_FullPipe_b{batch_idx}_i{i}.jpg")
            anno.save(path)
            print(f"    saved: {os.path.basename(path)}  ({n} boxes, {aug_type})")
    print("  done")


ALL_TESTS = {
    "OBBFlip": test_obb_flip,
    "OBBZoomOut": test_obb_zoomout,
    "OBBResize": test_obb_resize,
    "OBBIoUCrop": test_obb_ioucrop,
    "OBBConvertBoxes": test_obb_convert,
    "OBBSanitize": test_obb_sanitize,
    "FullPipeline": test_full_pipeline,
    "MosaicOBB": test_mosaic_obb,
    "MosaicMixUp": test_mosaic_mixup,
    "MosaicCopyBlend": test_mosaic_copyblend,
    "MixUp": test_mixup,
    "CopyBlend": test_copyblend,
    "MosaicAlignment": test_mosaic_alignment,
    "DataLoaderMosaic": test_dataloader_mosaic,
    "DataLoaderPipeline": test_dataloader_pipeline,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    print(f"Output: {OUTPUT_DIR}\nSeed: {args.seed}")

    if args.only:
        ALL_TESTS[args.only]()
    else:
        passed = failed = 0
        for name, fn in ALL_TESTS.items():
            try:
                fn()
                passed += 1
            except Exception as e:
                print(f"  FAILED: {e}")
                failed += 1
                import traceback

                traceback.print_exc()
        print(f"\n{'='*50}")
        print(f"{passed} passed, {failed} failed")
        if failed:
            sys.exit(1)
    print(f"\nOutputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
