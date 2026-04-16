import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
from PIL import Image
import time
from tqdm import tqdm


from deim_wapper.deimv2_det import DEIMv2Det
from deim_wapper.deimv2_model_config import DEIMV2_VITL16P_CFG


def benchmark_model(
    model,
    img_dir,
    score_threshold=0.5,
    num_warmup=10,
    num_test=-1,
    device="cuda:1",
):

    img_list = [
        os.path.join(img_dir, f)
        for f in os.listdir(img_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    if not img_list:
        print(f"No images found in {img_dir}")
        return

    print(f"Found {len(img_list)} images for testing")
    print(f"Warmup iterations: {num_warmup}")

    print("-" * 80)

    total_inference_time = 0
    total_preprocess_time = 0
    total_postprocess_time = 0

    inference_times = []
    memory_allocated = []
    memory_reserved = []

    if torch.cuda.is_available():
        torch.cuda.set_device(device)
        torch.cuda.empty_cache()

    print("\n=== Warmup Phase ===")
    for i in tqdm(range(num_warmup), desc="Warmup"):
        img_path = img_list[i % len(img_list)]
        image = Image.open(img_path).convert("RGB")

        start_time = time.time()
        outputs = model.infer(image, score_threshold)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = time.time()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    print("\n=== Testing Phase ===")
    test_count = min(num_test, len(img_list)) if num_test > 0 else len(img_list)
    print(f"Test iterations: {test_count}")
    for i in tqdm(range(test_count), desc="Testing"):
        img_path = img_list[i % len(img_list)]
        image = Image.open(img_path).convert("RGB")

        if torch.cuda.is_available():
            torch.cuda.synchronize()

            start_preprocess = time.time()
            input_tensor = model.transform(image).unsqueeze(0).to(model.device)
            orig_target_sizes = torch.Tensor(model.imgsz).to(model.device)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            preprocess_time = time.time() - start_preprocess

            start_inference = time.time()
            with torch.no_grad():
                outputs_raw = model.dimv2(input_tensor, orig_target_sizes)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            inference_time = time.time() - start_inference

            start_postprocess = time.time()
            outputs = model.infer(image, score_threshold)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            postprocess_time = time.time() - start_postprocess

            memory_allocated.append(torch.cuda.memory_allocated() / 1024**2)
            memory_reserved.append(torch.cuda.memory_reserved() / 1024**2)
        else:
            start_total = time.time()
            outputs = model.infer(image, score_threshold)
            total_time = time.time() - start_total

            preprocess_time = 0
            inference_time = total_time
            postprocess_time = 0

        total_inference_time += inference_time
        total_preprocess_time += preprocess_time
        total_postprocess_time += postprocess_time
        inference_times.append(inference_time)

    avg_inference_time = total_inference_time / test_count
    avg_preprocess_time = total_preprocess_time / test_count
    avg_postprocess_time = total_postprocess_time / test_count
    avg_total_time = (
        total_inference_time + total_preprocess_time + total_postprocess_time
    ) / test_count

    fps = 1.0 / avg_inference_time if avg_inference_time > 0 else 0
    total_fps = 1.0 / avg_total_time if avg_total_time > 0 else 0

    std_inference_time = (
        torch.std(torch.tensor(inference_times)).item() if inference_times else 0
    )

    print("\n" + "=" * 80)
    print("Performance Benchmark Results")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Image Size: {model.imgsz}")
    print(f"Number of Test Images: {test_count}")
    print(f"Score Threshold: {score_threshold}")
    print("-" * 80)
    print(f"Average Preprocessing Time:  {avg_preprocess_time * 1000:.2f} ms")
    print(f"Average Model Inference Time: {avg_inference_time * 1000:.2f} ms")
    print(f"Average Postprocessing Time: {avg_postprocess_time * 1000:.2f} ms")
    print(f"Average Total Time:          {avg_total_time * 1000:.2f} ms")
    print("-" * 80)
    print(f"Inference FPS:        {fps:.2f}")
    print(f"Total FPS (with I/O): {total_fps:.2f}")
    print(f"Inference Time Std:   {std_inference_time * 1000:.2f} ms")
    print("-" * 80)

    if torch.cuda.is_available():
        peak_allocated = max(memory_allocated) if memory_allocated else 0
        peak_reserved = max(memory_reserved) if memory_reserved else 0
        current_allocated = torch.cuda.memory_allocated() / 1024**2
        current_reserved = torch.cuda.memory_reserved() / 1024**2

        print(f"Peak Memory Allocated:   {peak_allocated:.2f} MB")
        print(f"Peak Memory Reserved:    {peak_reserved:.2f} MB")
        print(f"Current Memory Allocated: {current_allocated:.2f} MB")
        print(f"Current Memory Reserved:  {current_reserved:.2f} MB")
        print("-" * 80)

        try:
            from thop import profile, clever_format

            dummy_input = torch.randn(1, 3, *model.imgsz).to(model.device)
            orig_target_sizes = torch.Tensor(model.imgsz).to(model.device)
            flops, params = profile(
                model.dimv2, inputs=(dummy_input, orig_target_sizes), verbose=False
            )
            flops_str, params_str = clever_format([flops, params], "%.3f")
            print(f"Model Parameters: {params_str}")
            print(f"FLOPs: {flops_str}")
            print("-" * 80)
        except ImportError:
            print("thop not installed. Install with: pip install thop")
            print("-" * 80)

    print("=" * 80)
    print("Benchmark completed successfully!")
    print("=" * 80)


def __test():
    img_dir = (
        "/home/cx/cx_dir/data/deimv2_train_data/dlzdt_dataset_20260331_hbb/images/val"
    )

    model_weight = "outputs/dlzdt_vitl16_freeze_extend/best_stg2.pth"
    num_classes = 3
    imgsz = (640, 640)
    max_det = 20
    config = DEIMV2_VITL16P_CFG
    model = DEIMv2Det(
        model_weight, num_classes, imgsz, max_det, DEIMV2_VITL16P_CFG, "cuda:1"
    )

    benchmark_model(
        model=model,
        img_dir=img_dir,
        score_threshold=0.5,
        num_warmup=10,
        num_test=-1,
        device="cuda:1",
    )


if __name__ == "__main__":
    __test()
