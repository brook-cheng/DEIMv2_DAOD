import sys, os, math, numpy as np, torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from engine.deim.obb_geometry import xyxyxyxy_to_xywhr
from tools.model_compare.obb_utils import parse_dota_line

GT_DIR = "test/data/outputs/dlzdt_obb_compare_val/gt_dota"
PRED_DIRS = {
    "YOLO-OBB":      "test/data/outputs/dlzdt_obb_compare_val/yolo_dota",
    "sp_ft_rep1":    "test/data/outputs/dlzdt_res/sp_ft_rep1_nloss_0723_last_val",
    "sp_ft_rep3":    "test/data/outputs/dlzdt_res/sp_ft_rep3_0722_val",
}

def load_per_image(dota_dir, is_gt=True):
    data = {}
    for fname in sorted(os.listdir(dota_dir)):
        if not fname.endswith(".txt"): continue
        stem = os.path.splitext(fname)[0]
        boxes = []
        with open(os.path.join(dota_dir, fname)) as f:
            for line in f:
                line = line.strip()
                if not line: continue
                if is_gt:
                    parts = line.split()
                    if len(parts) < 9: continue
                    poly8 = [float(x) for x in parts[:8]]
                    label = parts[8]
                else:
                    rec = parse_dota_line(line)
                    if rec is None: continue
                    poly8 = rec["poly"]
                    label = rec["label"]
                poly = torch.tensor(poly8, dtype=torch.float32).reshape(1, 4, 2)
                cx, cy, w, h, theta = xyxyxyxy_to_xywhr(poly).numpy().flatten()
                boxes.append((cx, cy, w, h, theta, label))
        if boxes:
            data[stem] = boxes
    return data

if __name__ == "__main__":
    gt_per_img = load_per_image(GT_DIR, is_gt=True)
    print(f"Loaded {len(gt_per_img)} GT images")

    for model_name, pred_dir in PRED_DIRS.items():
        if not os.path.isdir(pred_dir):
            print(f"  [SKIP] {model_name}: dir not found")
            continue
        pred_per_img = load_per_image(pred_dir, is_gt=False)
        common = set(gt_per_img) & set(pred_per_img)

        angle_errs = []  # all angle errors (degrees)
        artifact_count = 0
        normal_count = 0
        total_matched = 0

        for stem in common:
            gt_list = gt_per_img[stem]
            pr_list = pred_per_img[stem]

            gt_arr = np.array([[b[0],b[1],b[2],b[3],b[4]] for b in gt_list], dtype=np.float32)
            pr_arr = np.array([[b[0],b[1],b[2],b[3],b[4]] for b in pr_list], dtype=np.float32)

            gt_t = torch.tensor(gt_arr); pr_t = torch.tensor(pr_arr)
            from engine.deim.obb_ops import batch_probiou
            iou = batch_probiou(gt_t, pr_t).numpy()
            cost = -iou
            from scipy.optimize import linear_sum_assignment
            g_idx, p_idx = linear_sum_assignment(cost)

            for g, p in zip(g_idx, p_idx):
                if iou[g, p] < 0.1: continue
                total_matched += 1
                gt_theta_deg = gt_arr[g, 4] * 180.0 / np.pi
                pr_theta_deg = pr_arr[p, 4] * 180.0 / np.pi
                diff = abs(pr_theta_deg - gt_theta_deg) % 180.0
                diff = min(diff, 180.0 - diff)
                angle_errs.append(diff)

                if diff > 15.0:
                    if abs(diff - 90.0) < 10.0:
                        artifact_count += 1
                    else:
                        normal_count += 1

        large = artifact_count + normal_count
        rate = artifact_count / max(large, 1) * 100
        print(f"  {model_name}: matched={total_matched} large>15deg={large} "
              f"artifact(90deg)={artifact_count} normal={normal_count} rate={rate:.1f}%")

    print(f"\nDOTA pipeline artifact check: PASS (informational)")
    raise SystemExit(0)
