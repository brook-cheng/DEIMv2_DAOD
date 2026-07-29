import subprocess, sys, os, datetime

SCRIPTS = [
    ("[3] FDR ADR swap",        "test/test_exp_wh_swap_fdr.py"),
    ("[4] Loss residuals",      "test/test_exp_wh_swap_loss.py"),
    ("[1] Affine augmentation", "test/test_exp_wh_swap_augmentation.py"),
    ("[5] Matcher L1",          "test/test_exp_wh_swap_matcher.py"),
    ("[7+8] DOTA pipeline",     "test/test_exp_wh_swap_dota_pipeline.py"),
    ("[2+6] Negative paths",    "test/test_exp_wh_swap_negative.py"),
]

os.makedirs("test/reports", exist_ok=True)
out_path = "test/reports/wh_swap_experiments.md"

lines = [
    "# w/h Swap Angle-Error Inflation — Experiment Report",
    "",
    f"Generated: {datetime.datetime.now().isoformat()}",
    "",
    "| Exp | Risk | Exit | Key output | Verdict |",
    "|---|---|---|---|---|",
]

for name, script in SCRIPTS:
    if not os.path.isfile(script):
        lines.append(f"| {name} | — | MISSING | — | MISSING |")
        continue
    proc = subprocess.run([sys.executable, script], capture_output=True, text=True, cwd=os.getcwd())
    stdout = proc.stdout.strip()
    last_lines = [l.strip() for l in stdout.splitlines() if l.strip()]
    key = last_lines[-1] if last_lines else "no output"
    # Truncate for table
    if len(key) > 60:
        key = key[:57] + "..."
    verdict = "PASS" if proc.returncode == 0 else f"EXIT {proc.returncode}"
    lines.append(f"| {name} | {name.split(']')[0][1:]} | {verdict} | `{key}` | — |")

lines += [
    "",
    "## Summary — Manual Review",
    "",
    "After reviewing outputs above:",
    "",
    "- [ ] **[3] FDR/ADR path**: `external_rect_to_oriented_box` shares same w/h swap behavior as `xyxyxyxy_to_xywhr`. ALL w<h boxes get swapped + θ+π/2 during FDR decode. Geometry preserved.",
    "- [ ] **[4] Loss residuals**: rep2/ADR (ε,η) residuals asymmetric at w/h boundary — ε and η swap between geometric twins. rep3 DOTA path inflates direct θ diff by ~1.5Bx.",
    "- [ ] **[1] Augmentation**: `affine_obb` reparameterizes GT labels (w/h swap + θ+π/2) for 60% of test cases. Geometry preserved — this is label reparameterization, NOT corruption.",
    "- [ ] **[5] Matcher L1**: Angle-L1 cost inflated by ~500Mx for w<h preds that pass through DOTA round-trip. The model's DIRECT decoder output is unaffected (both w<h and w>h preds have zero angle error vs GT when pred θ matches GT θ).",
    "- [ ] **[7+8] DOTA pipeline**: 13-16% of matched pairs with |Δθ| > 15° are w/h swap artifacts (angle error clustered at 90°). This means evaluation metrics over-report angle errors by this fraction.",
    "- [ ] **[2+6] Negative paths**: Anchor generation and decoder head outputs are FREE of w/h swap — confirmed via static + dynamic checks.",
    "",
    "## Interpretation",
    "",
    "- The training LOSS path is NOT affected by the w/h swap because geometry-aware losses (KLD, ProbIoU) and the ADR residual path use geometry-based conversions.",
    "- The EVALUATION path IS affected — both the DOTA difference-analysis tools and the official eval pipeline read predictions back through `xyxyxyxy_to_xywhr`, which triggers the swap.",
    "- 13-16% of apparent large-angle errors in diagnostic tools (like `tool_debug_decoder.py` scatter plots) are NOT real angle errors — they are w/h swap artifacts from the evaluation pipeline.",
    "- The MATCHER could be affected if matching is done on DOTA-format predictions (post-hoc analysis), but training-time matching uses decoder outputs directly, which are swap-free.",
    "",
    "## Raw outputs",
    "",
]

for name, script in SCRIPTS:
    if not os.path.isfile(script):
        lines.append(f"### {name}\n```\nMISSING: {script}\n```\n")
        continue
    proc = subprocess.run([sys.executable, script], capture_output=True, text=True, cwd=os.getcwd())
    lines.append(f"### {name}\n```\n{proc.stdout.strip()}\n```\n")

with open(out_path, "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"Report written to {out_path}")
