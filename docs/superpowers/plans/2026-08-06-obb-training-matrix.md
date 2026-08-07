# DEIMv2 OBB Stage 1/2 功能稳定性训练矩阵 — 执行计划

> **For agentic workers:** 本计划由用户在自有硬件上执行（spec §1：「由用户执行」）。每个 Task 是独立的可验收单元，按序推进；可用 `subagent-driven-development` 或 `executing-plans` 辅助跟踪勾选框进度。

**Goal:** 执行 8-run × 80 epoch 功能稳定性训练矩阵，验证 DEIMv2 OBB Stage 1/2 改动在 train / resume / eval / infer 全链路的稳定性（无崩溃、无 NaN/Inf、输出契约合法）。

**Architecture:** 三层门控：Gate 1（8 配置 × 1 epoch 预检）→ Gate 2（7 配置全量 80 epoch + R2 前 40 epoch）→ Gate 3（R2 resume 续训、4 代表 eval、R2/R0 infer）。以 `synthetic_exp_020_anrep0_offset_per.yml` 为基线做单变量配置；R0–R3 用已有配置，R2-DN0/R2-BN0/R2-POST 新建。

**Tech Stack:** PyTorch + DEIMv2（`deimv2_daod/`）、`train.py` CLI、`test/tool_deimv2_obb_infer.py`（DOTA 导出）。

## Global Constraints

- 所有命令在 `deimv2_daod/` 根目录执行。
- **禁止修改** `engine/` 下任何代码；禁止修改已有配置 YAML；仅允许：新建 3 个配置 YAML、临时编辑 `test/tool_deimv2_obb_infer.py` 的 `infoes`（Task 7，跑完还原 `git checkout`）、创建结果记录文档。
- 验收仅限功能稳定性：无 AP/损失回归门槛；`mAP50_95 > 0` 为 sanity。
- NaN/Inf 处理：任一 run 出现 → 该配置重跑 1 次区分偶发/缺陷；持续存在 → 上报，不掩盖。
- resume 断点回退链：`checkpoint0039.pth` → 最近的 `checkpoint0029/0019/0009.pth` → 全缺失则从头重训并记录。
- 不涉及 `use_focal_loss=False`（spec §7 排除项，head 无背景通道、criterion 仅 focal）。
- git commit 仅在用户明确指示时执行。
- epoch 语义（已核实 `engine/core/_config.py:58` + `engine/solver/det_solver.py:100,133`）：`last_epoch` 默认 -1 → `start_epoch = last_epoch + 1 = 0` → 循环 `range(start_epoch, epoches)` 0-indexed；`epoches: 80` 恰训 80 个 epoch（epoch 0..79）；resume 后 `start_epoch = last_epoch + 1`；resume 且 `last_epoch > 0` 时续训前自动执行一次 eval。
- checkpoint（`engine/solver/det_solver.py:188-196`）：每 epoch 末写 `last.pth`；`(epoch+1) % checkpoint_freq == 0` 写 `checkpoint{epoch:04}.pth`（freq=10 → 0009/0019/0029/0039/0049/0059/0069/0079）。`checkpoint0039.pth` = epoch 39 结束 = 已训 40 epoch。
- Comet 干扰抑制（可选）：预检/训练前 `COMET_API_KEY=` 置空可跳过 `init_comet_experiment`（`train.py:60-92`，空 key 走 falsy 分支）。
- 结果统一记录到 `docs/superpowers/records/2026-08-06-obb-matrix-results.md`（Task 8 模板，各 Task 分步填写）。

---

### Task 1: 创建 3 个新配置 YAML

**Files:**
- Create: `configs/custom_obb/synthetic_configs/synthetic_exp_020_anrep2_dn0.yml`
- Create: `configs/custom_obb/synthetic_configs/synthetic_exp_020_anrep2_bn0.yml`
- Create: `configs/custom_obb/synthetic_configs/synthetic_exp_020_anrep2_offset_post.yml`

**Interfaces:**
- Consumes: `configs/custom_obb/synthetic_configs/synthetic_exp_020_anrep2_offset_per.yml`（模板，已核实 `angle_rep: 2`、`output_dir: ./outputs/synthetic_exp_020_anrep2_offset_per`）
- Produces: 3 个单字段差异配置，供 Task 2/4/6 引用

- [ ] **Step 1: 复制模板为 3 个新文件**

```bash
cd /mnt/d/cx/thired/deimv2_daod
cp configs/custom_obb/synthetic_configs/synthetic_exp_020_anrep2_offset_per.yml configs/custom_obb/synthetic_configs/synthetic_exp_020_anrep2_dn0.yml
cp configs/custom_obb/synthetic_configs/synthetic_exp_020_anrep2_offset_per.yml configs/custom_obb/synthetic_configs/synthetic_exp_020_anrep2_bn0.yml
cp configs/custom_obb/synthetic_configs/synthetic_exp_020_anrep2_offset_per.yml configs/custom_obb/synthetic_configs/synthetic_exp_020_anrep2_offset_post.yml
```

- [ ] **Step 2: 编辑 dn0 配置（2 处修改）**

`configs/custom_obb/synthetic_configs/synthetic_exp_020_anrep2_dn0.yml`：
- 第 2 行 `output_dir: ./outputs/synthetic_exp_020_anrep2_offset_per` → `output_dir: ./outputs/synthetic_exp_020_anrep2_dn0`
- 第 241 行 `num_denoising: 100` → `num_denoising: 0`

- [ ] **Step 3: 编辑 bn0 配置（2 处修改）**

`configs/custom_obb/synthetic_configs/synthetic_exp_020_anrep2_bn0.yml`：
- 第 2 行 `output_dir: ./outputs/synthetic_exp_020_anrep2_offset_per` → `output_dir: ./outputs/synthetic_exp_020_anrep2_bn0`
- 第 243 行 `box_noise_scale: 0.5` → `box_noise_scale: 0`

- [ ] **Step 4: 编辑 offset_post 配置（3 处修改）**

`configs/custom_obb/synthetic_configs/synthetic_exp_020_anrep2_offset_post.yml`：
- 第 2 行 `output_dir: ./outputs/synthetic_exp_020_anrep2_offset_per` → `output_dir: ./outputs/synthetic_exp_020_anrep2_offset_post`
- 第 258 行 `DEIMTransformer.offset_scale_source: "pre"` → `"post"`
- 第 303 行 `DEIMCriterion.offset_scale_source: "pre"` → `"post"`

- [ ] **Step 5: 验证差异仅为目标字段**

```bash
diff configs/custom_obb/synthetic_configs/synthetic_exp_020_anrep2_offset_per.yml configs/custom_obb/synthetic_configs/synthetic_exp_020_anrep2_dn0.yml
diff configs/custom_obb/synthetic_configs/synthetic_exp_020_anrep2_offset_per.yml configs/custom_obb/synthetic_configs/synthetic_exp_020_anrep2_bn0.yml
diff configs/custom_obb/synthetic_configs/synthetic_exp_020_anrep2_offset_per.yml configs/custom_obb/synthetic_configs/synthetic_exp_020_anrep2_offset_post.yml
python -c "import yaml,sys; [yaml.safe_load(open(f)) for f in sys.argv[1:]]" configs/custom_obb/synthetic_configs/synthetic_exp_020_anrep2_dn0.yml configs/custom_obb/synthetic_configs/synthetic_exp_020_anrep2_bn0.yml configs/custom_obb/synthetic_configs/synthetic_exp_020_anrep2_offset_post.yml
```

Expected: 每个 diff 只显示 output_dir + 目标字段；YAML 解析无异常。

- [ ] **Step 6: 记录（可选 commit，需用户指示）**

在 `docs/superpowers/records/2026-08-06-obb-matrix-results.md` 记录 3 文件路径与 diff 摘要。

**Deliverable / PASS:** 3 个配置 YAML 存在、YAML 可解析、与模板 diff 仅含 spec §5 列出的字段。

---

### Task 2: Gate 1 预检（8 配置 × 1 epoch）

**Files:**
- Create (output): `./outputs/preflight/{anrep0,anrep1,anrep2,anrep3,anrep2_dn0,anrep2_bn0,anrep2_offset_post}/`

**Interfaces:**
- Consumes: Task 1 的 3 个新配置 + 4 个已有配置
- Produces: 每配置 scratch `last.pth`（供 Task 3 resume 冒烟复用 anrep2 的）

- [ ] **Step 1: 逐配置跑预检（8 条命令，可并行）**

```bash
cd /mnt/d/cx/thired/deimv2_daod
mkdir -p ./outputs/preflight   # 先建目录，避免 tee 与 train.py 建目录竞态
for name in anrep0 anrep1 anrep2 anrep3 anrep2_dn0 anrep2_bn0 anrep2_offset_post; do
  cfg="configs/custom_obb/synthetic_configs/synthetic_exp_020_${name}_offset_per.yml"
  [ "$name" = "anrep2_dn0" ] && cfg="configs/custom_obb/synthetic_configs/synthetic_exp_020_anrep2_dn0.yml"
  [ "$name" = "anrep2_bn0" ] && cfg="configs/custom_obb/synthetic_configs/synthetic_exp_020_anrep2_bn0.yml"
  [ "$name" = "anrep2_offset_post" ] && cfg="configs/custom_obb/synthetic_configs/synthetic_exp_020_anrep2_offset_post.yml"
  echo "=== PREFLIGHT $name ==="
  COMET_API_KEY= python train.py -c "$cfg" -u epoches=1 print_freq=10 output_dir=./outputs/preflight/$name 2>&1 | tee ./outputs/preflight/$name.log
done
```

- [ ] **Step 2: 检查每配置通过条件**

```bash
for name in anrep0 anrep1 anrep2 anrep3 anrep2_dn0 anrep2_bn0 anrep2_offset_post; do
  echo "=== $name ==="
  ls -la ./outputs/preflight/$name/last.pth 2>&1
  grep -iE "nan|inf" ./outputs/preflight/$name.log | head -3 || true
  grep -E "loss" ./outputs/preflight/$name.log | head -2
done
```

Expected（PASS）: 每配置 `last.pth` 存在；log 无 `nan/inf` 字样；≥1 条有限 loss 行；无异常堆栈。任一项失败 → 该配置不进 Gate 2，上报排查（区分配置错误与代码缺陷）。

- [ ] **Step 3: 记录**

`docs/superpowers/records/2026-08-06-obb-matrix-results.md` 记录 8 配置 Gate 1 pass/fail + 首条 loss。

**Deliverable / PASS:** 8 配置全部通过；结果表已记录。

---

### Task 3: Gate 1b resume 冒烟（R2-RESUME 专属）

**Files:**
- Consume: `./outputs/preflight/anrep2/last.pth`（Task 2 产物，`last_epoch=0`）
- Create (output): `./outputs/preflight/anrep2/`（追加 epoch 1 记录）

**Interfaces:**
- Consumes: Task 2 anrep2 预检产物
- Produces: resume 机制冒烟验证（供 Task 5 正式续训的预演）

- [ ] **Step 1: 从 scratch last.pth 续训 1 epoch**

```bash
cd /mnt/d/cx/thired/deimv2_daod
COMET_API_KEY= python train.py \
  -c configs/custom_obb/synthetic_configs/synthetic_exp_020_anrep2_offset_per.yml \
  -u epoches=2 print_freq=10 output_dir=./outputs/preflight/anrep2 \
  -r ./outputs/preflight/anrep2/last.pth 2>&1 | tee ./outputs/preflight/anrep2_resume.log
```

Expected: 启动时打印 `Load last_epoch:0`；续训 epoch 1（`range(1,2)` 恰 1 个真实 epoch）；loss 有限；无异常。

- [ ] **Step 2: 验证续训生效**

```bash
grep -E "Load last_epoch|epoch" ./outputs/preflight/anrep2_resume.log | head -5
grep -iE "nan|inf" ./outputs/preflight/anrep2_resume.log | head -3 || echo "NO-NAN-INF"
```

Expected（PASS）: `last_epoch` 从 0 恢复（start_epoch=1）；log 出现 epoch 1 训练记录；无 nan/inf。

- [ ] **Step 3: 记录**

结果表记录 pass/fail + 恢复的 last_epoch 值。

**Deliverable / PASS:** resume 冒烟通过 → 解锁 Task 5 正式续训。

---

### Task 4: Gate 2 完整训练（7 配置全量 + R2 前 40 epoch）

**Files:**
- Create (output): `./outputs/synthetic_exp_020_{anrep0,anrep1,anrep2,anrep3}_offset_per/`、`./outputs/synthetic_exp_020_anrep2_{dn0,bn0,offset_post}/`
- Create (artifact): 每 run `log.txt`、`last.pth`、`checkpoint*.pth`、`best_stg1/best_stg2.pth`、`eval/`

**Interfaces:**
- Consumes: Task 1 配置 + Task 2 预检通过结论
- Produces: 7 个完整 run 产物 + R2 的 `checkpoint0039.pth`（供 Task 5）

- [ ] **Step 1: R0 全量训练**

```bash
cd /mnt/d/cx/thired/deimv2_daod
# 预建全部输出目录，避免 tee 与 train.py 建目录竞态
mkdir -p ./outputs/synthetic_exp_020_anrep0_offset_per ./outputs/synthetic_exp_020_anrep1_offset_per ./outputs/synthetic_exp_020_anrep2_offset_per ./outputs/synthetic_exp_020_anrep3_offset_per ./outputs/synthetic_exp_020_anrep2_dn0 ./outputs/synthetic_exp_020_anrep2_bn0 ./outputs/synthetic_exp_020_anrep2_offset_post
COMET_API_KEY= python train.py -c configs/custom_obb/synthetic_configs/synthetic_exp_020_anrep0_offset_per.yml 2>&1 | tee ./outputs/synthetic_exp_020_anrep0_offset_per/train.log
```

- [ ] **Step 2: R1 全量训练**

```bash
COMET_API_KEY= python train.py -c configs/custom_obb/synthetic_configs/synthetic_exp_020_anrep1_offset_per.yml 2>&1 | tee ./outputs/synthetic_exp_020_anrep1_offset_per/train.log
```

- [ ] **Step 3: R3 全量训练**

```bash
COMET_API_KEY= python train.py -c configs/custom_obb/synthetic_configs/synthetic_exp_020_anrep3_offset_per.yml 2>&1 | tee ./outputs/synthetic_exp_020_anrep3_offset_per/train.log
```

- [ ] **Step 4: R2-DN0 全量训练**

```bash
COMET_API_KEY= python train.py -c configs/custom_obb/synthetic_configs/synthetic_exp_020_anrep2_dn0.yml 2>&1 | tee ./outputs/synthetic_exp_020_anrep2_dn0/train.log
```

- [ ] **Step 5: R2-BN0 全量训练**

```bash
COMET_API_KEY= python train.py -c configs/custom_obb/synthetic_configs/synthetic_exp_020_anrep2_bn0.yml 2>&1 | tee ./outputs/synthetic_exp_020_anrep2_bn0/train.log
```

- [ ] **Step 6: R2-POST 全量训练**

```bash
COMET_API_KEY= python train.py -c configs/custom_obb/synthetic_configs/synthetic_exp_020_anrep2_offset_post.yml 2>&1 | tee ./outputs/synthetic_exp_020_anrep2_offset_post/train.log
```

- [ ] **Step 7: R2 训练至 epoch 39 结束（40 epochs）后中断**

```bash
COMET_API_KEY= python train.py -c configs/custom_obb/synthetic_configs/synthetic_exp_020_anrep2_offset_per.yml 2>&1 | tee ./outputs/synthetic_exp_020_anrep2_offset_per/train.log
```

监视 `./outputs/synthetic_exp_020_anrep2_offset_per/`：`checkpoint0039.pth` 落盘（epoch 39 训练结束、该 epoch eval 前后均可）后中断进程（Ctrl-C）。checkpoint 在训练后、eval 前写入（`det_solver.py:188-196`），中断不丢数据；Task 5 resume 时会先自动 eval。

- [ ] **Step 8: 逐 run 验证 Gate 2 通过条件**

```bash
for d in synthetic_exp_020_anrep0_offset_per synthetic_exp_020_anrep1_offset_per synthetic_exp_020_anrep2_offset_per synthetic_exp_020_anrep3_offset_per synthetic_exp_020_anrep2_dn0 synthetic_exp_020_anrep2_bn0 synthetic_exp_020_anrep2_offset_post; do
  echo "=== $d ==="
  ls outputs/$d/last.pth outputs/$d/checkpoint0079.pth outputs/$d/best_stg2.pth 2>&1
  grep -iE "nan|inf" outputs/$d/log.txt | head -3 || echo "NO-NAN-INF"
  tail -1 outputs/$d/log.txt
done
```

Expected（PASS）: `last.pth` + `checkpoint0079.pth` + `best_stg2.pth` 落盘（R2 当前仅需 `checkpoint0039.pth` + `last.pth`）；`log.txt` 无 nan/inf；末条 `log.txt` 含有限 loss 与 `mAP50_95`（>0）。

- [ ] **Step 9: 记录**

结果表记录 8 run 完成状态、末 epoch loss、`mAP50_95`（不设门槛）、checkpoint 清单。

**Deliverable / PASS:** 7 个 run 完成 80 epoch（epoch 0..79）且产物齐全；R2 已训 40 epoch 且 `checkpoint0039.pth` 就绪。

---

### Task 5: Gate 3a R2-RESUME 续训至完成

**Files:**
- Consume: `./outputs/synthetic_exp_020_anrep2_offset_per/checkpoint0039.pth`
- Create (output): `./outputs/synthetic_exp_020_anrep2_offset_per/`（续训 epoch 40..79，覆盖 `last.pth`，追加 `checkpoint0049..0079.pth`）

**Interfaces:**
- Consumes: Task 4 Step 7 的 `checkpoint0039.pth`
- Produces: R2 最终 `last.pth` / `checkpoint0079.pth` / `best_stg2.pth`（供 Task 6/7）

- [ ] **Step 1: 确认断点存在，否则回退**

```bash
ls -la ./outputs/synthetic_exp_020_anrep2_offset_per/checkpoint0039.pth || \
ls -la ./outputs/synthetic_exp_020_anrep2_offset_per/checkpoint00{29,19,09}.pth
```

Expected: 找到任一断点；全缺失 → 从头重训 R2 并记录。

- [ ] **Step 2: resume 续训**

```bash
cd /mnt/d/cx/thired/deimv2_daod
COMET_API_KEY= python train.py \
  -c configs/custom_obb/synthetic_configs/synthetic_exp_020_anrep2_offset_per.yml \
  -r ./outputs/synthetic_exp_020_anrep2_offset_per/checkpoint0039.pth 2>&1 | tee ./outputs/synthetic_exp_020_anrep2_offset_per/resume.log
```

Expected: 启动打印 `Load last_epoch:39`；resume 前自动 eval（`last_epoch=39 > 0`）无异常；续训 epoch 40..79；结束态与 Task 4 其他 run 一致。

- [ ] **Step 3: 验证最终产物**

```bash
ls -la ./outputs/synthetic_exp_020_anrep2_offset_per/last.pth ./outputs/synthetic_exp_020_anrep2_offset_per/checkpoint0079.pth ./outputs/synthetic_exp_020_anrep2_offset_per/best_stg2.pth
grep -iE "nan|inf" ./outputs/synthetic_exp_020_anrep2_offset_per/log.txt | head -3 || echo "NO-NAN-INF"
```

Expected（PASS）: 三文件就绪；log 无 nan/inf；resume.log 含 epoch 40 起始记录。

- [ ] **Step 4: 记录**

结果表记录 resume 状态、续训起止 epoch、最终 `mAP50_95`。

**Deliverable / PASS:** R2 经 40+40 两段完成 80 epoch，产物与全量 run 等价。

---

### Task 6: Gate 3b test-only eval（4 个代表终检）

**Files:**
- Consume: 4 个 `last.pth`
- Create (output): 各 `output_dir` 追加 eval 输出

**Interfaces:**
- Consumes: Task 4/5 的 `last.pth`（R0、R2、R2-DN0、R2-POST）
- Produces: 4 组独立 eval 指标

- [ ] **Step 1: R0 eval**

```bash
cd /mnt/d/cx/thired/deimv2_daod
COMET_API_KEY= python train.py -c configs/custom_obb/synthetic_configs/synthetic_exp_020_anrep0_offset_per.yml -r ./outputs/synthetic_exp_020_anrep0_offset_per/last.pth --test-only
```

- [ ] **Step 2: R2 eval**

```bash
COMET_API_KEY= python train.py -c configs/custom_obb/synthetic_configs/synthetic_exp_020_anrep2_offset_per.yml -r ./outputs/synthetic_exp_020_anrep2_offset_per/last.pth --test-only
```

- [ ] **Step 3: R2-DN0 eval**

```bash
COMET_API_KEY= python train.py -c configs/custom_obb/synthetic_configs/synthetic_exp_020_anrep2_dn0.yml -r ./outputs/synthetic_exp_020_anrep2_dn0/last.pth --test-only
```

- [ ] **Step 4: R2-POST eval**

```bash
COMET_API_KEY= python train.py -c configs/custom_obb/synthetic_configs/synthetic_exp_020_anrep2_offset_post.yml -r ./outputs/synthetic_exp_020_anrep2_offset_post/last.pth --test-only
```

- [ ] **Step 5: 验证与记录**

Expected（PASS）: 4 次 eval 均无异常退出、产出有限数值指标（含 `mAP50_95`）。结果表记录各 run 指标（不设门槛）。

**Deliverable / PASS:** 4 组 eval 完成且指标有限。

---

### Task 7: Gate 3c OBB infer（R2 + R0）

**Files:**
- Modify (temporary): `test/tool_deimv2_obb_infer.py` 的 `__main__` 中 `infoes` 列表（Task 完成后 `git checkout` 还原）
- Create (output): `./test/data/outputs/synthetic_res/anrep2_val/`、`./test/data/outputs/synthetic_res/anrep0_val/`

**Interfaces:**
- Consumes: R2/R0 的 `last.pth`（Task 4/5 产物）
- Produces: 每图 DOTA txt（`x1 y1 x2 y2 x3 y3 x4 y4 class confidence`）

- [ ] **Step 1: 备份工具文件**

```bash
cd /mnt/d/cx/thired/deimv2_daod
cp test/tool_deimv2_obb_infer.py /tmp/opencode/tool_deimv2_obb_infer.py.bak
```

- [ ] **Step 2: 替换 `__main__` 中 `img_dir` / `classes_txt` / `imgsz` / `infoes`**

将 `test/tool_deimv2_obb_infer.py` 第 240-283 行的 `__main__` 块替换为：

```python
if __name__ == "__main__":
    img_dir = (
        "/mnt/d/project_data/model_test/deimv2_obb_train_data/synthetic_ellipse/density_020/val"
    )
    classes_txt = (
        "/mnt/d/project_data/model_test/deimv2_obb_train_data/synthetic_ellipse/classes.txt"
    )
    imgsz = (256, 256)
    max_det = 300
    score_threshold = 0.2
    device = "cuda:0"

    infoes = [
        {
            "config": "configs/custom_obb/synthetic_configs/synthetic_exp_020_anrep2_offset_per.yml",
            "ckpt": "outputs/synthetic_exp_020_anrep2_offset_per/last.pth",
            "output_dir": "./test/data/outputs/synthetic_res/anrep2_val",
            "infer_flag": True,
        },
        {
            "config": "configs/custom_obb/synthetic_configs/synthetic_exp_020_anrep0_offset_per.yml",
            "ckpt": "outputs/synthetic_exp_020_anrep0_offset_per/last.pth",
            "output_dir": "./test/data/outputs/synthetic_res/anrep0_val",
            "infer_flag": True,
        },
    ]

    for info in infoes:
        infer_obb_and_export(
            img_dir=img_dir,
            ckpt=info["ckpt"],
            config=info["config"],
            output_dir=info["output_dir"],
            classes_txt=classes_txt,
            imgsz=imgsz,
            max_det=max_det,
            score_threshold=score_threshold,
            device=device,
        )
```

（`infer_obb_and_export` 签名无 `infer_flag` 参数，`tool_deimv2_obb_infer.py:115-125`；原 `__main__` 调用亦不传该参数，`infoes` 中的 `infer_flag` 为历史遗留死数据。）

- [ ] **Step 3: 运行 infer**

```bash
cd /mnt/d/cx/thired/deimv2_daod
python test/tool_deimv2_obb_infer.py 2>&1 | tee /tmp/opencode/obb_infer.log
```

- [ ] **Step 4: 验证 DOTA 输出契约**

```bash
# 两个输出目录都应非空且含每图 txt
ls ./test/data/outputs/synthetic_res/anrep2_val/ | head -5
ls ./test/data/outputs/synthetic_res/anrep2_val/ | wc -l
ls ./test/data/outputs/synthetic_res/anrep0_val/ | wc -l
# 抽查首行格式与置信度范围
head -1 "$(ls ./test/data/outputs/synthetic_res/anrep2_val/*.txt | head -1)"
```

Expected（PASS）: 全部图像处理完成（`Inference completed: N images with detections`）；每行 10 字段（8 坐标 + 类别名 + 置信度）；置信度 ∈ [0,1]；类别名 ∈ `classes.txt`；无异常堆栈。

- [ ] **Step 5: 还原工具文件**

```bash
cd /mnt/d/cx/thired/deimv2_daod
git checkout -- test/tool_deimv2_obb_infer.py
git diff --stat   # 应无输出
```

- [ ] **Step 6: 记录**

结果表记录 infer 成功图数 / 总图数、输出样例、异常数。

**Deliverable / PASS:** R2/R0 infer 输出合法 DOTA txt；工具文件已还原。

---

### Task 8: 结果归档与结论

**Files:**
- Create: `docs/superpowers/records/2026-08-06-obb-matrix-results.md`

**Interfaces:**
- Consumes: Task 1-7 各步骤记录

- [ ] **Step 1: 汇总结果表**

`docs/superpowers/records/2026-08-06-obb-matrix-results.md` 按以下模板汇总（含各 Task 已填记录）：

| run | 配置 | Gate 1 | Gate 2 完成 | 末 epoch loss | mAP50_95 | 产物齐全 | Gate 3 eval | infer | 异常 |
|---|---|---|---|---|---|---|---|---|---|
| R0 | anrep0_offset_per | ✅/❌ | epoch 0..79 | 值/NaN | 值 | last+0079+best_stg2 | ✅/❌ | ✅/❌ | 摘要 |
| R1 | anrep1_offset_per | ... | | | | | | | |
| R2 | anrep2_offset_per | ... | 40+40（含 resume） | | | | | | |
| R3 | anrep3_offset_per | ... | | | | | | | |
| R2-DN0 | anrep2_dn0 | ... | | | | | | | |
| R2-BN0 | anrep2_bn0 | ... | | | | | | | |
| R2-POST | anrep2_offset_post | ... | | | | | | | |

- [ ] **Step 2: NaN/Inf 事件清单**

列出每次 NaN/Inf 出现位置、重跑结果、结论（偶发 / 缺陷）。

- [ ] **Step 3: 结论**

- 全绿 → 声明「Stage 1/2 功能稳定性 PASS」
- 有缺陷 → 逐项列缺陷现象、复现 run、建议修复入口（engine/ 修改需另立任务，本计划不改代码）

**Deliverable / PASS:** 结果文档完整、结论明确（通过 / 缺陷项列表）。

---

## Self-Review（写后自查）

**Spec 覆盖：**
- spec §2 验收标准 → Task 2（Gate 1）/ Task 3（resume 冒烟）/ Task 4（Gate 2）/ Task 5（Gate 3a）/ Task 6（Gate 3b）/ Task 7（Gate 3c）✓
- spec §4 运行矩阵 8 runs → Task 1（3 新配置）+ Task 4（R0/R1/R3/3 新配置全量 + R2 前半）+ Task 5（R2-RESUME）✓
- spec §5 新建配置字段 → Task 1 Step 2-4（含行号）✓
- spec §6 命令清单 → Task 2-7 展开为逐条命令 ✓
- spec §7 non-focal 排除 → Global Constraints 声明 ✓
- spec §8 结果记录 → Task 8 模板 ✓
- spec §9 风险回退 → Global Constraints + Task 5 Step 1 回退链 ✓
- spec §10 范围 → Global Constraints（不改代码/已有配置）✓

**占位符扫描:** 无 TBD/TODO；`<name>`/`<CFG>` 类引用已在 Task 内展开为字面值。✓

**一致性:** Task 4 Step 7 中断点（`checkpoint0039.pth`，epoch 39 结束=已训 40 epoch）与 spec §3 epoch 语义、Task 5 恢复逻辑（`last_epoch=39 → start_epoch=40`）一致；Task 3 冒烟（预检 `last_epoch=0 → epoches=2 → epoch 1`）与 spec §6 Gate 1b 一致。✓
