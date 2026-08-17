# DEIMv2 真实数据验收测试方案

> 状态：**待用户确认训练配置**（确认后开始执行）
> 目标：工程化改造完成后，用真实数据验证功能性 CLI 与 API 接口的端到端可用性
> 范围调整（2026-08-15）：**移除全部 export 及导出后模型测试**（ONNX/TensorRT/OpenVINO）——该功能未完善，不在本轮验收范围

## 一、数据与环境（已确认）

| 项 | 值 |
|---|---|
| HBB 数据集 | `/mnt/d/project_data/model_test/deimv2_train_data/hook_coco`（COCO 格式，1 类 `bh_guagoutou`；train 860 图/1326 框，val 97/153，test 4353） |
| HBB 标注修正 | 原始类别 ID=1 不满足 app 层 0 起连续契约 → 已生成 0 起副本 `test/data/acceptance/hook_coco_annotations/`（原数据未动） |
| OBB 数据集 | `/mnt/d/project_data/model_test/deimv2_obb_train_data/synthetic_ellipse/density_002`（DOTA 格式，3 类 r/g/b；train 400 图，val 100 图，**val 兼作推理测试集**） |
| 主干权重 | `ckpts/dinov3_vits16_pretrain_lvd1689m-08c60483.pth`（ViT-S/16，384d，LVD-1689M 预训练） |
| GPU | 单卡 RTX 4060 Ti 16GB（已用 2.3GB）——**无多卡，torchrun 用例裁剪** |
| Comet | `COMET_API_KEY` 已入 `~/.bashrc`（可用） |

## 二、已创建的配置（待确认）

| 文件 | 内容 |
|---|---|
| `configs/app/presets/deimv2_dinov3_vits16_freeze_hbb.yml` | HBB preset 变体：`DINOv3STAs.finetune=False`（冻结主干），其余同 sp_hbb |
| `configs/app/presets/deimv2_dinov3_vits16_freeze_obb.yml` | OBB preset 变体：`DINOv3STAsResAtten` 换 `dinov3_vits16` 主干 + 冻结；维度不变（vits16 与 vits16plus 同为 384d） |
| `configs/app/base/{hbb,obb}_freeze_app.yml` | 对应 base 变体（public 默认值与原 base 一致） |
| `deim_app/config/loader.py` | `_APPROVED_BASE_PATHS` 登记两个新 base（信任边界保持仓库内策展） |
| `configs/app/acceptance/hook_coco_hbb.yml` | HBB 验收配置（数据路径见上，epochs=2） |
| `configs/app/acceptance/synthetic_obb.yml` | OBB 验收配置（epochs=2） |

### 关键训练参数（请确认）

| 参数 | HBB | OBB | 说明 |
|---|---|---|---|
| backbone | DINOv3STAs + vits16，冻结 | DINOv3STAsResAtten + vits16，冻结 | 用户指定；HBB 沿用 preset 原生 DINOv3STAs 类（ResAtten 是 OBB 矩阵用的类），**若 HBB 也想换 ResAtten 请指出** |
| epochs | 2 | 2 | 验收短训；质量训练再调 |
| batch_size | 4 | 4 | 16GB 单卡 + 冻结主干，预计显存富余 |
| learning_rate | 5e-4 | 5e-4 | preset 默认（backbone lr 组因冻结不生效） |
| amp | True | True | |
| early_stopping | 关 | 关 | 2 epoch 无意义 |
| eval batch | 4 | 2 | |
| 预检 | ✅ 构建OK 32.1M 参数，backbone 175/196 冻结 | ✅ 构建OK 43.2M 参数，backbone 175/202 冻结（余为 STA 适配器，设计如此） | CPU 级 YAML→resolver→引擎构建全链通过 |

**留白（用户补充）**：
- [ ] epochs=2 是否符合预期？希望改几轮：______
- [ ] HBB backbone 类选择：维持 DINOv3STAs（默认）/ 换 DINOv3STAsResAtten：______
- [ ] 短训是否需要在 mAP 上有任何下限预期（默认无，仅验 loss 下降）：______

## 三、接口清单（export 系已移除）

### L1 应用层 `deim_app`（主验收面）

| # | 接口 | 用例 |
|---|---|---|
| A1 | `python -m deim_app train -c <app.yml>` | T01 |
| A2 | `python -m deim_app eval -c <app.yml> -r <ckpt>` | T02 |
| A3 | `python -m deim_app infer -c <app.yml> -r <ckpt> -i <imgs> -o <dir> --format json/dota/visualization` | T03 |
| A5 | `DetectionModel.from_config().load().predict()/predict_filtered()` | T05 |
| A6 | API 负路径（未 load / 未知类名） | T06 |

### L2 引擎层

| # | 接口 | 用例 |
|---|---|---|
| E1 | `tools/train/train.py`（从零短训 / resume / `--test-only`） | T07/T08/T10 |
| E2 | `tools/inference/torch_inf.py`（HBB 推理） | T11a |
| E3 | `tools/inference/torch_inf_vis.py`（可视化推理） | T11b |
| E5 | `tools/compare/run_infer.py`（OBB 研究：DOTA 导出 + 可视化） | T13 |

### L4 脚本（单卡版）
- T15：`scripts/single_gpu_train.sh`、`single_gpu_val.sh`（多卡脚本本轮裁剪）

## 四、测试用例矩阵

**Checkpoint 来源**：全部由 T01/T07 短训产出（当前代码保存，自带 `shifted_v1` 标记）。

| 用例 | 层 | 命令要点 | 通过标准 |
|---|---|---|---|
| T01 | A1 | `train -c hook_coco_hbb.yml` | 退出 0；loss 有限且末轮<首轮；`outputs/acceptance/hook_hbb/` 产出 last/best checkpoint |
| T02 | A2 | `eval -r T01产物` | 退出 0；产出 mAP50/95 数值（无下限） |
| T03 | A3 | `infer` 三格式 ×（HBB json+vis / OBB json+dota+vis）；输入 HBB=hook val 97 图、OBB=density_002 val 100 图 | json 可解析且含 box_mode 字段；dota 行 8 坐标+类名+分值；vis 图片非空 |
| T05 | A5 | API predict/predict_filtered 同图同 ckpt | 框/分数与 T03 CLI 一致（1e-5 容差）；filter 链 score→class→top_k 生效 |
| T06 | A6 | 未 load 即 predict；class_filter 未知名 | 分别抛 InferenceBackendError / AppConfigError 且 CLI 退出码非 0 |
| T07 | E1 | `train.py -c`（OBB 引擎层等价短训——执行时按最小改动选定配置并记录） | 同 T01；checkpoint 含 meta.obb_angle_contract=shifted_v1 |
| T08 | E1 | `-r T07产物` 续训 1 epoch | 退出 0；last_epoch 递增 |
| T10 | E1 | `--test-only -r` | 产出 mAP |
| T11a/b | E2/E3 | HBB：hook val 抽样 10 图推理+可视化 | 标签/框/分数张量形状合法；vis 非空 |
| T13 | E5 | OBB：density_002 val → DOTA 导出 + vis | per-image .txt 行格式正确，θ 分布∈[0,π) |
| T15 | L4 | 两个单卡脚本各一次（短训参数） | 退出 0 |
| T16 | 一致性 | 同一 OBB ckpt：A3 vs E5 vs T05 三方对比 | 框/分数一致（容差内） |
| T17 | 契约负路径 | 无标记旧 ckpt（`outputs/dlzdt_ablation/abl_rep0.pth` 本机已有）过 resume/infer | 显式拒绝，报错含 obb_angle_contract |

**裁剪记录**：A4/T04（app export）、T12（engine ONNX 导出）、T14/D1-D3（onnx/trt/openvino 推理）、multi_gpu torchrun——功能未完善/无多卡，不计失败。

## 五、执行顺序与产物

1. 用户确认第二节配置 → 2. T01/T07 短训产出 checkpoint → 3. L1 其余用例 → 4. L2 → 5. 一致性与负路径 → 6. 汇总表

产物根：`outputs/acceptance/`（训练）与 `test/data/outputs/acceptance/<用例号>/`（推理/导出）。
汇总表字段：用例号 | 命令 | 退出码 | 关键指标 | 产物路径 | 结论（通过/失败/裁剪+原因）。

## 六、风险与已知约束

1. 16GB 单卡：OBB batch=4 + AMP 预计 <10GB；若 OOM 降 batch=2 重跑（记录即可）
2. `weights_only=True` 加载：本方案 checkpoint 均为当前代码产物，兼容
3. T07 的引擎层等价配置：OBB 短训走 app 配置展开或 jyz 引擎配置，执行时按最小改动选择并记录
4. hook_coco 的 test 集（4353 图）不用于 eval（时间考虑），仅作为可选推理压力输入
