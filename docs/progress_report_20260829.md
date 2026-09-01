# Slip Detection CNN 阶段汇报

日期：2026-08-29

## 1. 项目目标

本阶段目标是在 NLiPsTac 视觉触觉传感器采集的 RGB 图像序列上，训练一个二分类 CNN，用于判断当前接触状态是否发生滑动。模型输入为连续 8 帧触觉图像窗口，输出为 slip / non-slip 判断。

当前重点不是追求单一训练集上的最高准确率，而是验证模型在跨 session、跨材料、跨接触行为条件下的泛化能力。

## 2. 方法概述

当前主模型为 `slip_cnn_v2.py` 中的 stacked-frame CNN：

- 输入模式：`refdiff`，即当前帧减去该 session 的参考帧；
- 窗口长度：8 帧；
- 输入尺寸：320 x 180；
- 归一化：InstanceNorm；
- checkpoint 选择：使用 3-epoch smoothed balanced accuracy，避免选中单 epoch 噪声峰值；
- 数据增强：训练阶段加入随机平移增强，幅度 +/-8%，概率 0.5，前 3 个 epoch 不增强作为 warmup。

采用 session-level 划分，保证训练集、验证集、测试集之间没有窗口级泄漏。

## 3. 数据集情况

当前数据覆盖 4 类材料：

- plastic
- A4 paper
- soft foam earplug
- USB cable

最终训练/验证集来自 59 个 session，包含原有直线/旋转滑动行为，以及后续补充的 diagonal 行为。最终独立测试集 `test_windows_final_plus_diag.csv` 包含 22 个 session。

| Split | Windows | Sessions | Label 0 | Label 1 |
|---|---:|---:|---:|---:|
| Train | 2798 | 49 | 1399 | 1399 |
| Val | 294 | 10 | 160 | 134 |
| Test final_plus_diag | 1439 | 22 | 724 | 715 |

测试集按材料分布如下：

| Material | Windows |
|---|---:|
| plastic | 654 |
| A4 paper | 258 |
| soft foam earplug | 264 |
| USB cable | 263 |

## 4. 交叉验证结果

在 `cv_multi_material_augdiag_59` 上完成 5 折 session-level 交叉验证。各折 best smoothed balanced accuracy 如下：

| Fold | Best epoch | Best smoothed balanced_acc |
|---|---:|---:|
| 0 | 34 | 96.70% |
| 1 | 35 | 97.77% |
| 2 | 40 | 97.34% |
| 3 | 34 | 95.58% |
| 4 | 37 | 93.70% |

5 折平均 best smoothed balanced accuracy 为 **96.22%**，说明模型在当前训练分布和 session-level 验证设置下已经能够稳定学习 slip / non-slip 判别特征。

## 5. 最终模型与独立测试

最终模型训练完成于 **2026-08-27 20:42**。最佳 checkpoint 保存于 epoch 32：

- checkpoint：`model_v2_augdiag_59_augprob08_p05_warm3/slip_cnn_v2_best.pth`
- best smoothed balanced_acc：**96.55%**
- train samples：2798
- val samples：294

在独立测试集 `test_windows_final_plus_diag.csv` 上，默认阈值 0.5 的结果为：

| Metric | Value |
|---|---:|
| Accuracy | 87.77% |
| Balanced accuracy | 87.79% |
| F1 | 88.04% |
| Precision | 85.60% |
| Recall | 90.63% |
| Specificity | 84.94% |

混淆矩阵：

| TP | TN | FP | FN |
|---:|---:|---:|---:|
| 648 | 615 | 109 | 67 |

按材料划分：

| Material | Accuracy |
|---|---:|
| plastic | 84.40% |
| A4 paper | 77.91% |
| USB cable | 95.82% |
| soft foam earplug | 97.73% |

按运动类型划分：

| Motion type | Accuracy |
|---|---:|
| non-slip | 84.94% |
| rotation | 83.16% |
| translation | 99.69% |

结果表明：模型对 translational sliding 的识别非常稳定，但在 rotational sliding、A4 paper、plastic 以及部分非滑动强形变样本上仍存在混淆。

## 6. 阈值敏感性分析

最新阈值扫描完成于 **2026-08-29 21:16**。在 `test_windows_final_plus_diag.csv` 上，阈值从 0.5 调整到 0.7 后，balanced accuracy 从 **87.79%** 提升到 **88.32%**。

在旧测试集 `test_windows_final.csv` 上，阈值从 0.5 调整到 0.7 后，balanced accuracy 从 **85.41%** 提升到 **86.58%**。

该结果说明决策阈值对最终表现有小幅影响，但提升有限。报告中应将其表述为 threshold sensitivity analysis，而不是主要模型改进。

## 7. 当前问题

当前主要问题不是训练集拟合不足，而是独立测试集上的泛化差距：

- 5 折交叉验证平均 balanced accuracy 约 96.22%；
- 独立测试集 balanced accuracy 约 87.79%，最佳扫描阈值约 88.32%；
- 说明模型在已见分布附近表现稳定，但面对更难的 held-out session 时仍有下降。

主要误差来源包括：

- `diagonal constant-force pressing`：非滑动但存在明显形变，容易被误判为滑动；
- `rotational sliding`：召回仍不如 translational sliding；
- `A4 paper` 和 `plastic`：相对 USB cable、soft foam earplug 泛化更弱。

## 8. 下一步计划

在不增加新数据的前提下，后续可以优先完善以下内容：

1. 使用验证集选择部署阈值，再在测试集上固定评估；
2. 加入简单 temporal smoothing 或连续窗口投票，减少单窗口误判；
3. 在报告中补充 failure case 分析，重点解释非滑动强形变与旋转滑动的混淆；
4. 整理最终提交包，包含最终 checkpoint、训练日志、独立测试日志、阈值扫描日志和本报告。

## 9. 阶段结论

本阶段已经完成从数据采集、窗口构建、session-level 交叉验证、最终模型训练到独立测试评估的完整流程。当前模型在 5 折交叉验证中达到约 **96.22%** 的平均 balanced accuracy，在更严格的独立测试集上达到约 **87.79%** 的 balanced accuracy，阈值扫描后最高约 **88.32%**。

整体来看，该模型已经形成一个可复现、可提交的 slip detection baseline。现阶段的主要价值在于证明 RGB 视觉触觉序列可以有效区分 slip / non-slip，同时明确了下一阶段需要重点解决的泛化问题：旋转滑动、非滑动强形变以及部分材料上的跨 session 稳定性。

