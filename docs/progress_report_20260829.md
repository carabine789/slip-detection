# Slip Detection CNN 阶段汇报

日期：2026-08-29

## 1. 项目目标

本阶段目标是在 NLiPsTac 视觉触觉传感器采集的 RGB 图像序列上，训练一个二分类 CNN，用于判断当前接触状态是否发生滑动。模型输入为连续 8 帧触觉图像窗口，输出为 slip / non-slip 判断。

当前重点是验证模型在跨 session、跨材料、跨接触行为条件下的泛化能力。

## 2. 方法概述

当前主模型为 `slip_cnn_v2.py` 中的 stacked-frame CNN：

- 输入模式：`refdiff`，即当前帧减去该 session 的参考帧；
- 窗口长度：8 帧；
- 输入尺寸：320 x 180；
- 归一化：InstanceNorm；
- checkpoint 选择：使用 3-epoch smoothed balanced accuracy，避免选中单 epoch 噪声峰值；
- 数据增强：训练阶段加入随机平移增强，幅度 +/-8%，概率 0.5，前 3 个 epoch 不增强作为 warmup。

采用 session-level 划分，保证训练集、验证集、测试集之间没有窗口级泄漏。

模型设计上主要参考 PPTac 中利用多帧触觉图像判断滑移的思路，先使用结构较简单的 CNN 作为 baseline。早期版本在训练后期出现过验证集指标明显震荡的问题，相邻 epoch 的 balanced accuracy 有时会相差较大；因此后续加入了梯度裁剪、学习率自动衰减、early stopping，以及使用平滑后的 balanced accuracy 选择 checkpoint。

另外，7 月底的可视化检查发现，部分误判和接触位置、session 间外观差异有关。例如物体按压在传感器边缘时更容易被判为 slip，而在中心区域时相对稳定。这说明模型可能学习到了一些和空间位置或单个 session 外观相关的 shortcut。基于这个观察，后续训练中加入了随机平移增强，并使用 `refdiff` 与 InstanceNorm 来减弱 session 背景差异的影响。

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

## 4. 数据划分与 Session Metadata

本阶段最终训练与验证使用 `data/raw_cnn/train_windows_multi_material_augdiag.csv` 和 `data/raw_cnn/val_windows_multi_material_augdiag.csv`。独立测试使用 `data/raw_cnn_test/test_windows_final_plus_diag.csv`。需要注意的是，训练/验证集与测试集位于不同数据根目录下，因此测试集中的 `session_001` 等编号不对应训练集中的同名 session。

原始 RGB 图像序列、窗口 CSV、reference image 和模型 checkpoint 文件体积较大，未直接纳入 Git 仓库。交接时建议将 `data/raw_cnn/`、`data/raw_cnn_test/` 和最终 checkpoint 作为单独数据包提供；Git 仓库中保留代码、报告和关键日志。

### 4.1 训练集 Sessions

| Session | Material | Non-slip behavior | Slip behavior | Motion | Frames | FPS | Windows | Label 0/1 |
|---|---|---|---|---|---:|---:|---:|---:|
| session_001 | plastic | continuous variable-force pressing | translational sliding | translation | 633 | 26.363 | 47 | 28/19 |
| session_002 | plastic | continuous variable-force pressing | translational sliding | translation | 632 | 26.312 | 43 | 24/19 |
| session_004 | plastic | continuous variable-force pressing | translational sliding | translation | 632 | 26.295 | 49 | 30/19 |
| session_005 | plastic | continuous variable-force pressing | translational sliding | translation | 637 | 26.510 | 48 | 29/19 |
| session_006 | plastic | constant-force pressing | translational sliding | translation | 650 | 27.081 | 49 | 29/20 |
| session_007 | plastic | constant-force pressing | translational sliding | translation | 634 | 26.385 | 48 | 28/20 |
| session_009 | plastic | constant-force pressing | translational sliding | translation | 655 | 27.252 | 45 | 26/19 |
| session_010 | plastic | constant-force pressing | translational sliding | translation | 646 | 26.909 | 46 | 27/19 |
| session_011 | plastic | diagonal constant-force pressing | rotational sliding | rotation | 651 | 27.105 | 49 | 30/19 |
| session_014 | plastic | diagonal constant-force pressing | rotational sliding | rotation | 653 | 27.189 | 52 | 33/19 |
| session_015 | plastic | diagonal constant-force pressing | rotational sliding | rotation | 651 | 27.096 | 51 | 31/20 |
| session_016 | plastic | continuous variable-force pressing | rotational sliding | rotation | 644 | 26.831 | 47 | 28/19 |
| session_017 | plastic | continuous variable-force pressing | rotational sliding | rotation | 646 | 26.909 | 51 | 32/19 |
| session_019 | plastic | continuous variable-force pressing | rotational sliding | rotation | 651 | 27.096 | 49 | 30/19 |
| session_020 | plastic | continuous variable-force pressing | rotational sliding | rotation | 645 | 26.871 | 46 | 27/19 |
| session_021 | A4 paper | static no motion | translational sliding | translation | 752 | 26.824 | 61 | 28/33 |
| session_022 | A4 paper | static no motion | translational sliding | translation | 750 | 26.768 | 61 | 29/32 |
| session_024 | A4 paper | static no motion | translational sliding | translation | 748 | 26.701 | 56 | 24/32 |
| session_025 | A4 paper | static no motion | translational sliding | translation | 760 | 27.121 | 62 | 28/34 |
| session_026 | A4 paper | continuous variable-force pressing | rotational sliding | rotation | 757 | 27.006 | 60 | 28/32 |
| session_027 | A4 paper | continuous variable-force pressing | rotational sliding | rotation | 745 | 26.597 | 61 | 28/33 |
| session_029 | A4 paper | continuous variable-force pressing | rotational sliding | rotation | 731 | 25.954 | 58 | 26/32 |
| session_030 | A4 paper | continuous variable-force pressing | rotational sliding | rotation | 722 | 25.761 | 59 | 28/31 |
| session_031 | soft foam earplug | constant-force pressing | translational sliding | translation | 717 | 25.600 | 57 | 26/31 |
| session_032 | soft foam earplug | constant-force pressing | translational sliding | translation | 745 | 26.603 | 59 | 26/33 |
| session_034 | soft foam earplug | constant-force pressing | translational sliding | translation | 750 | 26.754 | 60 | 28/32 |
| session_035 | soft foam earplug | constant-force pressing | translational sliding | translation | 758 | 27.038 | 60 | 27/33 |
| session_036 | soft foam earplug | continuous variable-force pressing | rotational sliding | rotation | 756 | 26.969 | 61 | 28/33 |
| session_037 | soft foam earplug | continuous variable-force pressing | rotational sliding | rotation | 735 | 26.215 | 62 | 31/31 |
| session_039 | soft foam earplug | continuous variable-force pressing | rotational sliding | rotation | 748 | 26.714 | 64 | 32/32 |
| session_040 | soft foam earplug | continuous variable-force pressing | rotational sliding | rotation | 755 | 26.951 | 64 | 31/33 |
| session_041 | USB cable | constant-force pressing | translational sliding | translation | 752 | 26.849 | 61 | 28/33 |
| session_042 | USB cable | constant-force pressing | translational sliding | translation | 755 | 26.949 | 60 | 27/33 |
| session_044 | USB cable | constant-force pressing | translational sliding | translation | 757 | 27.007 | 59 | 26/33 |
| session_045 | USB cable | constant-force pressing | translational sliding | translation | 746 | 26.637 | 62 | 29/33 |
| session_046 | USB cable | continuous variable-force pressing | rotational sliding | rotation | 752 | 26.844 | 60 | 27/33 |
| session_047 | USB cable | continuous variable-force pressing | rotational sliding | rotation | 755 | 26.929 | 61 | 28/33 |
| session_049 | USB cable | continuous variable-force pressing | rotational sliding | rotation | 755 | 26.955 | 63 | 30/33 |
| session_050 | USB cable | continuous variable-force pressing | rotational sliding | rotation | 753 | 26.867 | 65 | 32/33 |
| session_052 | plastic | diagonal constant-force pressing | diagonal translational sliding | translation | 736 | 26.276 | 65 | 32/33 |
| session_053 | plastic | diagonal constant-force pressing | diagonal translational sliding | translation | 759 | 27.101 | 65 | 32/33 |
| session_054 | plastic | diagonal constant-force pressing | diagonal translational sliding | translation | 761 | 27.172 | 61 | 28/33 |
| session_055 | plastic | diagonal constant-force pressing | diagonal translational sliding | translation | 755 | 26.932 | 61 | 28/33 |
| session_056 | plastic | diagonal constant-force pressing | diagonal translational sliding | translation | 755 | 26.953 | 61 | 28/33 |
| session_057 | plastic | diagonal changing-force pressing | diagonal rotational sliding | rotation | 755 | 26.950 | 60 | 27/33 |
| session_058 | plastic | diagonal changing-force pressing | diagonal rotational sliding | rotation | 756 | 26.985 | 63 | 30/33 |
| session_059 | plastic | diagonal changing-force pressing | diagonal rotational sliding | rotation | 759 | 27.090 | 58 | 25/33 |
| session_060 | plastic | diagonal changing-force pressing | diagonal rotational sliding | rotation | 761 | 27.153 | 64 | 31/33 |
| session_061 | plastic | diagonal changing-force pressing | diagonal rotational sliding | rotation | 759 | 27.084 | 64 | 31/33 |

### 4.2 验证集 Sessions

| Session | Material | Non-slip behavior | Slip behavior | Motion | Frames | FPS | Windows | Label 0/1 |
|---|---|---|---|---|---:|---:|---:|---:|
| session_003 | plastic | continuous variable-force pressing | translational sliding | translation | 639 | 26.597 | 26 | 16/10 |
| session_008 | plastic | constant-force pressing | translational sliding | translation | 647 | 26.942 | 25 | 16/9 |
| session_012 | plastic | diagonal constant-force pressing | rotational sliding | rotation | 645 | 26.864 | 25 | 16/9 |
| session_018 | plastic | continuous variable-force pressing | rotational sliding | rotation | 641 | 26.692 | 25 | 16/9 |
| session_023 | A4 paper | static no motion | translational sliding | translation | 749 | 26.723 | 32 | 16/16 |
| session_028 | A4 paper | continuous variable-force pressing | rotational sliding | rotation | 737 | 26.298 | 32 | 16/16 |
| session_033 | soft foam earplug | constant-force pressing | translational sliding | translation | 735 | 26.232 | 32 | 16/16 |
| session_038 | soft foam earplug | continuous variable-force pressing | rotational sliding | rotation | 745 | 26.572 | 33 | 16/17 |
| session_043 | USB cable | constant-force pressing | translational sliding | translation | 759 | 27.078 | 32 | 16/16 |
| session_048 | USB cable | continuous variable-force pressing | rotational sliding | rotation | 755 | 26.929 | 32 | 16/16 |

### 4.3 独立测试集 Sessions

| Session | Material | Non-slip behavior | Slip behavior | Motion | Frames | FPS | Windows | Label 0/1 |
|---|---|---|---|---|---:|---:|---:|---:|
| session_001 | A4 paper | continuous variable-force pressing | rotational sliding | rotation | 754 | 26.917 | 66 | 33/33 |
| session_002 | A4 paper | continuous variable-force pressing | rotational sliding | rotation | 751 | 26.792 | 65 | 33/32 |
| session_003 | soft foam earplug | constant-force pressing | translational sliding | translation | 759 | 27.075 | 66 | 33/33 |
| session_004 | soft foam earplug | constant-force pressing | translational sliding | translation | 758 | 27.065 | 66 | 33/33 |
| session_005 | soft foam earplug | continuous variable-force pressing | rotational sliding | rotation | 758 | 27.060 | 66 | 33/33 |
| session_006 | soft foam earplug | continuous variable-force pressing | rotational sliding | rotation | 756 | 26.987 | 66 | 33/33 |
| session_007 | USB cable | constant-force pressing | translational sliding | translation | 750 | 26.770 | 65 | 33/32 |
| session_008 | USB cable | constant-force pressing | translational sliding | translation | 753 | 26.880 | 66 | 33/33 |
| session_009 | A4 paper | constant-force pressing | translational sliding | translation | 726 | 25.917 | 63 | 32/31 |
| session_010 | A4 paper | constant-force pressing | translational sliding | translation | 741 | 26.447 | 64 | 32/32 |
| session_011 | USB cable | continuous variable-force pressing | rotational sliding | rotation | 752 | 26.846 | 66 | 33/33 |
| session_012 | USB cable | continuous variable-force pressing | rotational sliding | rotation | 756 | 26.978 | 66 | 33/33 |
| session_013 | plastic | constant-force pressing | translational sliding | translation | 756 | 26.981 | 66 | 33/33 |
| session_014 | plastic | constant-force pressing | translational sliding | translation | 750 | 26.780 | 65 | 33/32 |
| session_015 | plastic | continuous variable-force pressing | rotational sliding | rotation | 758 | 27.044 | 65 | 33/32 |
| session_016 | plastic | continuous variable-force pressing | rotational sliding | rotation | 758 | 27.044 | 66 | 33/33 |
| session_017 | plastic | diagonal constant-force pressing | rotational sliding | rotation | 748 | 26.694 | 65 | 33/32 |
| session_018 | plastic | diagonal constant-force pressing | rotational sliding | rotation | 748 | 26.692 | 65 | 33/32 |
| session_062 | plastic | diagonal changing-force pressing | diagonal rotational sliding | rotation | 758 | 27.057 | 66 | 33/33 |
| session_063 | plastic | diagonal changing-force pressing | diagonal rotational sliding | rotation | 759 | 27.099 | 66 | 33/33 |
| session_064 | plastic | diagonal constant-force pressing | diagonal translational sliding | translation | 750 | 26.751 | 65 | 33/32 |
| session_065 | plastic | diagonal constant-force pressing | diagonal translational sliding | translation | 745 | 26.598 | 65 | 33/32 |

## 5. 交叉验证结果

在 `cv_multi_material_augdiag_59` 上完成 5 折 session-level 交叉验证。各折 best smoothed balanced accuracy 如下：

| Fold | Best epoch | Best smoothed balanced_acc |
|---|---:|---:|
| 0 | 34 | 96.70% |
| 1 | 35 | 97.77% |
| 2 | 40 | 97.34% |
| 3 | 34 | 95.58% |
| 4 | 37 | 93.70% |

5 折平均 best smoothed balanced accuracy 为 **96.22%**，说明模型在当前训练分布和 session-level 验证设置下已经能够稳定学习 slip / non-slip 判别特征。

## 6. 最终模型与独立测试

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

从 session-level 和可视化结果看，测试集上的下降主要集中在少数较难样本中。旧测试集分析中，`session_017`、`session_010`、`session_015` 曾明显拉低整体结果；同材质、同行为组合下的另一些 session 表现相对正常。这说明误差可能和人工采集过程中的 session 间差异有关，例如接触位置、施力方式、非滑动阶段是否存在细微位移等。

## 7. 阈值敏感性分析

最新阈值扫描完成于 **2026-08-29 21:16**。在 `test_windows_final_plus_diag.csv` 上，阈值从 0.5 调整到 0.7 后，balanced accuracy 从 **87.79%** 提升到 **88.32%**。

在旧测试集 `test_windows_final.csv` 上，阈值从 0.5 调整到 0.7 后，balanced accuracy 从 **85.41%** 提升到 **86.58%**。

进行阈值扫描的主要原因是可视化时观察到连续窗口的 slip probability 存在一定抖动，默认 0.5 阈值不一定是最稳定的部署决策边界。该结果说明决策阈值对最终表现有小幅影响，但提升有限，因此这里表述为 threshold sensitivity analysis。

## 8. 阶段结论

本阶段已经完成从数据采集、窗口构建、session-level 交叉验证、最终模型训练到独立测试评估的完整流程。当前模型在 5 折交叉验证中达到约 **96.22%** 的平均 balanced accuracy，在更严格的独立测试集上达到约 **87.79%** 的 balanced accuracy，阈值扫描后最高约 **88.32%**。

整体来看，该模型已经形成一个可复现、可提交的 slip detection baseline。现阶段的主要价值在于证明 RGB 视觉触觉序列可以有效区分 slip / non-slip；同时，独立测试结果也显示模型在旋转滑动、非滑动强形变以及部分材料上的跨 session 稳定性仍存在一定局限。
