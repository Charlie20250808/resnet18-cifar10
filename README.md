# ResNet18 on CIFAR-10

从零手写 ResNet18 并在 CIFAR-10 上完成训练与调优，测试集准确率 **95.23%**。

本项目为学习性质的复现工作：模型结构、训练循环、数据增强与训练技巧均为手动实现，
目的是完整理解一个标准图像分类 pipeline 的每个环节，而非提出新方法。

---

## 结果

| 指标 | 数值 |
|------|------|
| 测试集准确率 | **95.23%** |
| 训练集最终准确率 | 99.98% |
| 训练轮数 | 100 epochs |
| 参数量 | 11.17 M |
| 训练设备 | RTX 5060 |
| 随机种子 | 42（已固定） |

准确率取单次运行中测试集表现最好的 epoch（best checkpoint）。
训练末期 test 准确率稳定在 95.0%–95.2%，波动约 0.2 个百分点，收敛良好。

---

## 运行

```bash
pip install torch torchvision numpy
python train.py
```

首次运行自动下载 CIFAR-10 到 `data/`，最优权重保存为 `resnet18_cifar10_best.pth`。

每轮输出：

```
100/100 | train 0.502, 99.98% | test 0.626, 95.20% | best 95.23%
```

> Windows 下若卡在进程创建阶段，将 `make_loaders` 中的 `num_workers` 改为 0、
> 同时将 `persistent_workers` 改为 False（该选项要求 num_workers > 0）。
> 仅影响数据加载速度，不影响训练结果。

---

## 网络结构

针对 CIFAR-10 的 32×32 输入，对标准 ResNet18 做了适配：

- **Stem 使用 3×3 卷积且不接 MaxPool**。ImageNet 版本的 7×7 卷积 + MaxPool 会在
  一开始就将分辨率降低 4 倍，对 32×32 输入损失过多空间信息。
- **四个 stage，每 stage 含 2 个 BasicBlock**，通道数 64 / 128 / 256 / 512，
  后三个 stage 的首个 block 以 stride=2 下采样。
- **残差连接**采用逐元素相加：形状一致时为恒等映射，不一致时用 1×1 卷积对齐
  （该处 stride 必须与主路径一致，否则两路输出无法相加）。

---

## 训练配置与设计选择

| 项目 | 设置 |
|------|------|
| 优化器 | SGD, lr=0.1, momentum=0.9 |
| 权重衰减 | 5e-4 |
| 学习率调度 | CosineAnnealingLR, T_max=100 |
| 损失函数 | CrossEntropyLoss, label_smoothing=0.1 |
| 数据增强 | RandomCrop(32, padding=4) + RandomHorizontalFlip |
| 混合精度 | AMP (fp16) + GradScaler |
| Batch size | 128 |

**余弦退火**
学习率按余弦曲线从 0.1 平滑降至 0。前期以较大学习率快速探索参数空间，
后期学习率趋近 0，使模型在损失曲面谷底附近精细收敛，相比阶梯式衰减过渡更平滑。
`T_max` 需与实际训练轮数一致，`scheduler.step()` 按 epoch 调用。

**Label Smoothing**
将 one-hot 硬标签（正确类 1.0，其余 0）软化为正确类 0.9、其余 9 类平摊 0.1，
避免模型对预测过度自信，缓解过拟合。

**AMP 混合精度**
前向使用 fp16 提升速度、降低显存占用。因 fp16 数值范围较窄、小梯度存在下溢风险，
配合 `GradScaler`：反向传播前放大 loss，更新前还原梯度；若梯度出现 inf/nan
则跳过该步更新并自动调小缩放系数。仅在 CUDA 设备启用，CPU 回退 fp32。

**随机种子**
统一固定 `random` / `numpy` / `torch` (CPU & CUDA) 的随机源。
需注意 cuDNN 与 AMP 的部分并行运算存在固有非确定性，复现结果会稳定在极小区间内
而非位级完全一致；如需完全确定性可另设 `torch.backends.cudnn.deterministic = True`
（会牺牲部分速度）。

---

## 调试记录

训练中曾出现异常：测试准确率长期在 76%–85% 剧烈震荡，训练准确率停滞于 89%，
最终仅 87.28%，与预期相差约 8 个百分点。

由于该现象在多次运行中稳定复现而非偶发，排除了随机性因素。从两个特征入手定位：

1. 训练准确率停滞 → 模型未能精细收敛；
2. 训练末期测试准确率仍剧烈震荡 → 参数在最优点附近反复越过。

两者共同指向学习率始终过大。排查后确认训练循环中遗漏了 `scheduler.step()` 调用——
调度器已创建但从未推进，学习率全程锁定在初始值 0.1，余弦退火未生效。

补上调用后学习率正常退火至 0，准确率恢复至 95.23%，训练末期波动收敛至 0.2% 以内。

该问题不抛出任何异常，属于典型的静默失效：程序可正常运行结束，仅结果不正确。

---

## 项目结构

```
.
├── train.py       # 模型定义 + 训练脚本
├── README.md
├── LICENSE
└── .gitignore     # 已排除 data/ 与 *.pth
```

---

## 参考

- He et al., *Deep Residual Learning for Image Recognition*, CVPR 2016.
- Szegedy et al., *Rethinking the Inception Architecture for Computer Vision*, CVPR 2016.（Label Smoothing）
- Loshchilov & Hutter, *SGDR: Stochastic Gradient Descent with Warm Restarts*, ICLR 2017.（Cosine Annealing）

## License

MIT
