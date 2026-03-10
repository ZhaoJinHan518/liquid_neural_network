# 液态神经网络与普通神经网络的对比研究

**项目类型：** 大学生创新训练项目（SITP）  
**文档版本：** v1.0  
**日期：** 2025年

---

## 一、研究背景

近年来，深度学习领域发展迅速，传统神经网络（如多层感知机 MLP、卷积神经网络 CNN、循环神经网络 RNN）在图像识别、自然语言处理等诸多任务上取得了显著成果。然而，传统神经网络普遍存在计算开销大、对时序数据建模能力有限、可解释性差等问题。

液态神经网络（Liquid Neural Network, LNN）是近年来提出的一种新型神经网络架构，其核心思想源自神经科学中对生物神经元动态特性的模拟。液态时间常数神经网络（Liquid Time-Constant Networks, LTC）及其高效变体——闭合形式连续时间神经网络（Closed-form Continuous-time Neural Networks, CfC）——通过引入连续时间常微分方程（ODE）来刻画神经元状态的动态变化，具有参数量少、可解释性强、时序建模能力突出等优势。

2024 年，液态神经网络在自动驾驶、机器人控制等实际落地场景中展现出良好性能，引发学术界与工业界的广泛关注。本项目旨在系统对比液态神经网络与普通神经网络在多类典型任务上的表现，深入分析其优劣势，为后续研究与应用提供参考。

---

## 二、研究目标

1. **理论梳理：** 系统整理 LTC、CfC 等液态神经网络模型的数学原理与实现细节，与传统 RNN、LSTM、GRU 进行理论层面的对比分析。
2. **实验对比：** 在时序预测、分类、控制等典型任务上，定量比较液态神经网络与常规神经网络的性能指标（准确率、MSE、参数量、训练速度等）。
3. **优劣分析：** 归纳两类网络在不同任务场景下的适用范围、计算效率与可解释性差异。
4. **文档输出：** 形成完整的研究报告与可复现的实验代码，供后续研究者参考。

---

## 三、研究内容与技术路线

### 3.1 研究内容

| 模块 | 内容 |
|------|------|
| 文献调研 | 系统阅读 LTC、CfC、NCP 等相关论文，梳理液态神经网络发展脉络 |
| 模型实现 | 基于 PyTorch 实现 LTC、CfC，以及对照组 LSTM、GRU、标准 RNN |
| 实验设计 | 设计时序预测（正弦波、金融时序）、序列分类（活动识别）、控制任务（CartPole）等实验 |
| 结果分析 | 多维度对比实验结果，绘制学习曲线、性能对比图表 |
| 报告撰写 | 整理实验结论，撰写研究报告并开源代码 |

### 3.2 技术路线

```
文献调研
    ↓
模型实现（LTC / CfC / LSTM / GRU / RNN）
    ↓
数据集准备（时序、分类、控制）
    ↓
实验训练与评估
    ↓
结果对比与可视化
    ↓
报告撰写与代码开源
```

### 3.3 实验任务

1. **时序预测：** 使用正弦波合成数据及真实金融时间序列，评估各模型的预测精度（MSE / MAE）。
2. **序列分类：** 使用公开活动识别数据集（如 UCI HAR），评估分类准确率与模型参数量。
3. **强化学习控制：** 在 OpenAI Gymnasium 的 CartPole 环境中，比较各网络作为策略网络的收敛速度与最终奖励。

---

## 四、创新点

1. **系统性对比框架：** 在统一实验平台上同时评估液态神经网络与多种主流神经网络，提供横向可比的性能基准。
2. **多任务覆盖：** 涵盖预测、分类、控制三类典型任务，全面展示两类网络的适用边界。
3. **可复现性保障：** 所有实验代码、超参数配置及随机种子均公开，确保实验结果可复现。
4. **中文社区贡献：** 以中文撰写系统性对比文档，降低国内研究者的学习门槛。

---

## 五、预期成果

- **研究报告：** 完整的对比研究报告（中文），包含理论分析与实验结论。
- **开源代码：** 基于 PyTorch 的模型实现与实验脚本，托管于本仓库。
- **可视化结果：** 学习曲线、性能雷达图、参数效率对比图等。
- **文档资料：** 项目提案（本文档）及 README 使用说明。

---

## 六、研究计划

| 阶段 | 时间 | 主要任务 |
|------|------|----------|
| 第一阶段 | 第 1–2 周 | 文献调研，阅读核心论文，整理笔记 |
| 第二阶段 | 第 3–5 周 | 实现 LTC、CfC、LSTM、GRU、RNN 模型 |
| 第三阶段 | 第 6–9 周 | 设计并运行实验，收集实验数据 |
| 第四阶段 | 第 10–11 周 | 结果分析、可视化，撰写报告 |
| 第五阶段 | 第 12 周 | 代码整理、文档完善、项目提交 |

---

## 七、参考文献

1. Hasani, R., Lechner, M., Amini, A., Rus, D., & Grosu, R. (2021). **Liquid Time-constant Networks**. *Proceedings of the AAAI Conference on Artificial Intelligence*, 35(9), 7657–7666.

2. Hasani, R., Lechner, M., Amini, A., Liebenwein, L., Ray, A., Tschaikowski, M., Teschl, G., & Rus, D. (2022). **Closed-form Continuous-time Neural Networks**. *Nature Machine Intelligence*, 4, 992–1003.

3. Lechner, M., Hasani, R., Amini, A., Henzinger, T. A., Rus, D., & Grosu, R. (2020). **Neural Circuit Policies Enabling Auditable Autonomy**. *Nature Machine Intelligence*, 2, 642–652.

4. **arXiv:2510.07578v1** — 关于液态神经网络最新进展的预印本论文。  
   [https://arxiv.org/abs/2510.07578](https://arxiv.org/abs/2510.07578)

5. Hochreiter, S., & Schmidhuber, J. (1997). **Long Short-Term Memory**. *Neural Computation*, 9(8), 1735–1780.

6. Cho, K., van Merriënboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk, H., & Bengio, Y. (2014). **Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation**. *EMNLP 2014*.
