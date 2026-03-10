# SITP Experiment Log
> 液态神经网络与普通神经网络的对比研究  
> 同济大学 SITP 项目实验超参数记录

---

## 通用配置

| 参数 | 值 |
|------|----|
| 随机种子 (Random Seed) | 42 |
| 优化器 (Optimizer) | Adam |
| 梯度裁剪 (Gradient Clipping) | 1.0 |
| 设备 (Device) | CUDA (若可用) / CPU |
| 框架 (Framework) | PyTorch ≥ 2.0, ncps ≥ 0.0.6 |

---

## 实验 1：阻尼正弦波 1-step 预测（复现论文 Fig 6）

**脚本：** `experiments/damped_sine.py`  
**结果目录：** `results/damped_sine/`

| 超参数 | 值 |
|--------|-----|
| 模型 | LTC, GRU |
| 隐藏单元数 (Units) | 32 |
| 输入维度 | 1 |
| 输出维度 | 1 |
| 序列长度 (Seq Len) | 100 |
| 训练样本数 | 1000 |
| 测试样本数 | 200 |
| 学习率 (LR) | 1e-3 |
| 训练轮次 (Epochs) | 200 |
| Batch Size | 32 |
| 数据生成 | 阻尼正弦波：A·exp(-γt)·sin(ωt+φ)，参数随机采样 |

**信号参数范围：**
- 振幅 A ∈ [0.5, 1.5]
- 衰减系数 γ ∈ [0.005, 0.02]
- 角频率 ω ∈ [0.5, 2.0]
- 初相位 φ ∈ [0, 2π]

---

## 实验 2：Walker2d 轨迹预测（复现论文 Section 4.1）

**脚本：** `experiments/walker2d.py`  
**结果目录：** `results/walker2d/`

| 超参数 | 值 |
|--------|-----|
| 模型 | LTC, LSTM |
| 隐藏单元数 (Units) | 64 |
| 输入维度 | 17（关节位置 8 维 + 关节速度 9 维）|
| 输出维度 | 17 |
| 输入序列长度 | 10 steps |
| 数据模式 | 合成数据（默认）/ minari Walker2d-medium-v2（--use-minari） |
| 训练样本数 | 4800（合成模式） |
| 测试样本数 | 1200（合成模式） |
| 学习率 | 1e-3 |
| 训练轮次 | 100 |
| Batch Size | 64 |

---

## 实验 3：校园人流预测（SITP 创新模块）

**脚本：** `experiments/campus_flow_exp.py`  
**数据生成：** `src/data/campus_flow.py`  
**结果目录：** `results/campus_flow/`

### 数据生成参数

| 参数 | 值 |
|------|-----|
| 模拟天数 | 60 天 |
| 采样间隔 | 5 分钟 |
| 基础信号 | 24h 周期正弦（含谐波） |
| 下课时间点 | 10:00, 11:45, 13:30, 15:15, 17:00, 18:30 |
| 尖峰幅度 | 1.5 × 基础幅度 |
| 尖峰宽度 | 3 分钟（高斯标准差） |
| 缺失率（不规则采样）| 15% |
| 噪声标准差 | 0.02 |

### 模型训练参数

| 超参数 | 值 |
|--------|-----|
| 模型 | CfC, LSTM |
| 隐藏单元数 (Units) | 32 |
| 输入维度 | 2（观测值 + 掩码位） |
| 输出维度 | 1 |
| 序列长度 | 48 steps（约 4 小时） |
| 学习率 | 1e-3 |
| 训练轮次 | 100 |
| Batch Size | 32 |
| Rollout 步数 | 5 |

---

## 实验 4：多维度量化评估

**脚本：** `experiments/evaluation.py`  
**结果目录：** `results/evaluation/`

| 超参数 | 值 |
|--------|-----|
| 模型 | CfC, LSTM, LTC, GRU |
| 隐藏单元数 | 32 |
| 评估噪声水平 σ | 0.0, 0.01, 0.05 |
| 训练轮次（快速） | 80 |
| 推理延迟测量重复次数 | 200 |
| 推理设备 | CPU（为公平对比）|

---

## 实验 5：金融时序预测（纳斯达克指数，30→5天）

**脚本：** `experiments/timeseries_finance.py`  
**结果目录：** `results/timeseries_finance/`

| 超参数 | 值 |
|--------|-----|
| 模型 | CfC, LSTM, GRU |
| 数据源 | 纳斯达克综合指数 ^IXIC（yfinance，近5年）；离线时使用合成GBM序列 |
| 特征 | 对数收益率（归一化） |
| 输入窗口 (Input_len) | 30 个交易日 |
| 预测步数 (Pred_len) | 5 个交易日 |
| 隐藏单元数 (Units) | 32 |
| 输入维度 | 1（单变量） |
| 输出维度 | 5（多步预测） |
| 数据划分 | 70% 训练 / 15% 验证 / 15% 测试 |
| 学习率 | 1e-3 |
| 训练轮次 (Epochs) | 150 |
| Batch Size | 32 |
| 梯度裁剪 | 1.0 |
| 评估指标 | MSE, MAE（归一化对数收益率空间） |

---

## 实验 6：UCI HAR 序列分类（人体活动识别）

**脚本：** `experiments/uci_har.py`  
**结果目录：** `results/uci_har/`

| 超参数 | 值 |
|--------|-----|
| 模型 | LTC, CfC, LSTM, GRU, RNN |
| 数据集 | UCI HAR Dataset（6 类活动，下载失败时使用含类别信号的合成正弦数据）|
| 最大训练样本数 | 2000（CPU 上限制 LTC 训练时间，可通过 MAX_TRAIN_SAMPLES=None 使用全量数据）|
| 输入维度 | 9（身体加速度×3 + 陀螺仪×3 + 总加速度×3） |
| 序列长度 | 128 步（约 2.56 秒，50Hz 采样） |
| 类别数 | 6（行走/上楼/下楼/坐/站/躺） |
| 隐藏单元数 (Units) | 32 |
| 分类头 | 线性层（Units → 6） |
| 归一化 | 逐特征 Z-score（在训练集上拟合） |
| 学习率 | 1e-3 |
| 训练轮次 (Epochs) | 80 |
| Batch Size | 64 |
| 权重衰减 | 1e-4 |
| 评估指标 | 测试集准确率（%），参数量 |

---

## 实验 7：CartPole-v1 强化学习（REINFORCE）

**脚本：** `experiments/cartpole_rl.py`  
**结果目录：** `results/cartpole/`

| 超参数 | 值 |
|--------|-----|
| 模型 | CfC 策略网络, LSTM 策略网络 |
| 环境 | CartPole-v1（gymnasium） |
| 算法 | REINFORCE（Policy Gradient，无基线） |
| 隐藏单元数 (Units) | 32 |
| 状态维度 | 4（杆角、角速度、车位置、车速度） |
| 动作空间 | 2（左/右） |
| 折扣因子 (γ) | 0.99 |
| 收益归一化 | 逐 episode 均值/标准差 |
| 学习率 | 2e-3 |
| 最大 Episodes | 500 |
| 梯度裁剪 | 0.5 |
| 熵奖励系数 | 0.01（提升探索能力）|
| 基线估计 | 指数移动平均（α=0.05，方差缩减）|
| 解决阈值 | 连续 20 episode 平均奖励 ≥ 195 |
| 评估指标 | 最终平均奖励，首次达到解决阈值的 episode 数 |

---

## 文件结构

```
lnn-project/
├── src/
│   ├── models/
│   │   └── registry.py        # 统一模型接口（LTC, CfC, LSTM, GRU, RNN）
│   └── data/
│       └── campus_flow.py     # 校园人流数据生成器
├── experiments/
│   ├── damped_sine.py         # 实验 1：阻尼正弦
│   ├── walker2d.py            # 实验 2：Walker2d 轨迹
│   ├── campus_flow_exp.py     # 实验 3：校园人流
│   ├── evaluation.py          # 实验 4：综合评估
│   ├── timeseries_finance.py  # 实验 5：金融时序预测
│   ├── uci_har.py             # 实验 6：UCI HAR 序列分类
│   └── cartpole_rl.py         # 实验 7：CartPole-v1 强化学习
├── results/
│   ├── damped_sine/           # 实验 1 图表与指标
│   ├── walker2d/              # 实验 2 图表与指标
│   ├── campus_flow/           # 实验 3 图表与指标
│   ├── evaluation/            # 实验 4 图表与性能表格
│   ├── timeseries_finance/    # 实验 5 图表与指标
│   ├── uci_har/               # 实验 6 图表与指标
│   └── cartpole/              # 实验 7 图表与指标
├── docs/
│   ├── proposal.md
│   ├── SITP_Log.md            # 本文档
│   └── Final_Report_SITP.md   # 中文结题报告
└── requirements.txt
```
