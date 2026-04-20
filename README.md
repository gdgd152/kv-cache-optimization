# KV Cache Compression for Efficient LLM Inference (Pythia-70M)

## 项目简介

本项目基于 **Pythia-70M** 模型，在 **CPU 环境**下实现了 **StreamingLLM** 优化方法：

* **Baseline**（完整 KV Cache）
* **Truncate**（简单截断）
* **StreamingLLM**（保留 sink tokens + 滑动窗口）

在以下数据集上进行评估：

* **WikiText-2**（短文本）
* **PG19**（长文本）

评估内容包括：

* 语言建模质量（Perplexity, PPL）
* 推理效率（TTFT / TPOT / Throughput）
* TPOT 随生成长度变化曲线

---

## 项目结构

```
.
├── main.py               # 主程序（运行所有实验 + 画图）
├── model.py              # 加载 Pythia-70M
├── generate.py           # 文本生成 + 时间统计
├── eval_ppl.py           # PPL 计算（滑动窗口 + KV Cache）
├── kv_cache.py           # KV 压缩方法实现
├── pg19_sample.txt       # PG19 长文本样本
├── tpot_pg19.png         # 长文本 TPOT 曲线
├── tpot_wikitext.png     # 短文本 TPOT 曲线
└── README.md
```

---

## 环境依赖

建议使用 Python ≥ 3.9

安装依赖：

```bash
pip install torch transformers datasets matplotlib tqdm
```

---

## 如何运行

只需运行主程序：

```bash
python main.py
```

程序将自动完成：

### 1️⃣ PPL 测试

* WikiText（滑动窗口 baseline）
* PG19（滑动窗口 baseline）
* KV Cache PPL（Baseline / Truncate / Streaming）

### 2️⃣ 生成速度测试（自动多次运行取平均）

* TTFT（首 token 延迟）
* TPOT（每 token 时间）
* Throughput（吞吐率）

### 3️⃣ 自动绘图

生成两张图：

```bash
tpot_pg19.png
tpot_wikitext.png
```

---

## 实验结果（简要报告）

### 1. PPL（语言建模质量）

| 方法        | WikiText | PG19   |
| --------- | -------- | ------ |
| Baseline  | 95.98    | 75.76  |
| Truncate  | 98.09    | 260.89 |
| Streaming | 97.92    | 231.80 |

**分析：**

* WikiText 为短文本，KV 压缩几乎不触发 → PPL 基本一致

* PG19 为长文本：
  
  * Truncate / Streaming 丢失历史信息 → PPL 显著上升
  * Streaming 略优于 Truncate（保留 sink tokens）

👉 说明：KV 压缩在长上下文中会带来明显的质量下降，这是方法本身的 trade-off。

---

### 2. 生成速度

#### PG19（长上下文）

| 方法        | TTFT (s) | TPOT (ms) | Throughput     |
| --------- | -------- | --------- | -------------- |
| Baseline  | 0.409    | 15.9      | 62.7 tok/s     |
| Truncate  | 0.383    | 14.2      | 70.5 tok/s     |
| Streaming | 0.388    | 14.0      | **71.2 tok/s** |

 **提升约 13%**

---

#### WikiText（短上下文）

| 方法        | TPOT (ms) | Throughput |
| --------- | --------- | ---------- |
| Baseline  | 15.3      | 65.2       |
| Truncate  | 15.0      | 66.8       |
| Streaming | 15.2      | 65.7       |

👉 几乎无差异（未触发压缩）

---

### 3. TPOT 曲线分析

#### PG19（长文本）

* Baseline：TPOT 随序列增长略有波动
* Truncate / Streaming：TPOT 更稳定
* Streaming 略优于 Truncate

👉 说明压缩有效控制了计算复杂度

---

#### WikiText（短文本）

* 三条曲线几乎重合
* 压缩未生效

---

## ⚖ 结论

本实验验证了 KV Cache 压缩方法的核心 trade-off：

### 优点

* 在长文本生成中：
  
  * 提升吞吐（≈10%~15%）
  * 降低 TPOT
  * 控制计算复杂度

### 缺点

* 显著增加 PPL（尤其是超长文本）
* 损失长距离依赖信息

---

## 方法总结

| 方法        | 特点              |
| --------- | --------------- |
| Baseline  | 完整上下文，质量最高      |
| Truncate  | 简单高效，但丢信息       |
| Streaming | 保留关键 token，效果更优 |

---

## 备注

* 所有实验均在 **CPU 上运行** ，此时速度的主要制约因素在于FFN，因此实验结果相对并不显著，仅作趋势上的分析。

* 使用 **Pythia-70M（无训练）**

* TPOT 曲线经过：
  
  * 多次运行取平均
  * 滑动窗口平滑

* 单个 pg_19 sample 从网络下载并保存为 `pg19_sample.txt` 

---
