"""motifscope —— DNA 调控序列模型的可解释性验证工具包。

本包提供一个端到端的基准框架，用于回答一个具体问题：

    当一个序列模型预测"这段 DNA 是增强子 / 剪接位点 / TF 结合位点"时，
    归因方法（gradient x input、ISM、integrated gradients ...）
    指出的重要位置，究竟有多可信？

由于真实生物数据没有逐碱基的"重要性真值"，本包使用**基序植入的合成数据**
来构造可精确计算的真值，并在此基础上定量评估归因方法的忠实度（faithfulness）。

主要子模块
----------
data        合成数据生成与真实数据加载
motifs      PWM 读写、扫描打分、基序库
models      可解释性友好的序列模型
attribution 归因方法
evaluate    忠实度、回收率与稳定性指标
report      中文 HTML 报告
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
