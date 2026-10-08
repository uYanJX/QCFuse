# 🔥 News

- **2026.10.08** 🚀 Qwen3-32B evaluation with Thinking enabled models practical reasoning across six tasks, including multi-hop QA and multi-value retrieval.
- **2026.10.02** 🚀 QCFuse is compatible with Huawei's [Unified Cache Management (UCM)](https://github.com/uYanJX/ucm-qcfuse) framework and supports multi-batch inference.

## Qwen3-32B results — Thinking: Enabled

Recomputation ratio: 0.5; TopK=20; 500 examples per task. Parentheses show relative changes from FullComp.

| Task | Metric | FullComp | QCFuse @ 0.5 | ProphetKV @ 0.5 |
| --- | --- | ---: | ---: | ---: |
| HotpotQA | F-score | 0.6956 <small>(baseline)</small> | 0.6863 <small>(−1.33%)</small> | 0.6829 <small>(−1.83%)</small> |
| 2WikiMQA | F-score | 0.6174 <small>(baseline)</small> | 0.6211 <small>(+0.59%)</small> | 0.5983 <small>(−3.10%)</small> |
| MuSiQue | F-score | 0.4147 <small>(baseline)</small> | 0.4266 <small>(+2.88%)</small> | 0.4239 <small>(+2.23%)</small> |
| RULER-MV | StringMatchAll | 0.9990 <small>(baseline)</small> | 0.9965 <small>(−0.25%)</small> | 0.9980 <small>(−0.10%)</small> |
| RULER-MQ | StringMatchAll | 1.0000 <small>(baseline)</small> | 1.0000 <small>(0.00%)</small> | 1.0000 <small>(0.00%)</small> |
| RULER-VT | StringMatchAll | 0.9960 <small>(baseline)</small> | 0.9908 <small>(−0.52%)</small> | 0.9972 <small>(+0.12%)</small> |

QCFuse at 50% recomputation achieves a +0.228% mean relative quality change across six tasks and a 1.90× TTFT speedup on QA.

## Effect of recomputation ratio

Quality changes and decode tokens are averaged over six tasks. TTFT speedups are relative to FullComp with SSD transfer at 10 GB/s.

| Ratio | ProphetKV quality change | QCFuse quality change | ProphetKV TTFT speedup | QCFuse TTFT speedup | ProphetKV decode tokens | QCFuse decode tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **FullComp (1.0)** | **0.000%** | **0.000%** | **1.00×** | **1.00×** | **547.3** | **547.3** |
| 0.1 | −9.250% | −8.226% | 4.81× | 6.54× | 621.7 | 581.6 |
| 0.2 | −4.192% | −3.166% | 3.29× | 4.11× | 576.0 | 561.7 |
| 0.3 | −1.868% | −1.359% | 2.50× | 2.95× | 570.7 | 563.1 |
| 0.4 | −0.596% | −0.827% | 2.03× | 2.31× | 551.1 | 557.3 |
| 0.5 | −0.447% | +0.228% | 1.70× | 1.90× | 554.8 | 560.9 |

QCFuse delivers higher TTFT speedups than ProphetKV across all evaluated ratios.
