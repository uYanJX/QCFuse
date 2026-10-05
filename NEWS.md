# 🔥 News

- **2026.10.02** 🚀 Qwen3-32B reasoning evaluation with Thinking enabled on HotpotQA, 2WikiMQA, and MuSiQue reports F-score and TTFT results for FullComp, Ours, and ProphetKV.
- **2026.10.02** 🚀 QCFuse is compatible with Huawei's [Unified Cache Management (UCM)](https://github.com/uYanJX/ucm-qcfuse) framework and supports multi-batch inference.

## Qwen3-32B results — Thinking: Enabled

Percentages are relative F-score changes from FullComp. The aggregate row is the unweighted mean over the three QA workloads, with its percentage calculated as the mean of the three dataset-level relative changes.

| Dataset | FullComp | Ours @ 0.5 | ProphetKV @ 0.5 |
| --- | ---: | ---: | ---: |
| HotpotQA | 0.6956 <small>(baseline)</small> | 0.6863 <small>(−1.34%)</small> | 0.6829 <small>(−1.83%)</small> |
| 2WikiMQA | 0.6174 <small>(baseline)</small> | 0.6211 <small>(+0.60%)</small> | 0.5983 <small>(−3.09%)</small> |
| MuSiQue | 0.4147 <small>(baseline)</small> | 0.4266 <small>(+2.87%)</small> | 0.4239 <small>(+2.22%)</small> |
| Aggregate (3 QA) | 0.5759 <small>(baseline)</small> | 0.5780 <small>(+0.71%)</small> | 0.5684 <small>(−0.90%)</small> |

QCFuse at blend ratio 0.5 achieves the highest aggregate F-score (0.5780, +0.71% vs. FullComp) with a 1.81× TTFT speedup.

## TTFT speedup

| Scope | FullComp | Ours @ 0.5 | ProphetKV @ 0.5 |
| --- | ---: | ---: | ---: |
| Aggregate (3 QA) | 1.00× | 1.81× | 1.65× |

All entries use 500 valid examples and TopK=20.
