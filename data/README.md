# Evaluation Data

This directory is the local target for the revised QCFuse evaluation data.
Dataset files are intentionally excluded from the GitHub repository and hosted
at [Yjx666/qcfuse-dataset](https://huggingface.co/datasets/Yjx666/qcfuse-dataset).

Download the seven JSONL files from Hugging Face with:

```bash
pip install -U huggingface_hub
hf download Yjx666/qcfuse-dataset \
  --repo-type dataset \
  --include "*.jsonl" \
  --local-dir data
```

```text
data/
├── musique.jsonl
├── 2wikimqa.jsonl
├── hotpotqa.jsonl
├── ruler_mq.jsonl
├── ruler_mv.jsonl
├── ruler_vt.jsonl
└── bird.jsonl
```

The evaluation contains seven workloads from three task families:

- Multi-hop QA with KET-RAG serialized contexts: `musique.jsonl`,
  `2wikimqa.jsonl`, `hotpotqa.jsonl`
- RULER: `ruler_mv.jsonl`, `ruler_mq.jsonl`, `ruler_vt.jsonl`
- BIRD Mini-Dev: `bird.jsonl`

Each workload contains 500 examples using the shared four-field JSONL schema:
`input`, `context`, `answers`, and `num_chunks`. The three multi-hop QA workloads
contain 9--11 losslessly serialized context chunks per example; RULER and BIRD
contain 10 chunks. Retrieved contexts average about 10K tokens.

Use `--data_dir data` in the Blend runner after downloading the files. For
example:

```bash
python blend/sglang_blend_ssd.py \
  --model qwen3-8b \
  --model_dir models \
  --data_dir data \
  --dataset musique \
  --baseline ours \
  --size 500 \
  --cache_dir cache/qcfuse
```

## BIRD execution accuracy

`bird.jsonl` contains the serialized schemas and gold SQL, but BIRD Execution
Accuracy also requires the official 500-row Mini-Dev metadata and its SQLite
databases. Download the original package from
[bird-bench/mini_dev](https://github.com/bird-bench/mini_dev), then run:

```bash
python blend/sglang_blend_ssd.py \
  --model qwen3-8b \
  --model_dir models \
  --data_dir data \
  --dataset bird \
  --baseline ours \
  --size 500 \
  --bird_data /path/to/mini_dev_sqlite.json \
  --bird_db_root /path/to/dev_databases \
  --cache_dir cache/qcfuse
```

The evaluator opens every database read-only and reports Execution Accuracy
(`EX`) by comparing the result sets produced by predicted and gold SQL.
