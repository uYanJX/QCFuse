"""
SSD-backed QCFuse blend runner.

Usage:
    python sglang_blend_ssd.py --model qwen3-8b --dataset hotpotqa \
        --model_dir models --baseline ours
"""

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional
from sglang.srt.utils.digest_index_manager import (
    DIGEST_INDEX_VERSION,
    DigestIndexManager,
)
from sglang.srt.utils.kv_ssd_manager import (
    PACKED_KV_FORMAT,
    QUERY_CACHE_FORMAT,
)

from utils import (
    load_dataset,
    build_prompt_for_dataset,
    evaluate_sample,
    get_metric_name,
    get_system_prompt,
    get_max_new_tokens,
)

from blend_common import (
    DEFAULT_DATA_DIR,
    BLEND_SEP,
    get_critical_layers,
    set_fuserag_layers,
    set_ours_layers,
    set_prophetkv_layers,
    BlendEngineBase,
)
from qcfuse_config import (
    BASELINE_DIGEST_RATIOS,
    BLEND_BASELINES,
    DEFAULT_BLEND_RATIO,
    DEFAULT_CONTEXT_N_SINK,
    DEFAULT_CRITICAL_LAYERS,
    DIGEST_INDEX_METHOD,
    DIGEST_RATIO,
    SUPPORTED_BASELINES,
)


DIGEST_ZIP_PROMPT = "\n\nRepeat the previous context exactly."


def print_final_metrics(
    model_name: str,
    dataset_name: str,
    baseline: str,
    result: dict,
    metric_name: str,
) -> None:
    if result["metric"]:
        score = sum(result["metric"]) / len(result["metric"])
        score_text = f"{score:.4f}"
        avg_ttft = sum(result["ttft"]) / len(result["ttft"])
        ttft_text = f"{avg_ttft:.4f}s"
    else:
        score_text = "nan"
        ttft_text = "nan"
    print(
        f"{model_name}\t{dataset_name}\t{baseline}\t"
        f"{metric_name}={score_text}\tavg_ttft={ttft_text}"
    )


@dataclass
class SSDSample:
    idx: int
    answers: List[str] = field(default_factory=list)
    prompt: str = ""
    plain_prompt: str = ""
    offline_prompt: str = ""
    query_sep: str = ""
    sample_dir_chunk: str = ""
    # query_cache stores materialized digest context KV plus critical-layer KV.
    sample_dir_query: str = ""
    has_cache: bool = False


class SSDPipelineEngine(BlendEngineBase):
    """Two-phase SSD pipeline runner."""

    def __init__(
        self,
        model_path: str,
        baseline: str = "ours",
        context_enhance: bool = False,
        cache_dir: str = "cache/qcfuse",
        digest_index_method: str = DIGEST_INDEX_METHOD,
        digest_ratio: float = DIGEST_RATIO,
        context_cache_source: str = "none",
        sample_evaluator=None,
    ):
        super().__init__(model_path, baseline)
        self.context_enhance = context_enhance
        self.context_n_sink = DEFAULT_CONTEXT_N_SINK
        self.digest_index_method = digest_index_method
        self.digest_ratio = digest_ratio
        self.context_cache_source = context_cache_source
        self.critical_layer_topk = DEFAULT_CRITICAL_LAYERS
        self.sample_evaluator = sample_evaluator

        self.cache_dir = cache_dir

    def _cache_paths(self, dataset_name: str, sample_idx: int) -> tuple[str, str]:
        chunk_dir, query_dir = self._cache_dataset_dirs(dataset_name)
        return (
            str(chunk_dir / f"sample_{sample_idx}"),
            str(query_dir / f"sample_{sample_idx}"),
        )

    def _cache_dataset_dirs(self, dataset_name: str) -> tuple[Path, Path]:
        base = Path(self.cache_dir)
        chunk_dir = base / "chunk_cache" / self.model_name / dataset_name
        query_dir = base / "query_cache" / self.model_name / dataset_name
        return chunk_dir, query_dir

    def _requires_query_cache(self) -> bool:
        return bool(self.context_enhance and self.context_cache_source == "query")

    def _offline_query_critical_layers(self) -> List[int]:
        if self.critical_layers:
            return [int(layer) for layer in self.critical_layers]
        return get_critical_layers(
            self.model_name,
            self._get_model_config()["num_layers"],
            critical_layers=self.critical_layer_topk,
        )

    @staticmethod
    def _has_packed_cache(sample_dir: Path) -> bool:
        meta_path = sample_dir / "kv_packed_meta.json"
        if not ((sample_dir / "kv_packed.bin").exists() and meta_path.exists()):
            return False
        meta = json.loads(meta_path.read_text())
        return meta.get("format") == PACKED_KV_FORMAT

    @staticmethod
    def _has_query_cache(sample_dir: Path) -> bool:
        meta_path = sample_dir / "query_packed_meta.json"
        if not ((sample_dir / "query_packed.bin").exists() and meta_path.exists()):
            return False
        meta = json.loads(meta_path.read_text())
        return meta.get("format") == QUERY_CACHE_FORMAT

    def _has_cache(
        self,
        sample_dir_chunk: str,
        sample_dir_query: str,
        require_query: bool = True,
        query_critical_layers: Optional[List[int]] = None,
    ) -> bool:
        chunk_dir = Path(sample_dir_chunk)
        query_dir = Path(sample_dir_query)
        if not self._has_packed_cache(chunk_dir):
            return False
        if not require_query:
            return True

        critical_layers = query_critical_layers
        if critical_layers is None:
            critical_layers = self.critical_layers
        if not critical_layers:
            return True
        if not self._has_query_cache(query_dir):
            return False

        query_meta = json.loads((query_dir / "query_packed_meta.json").read_text())
        meta = query_meta.get("metadata", {})
        method = DigestIndexManager.normalize_method(self.digest_index_method)
        digest_ratio = meta.get("digest_ratio")
        try:
            digest_ratio_matches = abs(float(digest_ratio) - self.digest_ratio) < 1e-9
        except (TypeError, ValueError):
            digest_ratio_matches = False
        critical_layers = [int(x) for x in critical_layers]
        expected_qcompute_end = max(critical_layers) + 1
        critical_layers_match = [
            int(x) for x in meta.get("critical_layers", [])
        ] == critical_layers
        qcompute_end_match = meta.get("qcompute_end") == expected_qcompute_end
        index_payload = query_meta.get("indices_by_method", {}).get(method, {})
        try:
            context_n_sink_match = (
                int(index_payload.get("context_n_sink", -1)) == self.context_n_sink
            )
        except (TypeError, ValueError):
            context_n_sink_match = False
        return (
            meta.get("digest_index_version") == DIGEST_INDEX_VERSION
            and meta.get("materialized_digest") is True
            and digest_ratio_matches
            and critical_layers_match
            and qcompute_end_match
            and query_meta.get("digest", {}).get("num_layers") == expected_qcompute_end
            and context_n_sink_match
            and set(critical_layers).issubset(
                set(int(x) for x in query_meta.get("critical", {}).get("layer_ids", []))
            )
        )

    def _ssd_args(self, sample: SSDSample) -> dict:
        return {
            "ssd_cache_path_chunk": sample.sample_dir_chunk,
            "ssd_cache_path_query": sample.sample_dir_query,
        }

    def _blend_args(
        self,
        blend_style: str,
        ratio: float,
        *,
        save_query_cache: bool = False,
        query_critical_layers: Optional[List[int]] = None,
    ) -> dict:
        args = {
            "blend_style": blend_style,
            "separator": BLEND_SEP,
            "start": self.start,
            "ratio": ratio,
            "method": self.method,
        }
        if self.method == "attn":
            args["attn_start"] = self.attn_start
            args["attn_end"] = self.attn_end
        uses_contextblend = save_query_cache or (
            self.context_enhance and blend_style != "KVCOMPUTE"
        )
        if uses_contextblend:
            args["is_contextblend"] = True
            if save_query_cache:
                args["context_cache_source"] = "query"
                args["digest_ratio"] = self.digest_ratio
                args["digest_index_method"] = self.digest_index_method
            else:
                args["context_cache_source"] = self.context_cache_source
            if not save_query_cache and self.context_cache_source == "query":
                args["digest_ratio"] = self.digest_ratio
                args["digest_index_method"] = self.digest_index_method
        critical_layers = (
            query_critical_layers
            if save_query_cache and query_critical_layers is not None
            else self.critical_layers
        )
        if critical_layers:
            args["critical_layers"] = [int(x) for x in critical_layers]
        if save_query_cache or blend_style == "KVCOMPUTE":
            args["context_n_sink"] = self.context_n_sink
        return args

    def _build_augmented_prompt(
        self, system_prompt: str, docs: List[str], q_prompt: List[str]
    ) -> str:
        sys_h, sys_e, asst_h = self._get_template()
        prefix = sys_h + system_prompt + sys_e
        suffix = "".join(q_prompt) + "\n\n## Answer\n" + asst_h
        parts = [prefix]
        for doc in docs:
            parts.extend([doc, DIGEST_ZIP_PROMPT])
        parts.append(suffix)
        return BLEND_SEP.join(parts)

    def _prepare_sample(
        self,
        example: Dict,
        dataset_name: str,
        sample_idx: int,
        system_prompt: str,
    ) -> SSDSample:
        answers = example.get("answers", [])
        if isinstance(answers, str):
            answers = [answers]

        sample_dir_chunk, sample_dir_query = self._cache_paths(dataset_name, sample_idx)
        docs, q_prompt = build_prompt_for_dataset(example, dataset_name)
        prompt, query_sep = self._build_prompt(
            system_prompt, docs, q_prompt, use_sep=True
        )
        plain_prompt, _ = self._build_prompt(
            system_prompt, docs, q_prompt, use_sep=False
        )
        offline_prompt = self._build_augmented_prompt(system_prompt, docs, q_prompt)

        return SSDSample(
            idx=sample_idx,
            answers=answers,
            prompt=prompt,
            plain_prompt=plain_prompt,
            offline_prompt=offline_prompt,
            query_sep=query_sep,
            sample_dir_chunk=sample_dir_chunk,
            sample_dir_query=sample_dir_query,
            has_cache=self._has_cache(
                sample_dir_chunk,
                sample_dir_query,
                require_query=self._requires_query_cache(),
            ),
        )

    def _prepare_samples(
        self,
        dataset: List[Dict],
        dataset_name: str,
    ) -> List[SSDSample]:
        system_prompt = get_system_prompt(dataset_name)
        return [
            self._prepare_sample(example, dataset_name, idx, system_prompt)
            for idx, example in enumerate(dataset)
        ]

    def _drain_generate(self, prompt: str, params: dict, **kwargs):
        for _ in self.llm.generate(prompt, params, stream=True, **kwargs):
            pass

    @staticmethod
    def _qcompute_params() -> dict:
        return {"temperature": 0, "max_new_tokens": 0}

    def _append_result(
        self,
        bucket: dict,
        result: dict,
        sample: SSDSample,
        dataset_name: str,
    ) -> float:
        if self.sample_evaluator is None:
            score = evaluate_sample(
                result["text"],
                sample.answers,
                dataset_name,
            )
        else:
            score = self.sample_evaluator.score(
                sample.idx,
                result["text"],
                sample.answers,
            )
        bucket["ttft"].append(result["ttft"])
        bucket["metric"].append(score)
        return score

    def warmup_blend(
        self,
        dataset: List[Dict],
        dataset_name: str,
        ratio: float,
        num_warmup: int = 3,
    ):
        if not dataset:
            return

        sample_data = self._prepare_samples(dataset, dataset_name)
        has_qcompute = self.method == "attn"
        is_fullcomp = self.baseline == "fullcomp"
        start_idx = max(0, len(sample_data) - num_warmup)

        for sample in sample_data[start_idx:]:
            if is_fullcomp:
                self._drain_generate(
                    sample.plain_prompt, {"temperature": 0, "max_new_tokens": 1}
                )
                continue
            if not sample.has_cache:
                continue
            if has_qcompute:
                self._drain_generate(
                    sample.query_sep,
                    self._qcompute_params(),
                    **self._blend_args("QCOMPUTE", ratio),
                    **self._ssd_args(sample),
                )
            self._drain_generate(
                sample.prompt,
                {"temperature": 0, "max_new_tokens": 1},
                **self._blend_args("DO_BLEND_FINISH", ratio),
                **self._ssd_args(sample),
            )

    def phase1_offline(
        self,
        dataset: List[Dict],
        dataset_name: str,
    ):
        """Run KVCOMPUTE for all samples, serialize to SSD. Skip if cache exists."""
        if self.first_style != "KVCOMPUTE":
            return
        system_prompt = get_system_prompt(dataset_name)
        query_critical_layers = self._offline_query_critical_layers()

        for idx, example in enumerate(dataset):
            sample_dir_chunk, sample_dir_query = self._cache_paths(dataset_name, idx)

            has_cache = self._has_cache(
                sample_dir_chunk,
                sample_dir_query,
                require_query=True,
                query_critical_layers=query_critical_layers,
            )
            if has_cache:
                continue

            sample = self._prepare_sample(example, dataset_name, idx, system_prompt)

            self._drain_generate(
                sample.offline_prompt,
                {"temperature": 0, "max_new_tokens": 1},
                **self._blend_args(
                    self.first_style,
                    0.0,
                    save_query_cache=True,
                    query_critical_layers=query_critical_layers,
                ),
                **self._ssd_args(sample),
            )

            if self._has_cache(
                sample_dir_chunk,
                sample_dir_query,
                require_query=True,
                query_critical_layers=query_critical_layers,
            ):
                continue
            if self._has_cache(
                sample_dir_chunk, sample_dir_query, require_query=False
            ):
                raise RuntimeError(
                    f"sample_{idx} query cache is incomplete: {sample_dir_query}"
                )
            raise RuntimeError(
                f"sample_{idx} SSD cache was not generated: {sample_dir_chunk}"
            )

    def phase2_online(
        self,
        dataset: List[Dict],
        dataset_name: str,
        ratio: float,
    ) -> dict:
        result_bucket = {
            "ttft": [],
            "metric": [],
        }

        has_qcompute = self.method == "attn"
        is_fullcomp = (self.baseline == "fullcomp")

        sample_data = self._prepare_samples(dataset, dataset_name)

        max_tokens = get_max_new_tokens(dataset_name)
        params = {"temperature": 0, "max_new_tokens": max_tokens}

        for sd in sample_data:
            if is_fullcomp:
                result = self._timed_generate(sd.plain_prompt, params)
                self._append_result(result_bucket, result, sd, dataset_name)
                continue

            if not sd.has_cache:
                continue

            q_time = 0
            if has_qcompute:
                start_q = time.time()
                self._drain_generate(
                    sd.query_sep,
                    self._qcompute_params(),
                    **self._blend_args("QCOMPUTE", ratio),
                    **self._ssd_args(sd),
                )
                q_time = time.time() - start_q

            result = self._timed_generate(
                sd.prompt,
                params,
                **self._blend_args("DO_BLEND_FINISH", ratio),
                **self._ssd_args(sd),
            )
            result["ttft"] += q_time

            self._append_result(result_bucket, result, sd, dataset_name)

        return result_bucket


def main():
    parser = argparse.ArgumentParser(description="SSD-backed QCFuse blend runner")
    parser.add_argument("--data_dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument(
        "--baseline",
        default="ours",
        choices=SUPPORTED_BASELINES,
        help="Baseline to run: fullcomp, ours, fuserag, or prophetkv",
    )
    parser.add_argument("--size", type=int, default=500)
    parser.add_argument(
        "--dataset",
        type=str,
        default="hotpotqa",
        help="Dataset name",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="qwen3-8b",
        help="Model name under --model_dir, or a full model path",
    )
    parser.add_argument(
        "--model_dir", type=str, default="", help="Base model directory"
    )
    parser.add_argument(
        "--cache_dir", type=str, default="cache/qcfuse",
        help="Base SSD directory for KV cache storage",
    )
    parser.add_argument(
        "--bird_data",
        "--bird-data",
        dest="bird_data",
        type=str,
        default=None,
        help="Official 500-row BIRD mini_dev_sqlite.json",
    )
    parser.add_argument(
        "--bird_db_root",
        "--bird-db-root",
        dest="bird_db_root",
        type=str,
        default=None,
        help="Directory containing the official BIRD Mini-Dev SQLite databases",
    )
    parser.add_argument(
        "--structured_eval_timeout",
        "--structured-eval-timeout",
        dest="structured_eval_timeout",
        type=float,
        default=30.0,
        help="Per-sample BIRD SQL execution timeout in seconds",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    model_arg = Path(args.model)
    model_path = str(
        model_arg if model_arg.is_absolute() else Path(args.model_dir) / args.model
    )
    model_name = Path(model_path).name
    dataset_name = args.dataset
    dataset_path = data_dir / f"{dataset_name}.jsonl"
    if not dataset_path.exists():
        raise FileNotFoundError(dataset_path)

    origin_dataset = load_dataset(str(dataset_path))
    dataset = origin_dataset[: min(args.size, len(origin_dataset))]
    metric_name = get_metric_name(dataset_name)
    sample_evaluator = None
    if dataset_name == "bird":
        from structured_eval import create_bird_evaluator

        sample_evaluator = create_bird_evaluator(
            dataset,
            data_dir,
            bird_data=args.bird_data,
            bird_db_root=args.bird_db_root,
            timeout_s=args.structured_eval_timeout,
        )

    with SSDPipelineEngine(
        model_path,
        baseline=args.baseline,
        context_enhance=False,
        cache_dir=args.cache_dir,
        digest_index_method=DIGEST_INDEX_METHOD,
        digest_ratio=DIGEST_RATIO,
        context_cache_source="none",
        sample_evaluator=sample_evaluator,
    ) as engine:
        engine.warmup(num_warmup=3)
        engine.set_baseline(args.baseline)
        if args.baseline in BLEND_BASELINES:
            engine.context_enhance = True
            engine.context_cache_source = "query"
            engine.digest_ratio = BASELINE_DIGEST_RATIOS[args.baseline]
            if args.baseline == "ours":
                set_ours_layers(engine, model_name)
            elif args.baseline == "fuserag":
                set_fuserag_layers(engine, model_name)
            elif args.baseline == "prophetkv":
                set_prophetkv_layers(engine, model_name)
        else:
            engine.context_enhance = False
            engine.context_cache_source = "none"
            engine.critical_layers = None

        if args.baseline in BLEND_BASELINES:
            engine.phase1_offline(dataset, dataset_name)
            ratio = DEFAULT_BLEND_RATIO
        else:
            ratio = 1.0

        engine.warmup_blend(dataset, dataset_name, ratio)
        result = engine.phase2_online(
            dataset,
            dataset_name,
            ratio,
        )
        print_final_metrics(
            model_name,
            dataset_name,
            args.baseline,
            result,
            metric_name,
        )


if __name__ == "__main__":
    main()
