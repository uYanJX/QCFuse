from enum import Enum
import math
import torch
from dataclasses import dataclass


DEFAULT_DIGEST_RATIO = 0.15


class BlendStyle(Enum):
    """Blend style enum"""

    KVCOMPUTE = 0
    QCOMPUTE = 1
    DO_BLEND = 2
    DO_BLEND_FINISH = 3

    @classmethod
    def parse(cls, value):
        if value is None:
            return None
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            name = value.upper()
            if name in cls.__members__:
                return cls[name]
            return None
        return None


class SelectMode(Enum):
    """Selection Strategy for Cache Blending"""

    ATTN = "attn"  # Attention based


@dataclass
class AttParams:
    """Parameters for Attention-based selection"""

    num_heads: int = 32
    num_kv_heads: int = 8
    head_dim: int = 128
    num_layers: int = 32


class HackBlendKVPool:
    k_buffer = []
    v_buffer = []
    q_buffer = []
    query_k_buffer = []
    q_lens = []
    q_offsets = []
    query_k_lens = []

    @classmethod
    def clear(cls):
        cls.k_buffer = []
        cls.v_buffer = []
        cls.q_buffer = []
        cls.query_k_buffer = []
        cls.q_lens = []
        cls.q_offsets = []
        cls.query_k_lens = []

    @classmethod
    def has_kv(cls, layer_id: int) -> bool:
        return (layer_id < len(cls.k_buffer)
                and cls.k_buffer[layer_id] is not None
                and isinstance(cls.k_buffer[layer_id], torch.Tensor)
                and cls.k_buffer[layer_id].numel() > 0)

    @classmethod
    def clear_layer(cls, layer_id: int):
        if layer_id < len(cls.k_buffer):
            cls.k_buffer[layer_id] = None
            cls.v_buffer[layer_id] = None

    @classmethod
    def init_buffers(cls, num_layers: int):
        cls.k_buffer = [None] * num_layers
        cls.v_buffer = [None] * num_layers
        cls.q_buffer = []
        cls.query_k_buffer = []
        cls.q_lens = []
        cls.q_offsets = []
        cls.query_k_lens = []

    @classmethod
    def compact_kv(cls, token_indices):
        if token_indices is None:
            return

        index_by_device = {}
        for layer_id, (k_tensor, v_tensor) in enumerate(zip(cls.k_buffer, cls.v_buffer)):
            if k_tensor is None or v_tensor is None:
                continue
            device = k_tensor.device
            index_t = index_by_device.get(device)
            if index_t is None:
                index_t = torch.as_tensor(
                    token_indices, dtype=torch.long, device=device
                )
                index_by_device[device] = index_t
            cls.k_buffer[layer_id] = k_tensor.index_select(0, index_t).contiguous()
            cls.v_buffer[layer_id] = v_tensor.index_select(0, index_t).contiguous()

    @classmethod
    def put_kv(cls, k: torch.Tensor, v: torch.Tensor, layer_id: int):
        # RotaryEmbedding CUDA path mutates K in-place after this call. Keep the
        # pool copy raw so SSD cache can be re-rotated at online positions.
        k = k.clone()
        while len(cls.k_buffer) <= layer_id:
            cls.k_buffer.append(None)
            cls.v_buffer.append(None)

        if cls.k_buffer[layer_id] is None:
            cls.k_buffer[layer_id] = k
        else:
            cls.k_buffer[layer_id] = torch.cat([cls.k_buffer[layer_id], k], dim=0)

        if cls.v_buffer[layer_id] is None:
            cls.v_buffer[layer_id] = v
        else:
            cls.v_buffer[layer_id] = torch.cat([cls.v_buffer[layer_id], v], dim=0)

    @classmethod
    def get_kv(cls, layer_id: int):
        return (cls.k_buffer[layer_id], cls.v_buffer[layer_id])

    @classmethod
    def get_all_kv(cls, start: int, end: int):
        return cls.k_buffer[start:end], cls.v_buffer[start:end]

    @classmethod
    def get_kv_layers(cls, layer_ids):
        return (
            [cls.k_buffer[int(layer_id)] for layer_id in layer_ids],
            [cls.v_buffer[int(layer_id)] for layer_id in layer_ids],
        )

    @classmethod
    def put_q(cls, q: torch.Tensor, layer_id: int):
        while len(cls.q_buffer) <= layer_id:
            cls.q_buffer.append(None)
        if cls.q_buffer[layer_id] is None:
            cls.q_buffer[layer_id] = q
        else:
            cls.q_buffer[layer_id] = torch.cat([cls.q_buffer[layer_id], q], dim=0)

    @classmethod
    def get_q(cls, layer_id: int):
        return cls.q_buffer[layer_id]

    @classmethod
    def get_all_q(cls, start: int, end: int):
        return cls.q_buffer[start:end]

    @classmethod
    def get_q_layers(cls, layer_ids):
        return [cls.q_buffer[int(layer_id)] for layer_id in layer_ids]

    @classmethod
    def put_query_k(cls, k: torch.Tensor, layer_id: int):
        while len(cls.query_k_buffer) <= layer_id:
            cls.query_k_buffer.append(None)
        if cls.query_k_buffer[layer_id] is None:
            cls.query_k_buffer[layer_id] = k
        else:
            cls.query_k_buffer[layer_id] = torch.cat(
                [cls.query_k_buffer[layer_id], k], dim=0
            )

    @classmethod
    def get_query_k(cls, layer_id: int):
        return cls.query_k_buffer[layer_id]

    @classmethod
    def get_all_query_k(cls, start: int, end: int):
        return cls.query_k_buffer[start:end]

    @classmethod
    def get_query_k_layers(cls, layer_ids):
        return [cls.query_k_buffer[int(layer_id)] for layer_id in layer_ids]

class ContextBlendPool:
    """Store digest-index metadata and runtime QCOMPUTE prefix KV."""

    k_buffer = []  # List[Tensor], runtime prefix K for each layer
    v_buffer = []  # List[Tensor], runtime prefix V for each layer
    ranked_indices_by_chunk = []  # List[List[int]], chunk-local anchor ranking
    ranked_indices_by_layer_chunk = []  # List[layer][chunk][ranked local idx]
    orig_chunk_ranges = []  # List[(start, end)], non-query chunk absolute ranges
    context_positions = []  # List[int], runtime loaded prefix token positions
    context_position_spans = []  # List[(start, end, out_start)] for packed KV reads
    context_positions_by_layer = []
    context_position_spans_by_layer = []
    num_index_layers = 0
    total_tokens = 0  # Query position offset in the original full prompt
    digest_ratio = None

    @classmethod
    def clear(cls):
        cls.k_buffer = []
        cls.v_buffer = []
        cls.ranked_indices_by_chunk = []
        cls.ranked_indices_by_layer_chunk = []
        cls.orig_chunk_ranges = []
        cls.context_positions = []
        cls.context_position_spans = []
        cls.context_positions_by_layer = []
        cls.context_position_spans_by_layer = []
        cls.num_index_layers = 0
        cls.total_tokens = 0
        cls.digest_ratio = None

    @classmethod
    def init_buffers(cls, num_layers: int):
        cls.k_buffer = [None] * num_layers
        cls.v_buffer = [None] * num_layers

    @classmethod
    def get(cls, layer_id: int):
        """Return compressed context K/V for one layer."""
        return cls.k_buffer[layer_id], cls.v_buffer[layer_id]

    @classmethod
    def get_kv_layers(cls, layer_ids):
        return (
            [cls.k_buffer[int(layer_id)] for layer_id in layer_ids],
            [cls.v_buffer[int(layer_id)] for layer_id in layer_ids],
        )

    @classmethod
    def get_all_kv(cls, start: int, end: int):
        return cls.k_buffer[start:end], cls.v_buffer[start:end]

    @classmethod
    def set_index_metadata(
        cls,
        *,
        ranked_indices_by_chunk=None,
        ranked_indices_by_layer_chunk=None,
        orig_chunk_ranges,
        total_tokens,
        num_layers=None,
    ):
        cls.ranked_indices_by_chunk = [
            list(x) for x in (ranked_indices_by_chunk or [])
        ]
        cls.ranked_indices_by_layer_chunk = [
            [list(chunk) for chunk in layer]
            for layer in (ranked_indices_by_layer_chunk or [])
        ]
        cls.orig_chunk_ranges = [tuple(x) for x in orig_chunk_ranges]
        cls.total_tokens = int(total_tokens)
        cls.num_index_layers = int(
            num_layers
            or len(cls.ranked_indices_by_layer_chunk)
            or len(cls.k_buffer)
            or 0
        )
        cls.context_positions = []
        cls.context_position_spans = []
        cls.context_positions_by_layer = []
        cls.context_position_spans_by_layer = []
        cls.digest_ratio = None

    @staticmethod
    def _coalesce_sorted_positions(positions):
        if not positions:
            return []
        spans = []
        run_start = int(positions[0])
        prev = run_start
        out_start = 0
        for raw_pos in positions[1:]:
            pos = int(raw_pos)
            if pos == prev + 1:
                prev = pos
                continue
            spans.append((run_start, prev + 1, out_start))
            out_start += prev - run_start + 1
            run_start = pos
            prev = pos
        spans.append((run_start, prev + 1, out_start))
        return spans

    @classmethod
    def _ranked_indices_for_layer(cls, layer_id: int):
        if (
            cls.ranked_indices_by_layer_chunk
            and 0 <= int(layer_id) < len(cls.ranked_indices_by_layer_chunk)
        ):
            return cls.ranked_indices_by_layer_chunk[int(layer_id)]
        return cls.ranked_indices_by_chunk

    @classmethod
    def _build_context_positions_for_layer(
        cls,
        layer_id: int = 0,
        digest_ratio: float = DEFAULT_DIGEST_RATIO,
    ):
        positions = []
        ranked_by_chunk = cls._ranked_indices_for_layer(layer_id)
        for chunk_idx, (orig_start, orig_end) in enumerate(cls.orig_chunk_ranges):
            orig_start = int(orig_start)
            orig_end = int(orig_end)
            chunk_len = orig_end - orig_start
            if chunk_len <= 0:
                continue

            # The first non-query chunk is the system prompt and is always kept.
            if chunk_idx == 0:
                positions.extend(range(orig_start, orig_end))
                continue

            ratio = min(1.0, max(0.0, float(digest_ratio)))
            n_left = int(math.ceil(chunk_len * ratio))
            n_left = min(chunk_len, n_left)
            local_selected = []
            seen = set()
            ranked = (
                ranked_by_chunk[chunk_idx]
                if chunk_idx < len(ranked_by_chunk)
                else []
            )
            for raw_idx in ranked[:n_left]:
                idx = int(raw_idx)
                if idx in seen or idx < 0 or idx >= chunk_len:
                    continue
                seen.add(idx)
                local_selected.append(idx)
            local_selected.sort()
            positions.extend(orig_start + idx for idx in local_selected)

        return positions

    @classmethod
    def build_context_positions(
        cls, digest_ratio: float = DEFAULT_DIGEST_RATIO
    ):
        num_layers = max(
            int(cls.num_index_layers or 0),
            len(cls.ranked_indices_by_layer_chunk),
            len(cls.k_buffer),
            1,
        )
        cls.context_positions_by_layer = []
        cls.context_position_spans_by_layer = []
        for layer_id in range(num_layers):
            positions = cls._build_context_positions_for_layer(
                layer_id=layer_id, digest_ratio=digest_ratio
            )
            cls.context_positions_by_layer.append(positions)
            cls.context_position_spans_by_layer.append(
                cls._coalesce_sorted_positions(positions)
            )

        positions = cls.context_positions_by_layer[0] if cls.context_positions_by_layer else []
        cls.context_positions = positions
        cls.context_position_spans = cls._coalesce_sorted_positions(positions)
        cls.digest_ratio = digest_ratio
        return positions

    @classmethod
    def set_materialized_positions(cls, positions_by_layer):
        cls.context_positions_by_layer = [
            [int(x) for x in layer_positions]
            for layer_positions in (positions_by_layer or [])
        ]
        cls.context_position_spans_by_layer = [
            cls._coalesce_sorted_positions(layer_positions)
            for layer_positions in cls.context_positions_by_layer
        ]
        cls.context_positions = (
            cls.context_positions_by_layer[0]
            if cls.context_positions_by_layer
            else []
        )
        cls.context_position_spans = cls._coalesce_sorted_positions(
            cls.context_positions
        )

    @classmethod
    def get_context_positions(cls, layer_id: int):
        if cls.context_positions_by_layer and 0 <= int(layer_id) < len(
            cls.context_positions_by_layer
        ):
            return cls.context_positions_by_layer[int(layer_id)]
        return cls.context_positions

    @classmethod
    def get_context_position_spans(cls, layer_id: int):
        if cls.context_position_spans_by_layer and 0 <= int(layer_id) < len(
            cls.context_position_spans_by_layer
        ):
            return cls.context_position_spans_by_layer[int(layer_id)]
        return cls.context_position_spans


class BatchBlendInfo:
    """Store cache blend info"""

    blend_style: BlendStyle = None
    select_mode: SelectMode = SelectMode.ATTN
    ratio: float = 0.3
    att_params: AttParams = None
    start: int = 0
    attn_start: int = 0
    attn_end: int = -1
    chunk_lens: torch.Tensor = None
    chunk_loc_list: torch.Tensor = None
    req_len_list: torch.Tensor = None
    blend_top_indices: torch.Tensor = None
    blend_top_lens: torch.Tensor = None
    fake_q: torch.Tensor = None
    quest_indices: torch.Tensor = None
    query_indices: torch.Tensor = None
    positions: torch.Tensor = None
    init_attmeta: bool = False
    is_contextblend: bool = False
    context_cache_source: str = "query"
    context_n_sink: int = 4
    digest_index_method: str = "kvzip"
    digest_ratio: float = DEFAULT_DIGEST_RATIO
    critical_layers: list = None
    critical_layers_set: set = None
    qcompute_end: int = None
    digest_keep_indices: list = None
    digest_original_chunk_loc_list: torch.Tensor = None
    digest_aug_sys_range: tuple = None
    digest_aug_doc_ranges: list = None
    digest_aug_zip_ranges: list = None
    # SSD: layers to keep during DO_BLEND selective clear
    keep_layers_set: set = None

    def should_collect_q(self, layer_id: int) -> bool:
        if self.blend_style != BlendStyle.QCOMPUTE:
            return False
        if self.critical_layers_set:
            return int(layer_id) in self.critical_layers_set
        attn_start = int(self.attn_start or 0)
        if self.attn_end is None:
            return layer_id >= attn_start
        attn_end = int(self.attn_end)
        if attn_end < 0:
            return layer_id >= attn_start
        return attn_start <= layer_id < attn_end
