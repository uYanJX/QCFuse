from typing import List, NamedTuple, Optional, Tuple

import torch

from sglang.srt.utils.cache_blender_info import (
    BatchBlendInfo,
    HackBlendKVPool,
    SelectMode,
)


class _ReqRegion(NamedTuple):
    req_start: int
    quest_start: int
    quest_end: int
    rag_start: int
    rag_len: int


def _compute_request_boundaries(info: BatchBlendInfo) -> Tuple[int, torch.Tensor]:
    chunk_loc = info.chunk_loc_list
    req_len_list = info.req_len_list
    device = chunk_loc.device

    num_reqs = len(req_len_list)
    req_boundaries = torch.zeros(num_reqs + 1, dtype=torch.long, device=device)
    req_boundaries[1:] = torch.cumsum(req_len_list, dim=0)
    return num_reqs, req_boundaries


def _get_request_region(
    chunk_loc: torch.Tensor, req_boundaries: torch.Tensor, req_idx: int
) -> _ReqRegion:
    req_start_chunk = req_boundaries[req_idx].item()
    req_end_chunk = req_boundaries[req_idx + 1].item()

    prefix_idx = req_start_chunk
    quest_idx = req_end_chunk - 1

    req_start = chunk_loc[prefix_idx].item()
    quest_start = chunk_loc[quest_idx].item()
    quest_end = chunk_loc[quest_idx + 1].item()

    if quest_idx > prefix_idx + 1:
        rag_start = chunk_loc[prefix_idx + 1].item()
        rag_len = quest_start - rag_start
    else:
        rag_start = quest_start
        rag_len = 0

    return _ReqRegion(req_start, quest_start, quest_end, rag_start, rag_len)


def _build_final_output(
    final_indices_list: List[torch.Tensor], req_lens_list: List[int], device
) -> Tuple[torch.Tensor, torch.Tensor]:
    return torch.cat(final_indices_list), torch.tensor(req_lens_list, device=device)


def _compute_budget(length: int, ratio: float) -> int:
    if ratio <= 0:
        return 0
    return min(length, max(1, int(length * ratio)))


class IndiceSelector:
    @staticmethod
    def select(
        info: BatchBlendInfo,
        old_k: Optional[List[torch.Tensor]] = None,
        old_q: Optional[List[torch.Tensor]] = None,
        positions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if info.ratio <= 0:
            return IndiceSelector._compute_query_only(info)
        if info.select_mode != SelectMode.ATTN:
            raise ValueError(f"Unsupported select mode for release: {info.select_mode}")
        if old_k is None or old_q is None:
            raise ValueError("ATTN mode requires old_k and old_q")
        return IndiceSelector._compute_layer_fusion(info, old_k, old_q, positions)

    @staticmethod
    def _compute_query_only(info: BatchBlendInfo) -> Tuple[torch.Tensor, torch.Tensor]:
        chunk_loc = info.chunk_loc_list
        device = chunk_loc.device

        num_reqs, req_boundaries = _compute_request_boundaries(info)
        final_indices_list = []
        req_lens_list = []

        for req_idx in range(num_reqs):
            r = _get_request_region(chunk_loc, req_boundaries, req_idx)
            quest_indices = torch.arange(r.quest_start, r.quest_end, device=device)
            final_indices_list.append(quest_indices)
            req_lens_list.append(quest_indices.numel())

        return _build_final_output(final_indices_list, req_lens_list, device)

    @staticmethod
    def _rotate_stacked(
        rotary_emb,
        token_positions: torch.Tensor,
        tensor: torch.Tensor,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        *,
        is_query: bool,
    ) -> torch.Tensor:
        pos = token_positions.repeat(num_layers)
        flat = tensor.reshape(-1, num_heads * head_dim)
        q_rot, k_rot = rotary_emb(pos, flat, flat)
        rotated = q_rot if is_query else k_rot
        return rotated.reshape(num_layers, -1, num_heads, head_dim)

    @staticmethod
    def _compute_layer_fusion(
        info: BatchBlendInfo,
        old_k: List[torch.Tensor],
        old_q: List[torch.Tensor],
        positions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        params = info.att_params
        chunk_loc = info.chunk_loc_list
        device = chunk_loc.device
        ratio = info.ratio
        layer_ids = [int(x) for x in (getattr(info, "critical_layers", None) or [])]
        if not layer_ids:
            layer_ids = list(range(int(info.attn_start), int(info.attn_end)))
        num_layers = len(layer_ids)
        rotary_emb = getattr(info, "rotary_emb", None)

        num_reqs, req_boundaries = _compute_request_boundaries(info)
        final_indices_list: List[torch.Tensor] = []
        req_lens_list: List[int] = []

        old_k_stacked = torch.stack(old_k, dim=0)
        old_q_stacked = torch.stack(old_q, dim=0)
        if not old_q_stacked.is_cuda:
            raise RuntimeError("ATTN selection requires CUDA.")

        if getattr(info, "critical_layers", None):
            query_k_layers = HackBlendKVPool.get_query_k_layers(layer_ids)
        else:
            query_k_layers = HackBlendKVPool.get_all_query_k(
                info.attn_start, info.attn_end
            )
        query_k_stacked = torch.stack(query_k_layers, dim=0)
        from sglang.srt.utils.triton_attention_score import (
            compute_att_full_softmax_importance,
        )

        q_lens = HackBlendKVPool.q_lens
        q_offsets = HackBlendKVPool.q_offsets
        query_k_lens = HackBlendKVPool.query_k_lens

        q_loc = 0
        query_k_loc = 0
        for req_idx in range(num_reqs):
            r = _get_request_region(chunk_loc, req_boundaries, req_idx)
            q_len = int(q_lens[req_idx])
            q_offset = int(q_offsets[req_idx])
            query_k_len = int(query_k_lens[req_idx])
            q_pos_start = r.quest_start + q_offset
            q_pos_end = q_pos_start + q_len
            k_abs_end = q_pos_end
            quest_indices = torch.arange(r.quest_start, r.quest_end, device=device)

            if r.rag_len > 0:
                prefix_len = r.quest_start - r.req_start
                target_start = r.rag_start - r.req_start
                full_q_start = prefix_len + q_offset

                q_chunk = old_q_stacked[:, q_loc : q_loc + q_len].reshape(
                    num_layers, q_len, params.num_heads, params.head_dim
                )
                prefix_k = old_k_stacked[:, r.req_start : r.quest_start]
                query_k = query_k_stacked[
                    :, query_k_loc : query_k_loc + query_k_len
                ]
                k_full = torch.cat([prefix_k, query_k], dim=1).reshape(
                    num_layers,
                    prefix_len + query_k_len,
                    params.num_kv_heads,
                    params.head_dim,
                )

                if rotary_emb is not None and positions is not None:
                    q_chunk = IndiceSelector._rotate_stacked(
                        rotary_emb,
                        positions[q_pos_start:q_pos_end],
                        q_chunk,
                        num_layers,
                        params.num_heads,
                        params.head_dim,
                        is_query=True,
                    )
                    k_full = IndiceSelector._rotate_stacked(
                        rotary_emb,
                        positions[r.req_start:k_abs_end],
                        k_full,
                        num_layers,
                        params.num_kv_heads,
                        params.head_dim,
                        is_query=False,
                    )

                # Average normalized attention over query tokens, query heads,
                # and selected layers before ranking context tokens.
                importance = compute_att_full_softmax_importance(
                    q_chunk,
                    k_full,
                    target_start=target_start,
                    target_len=r.rag_len,
                    q_start=full_q_start,
                )

                k_budget = _compute_budget(r.rag_len, ratio)
                if k_budget > 0:
                    _, top_idx = torch.topk(
                        importance, k=min(k_budget, importance.numel())
                    )
                    selected_rag, _ = torch.sort(top_idx + r.rag_start)
                    req_selection = torch.cat([selected_rag, quest_indices])
                else:
                    req_selection = quest_indices
            else:
                req_selection = quest_indices

            q_loc += q_len
            query_k_loc += query_k_len
            final_indices_list.append(req_selection)
            req_lens_list.append(req_selection.numel())

        return _build_final_output(final_indices_list, req_lens_list, device)
