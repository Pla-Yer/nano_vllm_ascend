import torch

from nanovllm_ascend.npu.paged_kv_cache import PagedKVCache


def test_prefill_metadata_cumulative_lengths_and_slots():
    cache = PagedKVCache(
        num_layers=1,
        num_blocks=8,
        block_size=4,
        max_num_seqs=3,
        num_kv_heads=2,
        head_dim=8,
        dtype=torch.float32,
        device="cpu",
    )

    metadata = cache.prepare_prefill_metadata(seq_slots=[0, 1, 2], seq_lens=[3, 5, 2])

    assert metadata.actual_seq_lengths_q == [3, 8, 10]
    assert metadata.actual_seq_lengths_kv == [3, 8, 10]
    assert cache.seq_lens == [3, 5, 2]
    assert metadata.slot_mapping.tolist() == [0, 1, 2, 4, 5, 6, 7, 8, 12, 13]


def test_decode_metadata_for_active_slots_and_positions():
    cache = PagedKVCache(
        num_layers=1,
        num_blocks=8,
        block_size=4,
        max_num_seqs=3,
        num_kv_heads=2,
        head_dim=8,
        dtype=torch.float32,
        device="cpu",
    )
    cache.prepare_prefill_metadata(seq_slots=[0, 1, 2], seq_lens=[3, 5, 2])

    metadata = cache.prepare_metadata(seq_slots=[1, 2], start_pos=[5, 2], q_len=1)

    assert cache.seq_lens == [3, 6, 3]
    assert metadata.context_lens.tolist() == [6, 3]
    assert metadata.slot_mapping.tolist() == [9, 14]

