import torch

from nanovllm_ascend.npu.block_manager import BlockManager


def test_prefill_metadata_cumulative_lengths_and_slots():
    manager = BlockManager(num_blocks=8, block_size=4, device="cpu")

    metadata = manager.prepare_prefill_metadata(
        slots=[0, 1, 2],
        seq_lens=[3, 5, 2],
        start_positions=[0, 0, 0],
        prefix_block_ids=[[], [], []],
    )

    assert metadata.actual_seq_lengths_q == [3, 8, 10]
    assert metadata.actual_seq_lengths_kv == [3, 8, 10]
    assert metadata.context_lens.tolist() == [3, 5, 2]
    assert metadata.slot_mapping.tolist() == [0, 1, 2, 4, 5, 6, 7, 8, 12, 13]


def test_decode_metadata_for_active_slots_and_positions():
    manager = BlockManager(num_blocks=8, block_size=4, device="cpu")
    manager.prepare_prefill_metadata(
        slots=[0, 1, 2],
        seq_lens=[3, 5, 2],
        start_positions=[0, 0, 0],
        prefix_block_ids=[[], [], []],
    )

    metadata = manager.prepare_decode_metadata(
        slots=[1, 2],
        start_positions=[5, 2],
        q_len=1,
    )

    assert metadata.context_lens.tolist() == [6, 3]
    assert metadata.slot_mapping.tolist() == [9, 14]


def test_prefix_cache_reuses_full_blocks():
    manager = BlockManager(num_blocks=8, block_size=4, device="cpu")
    token_ids = torch.tensor([10, 11, 12, 13, 20, 21], dtype=torch.long)

    manager.prepare_prefill_metadata(
        slots=[0],
        seq_lens=[6],
        start_positions=[0],
        prefix_block_ids=[[]],
    )
    manager.cache_full_blocks(slot=0, token_ids=token_ids)
    manager.free_slot(0)

    assert manager.find_longest_prefix_blocks(token_ids, max_cache_blocks=1) == [0]
