from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

from nanovllm_ascend.engine import LLM


class FakeTensor:
    def __init__(self, values=None):
        self.values = list(values or [])
        self.copied_from = None

    def clone(self):
        return FakeTensor(self.values)

    def copy_(self, other):
        self.copied_from = other
        self.values = list(other.values)
        return self

    def numel(self):
        return len(self.values)


class FakeGraph:
    def __init__(self):
        self.replays = 0

    def replay(self):
        self.replays += 1


class FakeStream:
    def __init__(self):
        self.waited_for = None

    def wait_stream(self, stream):
        self.waited_for = stream


class FakeExternalEvent:
    def __init__(self):
        self.recorded = None

    def record(self, stream):
        self.recorded = stream


class FakeGraphContext:
    def __init__(self, graph):
        self.graph = graph

    def __enter__(self):
        return self.graph

    def __exit__(self, exc_type, exc, tb):
        return False


def install_fake_graph_modules(monkeypatch):
    fake_torch = types.ModuleType("torch")
    fake_npu = types.SimpleNamespace(
        NPUGraph=FakeGraph,
        Stream=FakeStream,
        ExternalEvent=FakeExternalEvent,
        current_stream=lambda: FakeStream(),
        synchronize=lambda: None,
        graph=lambda graph: FakeGraphContext(graph),
        stream=lambda stream: FakeGraphContext(stream),
        graph_task_update_begin=lambda stream, handle: None,
        graph_task_update_end=lambda stream: None,
    )
    fake_torch.npu = fake_npu
    fake_torch.Tensor = FakeTensor
    fake_torch.device = str

    fake_torch_npu = types.ModuleType("torch_npu")
    fake_torch_npu._npu_paged_attention_get_workspace = lambda **kwargs: object()
    fake_torch_npu._npu_paged_attention = lambda **kwargs: None

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "torch_npu", fake_torch_npu)
    sys.modules.pop("nanovllm_ascend.npu.acl_graph", None)
    return importlib.import_module("nanovllm_ascend.npu.acl_graph")


class FakeOutput:
    def __init__(self, logits):
        self.logits = logits


class FakeModel:
    def __init__(self, graph_mod):
        self.graph_mod = graph_mod
        self.calls = 0

    def __call__(self, **kwargs):
        self.calls += 1
        self.graph_mod.record_paged_attention_task(
            self.graph_mod.PagedAttentionGraphTask(
                query=FakeTensor([1]),
                key_cache=FakeTensor([2]),
                value_cache=FakeTensor([3]),
                num_kv_heads=1,
                num_heads=1,
                scale=1.0,
                block_tables=kwargs["attn_metadata"].block_tables,
                context_lens=kwargs["attn_metadata"].context_lens,
                output=FakeTensor([4]),
                workspace=object(),
                handle=object(),
                event=FakeExternalEvent(),
            )
        )
        return FakeOutput(FakeTensor([99]))


def make_metadata(graph_mod, batch_size=1):
    return graph_mod.PagedKVCacheMetadata(
        block_tables=FakeTensor([7] * batch_size),
        context_lens=FakeTensor([8] * batch_size),
        slot_mapping=FakeTensor([9] * batch_size),
    )


def test_decode_graph_runner_captures_once_then_replays(monkeypatch):
    graph_mod = install_fake_graph_modules(monkeypatch)
    model = FakeModel(graph_mod)
    runner = graph_mod.DecodeGraphRunner(
        model=model,
        kv_cache=object(),
        batch_sizes=[1],
    )
    metadata = make_metadata(graph_mod)

    first = runner.forward(FakeTensor([1]), FakeTensor([2]), metadata)
    second = runner.forward(FakeTensor([3]), FakeTensor([4]), metadata)

    assert first.values == [99]
    assert second.values == [99]
    assert model.calls == 1
    assert runner.stats_dict()["captures"] == 1
    assert runner.stats_dict()["replays"] == 2
    assert runner.stats_dict()["updates"] == 2


def test_decode_graph_runner_disables_only_failed_batch_size(monkeypatch):
    graph_mod = install_fake_graph_modules(monkeypatch)

    class FailingModel:
        def __call__(self, **kwargs):
            raise RuntimeError("capture failed")

    runner = graph_mod.DecodeGraphRunner(
        model=FailingModel(),
        kv_cache=object(),
        batch_sizes=[1, 2],
    )

    result = runner.forward(
        FakeTensor([1]),
        FakeTensor([2]),
        make_metadata(graph_mod),
    )

    assert result is None
    assert 1 in runner.disabled_batch_sizes
    assert runner.supports(2)
    assert runner.stats_dict()["capture_failures"] == 1
    assert runner.stats_dict()["fallbacks"] == 1


def test_llm_decode_graph_api_leaves_default_sizes_to_model_runner(monkeypatch):
    captured = {}

    class FakeModelRunner:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.num_blocks = 16

    fake_module = types.ModuleType("nanovllm_ascend.model_runner")
    fake_module.ModelRunner = FakeModelRunner
    monkeypatch.setitem(sys.modules, "nanovllm_ascend.model_runner", fake_module)

    LLM(
        "model",
        max_num_seqs=3,
        enable_decode_graph=True,
    )

    assert captured["enable_decode_graph"] is True
    assert captured["decode_graph_batch_sizes"] is None


def test_model_runner_default_decode_graph_batch_sizes():
    source = Path("src/nanovllm_ascend/model_runner.py").read_text(encoding="utf-8")

    assert "DEFAULT_DECODE_GRAPH_BATCH_SIZES = [1, 2, 4, 8, 16]" in source
