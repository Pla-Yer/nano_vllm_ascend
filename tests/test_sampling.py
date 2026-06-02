import torch

from nanovllm_ascend.engine import LLM
from nanovllm_ascend.layers.sampler import Sampler
from nanovllm_ascend.sampling_params import SamplingParams


def test_sampling_params_defaults_are_valid():
    params = SamplingParams()
    assert params.temperature == 0.0
    assert params.top_k == 0
    assert params.top_p == 1.0
    assert params.is_greedy()


def test_sampling_params_validation():
    for kwargs in (
        {"temperature": -0.1},
        {"top_k": -1},
        {"top_p": 0.0},
        {"top_p": 1.1},
    ):
        try:
            SamplingParams(**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {kwargs}")


def test_sampler_greedy_matches_argmax():
    sampler = Sampler()
    logits = torch.tensor([[0.1, 3.0, 1.5], [5.0, 1.0, 4.0]])
    tokens = sampler.sample(logits, SamplingParams())
    assert tokens.tolist() == [1, 0]


def test_sampler_top_k_only_samples_from_top_k():
    torch.manual_seed(0)
    sampler = Sampler()
    logits = torch.tensor([[10.0, 9.0, 8.0, 7.0]])
    tokens = sampler.sample(logits, SamplingParams(temperature=1.0, top_k=2))
    assert int(tokens.item()) in {0, 1}


def test_sampler_top_p_only_samples_from_nucleus():
    torch.manual_seed(0)
    sampler = Sampler()
    logits = torch.tensor([[10.0, 9.0, -10.0, -10.0]])
    tokens = sampler.sample(logits, SamplingParams(temperature=1.0, top_p=0.9))
    assert int(tokens.item()) in {0, 1}


def test_sampler_temperature_sampling_shape():
    torch.manual_seed(0)
    sampler = Sampler()
    logits = torch.tensor([[1.0, 2.0, 3.0], [1.0, 4.0, 2.0]])
    tokens = sampler.sample(logits, SamplingParams(temperature=0.7))
    assert tokens.shape == (2,)


def test_generate_sampling_params_overrides():
    llm = object.__new__(LLM)
    resolved = llm._resolve_sampling_params(
        SamplingParams(temperature=0.8, top_k=5, top_p=0.9),
        temperature=None,
        top_k=10,
        top_p=None,
    )
    assert resolved.temperature == 0.8
    assert resolved.top_k == 10
    assert resolved.top_p == 0.9


def test_generate_default_sampling_params_are_greedy():
    llm = object.__new__(LLM)
    resolved = llm._resolve_sampling_params(
        None,
        temperature=None,
        top_k=None,
        top_p=None,
    )
    assert resolved == SamplingParams()
