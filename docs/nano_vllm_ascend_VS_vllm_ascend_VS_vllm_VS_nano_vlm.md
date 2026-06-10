## background
NPU Info: Ascend 910B3
root@localhost:/home/player/models/Qwen3/nanovllm_ascend_new# npu-smi info
+------------------------------------------------------------------------------------------------+
| npu-smi 24.1.rc1                 Version: 24.1.rc1                                             |
+---------------------------+---------------+----------------------------------------------------+
| NPU   Name                | Health        | Power(W)    Temp(C)           Hugepages-Usage(page)|
| Chip                      | Bus-Id        | AICore(%)   Memory-Usage(MB)  HBM-Usage(MB)        |
+===========================+===============+====================================================+
| 6     910B3               | OK            | 98.9        44                0    / 0             |
| 0                         | 0000:41:00.0  | 0           0    / 0          54672/ 65536         |
GPU Info:
Nvidia 5090 laptop
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.142                Driver Version: 580.142        CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 5090 ...    Off |   00000000:02:00.0  On |                  N/A |
| N/A   47C    P8             12W /   95W |     766MiB /  24463MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+


## nano-vllm-ascend
python examples/bench_compare.py   --backend nano   --model-path /home/player/models/Qwen3/Qwen/Qwen3-0___6B/   --num-prompts 32   --input-len 256   --output-len 2048   --max-num-seqs 32   --max-model-len 4096   --ignore-eos   --output-json bench_outputs/nano_32x256_2048.json --enable-decode-graph
backend=nano num_prompts=32 input_len=[256,256] output_len=[2048,2048] max_num_seqs=32
The module name  (originally ) is not a valid Python identifier. Please rename the original module to avoid import issues.
{
  "backend": "nano",
  "model_path": "/home/player/models/Qwen3/Qwen/Qwen3-0___6B",
  "num_prompts": 32,
  "max_model_len": 4096,
  "max_num_seqs": 32,
  "max_num_batched_tokens": null,
  "block_size": 128,
  "seed": 0,
  "temperature": 0.0,
  "top_p": 1.0,
  "top_k": 0,
  "ignore_eos": true,
  "warmup_iters": 1,
  "profile_iters": 1,
  "elapsed_s": 50.56600925209932,
  "total_input_tokens": 8192,
  "requested_output_tokens": 65536,
  "total_output_tokens": 65536,
  "total_tokens": 73728,
  "request_throughput_req_s": 0.6328361773708981,
  "output_throughput_tok_s": 1296.0484912555994,
  "total_throughput_tok_s": 1458.0545526625492,
  "avg_input_len": 256,
  "avg_requested_output_len": 2048,
  "avg_actual_output_len": 2048,
  "actual_output_len_p50": 2048.0,
  "actual_output_len_p90": 2048.0,
  "actual_output_len_min": 2048,
  "actual_output_len_max": 2048,
  "dataset": {
    "vocab_size": 151936,
    "token_id_high_used": 10000,
    "input_lens_min": 256,
    "input_lens_max": 256,
    "output_lens_min": 2048,
    "output_lens_max": 2048
  },
  "memory": {
    "cuda_peak_allocated_gb": null,
    "cuda_peak_reserved_gb": null,
    "npu_peak_allocated_gb": 50.28274393081665,
    "npu_peak_reserved_gb": 50.611328125
  }
}

## vllm-ascend
root@localhost:/home/player/models/Qwen3/nanovllm_ascend_new# python examples/bench_compare.py   --backend vllm   --model-path /home/player/models/Qwen3/Qwen/Qwen3-0___6B/   --num-prompts 32   --input-len 256   --output-len 2048   --max-num-seqs 32   --max-model-len 4096   --dtype bfloat16   --ignore-eos   --output-json bench_outputs/vllm_ascend_32x256_2048.json
backend=vllm num_prompts=32 input_len=[256,256] output_len=[2048,2048] max_num_seqs=32
INFO 06-10 03:10:40 [__init__.py:44] Available plugins for group vllm.platform_plugins:
INFO 06-10 03:10:40 [__init__.py:46] - ascend -> vllm_ascend:register
INFO 06-10 03:10:40 [__init__.py:49] All plugins in this group will be loaded. Set `VLLM_PLUGINS` to control which plugins to load.
INFO 06-10 03:10:40 [__init__.py:239] Platform plugin ascend is activated
INFO 06-10 03:10:46 [__init__.py:110] Registered model loader `<class 'vllm_ascend.model_loader.netloader.netloader.ModelNetLoaderElastic'>` with load format `netloader`
INFO 06-10 03:10:46 [__init__.py:110] Registered model loader `<class 'vllm_ascend.model_loader.rfork.rfork_loader.RForkModelLoader'>` with load format `rfork`
INFO 06-10 03:10:46 [utils.py:233] non-default args: {'tokenizer': '/home/player/models/Qwen3/Qwen/Qwen3-0___6B/', 'trust_remote_code': True, 'dtype': 'bfloat16', 'max_model_len': 4096, 'block_size': 128, 'gpu_memory_utilization': 0.8, 'max_num_seqs': 32, 'disable_log_stats': True, 'model': '/home/player/models/Qwen3/Qwen/Qwen3-0___6B/'}
INFO 06-10 03:10:46 [model.py:533] Resolved architecture: Qwen3ForCausalLM
INFO 06-10 03:10:46 [model.py:1582] Using max model len 4096
INFO 06-10 03:10:46 [scheduler.py:231] Chunked prefill is enabled with max_num_batched_tokens=8192.
INFO 06-10 03:10:46 [vllm.py:754] Asynchronous scheduling is enabled.
WARNING 06-10 03:10:46 [platform.py:749] Parameter '--disable-cascade-attn' is a GPU-specific feature. Resetting to False for Ascend.
WARNING 06-10 03:10:46 [platform.py:838] Ignored parameter 'disable_flashinfer_prefill'. This is a GPU-specific feature not supported on Ascend. Resetting to False.
INFO 06-10 03:10:46 [ascend_config.py:425] Dynamic EPLB is False
INFO 06-10 03:10:46 [ascend_config.py:426] The number of redundant experts is 0
INFO 06-10 03:10:46 [platform.py:354] PIECEWISE compilation enabled on NPU. use_inductor not supported - using only ACL Graph mode
INFO 06-10 03:10:46 [utils.py:549] Calculated maximum supported batch sizes for ACL graph: 62
WARNING 06-10 03:10:46 [utils.py:550] Currently, communication is performed using FFTS+ method, which reduces the number of available streams and, as a result, limits the range of runtime shapes that can be handled. To both improve communication performance and increase the number of supported shapes, set HCCL_OP_EXPANSION_MODE=AIV.
INFO 06-10 03:10:46 [utils.py:582] No adjustment needed for ACL graph batch sizes: Qwen3ForCausalLM model (layers: 28) with 7 sizes
INFO 06-10 03:10:46 [platform.py:502] Set PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
INFO 06-10 03:10:46 [compilation.py:289] Enabled custom fusions: norm_quant, act_quant
(EngineCore pid=18225) INFO 06-10 03:10:47 [core.py:103] Initializing a V1 LLM engine (v0.18.0) with config: model='/home/player/models/Qwen3/Qwen/Qwen3-0___6B/', speculative_config=None, tokenizer='/home/player/models/Qwen3/Qwen/Qwen3-0___6B/', skip_tokenizer_init=False, tokenizer_mode=auto, revision=None, tokenizer_revision=None, trust_remote_code=True, dtype=torch.bfloat16, max_seq_len=4096, download_dir=None, load_format=auto, tensor_parallel_size=1, pipeline_parallel_size=1, data_parallel_size=1, decode_context_parallel_size=1, dcp_comm_backend=ag_rs, disable_custom_all_reduce=True, quantization=None, enforce_eager=False, enable_return_routed_experts=False, kv_cache_dtype=auto, device_config=npu, structured_outputs_config=StructuredOutputsConfig(backend='auto', disable_any_whitespace=False, disable_additional_properties=False, reasoning_parser='', reasoning_parser_plugin='', enable_in_reasoning=False), observability_config=ObservabilityConfig(show_hidden_metrics_for_version=None, otlp_traces_endpoint=None, collect_detailed_traces=None, kv_cache_metrics=False, kv_cache_metrics_sample=0.01, cudagraph_metrics=False, enable_layerwise_nvtx_tracing=False, enable_mfu_metrics=False, enable_mm_processor_stats=False, enable_logging_iteration_details=False), seed=0, served_model_name=/home/player/models/Qwen3/Qwen/Qwen3-0___6B/, enable_prefix_caching=True, enable_chunked_prefill=True, pooler_config=None, compilation_config={'mode': <CompilationMode.VLLM_COMPILE: 3>, 'debug_dump_path': None, 'cache_dir': '', 'compile_cache_save_format': 'binary', 'backend': 'vllm_ascend.compilation.compiler_interface.AscendCompiler', 'custom_ops': ['all'], 'splitting_ops': ['vllm::unified_attention', 'vllm::unified_attention_with_output', 'vllm::unified_mla_attention', 'vllm::unified_mla_attention_with_output', 'vllm::mamba_mixer2', 'vllm::mamba_mixer', 'vllm::short_conv', 'vllm::linear_attention', 'vllm::plamo2_mamba_mixer', 'vllm::gdn_attention_core', 'vllm::olmo_hybrid_gdn_full_forward', 'vllm::kda_attention', 'vllm::sparse_attn_indexer', 'vllm::rocm_aiter_sparse_attn_indexer', 'vllm::unified_kv_cache_update', 'vllm::unified_mla_kv_cache_update', 'vllm::mla_forward'], 'compile_mm_encoder': False, 'compile_sizes': [], 'compile_ranges_endpoints': [8192], 'inductor_compile_config': {'enable_auto_functionalized_v2': False, 'combo_kernels': True, 'benchmark_combo_kernel': True}, 'inductor_passes': {}, 'cudagraph_mode': <CUDAGraphMode.PIECEWISE: 1>, 'cudagraph_num_of_warmups': 1, 'cudagraph_capture_sizes': [1, 2, 4, 8, 16, 24, 32], 'cudagraph_copy_inputs': False, 'cudagraph_specialize_lora': True, 'use_inductor_graph_partition': False, 'pass_config': {'fuse_norm_quant': True, 'fuse_act_quant': True, 'fuse_attn_quant': False, 'enable_sp': False, 'fuse_gemm_comms': False, 'fuse_allreduce_rms': False}, 'max_cudagraph_capture_size': 32, 'dynamic_shapes_config': {'type': <DynamicShapesType.BACKED: 'backed'>, 'evaluate_guards': False, 'assume_32_bit_indexing': False}, 'local_cache_dir': None, 'fast_moe_cold_start': True, 'static_all_moe_layers': []}
(EngineCore pid=18225) WARNING 06-10 03:10:48 [warnings.py:110] /vllm-workspace/vllm-ascend/vllm_ascend/patch/worker/patch_weight_utils.py:80: DeprecationWarning: pkg_resources is deprecated as an API. See https://setuptools.pypa.io/en/latest/pkg_resources.html
(EngineCore pid=18225) INFO 06-10 03:10:52 [parallel_state.py:1395] world_size=1 rank=0 local_rank=0 distributed_init_method=tcp://192.168.10.4:45093 backend=hccl
[Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
[Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
[Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
[Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
[Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
[Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
(EngineCore pid=18225) INFO 06-10 03:10:52 [parallel_state.py:1717] rank 0 in world size 1 is assigned as DP rank 0, PP rank 0, PCP rank 0, TP rank 0, EP rank N/A, EPLB rank N/A
[Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
(EngineCore pid=18225) INFO 06-10 03:10:54 [cpu_binding.py:320] [cpu_bind_mode] mode=topo_affinity rank=0 visible_npus=[0]
(EngineCore pid=18225) INFO 06-10 03:10:54 [cpu_binding.py:367] The CPU allocation plan is as follows:
(EngineCore pid=18225) INFO 06-10 03:10:54 [cpu_binding.py:372] NPU0: main=[50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93]  acl=[94]  release=[[95]]
(EngineCore pid=18225) INFO 06-10 03:10:54 [cpu_binding.py:394] [migrate] NPU:0 -> NUMA [2]
(EngineCore pid=18225) INFO 06-10 03:10:54 [model_runner_v1.py:2562] Starting to load model /home/player/models/Qwen3/Qwen/Qwen3-0___6B/...
(EngineCore pid=18225) INFO 06-10 03:10:55 [compilation.py:942] Using OOT custom backend for compilation.
(EngineCore pid=18225) INFO 06-10 03:10:55 [compilation.py:942] Using OOT custom backend for compilation.
Loading safetensors checkpoint shards:   0% Completed | 0/1 [00:00<?, ?it/s]
Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  4.74it/s]
Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  4.73it/s]
(EngineCore pid=18225)
(EngineCore pid=18225) INFO 06-10 03:10:55 [default_loader.py:384] Loading weights took 0.26 seconds
(EngineCore pid=18225) INFO 06-10 03:10:56 [model_runner_v1.py:2589] Loading model weights took 1.1397 GB
(EngineCore pid=18225) INFO 06-10 03:11:04 [backends.py:988] Using cache directory: /root/.cache/vllm/torch_compile_cache/751734dbef/rank_0_0/backbone for vLLM's torch.compile
(EngineCore pid=18225) INFO 06-10 03:11:04 [backends.py:1048] Dynamo bytecode transform time: 6.84 s
(EngineCore pid=18225) INFO 06-10 03:11:19 [backends.py:387] Compiling a graph for compile range (1, 8192) takes 13.29 s
(EngineCore pid=18225) INFO 06-10 03:11:22 [monitor.py:48] torch.compile and initial profiling/warmup run together took 24.70 s in total
(EngineCore pid=18225) INFO 06-10 03:11:23 [worker.py:357] Available KV cache memory: 47.39 GiB
(EngineCore pid=18225) INFO 06-10 03:11:23 [kv_cache_utils.py:1316] GPU KV cache size: 443,648 tokens
(EngineCore pid=18225) INFO 06-10 03:11:23 [kv_cache_utils.py:1321] Maximum concurrency for 4,096 tokens per request: 108.31x
Capturing CUDA graphs (mixed prefill-decode, PIECEWISE): 100%|█████████████████████████████████████████████████████████████| 7/7 [00:00<00:00,  8.03it/s]
(EngineCore pid=18225) INFO 06-10 03:11:25 [gpu_model_runner.py:5746] Graph capturing finished in 2 secs, took 0.03 GiB
(EngineCore pid=18225) INFO 06-10 03:11:25 [core.py:281] init engine (profile, create kv cache, warmup model) took 28.67 seconds
(EngineCore pid=18225) INFO 06-10 03:11:26 [platform.py:354] PIECEWISE compilation enabled on NPU. use_inductor not supported - using only ACL Graph mode
(EngineCore pid=18225) INFO 06-10 03:11:26 [utils.py:549] Calculated maximum supported batch sizes for ACL graph: 62
(EngineCore pid=18225) WARNING 06-10 03:11:26 [utils.py:550] Currently, communication is performed using FFTS+ method, which reduces the number of available streams and, as a result, limits the range of runtime shapes that can be handled. To both improve communication performance and increase the number of supported shapes, set HCCL_OP_EXPANSION_MODE=AIV.
(EngineCore pid=18225) INFO 06-10 03:11:26 [utils.py:582] No adjustment needed for ACL graph batch sizes: Qwen3ForCausalLM model (layers: 28) with 7 sizes
(EngineCore pid=18225) INFO 06-10 03:11:26 [platform.py:502] Set PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
INFO 06-10 03:11:26 [llm.py:391] Supported tasks: ['generate']
(EngineCore pid=18225) INFO 06-10 03:11:26 [acl_graph.py:192] Replaying aclgraph
{
  "backend": "vllm",
  "model_path": "/home/player/models/Qwen3/Qwen/Qwen3-0___6B",
  "num_prompts": 32,
  "max_model_len": 4096,
  "max_num_seqs": 32,
  "max_num_batched_tokens": null,
  "block_size": 128,
  "seed": 0,
  "temperature": 0.0,
  "top_p": 1.0,
  "top_k": 0,
  "ignore_eos": true,
  "warmup_iters": 1,
  "profile_iters": 1,
  "elapsed_s": 52.203962800092995,
  "total_input_tokens": 8192,
  "requested_output_tokens": 65536,
  "total_output_tokens": 65536,
  "total_tokens": 73728,
  "request_throughput_req_s": 0.6129802850894491,
  "output_throughput_tok_s": 1255.3836238631918,
  "total_throughput_tok_s": 1412.3065768460908,
  "avg_input_len": 256,
  "avg_requested_output_len": 2048,
  "avg_actual_output_len": 2048,
  "actual_output_len_p50": 2048.0,
  "actual_output_len_p90": 2048.0,
  "actual_output_len_min": 2048,
  "actual_output_len_max": 2048,
  "dataset": {
    "vocab_size": 151936,
    "token_id_high_used": 10000,
    "input_lens_min": 256,
    "input_lens_max": 256,
    "output_lens_min": 2048,
    "output_lens_max": 2048
  },
  "memory": {
    "cuda_peak_allocated_gb": null,
    "cuda_peak_reserved_gb": null,
    "npu_peak_allocated_gb": 0.0,
    "npu_peak_reserved_gb": 0.0
  }
}
saved: bench_outputs/vllm_ascend_32x256_2048.json

## vllm
vllm) test@test-Legion-Y9000P-IAX10H:~/py/vllm$ python bench_compare.py --model-path /home/test/huggingface/Qwen3-0.6B/qwen/Qwen3-0.6B/ --input-len 256 --output-len 2048 --max-num-seqs 32 --num-prompts 32 --backend vllm --output-json vllm_32*256_2048
backend=vllm num_prompts=32 input_len=[256,256] output_len=[2048,2048] max_num_seqs=32
INFO 06-10 14:41:28 [utils.py:233] non-default args: {'tokenizer': '/home/test/huggingface/Qwen3-0.6B/qwen/Qwen3-0.6B/', 'trust_remote_code': True, 'max_model_len': 4096, 'block_size': 128, 'gpu_memory_utilization': 0.8, 'max_num_seqs': 32, 'disable_log_stats': True, 'model': '/home/test/huggingface/Qwen3-0.6B/qwen/Qwen3-0.6B/'}
INFO 06-10 14:41:28 [model.py:533] Resolved architecture: Qwen3ForCausalLM
INFO 06-10 14:41:28 [model.py:1582] Using max model len 4096
INFO 06-10 14:41:28 [scheduler.py:231] Chunked prefill is enabled with max_num_batched_tokens=8192.
INFO 06-10 14:41:28 [vllm.py:754] Asynchronous scheduling is enabled.
(EngineCore pid=1730732) INFO 06-10 14:41:29 [core.py:103] Initializing a V1 LLM engine (v0.18.1.dev0+gbcf2be961.d20260324) with config: model='/home/test/huggingface/Qwen3-0.6B/qwen/Qwen3-0.6B/', speculative_config=None, tokenizer='/home/test/huggingface/Qwen3-0.6B/qwen/Qwen3-0.6B/', skip_tokenizer_init=False, tokenizer_mode=auto, revision=None, tokenizer_revision=None, trust_remote_code=True, dtype=torch.bfloat16, max_seq_len=4096, download_dir=None, load_format=auto, tensor_parallel_size=1, pipeline_parallel_size=1, data_parallel_size=1, decode_context_parallel_size=1, dcp_comm_backend=ag_rs, disable_custom_all_reduce=False, quantization=None, enforce_eager=False, enable_return_routed_experts=False, kv_cache_dtype=auto, device_config=cuda, structured_outputs_config=StructuredOutputsConfig(backend='auto', disable_any_whitespace=False, disable_additional_properties=False, reasoning_parser='', reasoning_parser_plugin='', enable_in_reasoning=False), observability_config=ObservabilityConfig(show_hidden_metrics_for_version=None, otlp_traces_endpoint=None, collect_detailed_traces=None, kv_cache_metrics=False, kv_cache_metrics_sample=0.01, cudagraph_metrics=False, enable_layerwise_nvtx_tracing=False, enable_mfu_metrics=False, enable_mm_processor_stats=False, enable_logging_iteration_details=False), seed=0, served_model_name=/home/test/huggingface/Qwen3-0.6B/qwen/Qwen3-0.6B/, enable_prefix_caching=True, enable_chunked_prefill=True, pooler_config=None, compilation_config={'mode': <CompilationMode.VLLM_COMPILE: 3>, 'debug_dump_path': None, 'cache_dir': '', 'compile_cache_save_format': 'binary', 'backend': 'inductor', 'custom_ops': ['none'], 'splitting_ops': ['vllm::unified_attention', 'vllm::unified_attention_with_output', 'vllm::unified_mla_attention', 'vllm::unified_mla_attention_with_output', 'vllm::mamba_mixer2', 'vllm::mamba_mixer', 'vllm::short_conv', 'vllm::linear_attention', 'vllm::plamo2_mamba_mixer', 'vllm::gdn_attention_core', 'vllm::olmo_hybrid_gdn_full_forward', 'vllm::kda_attention', 'vllm::sparse_attn_indexer', 'vllm::rocm_aiter_sparse_attn_indexer', 'vllm::unified_kv_cache_update', 'vllm::unified_mla_kv_cache_update'], 'compile_mm_encoder': False, 'compile_sizes': [], 'compile_ranges_endpoints': [8192], 'inductor_compile_config': {'enable_auto_functionalized_v2': False, 'combo_kernels': True, 'benchmark_combo_kernel': True}, 'inductor_passes': {}, 'cudagraph_mode': <CUDAGraphMode.FULL_AND_PIECEWISE: (2, 1)>, 'cudagraph_num_of_warmups': 1, 'cudagraph_capture_sizes': [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64], 'cudagraph_copy_inputs': False, 'cudagraph_specialize_lora': True, 'use_inductor_graph_partition': False, 'pass_config': {'fuse_norm_quant': False, 'fuse_act_quant': False, 'fuse_attn_quant': False, 'enable_sp': False, 'fuse_gemm_comms': False, 'fuse_allreduce_rms': False}, 'max_cudagraph_capture_size': 64, 'dynamic_shapes_config': {'type': <DynamicShapesType.BACKED: 'backed'>, 'evaluate_guards': False, 'assume_32_bit_indexing': False}, 'local_cache_dir': None, 'fast_moe_cold_start': True, 'static_all_moe_layers': []}
(EngineCore pid=1730732) INFO 06-10 14:41:29 [parallel_state.py:1395] world_size=1 rank=0 local_rank=0 distributed_init_method=tcp://192.168.43.114:49261 backend=nccl
(EngineCore pid=1730732) INFO 06-10 14:41:29 [parallel_state.py:1717] rank 0 in world size 1 is assigned as DP rank 0, PP rank 0, PCP rank 0, TP rank 0, EP rank N/A, EPLB rank N/A
(EngineCore pid=1730732) INFO 06-10 14:41:30 [gpu_model_runner.py:4481] Starting to load model /home/test/huggingface/Qwen3-0.6B/qwen/Qwen3-0.6B/...
(EngineCore pid=1730732) INFO 06-10 14:41:30 [cuda.py:317] Using FLASH_ATTN attention backend out of potential backends: ['FLASH_ATTN', 'FLASHINFER', 'TRITON_ATTN', 'FLEX_ATTENTION'].
(EngineCore pid=1730732) INFO 06-10 14:41:30 [flash_attn.py:598] Using FlashAttention version 2
Loading safetensors checkpoint shards:   0% Completed | 0/1 [00:00<?, ?it/s]
Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  5.75it/s]
Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  5.75it/s]
(EngineCore pid=1730732) 
(EngineCore pid=1730732) INFO 06-10 14:41:30 [default_loader.py:384] Loading weights took 0.18 seconds
(EngineCore pid=1730732) INFO 06-10 14:41:31 [gpu_model_runner.py:4566] Model loading took 1.12 GiB memory and 0.517589 seconds
(EngineCore pid=1730732) INFO 06-10 14:41:34 [backends.py:988] Using cache directory: /home/test/.cache/vllm/torch_compile_cache/d981fb7afa/rank_0_0/backbone for vLLM's torch.compile
(EngineCore pid=1730732) INFO 06-10 14:41:34 [backends.py:1048] Dynamo bytecode transform time: 2.91 s
(EngineCore pid=1730732) INFO 06-10 14:41:35 [backends.py:284] Directly load the compiled graph(s) for compile range (1, 8192) from the cache, took 0.649 s
(EngineCore pid=1730732) INFO 06-10 14:41:36 [monitor.py:48] torch.compile and initial profiling/warmup run together took 4.57 s in total
(EngineCore pid=1730732) INFO 06-10 14:41:40 [kv_cache_utils.py:826] Overriding num_gpu_blocks=0 with num_gpu_blocks_override=64
(EngineCore pid=1730732) INFO 06-10 14:41:40 [gpu_model_runner.py:5607] Profiling CUDA graph memory: PIECEWISE=11 (largest=64), FULL=7 (largest=32)
(EngineCore pid=1730732) INFO 06-10 14:41:40 [gpu_model_runner.py:5686] Estimated CUDA graph memory: 0.08 GiB total
(EngineCore pid=1730732) INFO 06-10 14:41:41 [gpu_worker.py:456] Available KV cache memory: 17.23 GiB
(EngineCore pid=1730732) INFO 06-10 14:41:41 [gpu_worker.py:490] In v0.19, CUDA graph memory profiling will be enabled by default (VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1), which more accurately accounts for CUDA graph memory during KV cache allocation. To try it now, set VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 and increase --gpu-memory-utilization from 0.8000 to 0.8033 to maintain the same effective KV cache size.
(EngineCore pid=1730732) INFO 06-10 14:41:41 [kv_cache_utils.py:1316] GPU KV cache size: 161,280 tokens
(EngineCore pid=1730732) INFO 06-10 14:41:41 [kv_cache_utils.py:1321] Maximum concurrency for 4,096 tokens per request: 39.38x
(EngineCore pid=1730732) 2026-06-10 14:41:41,084 - INFO - autotuner.py:262 - flashinfer.jit: [Autotuner]: Autotuning process starts ...
(EngineCore pid=1730732) 2026-06-10 14:41:41,091 - INFO - autotuner.py:268 - flashinfer.jit: [Autotuner]: Autotuning process ends
Capturing CUDA graphs (mixed prefill-decode, PIECEWISE): 100%|█| 11/11 [00:00<00
Capturing CUDA graphs (decode, FULL): 100%|███████| 7/7 [00:00<00:00, 68.33it/s]
(EngineCore pid=1730732) INFO 06-10 14:41:41 [gpu_model_runner.py:5746] Graph capturing finished in 1 secs, took 0.04 GiB
(EngineCore pid=1730732) INFO 06-10 14:41:41 [gpu_worker.py:617] CUDA graph pool memory: 0.04 GiB (actual), 0.08 GiB (estimated), difference: 0.04 GiB (85.7%).
(EngineCore pid=1730732) INFO 06-10 14:41:41 [core.py:281] init engine (profile, create kv cache, warmup model) took 10.54 seconds
INFO 06-10 14:41:42 [llm.py:391] Supported tasks: ['generate']
{
  "backend": "vllm",
  "model_path": "/home/test/huggingface/Qwen3-0.6B/qwen/Qwen3-0.6B",
  "num_prompts": 32,
  "max_model_len": 4096,
  "max_num_seqs": 32,
  "max_num_batched_tokens": null,
  "block_size": 128,
  "seed": 0,
  "temperature": 0.0,
  "top_p": 1.0,
  "top_k": 0,
  "ignore_eos": false,
  "warmup_iters": 1,
  "profile_iters": 1,
  "elapsed_s": 28.961387279909104,
  "total_input_tokens": 8192,
  "requested_output_tokens": 65536,
  "total_output_tokens": 65536,
  "total_tokens": 73728,
  "request_throughput_req_s": 1.1049194463898773,
  "output_throughput_tok_s": 2262.8750262064686,
  "total_throughput_tok_s": 2545.734404482277,
  "avg_input_len": 256,
  "avg_requested_output_len": 2048,
  "avg_actual_output_len": 2048,
  "actual_output_len_p50": 2048.0,
  "actual_output_len_p90": 2048.0,
  "actual_output_len_min": 2048,
  "actual_output_len_max": 2048,
  "dataset": {
    "vocab_size": 151936,
    "token_id_high_used": 10000,
    "input_lens_min": 256,
    "input_lens_max": 256,
    "output_lens_min": 2048,
    "output_lens_max": 2048
  },
  "memory": {
    "cuda_peak_allocated_gb": 0.0,
    "cuda_peak_reserved_gb": 0.0,
    "npu_peak_allocated_gb": null,
    "npu_peak_reserved_gb": null
  }
}
saved: vllm_32*256_2048.json
(EngineCore pid=1730732) INFO 06-10 14:42:39 [core.py:1201] Shutdown initiated (timeout=0)
(EngineCore pid=1730732) INFO 06-10 14:42:39 [core.py:1224] Shutdown complete

## nano-vllm
{
  "backend": "nanovllm",
  "model_path": "/home/test/huggingface/Qwen3-0.6B/qwen/Qwen3-0.6B",
  "num_prompts": 32,
  "max_model_len": 4096,
  "max_num_seqs": 32,
  "max_num_batched_tokens": null,
  "block_size": 128,
  "seed": 0,
  "temperature": 0.001,
  "top_p": 1.0,
  "top_k": 0,
  "ignore_eos": true,
  "warmup_iters": 1,
  "profile_iters": 1,
  "elapsed_s": 29.029644688940607,
  "total_input_tokens": 8192,
  "requested_output_tokens": 65536,
  "total_output_tokens": 65536,
  "total_tokens": 73728,
  "request_throughput_req_s": 1.1023214490872155,
  "output_throughput_tok_s": 2257.5543277306174,
  "total_throughput_tok_s": 2539.7486186969445,
  "avg_input_len": 256,
  "avg_requested_output_len": 2048,
  "avg_actual_output_len": 2048,
  "actual_output_len_p50": 2048.0,
  "actual_output_len_p90": 2048.0,
  "actual_output_len_min": 2048,
  "actual_output_len_max": 2048,
  "dataset": {
    "vocab_size": 151936,
    "token_id_high_used": 10000,
    "input_lens_min": 256,
    "input_lens_max": 256,
    "output_lens_min": 2048,
    "output_lens_max": 2048
  },
  "memory": {
    "cuda_peak_allocated_gb": 17.16930055618286,
    "cuda_peak_reserved_gb": 17.314453125,
    "npu_peak_allocated_gb": null,
    "npu_peak_reserved_gb": null
  }
}