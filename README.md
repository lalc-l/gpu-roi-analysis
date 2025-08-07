# GPU ROI Analysis: Price Performance Metrics
Executive-Facing General Benchmarks H100 vs B200 (SHARP) to evaluate the price performance relationship overtime for a comprehensive set of model test cases. 

## Goals: 
Measure the performance of B200 vs H100 over a comprehensive set of open-source models


### Cloud Setup:

1. GPU:
    - spin up a B200 cluster 
    - start a H100 instance

2. File System:
    - Setup a file system in the same location as your cluster or instance for lower latency

3. SSH into your GPU
    - add your config file to local
    - include your github personal token
    - add and store your credentials in your github config
    - add user to account

4. Virtual Environment 
    - setup a virtual environment 

5. Version Control:
    - github: create a github repo
    - clone to both GPUs
    - separate logging and environments from different gpus but everything else you can continue to pull requests

6. Huggingface CLI:
    - run this in terminal: pip install huggingface_hub
    - then run hf auth login
    - generate huggingface token (fine grained w/ repo access), no need for git credential

### Benchmarking:

#### Versioning:
*Software stack:*

    PyTorch 2.7
    
    transformers, accelerate, deepspeed (for multi-GPU)

    tgi, vllm or tensorrt-llm for inference

    Profiling tools: torch.profiler, nvprof, nsys, nvidia-smi

*Cluster setup:*

- Run 1-GPU tests on a single H100 and single B200

- Run 8-GPU B200 inference with tensor parallelism + model parallelism

#### Metrics:

- Memory
    - MFU
    - Utilization
- Throughput
    - FLOPS
    - Speedup
- Accuracy
    - Perplexity: how accurate a prompt answer is for autoregressive or causal language models
- Financials
    - Cost / 1M tokens
        runtime (sec) * $/sec /tokens * 1000
    - Total cost
