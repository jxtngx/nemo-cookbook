# NeMo Cookbook

> [!IMPORTANT]
> NeMo Cookbook is under active development

NeMo Cookbook is an example template for Generative AI with [NVIDIA NeMo 2.0](https://www.nvidia.com/en-us/ai-data-science/products/nemo/).

[NVIDA NeMo](https://www.nvidia.com/en-us/ai-data-science/products/nemo/) is an accelerated end-to-end platform that is flexible and production ready. NeMo is comprised of several component frameworks which enable teams to build, customize, and deploy Generative AI solutions for:

- large language models
- vision language models
- video models
- speech models

> [!TIP]
> Get started with the quick start [tutorials](docs/tutorials/quickstarts) and [scripts](scripts/tutorials/nemo)

## Focus

NeMo Cookbook is inspired by [`NeMo tutorials`](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/starthere/tutorials.html) and focuses on using NeMo to tune generative models.

### Additional Concepts

- Code profiling
- Logging training and tuning runs with [Weights & Biases](https://wandb.ai/site)
- Model alignment with [NeMo Aligner](https://github.com/NVIDIA/NeMo-Aligner)
- Model output control with [NeMo Guardrails](https://github.com/NVIDIA/NeMo-Guardrails)
- Containerization with Docker

## Models

We will use NVIDIA and Meta models including, but not limited to:

- NVIDIA Llama variants, Mistral variants, Megatron distillations, and Minitron
- NVIDIA embedding, reranking, and retrieval models
- NVIDIA Cosmos tokenizers
- NeMo compatible Meta Llama variants

## System Requirements

- a CUDA compatible OS and device (GPU) with at least 48GB of VRAM (e.g. an L40S).
- CUDA 12.1
- Python 3.10.10
- Pytorch 2.2.1

> [!TIP]
> See https://nemo.theosis.ai/cookbook/hardware for more regarding VRAM requirements of particular models

## Core Dependencies

- [Megatron Core](https://github.com/NVIDIA/Megatron-LM)
- [APEX](https://github.com/NVIDIA/apex)
- [Transformer Engine](https://github.com/NVIDIA/TransformerEngine)
- [Flash Attention](https://github.com/Dao-AILab/flash-attention)

## User Account Requirements

- [NVIDIA Developer Program](https://developer.nvidia.com/developer-program)
- [NVIDIA NGC](https://catalog.ngc.nvidia.com/) for NeMo and TensorRT-LLM containers
- [build.nvidia.com](https://build.nvidia.com/) for API calls to NVIDIA hosted endpoints
- [Hugging Face Hub](https://huggingface.co/) for model weights and datasets

## Setup

### On Host (local, no container)

To prepare a development environment, please run the following in terminal:

```sh
bash install_requirements.sh
```

Doing so will install `nemo_lab` along with the `nemo_run`, `megatron_core 0.10.0rc0`, and the `nvidia/apex` PyTorch extension. 

> [!NOTE]
> `megatron_core 0.10.0rc0` is required for compatibility with NeMo 2.0

> [!NOTE]
> NVIDIA Apex is required for RoPE Scaling in NeMo 2.0.
> NVIDIA Apex is built with CUDA and C++ extensions for performance and full functionality.
> please be aware that the build process may take several minutes

### Docker

> [!IMPORTANT]
> running the images requires for the host machine to have access to NVIDIA GPUs

Two Docker images have been created for the quick start tutorials. One for pretraining, and one for finetuning.

To run pretraining, do the following in terminal:

```sh
docker pull jxtngx/nemo-lab:pretrain
docker run --rm --gpus 1 -it jxtngx/nemo-lab:pretrain
python pretrain_nemotron3_4b.py
```

To run finetuning, do the following in terminal:

```sh
docker pull jxtngx/nemo-lab:finetune
docker run --rm --gpus 1 -it jxtngx/nemo-lab:finetune
# WAIT FOR CONTAINER TO START 
huggingface-cli login
# ENTER HF KEY WHEN PROMPTED
python finetune_llama3_8b.py
```

> [!IMPORTANT]
> Finetuning requires a Hugging Face key and access to Llama 3 8B <br/>
> For keys, see: https://huggingface.co/docs/hub/en/security-tokens <br/>
> For Llama 3 8B access, see: https://huggingface.co/meta-llama/Meta-Llama-3-8B

## Resources

### Quickstart Images and Containers

<table>
    <tr>
        <th style="font-size: small;">Quickstart</th>
        <th style="font-size: small;">Docker</th>
        <th style="font-size: small;">NVIDIA</th>
    </tr>
    <tr>
        <td>Pretrain</td>
        <td><a target="_blank" href="https://hub.docker.com/repository/docker/jxtngx/nemo-lab/tags/train/sha256-e056f7988b875676e01a0ba2ae8fa4623ae28f0295a012f809e98158f413c2a2"><img src="https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=fff"/></a></td>
        <td><a target="_blank" href="https://console.brev.dev/launchable/deploy?launchableID=env-2qcfxLVNihSWulFbpOYJmyy8kBB"><img src="https://brev-assets.s3.us-west-1.amazonaws.com/nv-lb-dark.svg"/></a></td>   
    </tr>
    <tr>
        <td>Finetune</td>
        <td><a target="_blank" href="https://hub.docker.com/repository/docker/jxtngx/nemo-lab/tags/tune/sha256-29c27b9c41d633f48ed92aec40e33438b84fb3751e764df4655c9b2e697650d7"><img src="https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=fff"/></a></td>
        <td><a target="_blank" href="https://console.brev.dev/launchable/deploy?launchableID=env-2qcfslJNcRDEkHNGfhpW625D2SN"><img src="https://brev-assets.s3.us-west-1.amazonaws.com/nv-lb-dark.svg"/></a></td>
    </tr>
</table>

> [!IMPORTANT]
> regarding the NVIDIA Launchable: use the following command in terminal to run the: <br/>
> tuning: `python /workspace/finetune_llama3_8b.py` <br/>
> training: `python /workspace/pretrain_nemotron3_4b.py` <br/>

> [!IMPORTANT]
> regarding the NVIDIA Launchable: to avoid data storage costs, be certain to delete the demo instance once the demo is complete 

## NeMo References

- [NeMo documentation](https://docs.nvidia.com/nemo-framework/user-guide/latest/overview.html)
- [NeMo tutorials](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/starthere/tutorials.html)
- [NeMo Guardrails documentation](https://docs.nvidia.com/nemo/guardrails/index.html)
- [Deploy on a SLURM cluster](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemo-2.0/quickstart.html#execute-on-a-slurm-cluster)
- [Mixed Precision Training](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/features/mixed_precision.html)
- [CPU Offloading](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/features/optimizations/cpu_offloading.html)
- [Communication Overlap](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/features/optimizations/communication_overlap.html)
