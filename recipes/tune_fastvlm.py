# see https://docs.nvidia.com/nemo-framework/user-guide/latest/vlms/neva.html
# code adapted from above for apple fastvlm variants

import os
from dataclasses import dataclass, field
from typing import Union, Optional
from pathlib import Path

import torch
import nemo.lightning as nl
import nemo_run as run

from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.transformer.transformer_config import TransformerConfig
from nemo.collections import llm, vlm
from nemo.collections.llm import import_ckpt, LlamaConfig
from nemo.collections.llm.recipes.finetune_default import nemo_resume
from nemo.collections.llm.recipes.precision.mixed_precision import bf16_mixed
from nemo.collections.llm.recipes.optim.adam import (
    distributed_fused_adam_with_cosine_annealing,
)
from nemo.collections.llm.recipes.log.default import wandb_logger
from nemo.collections.vlm.vision.base import (
    HFCLIPVisionConfig,
    MultimodalProjectorConfig,
)
from nemo.lightning.pytorch.callbacks.megatron_comm_overlap import (
    MegatronCommOverlapCallback,
)
from nemo.utils.exp_manager import TimingCallback

# simple resolve to set cache dir
filepath = Path(__file__)
rootpath = filepath.parents[1]
cache_dir = os.path.join(rootpath, ".lab-models", "hugging-face")

# name for NeMo run recipe
NAME = "fastvlm_1B"
# Specify the Hugging Face model ID
HF_MODEL_ID = "apple/FastVLM-1.5B"


@dataclass
class FastVLMConfig(vlm.LlavaConfig):
    """FastVLM Config 1.5B"""

    from transformers import PretrainedConfig

    language_transformer_config: TransformerConfig = field(
        default_factory=lambda: LlamaConfig()
    )
    vision_transformer_config: Union[TransformerConfig, PretrainedConfig] = field(
        default_factory=lambda: HFCLIPVisionConfig(
            pretrained_model_name_or_path="openai/clip-vit-large-patch14-336"
        )
    )
    vision_projection_config: TransformerConfig = field(
        default_factory=lambda: MultimodalProjectorConfig(
            input_size=1024, hidden_size=4096, ffn_hidden_size=4096
        )
    )


@run.cli.factory(target=llm.finetune, name=NAME)
def finetune_recipe(
    dir: Optional[str] = None,
    name: str = "default",
    num_nodes: int = 1,
    num_gpus_per_node: int = 8,
    peft_scheme: Optional[str] = "none",
) -> run.Partial:
    """
    Create a fine-tuning recipe for Llava1.5 7B model.

    This function sets up a complete configuration for fine-tuning, including
    model, trainer, data, logging, optimization, and resumption settings.
    The recipe uses LoRA (Low-Rank Adaptation) for efficient fine-tuning, unless peft_scheme is set to None.

    Args:
        dir (Optional[str]): Directory for saving logs and checkpoints.
        name (str): Name of the fine-tuning run.
        num_nodes (int): Number of compute nodes to use.
        num_gpus_per_node (int): Number of GPUs per node.

    Returns:
        run.Partial: Partial configuration for fine-tuning.

    Examples:
        CLI usage:
            $ nemo llm finetune --factory llava15_7b

        Python API usage:
            >>> recipe = finetune_recipe(name="llava15_7b_finetune", num_nodes=1)
            >>> print(recipe)

    Note:
        This recipe uses the SQuAD dataset for fine-tuning. For more information
        on fine-tuning LLMs with NeMo, see the fine-tuning guide in the
        `examples/llm/finetune/` directory.
    """

    strategy = run.Config(
        nl.MegatronStrategy,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        encoder_pipeline_model_parallel_size=0,
        sequence_parallel=True,
        pipeline_dtype=torch.bfloat16,
        ddp=run.Config(
            DistributedDataParallelConfig,
            check_for_nan_in_grad=True,
            grad_reduce_in_fp32=True,
            overlap_grad_reduce=True,
            overlap_param_gather=True,
            average_in_collective=True,
        ),
    )

    trainer = run.Config(
        nl.Trainer,
        accelerator="gpu",
        accumulate_grad_batches=1,
        devices=num_gpus_per_node,
        limit_val_batches=10,
        log_every_n_steps=1,
        max_steps=5190,
        num_nodes=num_nodes,
        plugins=bf16_mixed(),
        strategy=strategy,
        val_check_interval=1000,
        callbacks=[
            run.Config(TimingCallback),
            run.Config(MegatronCommOverlapCallback, tp_comm_overlap=True),
        ],
    )

    recipe = run.Partial(
        llm.finetune,
        model=model(),
        trainer=trainer,
        data=run.Config(
            MockDataModule,
            seq_length=4096,
            global_batch_size=128,
            micro_batch_size=2,
            tokenizer=None,
            image_processor=None,
            num_workers=4,
        ),
        log=llm.default_log(dir=dir, name=name, wandb_logger=wandb_logger(name=name)),
        optim=distributed_fused_adam_with_cosine_annealing(
            max_lr=2.0e-05, min_lr=2.0e-07, warmup_steps=150
        ),
        resume=nemo_resume(HF_MODEL_ID),
    )

    if peft_scheme is None or peft_scheme.lower() == "none":
        recipe.trainer.strategy.tensor_model_parallel_size = 2
        recipe.optim.config.lr = 2e-05
    elif peft_scheme.lower() == "lora":
        recipe.peft = run.Config(
            vlm.LoRA,
            freeze_vision_model=False,
            target_modules=[
                "*.language_model.*.linear_qkv",
                "*.language_model.*.linear_q",
                "*.language_model.*.linear_kv",
                "*.language_model.*.linear_proj",
                "*.language_model.*.linear_fc1",
                "*.language_model.*.linear_fc2",
            ],
        )
        recipe.optim.config.lr = 1e-4
    else:
        raise ValueError(f"Unrecognized peft scheme: {peft_scheme}")

    return recipe


if __name__ == "__main__":

    # Import the model and convert to NeMo 2.0 format
    import_ckpt(
        model=vlm.LlavaModel(FastVLMConfig()),  # Model configuration
        source=f"hf://{HF_MODEL_ID}",  # Hugging Face model source
    )

    finetune = finetune_recipe(
        name="fastvlm_1b_finetune",
        dir=cache_dir,
        num_nodes=1,
        num_gpus_per_node=1,
        peft_scheme="lora",  # 'lora', 'none'
    )
