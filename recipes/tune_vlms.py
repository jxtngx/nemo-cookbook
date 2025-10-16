# see https://docs.nvidia.com/nemo-framework/user-guide/latest/vlms/neva.html
# code adapted from above for apple fastvlm variants

from dataclasses import dataclass, field
from typing import Union

from nemo.collections.llm import import_ckpt
from nemo.collections import vlm
from megatron.core.transformer.transformer_config import TransformerConfig


@dataclass
class FastVLMConfig(vlm.LlavaConfig):
    """Llava v1.5 Config 7B"""

    from transformers import PretrainedConfig

    language_transformer_config: TransformerConfig = field(
        default_factory=lambda: Llama2Config7B()
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


if __name__ == "__main__":
    # Specify the Hugging Face model ID
    hf_model_id = "apple/FastVLM-1.5B"

    # Import the model and convert to NeMo 2.0 format
    import_ckpt(
        model=vlm.LlavaModel(FastVLMConfig()),  # Model configuration
        source=f"hf://{hf_model_id}",  # Hugging Face model source
    )
