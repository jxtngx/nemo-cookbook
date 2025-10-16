import os
from pathlib import Path

from transformers import AutoModelForCausalLM

# simple resolve to set cache dir
filepath = Path(__file__)
rootpath = filepath.parents[1]
cache_dir = os.path.join(rootpath, ".lab-models", "hugging-face")

# before download, .from_pretrained will check the cache dir
# for existing downloads and avoid overwriting an existing model

# Load model directly
AutoModelForCausalLM.from_pretrained(
    "apple/FastVLM-1.5B",
    trust_remote_code=True,
    torch_dtype="auto",
    cache_dir=cache_dir,
)
