# configs/llama7b.py
from mmengine.config import read_base

with read_base():
    # Read the required dataset configurations directly from the preset dataset configurations
    from .datasets.piqa.piqa_ppl import piqa_datasets
    from .datasets.siqa.siqa_gen import siqa_datasets

# Concatenate the datasets to be evaluated into the datasets field
datasets = [*piqa_datasets, *siqa_datasets]

# Evaluate models supported by HuggingFace's `AutoModelForCausalLM` using `HuggingFaceCausalLM`
from opencompass.models import HuggingFaceCausalLM

models = [
    dict(
        type=HuggingFaceCausalLM,
        # Initialization parameters for `HuggingFaceCausalLM`
        path='huggyllama/llama-7b',
        tokenizer_path='huggyllama/llama-7b',
        tokenizer_kwargs=dict(padding_side='left', truncation_side='left'),
        max_seq_len=2048,
        # Common parameters for all models, not specific to HuggingFaceCausalLM's initialization parameters
        abbr='llama-7b',            # Model abbreviation for result display
        max_out_len=100,            # Maximum number of generated tokens
        batch_size=16,
        run_cfg=dict(num_gpus=1),   # Run configuration for specifying resource requirements
    )
]