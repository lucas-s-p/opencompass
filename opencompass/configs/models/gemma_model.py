# model_cfg.py
from opencompass.models import HuggingFaceCausalLM

models = [
    dict(
        type=HuggingFaceCausalLM,
        path='huggyllama/llama-7b',
        model_kwargs=dict(device_map='auto'),
        tokenizer_path='huggyllama/llama-7b',
        tokenizer_kwargs=dict(padding_side='left', truncation_side='left'),
        max_seq_len=2048,
        max_out_len=50,
        run_cfg=dict(num_gpus=8, num_procs=1),
    )
]