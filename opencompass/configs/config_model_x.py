from mmengine.config import read_base
from opencompass.partitioners import NaivePartitioner
from opencompass.runners.local_api import LocalAPIRunner
from opencompass.tasks import OpenICLInferTask

with read_base():
    from opencompass.configs.datasets.gsm8k.gsm8k_gen import gsm8k_datasets

# Modifique o conjunto de dados conforme necessário
gsm8k_datasets[0]['abbr'] = 'demo_' + gsm8k_datasets[0]['abbr']
gsm8k_datasets[0]['reader_cfg']['test_range'] = '[0:64]'  # Limita o intervalo de teste

datasets = [
    *gsm8k_datasets,
]

models = [
    dict(
        abbr='hf_llama_3_2_1b',
        type='HuggingFaceCausalLM',
        path='meta-llama/Llama-3.2-1B',
        max_out_len=1024,
        batch_size=8,
        run_cfg=dict(num_gpus=1),
    )
]

work_dir = './output/hf_llama_3_2_1b'
