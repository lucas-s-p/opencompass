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
        abbr='granite-3.0-2b-base',
        type='HuggingFaceCausalLM',
        path='ibm-granite/granite-3.0-2b-base',
        max_out_len=1024,
        batch_size=8,
        run_cfg=dict(num_gpus=1),
    )
]

work_dir = './output/granite-3.0-2b-base'
