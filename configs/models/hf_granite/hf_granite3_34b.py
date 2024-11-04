from opencompass.models import HuggingFaceBaseModel

models = [
    dict(
        type=HuggingFaceBaseModel,
        abbr='granite-34b-code-base-8k',
        path='ibm-granite/granite-34b-code-base-8k',
        max_out_len=2048,
        batch_size=8,
        run_cfg=dict(num_gpus=2),
    )
]
