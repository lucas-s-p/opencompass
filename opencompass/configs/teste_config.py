from configs.datasets.teste_dataset import custom_datasets
from configs.models.gemma_model import custom_models

# Combinações de dataset e modelo
configurations = [
    dict(
        datasets=custom_datasets,
        models=custom_models,
    ),
]
