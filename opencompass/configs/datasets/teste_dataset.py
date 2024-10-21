from opencompass.datasets import JSONLDataset

# Configuração do leitor para o seu dataset
custom_reader_cfg = dict(
    input_columns=['input_text'],
    output_column='label',
    train_split='train',
    test_split='test'
)

# Configuração de inferência
custom_infer_cfg = dict(
    prompt_template=dict(
        type='cloze',
        template='Complete the following: {input_text}',
    ),
    retriever=None,
)

# Configuração de avaliação
custom_eval_cfg = dict(
    evaluator=dict(
        type='accuracy',
    ),
)

# Configuração do seu dataset
custom_datasets = [
    dict(
        abbr="teste",
        type=JSONLDataset,  # Use o formato correto
        path="./data/teste/teste.jsonl",  # Caminho atualizado para o seu dataset
        reader_cfg=custom_reader_cfg,
        infer_cfg=custom_infer_cfg,
        eval_cfg=custom_eval_cfg,
    ),
]
