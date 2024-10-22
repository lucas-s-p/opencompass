# Configuração do leitor para o dataset
custom_reader_cfg = dict(
    input_columns=['question'],
    output_column='answer',
)

custom_infer_cfg = dict(
    prompt_template=dict(
        template='Consider the following question and provide a detailed answer: {question}',
    ),
)


# Configuração de avaliação
custom_eval_cfg = dict(
    evaluator=dict(
        # Defina o método de avaliação
        type='accuracy',
    ),
)

# Configuração do dataset
teste_datasets = [
    dict(
        abbr="teste",
        #type=JSONLDataset,  # Tipo do dataset
        path="/content/opencompass/opencompass/data_personalizado/teste/teste.jsonl",
        reader_cfg=custom_reader_cfg,
        infer_cfg=custom_infer_cfg,
        eval_cfg=custom_eval_cfg,
    ),
]
