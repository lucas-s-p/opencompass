custom_reader_cfg = dict(
    input_columns=['input_text'],
    output_column='label',
    train_split='train',
    test_split='test'
)

custom_infer_cfg = dict(
    prompt_template=dict(
        type='cloze',
        template='Complete the following: {input_text}',
    ),
    retriever=None,
)

custom_eval_cfg = dict(
    evaluator=dict(
        type='accuracy',
    ),
)

custom_datasets = [
    dict(
        abbr="teste",
        type="JSONL",  # Formato do seu dataset
        path="./data/teste/teste.jsonl",
        reader_cfg=custom_reader_cfg,
        infer_cfg=custom_infer_cfg,
        eval_cfg=custom_eval_cfg,
    ),
]
