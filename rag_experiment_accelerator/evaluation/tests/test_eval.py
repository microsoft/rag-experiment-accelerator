from rag_experiment_accelerator.evaluation.eval import bleu


def test_bleu():
    predictions = [
        "Transformers Transformers are fast plus efficient",
        "Good Morning",
        "I am waiting for new Transformers",
    ]
    references = [
        [
            "HuggingFace Transformers are quick, efficient and awesome",
            "Transformers are awesome because they are fast to execute",
        ],
        ["Good Morning Transformers", "Morning Transformers"],
        [
            "People are eagerly waiting for new Transformer models",
            "People are very excited about new Transformers",
        ],
    ]
    score = bleu(predictions, references)
    assert round(score) == 50
