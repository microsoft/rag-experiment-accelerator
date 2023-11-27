from rag_experiment_accelerator.evaluation.eval import (
    bleu,
    answer_relevance,
    context_precision,
    get_result_from_model,
)

from langchain.prompts import ChatPromptTemplate
from rag_experiment_accelerator.llm.prompts import (
    answer_relevance_instruction,
)
from unittest.mock import patch
from langchain.schema.output import LLMResult, Generation
from rag_experiment_accelerator.evaluation.eval import (
    bleu,
    answer_relevance,
    context_precision,
    get_result_from_model,
)

from langchain.prompts import ChatPromptTemplate
from rag_experiment_accelerator.llm.prompts import (
    answer_relevance_instruction,
)
from unittest.mock import patch
from langchain.schema.output import LLMResult, Generation


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


@patch("rag_experiment_accelerator.evaluation.eval.get_result_from_model")
def test_answer_relevance(mock_get_result_from_model):
    mock_get_result_from_model.return_value = "What is the name of the largest bone in the human body?"

    question = "What is the name of the largest bone in the human body?"
    answer = (
        "The largest bone in the human body is the femur, also known as the thigh bone. It is about 19.4 inches (49.5 cm) long on average and can support up to 30 times the weight of a person’s body.",
    )
    score = answer_relevance(question, answer)
    assert round(score) == 1.0


@patch("rag_experiment_accelerator.evaluation.eval.get_result_from_model")
def test_context_precision(mock_get_result_from_model):
    mock_get_result_from_model.return_value = "Yes"
    question = "What is the name of the largest bone in the human body?"
    context = "According to the Cleveland Clinic, \"The femur is the largest and strongest bone in the human body. It can support as much as 30 times the weight of your body. The average adult male femur is 48 cm (18.9 in) in length and 2.34 cm (0.92 in) in diameter. The average weight among adult males in the United States is 196 lbs (872 N). Therefore, the adult male femur can support roughly 6,000 lbs of compressive force.\""
    
    score = context_precision(question, context)
    assert score == 1.0


@patch("rag_experiment_accelerator.evaluation.eval.AzureChatOpenAI")
@patch("rag_experiment_accelerator.evaluation.eval.Config")
def test_get_result_from_model(mock_config, mock_azure_chat_open_ai):
    generation = [[Generation(text="response")]]
    llm_result = LLMResult(generations=generation)
    mock_config().OpenAICredentials.OPENAI_API_TYPE = 'azure'

    mock_azure_chat_open_ai().generate.return_value = llm_result

    human_prompt = answer_relevance_instruction.format(answer="answer")
    prompt = [ChatPromptTemplate.from_messages([human_prompt]).format_messages()]

    result = get_result_from_model(prompt)
    assert result == "response"