import json
from rag_experiment_accelerator.llm.prompt.prompt import (
    Prompt,
    StructuredPrompt,
    PromptTag,
)


def validate_hypothetical_questions(text: str) -> bool:
    json_output = json.loads(text)
    return isinstance(json_output, list) and all(
        isinstance(item, str) for item in json_output
    )


prompt_generate_hypothetical_answer = Prompt(
    system_message="prompt_generate_hypothetical_answer.txt",
    user_template="${text}",
)

prompt_generate_hypothetical_document = Prompt(
    system_message="prompt_generate_hypothetical_document.txt",
    user_template="${text}",
)

prompt_generate_hypothetical_questions = StructuredPrompt(
    system_message="prompt_generate_hypothetical_questions.txt",
    user_template="${text}",
    validator=validate_hypothetical_questions,
    tags={PromptTag.JSON, PromptTag.NonStrict},
)
