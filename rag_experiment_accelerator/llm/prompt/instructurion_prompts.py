import json
from rag_experiment_accelerator.llm.prompt.prompt import (
    Prompt,
    StructuredPrompt,
    PromptTag,
)


def validate_instruction_keyword(text: str) -> bool:
    json_output = json.loads(text)
    return isinstance(json_output, list) and all(
        isinstance(item, str) for item in json_output
    )


def validate_instruction_entities(text: str) -> bool:
    json_output = json.loads(text)
    return isinstance(json_output, list)


_main_response_template: str = """
Context:
{context}

Question:
{question}
"""

prompt_instruction_entities = StructuredPrompt(
    system_message="prompt_instruction_entities.txt",
    user_template="{text}",
    validator=validate_instruction_entities,
    tags=[PromptTag.JSON],
)

prompt_instruction_keywords = StructuredPrompt(
    system_message="prompt_instruction_keywords.txt",
    user_template="{text}",
    validator=validate_instruction_keyword,
    tags=[PromptTag.JSON],
)

prompt_instruction_title = Prompt(
    system_message="prompt_instruction_title.txt",
    user_template="{text}",
)

prompt_instruction_summary = Prompt(
    system_message="prompt_instruction_summary.txt",
    user_template="{text}",
)

# TODO: Add selector for usage of long/short prompts
main_instruction_short = Prompt(
    system_message="main_instruction_short.txt",
    user_template=_main_response_template,
)

# TODO: Add selector for usage of long/short prompts
main_instruction_long = Prompt(
    system_message="main_instruction_long.txt",
    user_template=_main_response_template,
)

main_instruction = main_instruction_short