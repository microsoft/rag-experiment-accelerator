import json
from rag_experiment_accelerator.llm.prompt.prompt import (
    StructuredWithCoTPrompt,
    StructuredPrompt,
    PromptTag,
)


def qna_generation_validate(response: str) -> bool:
    response_json = json.loads(response)
    return (
        isinstance(response_json, dict)
        and "question" in response_json
        and "answer" in response_json
    )


_response_template: str = """
Context:
{context}
"""

# TODO: Add selector for usage of long/short prompts
generate_qna_long_single_context_instruction_prompt = StructuredWithCoTPrompt(
    system_message="generate_qna_long_single_context.txt",
    user_template=_response_template,
    tags=[PromptTag.JSON, PromptTag.NonStrict],
    validator=qna_generation_validate,
)

# TODO: Add selector for usage of long/short prompts
generate_qna_short_single_context_instruction_prompt = StructuredWithCoTPrompt(
    system_message="generate_qna_short_single_context.txt",
    user_template=_response_template,
    tags=[PromptTag.JSON, PromptTag.NonStrict],
    validator=qna_generation_validate,
)

# TODO: Add selector for usage of long/short prompts
generate_qna_long_multiple_context_instruction_prompt = StructuredWithCoTPrompt(
    system_message="generate_qna_long_multi_context.txt",
    user_template=_response_template,
    tags=[PromptTag.JSON, PromptTag.NonStrict],
    validator=qna_generation_validate,
)

# TODO: Add selector for usage of long/short prompts
generate_qna_short_multiple_context_instruction_prompt = StructuredWithCoTPrompt(
    system_message="generate_qna_short_multi_context.txt",
    user_template=_response_template,
    tags=[PromptTag.JSON, PromptTag.NonStrict],
    validator=qna_generation_validate,
)

# TODO: Add selector for usage of long/short prompts
generate_qna_short_single_context_no_cot_instruction_prompt = StructuredPrompt(
    system_message="generate_qna_short_single_context_no_cot.txt",
    user_template=_response_template,
    tags=[PromptTag.JSON, PromptTag.NonStrict],
    validator=qna_generation_validate,
)

qna_generation_prompt = generate_qna_short_single_context_no_cot_instruction_prompt
