import json
from rag_experiment_accelerator.llm.prompt.prompt import StructuredPrompt, PromptTag


def validate_do_we_need_multiple(text: str) -> bool:
    return text.lower().strip() in ["complex", "simple"]


def validate_multiple_prompt(text: str) -> bool:
    json_output = json.loads(text)
    return isinstance(json_output, list) and all(
        isinstance(item, str) for item in json_output
    )


do_need_multiple_prompt_instruction = StructuredPrompt(
    system_message="do_need_multiple_prompt_instruction.txt",
    user_template="${text}",
    validator=validate_do_we_need_multiple,
    tags=[PromptTag.NonStrict],
)

multiple_prompt_instruction = StructuredPrompt(
    system_message="multiple_prompt_instruction.txt",
    user_template="${text}",
    validator=validate_multiple_prompt,
    tags=[PromptTag.JSON, PromptTag.NonStrict],
)
