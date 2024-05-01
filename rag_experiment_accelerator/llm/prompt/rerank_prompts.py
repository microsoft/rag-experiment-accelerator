import re
import json
from rag_experiment_accelerator.llm.prompt.prompt import StructuredPrompt, PromptTag


def validate_rerank(text: str) -> bool:
    json_output = json.loads(text)

    def key_matches(key: str) -> bool:
        return bool(re.match(r"^document_\d+$", key))

    return isinstance(json_output, dict) and all(
        isinstance(key, str) and isinstance(value, int) and key_matches(key)
        for key, value in json_output.items()
    )


_rerank_template: str = """
${documents}

Question: ${question}
"""

rerank_prompt_instruction = StructuredPrompt(
    system_message="prompt_instruction_keywords.txt",
    user_template=_rerank_template,
    validator=validate_rerank,
    tags=[PromptTag.JSON, PromptTag.NonStrict],
)
