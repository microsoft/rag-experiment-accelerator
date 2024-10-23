import re
import json
from rag_experiment_accelerator.llm.prompt.prompt import (
    Prompt,
    StructuredPrompt,
    PromptTag,
)


def validate_context_precision(text: str) -> bool:
    return text.lower().strip() in ["yes", "no"]


def validate_context_recall(text: str) -> bool:
    json_text = json.loads(text)

    def is_valid_entry(entry):
        statement_key_pattern = re.compile(r"^statement_\d+$")
        return all(
            key in ["reason", "attributed"] or statement_key_pattern.match(key)
            for key in entry.keys()
        )

    return isinstance(json_text, list) and all(
        is_valid_entry(entry) for entry in json_text
    )


_context_precision_input = """
Context:
${context}

Question:
${question}
"""

_context_recall_input = """
question: ${question}
context: ${context}
answer: ${answer}
"""

llm_answer_relevance_instruction = Prompt(
    system_message="llm_answer_relevance_instruction.txt",
    user_template="${text}",
    tags={PromptTag.NonStrict},
)

llm_context_precision_instruction = StructuredPrompt(
    system_message="llm_context_precision_instruction.txt",
    user_template=_context_precision_input,
    validator=validate_context_precision,
    tags={PromptTag.NonStrict},
)

llm_context_recall_instruction = StructuredPrompt(
    system_message="llm_context_recall_instruction.txt",
    user_template=_context_recall_input,
    validator=validate_context_recall,
    tags={PromptTag.JSON, PromptTag.NonStrict},
)
