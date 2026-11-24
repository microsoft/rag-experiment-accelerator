from rag_experiment_accelerator.llm.response_generator import ResponseGenerator
from transformers import AutoTokenizer, AutoModelForCausalLM

from rag_experiment_accelerator.config.llm_config import BaseLLMConfig
from rag_experiment_accelerator.utils.logging import get_logger
from rag_experiment_accelerator.llm.prompt.prompt import (
    Prompt,
    PromptTag,
)

logger = get_logger(__name__)


class HuggingfaceResponseGenerator(ResponseGenerator):
    def __init__(self, config: BaseLLMConfig):
        super().__init__(config)

        self._tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self._model = AutoModelForCausalLM.from_pretrained(self.config.model_name)

    def _get_response(
        self,
        messages,
        prompt: Prompt,
    ) -> any:
        kwargs = {}

        if self.json_object_supported and PromptTag.JSON in prompt.tags:
            kwargs["response_format"] = {"type": "json_object"}

        input_ids = self._tokenizer.encode(messages, return_tensors="pt")
        output_ids = self._model.generate(
            input_ids,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            temperature=self.config.temperature,
            max_length=self.config.max_tokens,
        )
        response_text = self._tokenizer.decode(output_ids[0], skip_special_tokens=True)

        return self._interpret_response(response_text, prompt)
