import os
from importlib import resources
import re

from string import Template

from enum import StrEnum
from rag_experiment_accelerator.utils.logging import get_logger

logger = get_logger(__name__)


def _default_validation(x: any) -> bool:
    return True


class PromptTag(StrEnum):
    ChainOfThought = "chain_of_thought"  # Prompt is a chain of thought prompt
    Structured = "structured"  # Output of the prompt is expected to be structured
    JSON = "json"  # Output of the prompt is expected to be in JSON format
    NonStrict = "non_strict"  # Do not raise an exception if inference failed


class Prompt:
    _base_tags: list = list()

    def __init__(
        self,
        system_message: str,
        user_template: str,
        tags: list[str] | None = None,
    ) -> None:
        self.system_message = system_message
        self.user_template = user_template

        self.tags = self._base_tags + tags if tags else []

        if PromptTag.JSON in self.tags:
            assert (
                PromptTag.Structured in self.tags
            ), "Structured tag must be present for JSON prompts"

        self.system_message = self._try_load_prompt_file(self.system_message)
        self.user_template = self._try_load_prompt_file(self.user_template)

    @staticmethod
    def arguments_in_prompt(prompt: str) -> set[str]:
        pattern = re.compile(r"\$\{([a-zA-Z_][a-zA-Z0-9_]*)\}")
        matches = pattern.findall(prompt)
        return set(matches)

    @staticmethod
    def check_formatting_argument(text: str, field: str) -> bool:
        return field in Prompt.arguments_in_prompt(text)

    @staticmethod
    def _get_prompt_file_path(prompt_file: str) -> str:
        base_path = os.path.join("llm", "prompts_text", prompt_file)
        return (
            resources.files("rag_experiment_accelerator").joinpath(base_path).resolve()
        )

    @staticmethod
    def _try_load_prompt_file(prompt_file: str) -> str:
        if re.match(r'^[^\/:*?"<>|\r\n]+\.txt$', prompt_file):
            prompt_file = Prompt._get_prompt_file_path(prompt_file)
            if os.path.isfile(prompt_file):
                logger.debug(f"Reading prompt from file: {prompt_file}")
                with open(prompt_file, "r") as f:
                    return f.read()
            else:
                logger.error(f"Prompt file not found: {prompt_file}")
                raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
        else:
            return prompt_file

    def update_system_prompt(self, system_message: str) -> "Prompt":
        system_message = self._try_load_prompt_file(system_message)
        self.system_message = system_message
        return self

    def update_user_prompt(self, user_template: str) -> "Prompt":
        user_template = self._try_load_prompt_file(user_template)
        self.user_template = user_template
        return self


class CoTPrompt(Prompt):
    _base_tags: list = [PromptTag.ChainOfThought]

    def __init__(
        self,
        system_message: str,
        user_template: str,
        tags: list[str] | None = None,
        separator: str = "##RESPONSE##",
    ) -> None:
        super().__init__(system_message, user_template, tags)
        assert (
            PromptTag.ChainOfThought in self.tags
        ), "CoTPrompt must have ChainOfThought tag"

        self.separator = separator

        has_system, has_user = self._check_separator_declaration(
            self.system_message, self.user_template
        )

        if has_system:
            template = Template(self.system_message)
            self.system_message = template.safe_substitute(separator=self.separator)

        if has_user:
            template = Template(self.user_template)
            self.user_template = template.safe_substitute(separator=self.separator)

    @staticmethod
    def _check_separator_declaration(
        system_message: str, user_template: str
    ) -> tuple[bool, bool]:
        has_sep_in_system = CoTPrompt.check_formatting_argument(
            system_message, "separator"
        )
        has_sep_in_user = CoTPrompt.check_formatting_argument(
            user_template, "separator"
        )

        if not has_sep_in_system:
            if has_sep_in_user:
                logger.warning(
                    "It is recommended to declare saparator in system message as well"
                )
            else:
                logger.error(
                    "Separator is not declared in system message or user template, this will cause issues"
                )

        return has_sep_in_system, has_sep_in_user

    def update_system_prompt(self, system_message: str) -> Prompt:
        system_message = self._try_load_prompt_file(system_message)
        if CoTPrompt.check_formatting_argument(system_message, "separator"):
            template = Template(system_message)
            system_message = template.safe_substitute(separator=self.separator)

        self.system_message = system_message


class StructuredPrompt(Prompt):
    """
    A prompt that expects a structured response, such as JSON.
    """

    _base_tags: list = [PromptTag.Structured]

    def __init__(
        self,
        system_message: str,
        user_template: str,
        tags: list[str] | None = None,
        validator: callable = _default_validation,
    ) -> None:
        super().__init__(system_message, user_template, tags)

        self.validator = validator


class StructuredWithCoTPrompt(CoTPrompt, StructuredPrompt):
    _base_tags: list = [PromptTag.ChainOfThought, PromptTag.Structured]

    def __init__(
        self,
        system_message: str,
        user_template: str,
        tags: list[str] | None = None,
        validator: callable = _default_validation,
        separator: str = "##RESPONSE##",
    ) -> None:
        StructuredPrompt.__init__(
            self,
            system_message=system_message,
            user_template=user_template,
            tags=tags,
            validator=validator,
        )
        CoTPrompt.__init__(
            self,
            system_message=system_message,
            user_template=user_template,
            tags=tags,
            separator=separator,
        )
