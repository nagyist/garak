"""Support `Anthropic <https://www.anthropic.com>`_ hosted Claude models"""

from typing import List, Tuple

import backoff

from garak import _config
from garak.exception import GeneratorBackoffTrigger
from garak.generators.base import Generator
from garak.attempt import Message, Conversation


class AnthropicGenerator(Generator):
    """Interface for Claude models served via the Anthropic API (api.anthropic.com).

    Expects an API key in the ``ANTHROPIC_API_KEY`` environment variable. Keys
    are issued at https://console.anthropic.com/.
    """

    generator_family_name = "anthropic"
    fullname = "Anthropic"
    supports_multiple_generations = False
    extra_dependency_names = ["anthropic"]

    ENV_VAR = "ANTHROPIC_API_KEY"
    DEFAULT_PARAMS = Generator.DEFAULT_PARAMS | {
        "name": "claude-sonnet-4-5",
        "max_tokens": 1024,
    }

    _unsafe_attributes = ["client"]

    def _load_unsafe(self):
        self.client = self.anthropic.Anthropic(api_key=self.api_key)

    def __init__(self, name="", config_root=_config):
        super().__init__(name, config_root)
        self._load_unsafe()

    @staticmethod
    def _split_system_and_messages(
        conversation: Conversation,
    ) -> Tuple[str | None, list[dict]]:
        """Split out system turns from the rest of the conversation.

        Anthropic's Messages API takes the system prompt as a top-level
        ``system`` parameter rather than a role inside ``messages``. Collect any
        ``system``-role turns, join them, and return the remaining turns as the
        messages payload.
        """
        system_parts = []
        messages = []
        for turn in conversation.turns:
            if turn.role == "system":
                system_parts.append(turn.content.text)
            else:
                messages.append({"role": turn.role, "content": turn.content.text})
        system = "\n\n".join(system_parts) if system_parts else None
        return system, messages

    @backoff.on_exception(backoff.fibo, GeneratorBackoffTrigger, max_value=70)
    def _call_model(
        self, prompt: Conversation, generations_this_call: int = 1
    ) -> List[Message | None]:
        system, messages = self._split_system_and_messages(prompt)

        call_kwargs = {
            "model": self.name,
            "max_tokens": self.max_tokens,
            "messages": messages,
        }
        if system is not None:
            call_kwargs["system"] = system
        if self.temperature is not None:
            call_kwargs["temperature"] = self.temperature
        if self.top_k is not None:
            call_kwargs["top_k"] = self.top_k

        try:
            response = self.client.messages.create(**call_kwargs)
        except Exception as e:
            backoff_exception_types = [
                self.anthropic.RateLimitError,
                self.anthropic.APIConnectionError,
                self.anthropic.APIStatusError,
            ]
            for backoff_exception in backoff_exception_types:
                if isinstance(e, backoff_exception):
                    raise GeneratorBackoffTrigger from e
            raise e

        # `content` is a list of blocks; pull the first text block we find.
        text_blocks = [
            block.text for block in response.content if hasattr(block, "text")
        ]
        if not text_blocks:
            return [None]
        return [Message(text_blocks[0])]


DEFAULT_CLASS = "AnthropicGenerator"
