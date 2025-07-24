import logging
from typing import Any, Callable, List, Tuple

from langchain_groq import ChatGroq

# from langchain_openai import ChatOpenAI
from omegaconf import DictConfig


class LLMManager:
    """
    Manages a list of LLM models (Groq, OpenAI, etc.) with fallback logic.
    Usage:
        llm_manager = LLMManager([
            ("meta-llama/llama-4-scout-17b-16e-instruct", "groq"),
            ("llama-3.1-8b-instant", "groq"),
            ("gpt-3.5-turbo", "openai"),
        ])
        # For a sequence of related calls, use reset=True for the first, then reset=False for subsequent calls
        result1 = llm_manager.invoke_with_fallback(chain1.invoke, ..., reset=True)
        result2 = llm_manager.invoke_with_fallback(chain2.invoke, ..., reset=False)
    """

    def __init__(self, model_configs: List[tuple], temperature: float = 0.0):
        """
        model_configs: List of tuples (model_id, provider), provider is 'groq' or 'openai'
        """
        self.model_configs = model_configs
        self.temperature = temperature
        self._build_clients()
        self.current_index = 0

    @classmethod
    def get_default_managers(cls) -> Tuple["LLMManager", "LLMManager"]:
        """
        Get default LLM managers for completions and tool calls.

        Returns:
            Tuple of (completion_manager, tool_manager)
        """
        # You can customize these lists independently if needed
        model_list = [
            ("mistral-saba-24b", "groq"),
            ("meta-llama/llama-4-maverick-17b-128e-instruct", "groq"),  # Slow
            ("meta-llama/llama-4-scout-17b-16e-instruct", "groq"),  # Slow
            # ("llama3-8b-8192", "groq"),
        ]
        tool_model_list = [
            ("moonshotai/kimi-k2-instruct", "groq"),  # fast
            ("qwen/qwen3-32b", "groq"),  # fast
            ("llama-3.3-70b-versatile", "groq"),  # Slow
            ("llama3-70b-8192", "groq"),  # Slow
            ("deepseek-r1-distill-llama-70b", "groq"),  # Slow
            # ("gemma2-9b-it", "groq"), # tool fails
            # ("llama-3.1-8b-instant", "groq"), # tool fails
            # ("llama3-8b-8192", "groq"), # tool fails
            # ("gpt-3.5-turbo", "openai"),
        ]
        return cls(model_list), cls(tool_model_list)

    @classmethod
    def from_config(
        cls, config: DictConfig, use_tool_models: bool = False
    ) -> "LLMManager":
        """
        Create an LLMManager from a Hydra config DictConfig.
        Args:
            config: The loaded config (from Hydra)
            use_tool_models: If True, use tool_models, else agent_models
        Returns:
            LLMManager instance
        """
        models = (
            config.models.tool_models if use_tool_models else config.models.agent_models
        )
        temperature = (
            config.models.temperature if "temperature" in config.models else 0.0
        )
        model_configs = [(m["id"], m["provider"]) for m in models]
        return cls(model_configs, temperature=temperature)

    def _build_clients(self):
        self.clients = []
        for model_id, provider in self.model_configs:
            if provider == "groq":
                self.clients.append(
                    ChatGroq(model=model_id, temperature=self.temperature)
                )
            elif provider == "openai":
                from langchain_openai import ChatOpenAI

                self.clients.append(
                    ChatOpenAI(model=model_id, temperature=self.temperature)
                )
            else:
                raise ValueError(f"Unknown provider: {provider}")
        # if self.clients:
        #     print(f"---- [LLMManager] Current selected model ---- : {model_id}")

    @property
    def current(self):
        return self.clients[self.current_index]

    @property
    def current_model_id(self):
        return self.model_configs[self.current_index][0]

    def advance(self) -> bool:
        if self.current_index + 1 < len(self.clients):
            self.current_index += 1
            print(
                f"[LLMManager] Switched to model: {self.model_configs[self.current_index]}"
            )
            return True
        return False

    def reset(self):
        self.current_index = 0
        if self.clients:
            print(
                f"[LLMManager] Reset to model: {self.model_configs[self.current_index]}"
            )

    def invoke_with_fallback(
        self, runnable: Callable, *args, reset: bool = True, **kwargs
    ) -> Any:
        """
        Try invoking the runnable (e.g., a chain or model call) with fallback on rate-limit/model errors.
        If reset is True (default), resets to the first model before trying. If False, continues from current model.
        """
        last_err = None
        if reset:
            self.reset()
        while True:
            try:
                return runnable(*args, **kwargs)
            except Exception as e:
                msg = str(e).lower()
                if (
                    "rate limit" in msg
                    or "rate_limit" in msg
                    or "quota" in msg
                    or "unavailable" in msg
                    or "over capacity" in msg
                ):
                    last_err = e
                    if self.advance():
                        logging.warning(
                            f"Rate-limit or error on model, switching to {self.current_model_id}"
                        )
                        continue
                raise last_err or e
