from dataclasses import dataclass
from typing import Dict, Any

import requests
import json

@dataclass
class LLMConfig:
    LLM_URL = "http://localhost:11434/api/generate"
    LLM_MODEL = "gpt-oss:20b"
    DEBUG_MODE = False

class LLM:
    """
    A class to interact with a local LLM server (e.g. Ollama).
    """

    def __init__(self, config: LLMConfig):
        """
        Initialize the LLM with configuration.

        Args:
            config: Configuration for the LLM.
        """
        self.config = config

    def call_llm(self, prompt: str) -> str:
        """
        Call local LLM server with prompt, return text response.

        Args:
            prompt: The prompt to send to the LLM.
        Returns:
            The text response from the LLM.
        """
        resp = requests.post(
            url=self.config.LLM_URL,
            json={"model": self.config.LLM_MODEL, "prompt": prompt},
            stream=True,
        )
        text = ""
        for line in resp.iter_lines():
            if line:
                text += json.loads(line)["response"]
        return text

class LLMAgent:
    """
    A class modeling an agent that uses an LLM that has a prompt template
    to perform specific tasks.
    """

    def __init__(self, llm_config: LLMConfig, prompt_template_yaml: str):
        """
        Initialize the LLMAgent with configuration and prompt template.

        Args:
            llm_config: Configuration for the LLM.
            prompt_template_yaml: Path to the YAML file containing the prompt template.
        """
        self.llm = LLM(llm_config)
        with open(prompt_template_yaml, 'r') as f:
            self.prompt_template = f.read()

    def _prepare_prompt(self, params: Dict[str, str]) -> str:
        """
        Prepare the prompt by formatting the template with provided parameters.

        Args:
            **params: Parameters to format the prompt template.
        Returns:
            str: The formatted prompt.
        """
        return self.prompt_template.format(**params)

    def ask(self, params: Dict[str, Any]) -> str:
        """
        Ask the LLM a question using the prepared prompt.

        Args:
            **params: Parameters to format the prompt template.
        Returns:
            str: The response from the LLM.
        """
        prompt = self._prepare_prompt(params=params)
        if self.llm.config.DEBUG_MODE:
            print('\n'+prompt+'\n')
        response = self.llm.call_llm(prompt)
        if self.llm.config.DEBUG_MODE:
            print("\nResponse:\n", response)
        return response
