"""
AILM API Client

A Python class that provides an abstraction layer for communicating with the AILM API.
This implementation uses the OpenAI SDK for compatibility with AILM's API.
"""
import logging
import time
from logging import DEBUG
from typing import Optional, List
from openai import OpenAI, APIError
from dataclasses import dataclass

@dataclass
class AILMResponse:
    """Data class to represent a structured response from the AILM API"""
    content: str
    raw_response: dict

class AILMClient:
    logger = logging.getLogger("TiktokSource")

    """
    A client for interacting with the AILM API.

    This class provides a clean abstraction for sending prompts and receiving responses
    from the AILM API, handling all the necessary configuration and API communication details.
    """

    def __init__(self, config):
        """
        Initialize the AILM API client.

        Args:
            config (Config): Nested config.
        """
        self.logger.setLevel(DEBUG)
        self.base_url = config.aiml_client.base_url
        self._api_keys = config.aiml_client.keys
        self._current_key_index = 0
        self._client = None

    def _initialize_client(self) -> None:
        """Initialize or reinitialize the OpenAI client with current configuration."""
        if not self._api_keys:
            raise ValueError("At least one API key must be set before initializing the client")
        self._client = OpenAI(api_key=self._api_keys[self._current_key_index], base_url=self.base_url)

    def send_prompt(
        self,
        prompt: str,
        system_prompt: str = "You are a helpful AI assistant.",
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        max_tokens: int = 256,
        stop: Optional[list[str]] = None,
        top_p: Optional[float] = None,
    ) -> AILMResponse:
        """
        Send a prompt to the AILM API and get a response.

        Args:
            prompt (str): The user's prompt/question.
            system_prompt (str): The system prompt that defines the AI's behavior.
            model (str): The model to use for generation.
            temperature (float): Controls randomness in the response (0.0 to 1.0).
            max_tokens (int): Maximum number of tokens in the response.
            stop (Optional[list[str]]): List of strings that signal the model to stop generating.
            top_p (Optional[float]): Nucleus sampling parameter (0.0 to 1.0).

        Returns:
            AILMResponse: A structured response containing the generated content and raw API response.

        Raises:
            ValueError: If the API key hasn't been set.
            Exception: If the API request fails for other reasons.
        """
        if not self._api_keys:
            raise ValueError("No API keys available. Please add at least one API key.")

        params = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        if stop is not None:
            params["stop"] = stop
        if top_p is not None:
            params["top_p"] = top_p

        # Keep track of rate limited keys
        rate_limited_keys = set()
        keys_tried = 0

        while keys_tried < len(self._api_keys):
            try:
                self.logger.debug(f"Attempt to creates a completion for the provided prompt with key {keys_tried}")
                self._initialize_client()
                completion = self._client.chat.completions.create(**params)
                return AILMResponse(
                    content=completion.choices[0].message.content,
                    raw_response=completion.model_dump()
                )

            except Exception as e:
                time.sleep(4)

                error_str = str(e)

                # Check for rate limit error - simplified logic
                is_rate_limit = isinstance(e, APIError) and (
                    "Rate limit exceeded" in str(e) or
                    "Free-tier limit" in str(e)
                )

                if is_rate_limit:
                    self.logger.debug(f"Rate limit reached with key {keys_tried}, trying next key")
                    rate_limited_keys.add(self._current_key_index)

                    # Move to next key
                    self._current_key_index = (self._current_key_index + 1) % len(self._api_keys)
                    keys_tried += 1

                # If we've tried all keys
                if keys_tried == len(self._api_keys):
                    # If current error is rate limit or we've tracked all keys as rate limited
                    if is_rate_limit or len(rate_limited_keys) == len(self._api_keys):
                        self.logger.info("All available API keys have reached their rate limits")
                        break
                    else:
                        raise Exception(f"Failed to get response from AILM API with any key: {error_str}")

                continue
