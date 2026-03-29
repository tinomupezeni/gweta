"""LLM Client for Gweta Intelligence.

Simple wrapper around LiteLLM to provide consistent access to
various LLM providers (OpenAI, Anthropic, etc.) for scouting,
extraction, and navigation.
"""

import os
from typing import Any, Dict, List, Optional, Union

try:
    import litellm
except ImportError:
    litellm = None

from gweta.core.logging import get_logger

logger = get_logger(__name__)


class LLMClient:
    """Wrapper for LLM operations using LiteLLM.

    Requires 'litellm' to be installed (pip install gweta[intelligence]).
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the LLM client.

        Args:
            model: Model name (e.g., 'gpt-4o', 'claude-3-5-sonnet-20240620')
            api_key: Optional API key. If None, looks for standard env vars.
            base_url: Optional base URL for the API.
            **kwargs: Additional arguments passed to litellm.completion.
        """
        if litellm is None:
            raise ImportError(
                "litellm is required for LLM features. "
                "Install with: pip install gweta[intelligence]"
            )

        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.kwargs = kwargs

        # Set API key if provided
        if api_key:
            # LiteLLM looks for provider-specific env vars or uses passed key
            # For simplicity, we can set it in the environment if it's a known provider
            if model.startswith("gpt"):
                os.environ["OPENAI_API_KEY"] = api_key
            elif model.startswith("claude"):
                os.environ["ANTHROPIC_API_KEY"] = api_key

    async def completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 1000,
        response_format: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> str:
        """Get a completion from the LLM.

        Args:
            messages: List of message dictionaries.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            response_format: Optional response format (e.g., {"type": "json_object"})
            **kwargs: Overrides for this specific call.

        Returns:
            The generated text content.
        """
        try:
            call_kwargs = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                **self.kwargs,
                **kwargs,
            }

            if response_format:
                call_kwargs["response_format"] = response_format

            response = await litellm.acompletion(**call_kwargs)
            return response.choices[0].message.content or ""

        except Exception as e:
            logger.error(f"LLM completion failed: {e}")
            raise

    async def ask(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Simple one-off prompt.

        Args:
            prompt: The user prompt.
            system_prompt: Optional system prompt.

        Returns:
            The generated text content.
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        return await self.completion(messages)

    async def extract_json(
        self,
        text: str,
        goal: str,
        schema: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Extract structured JSON from text based on a goal/schema.

        Args:
            text: The text to extract from.
            goal: What to extract.
            schema: Optional JSON schema or description of the output format.

        Returns:
            Extracted JSON data.
        """
        import json

        system_prompt = (
            "You are a precise data extraction assistant. "
            "Your task is to extract information from the provided text as JSON. "
            "Only return the JSON object, no other text or explanation."
        )

        user_prompt = f"Text to extract from:\n---\n{text}\n---\n\nGoal: {goal}\n"
        if schema:
            user_prompt += f"Output format/schema: {json.dumps(schema, indent=2)}\n"

        user_prompt += "\nReturn valid JSON."

        response_text = await self.completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
        )

        try:
            # Clean up potential markdown formatting if LLM didn't respect response_format
            clean_text = response_text.strip()
            if clean_text.startswith("```json"):
                clean_text = clean_text[7:-3].strip()
            elif clean_text.startswith("```"):
                clean_text = clean_text[3:-3].strip()

            return json.loads(clean_text)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from LLM: {e}\nResponse: {response_text}")
            return {"error": "JSON parse error", "raw": response_text}
