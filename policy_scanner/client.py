import json
import time
from typing import Any, Optional

GPT4O_INPUT_PRICE_PER_M = 2.50   # USD per million input tokens (proxy pricing)
GPT4O_OUTPUT_PRICE_PER_M = 10.00  # USD per million output tokens
CHARS_PER_TOKEN = 4  # conservative estimate for dense insurance prose


class AzureOpenAIClient:
    def __init__(
        self,
        mock: bool = False,
        mock_response: Optional[dict] = None,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        deployment_name: str = "gpt-4o",
        max_tokens: int = 16000,
    ):
        self.mock = mock
        self.mock_response = mock_response or {}
        self.deployment_name = deployment_name
        self.max_tokens = max_tokens

        if not mock:
            if not endpoint:
                raise ValueError("endpoint is required for non-mock client")
            self.endpoint = endpoint
            self.api_key = api_key

    def predict(self, prompt: str, stage: str = "unknown") -> dict:
        input_tokens = max(1, len(prompt) // CHARS_PER_TOKEN)
        start = time.monotonic()

        if self.mock:
            response = self.mock_response
            output_tokens = max(1, len(json.dumps(response)) // CHARS_PER_TOKEN)
            model_used = "mock"
        else:
            from openai import AzureOpenAI
            client = AzureOpenAI(
                azure_endpoint=self.endpoint,
                api_key=self.api_key,
                api_version="2024-02-01",
            )
            completion = client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You extract structured insurance data. Return valid JSON only.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"},
            )
            response = json.loads(completion.choices[0].message.content)
            usage = completion.usage
            input_tokens = usage.prompt_tokens
            output_tokens = usage.completion_tokens
            model_used = self.deployment_name

        duration_ms = int((time.monotonic() - start) * 1000)
        cost = (
            input_tokens / 1_000_000 * GPT4O_INPUT_PRICE_PER_M
            + (output_tokens if not self.mock else 0) / 1_000_000 * GPT4O_OUTPUT_PRICE_PER_M
        )

        return {
            "response": response,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "model_used": model_used,
            "duration_ms": duration_ms,
            "cost_estimate_usd": round(cost, 6),
            "stage": stage,
        }
