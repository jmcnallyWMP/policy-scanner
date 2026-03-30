import json
from policy_scanner.client import AzureOpenAIClient

FIXTURE = {"policy_number": "TEST-001", "carrier": "Test Carrier"}


def test_mock_client_returns_fixture():
    client = AzureOpenAIClient(mock=True, mock_response=FIXTURE)
    result = client.predict("any prompt", stage="declarations")
    assert result["response"] == FIXTURE
    assert result["input_tokens"] > 0
    assert result["output_tokens"] > 0
    assert result["model_used"] == "mock"


def test_mock_client_estimates_tokens_from_prompt():
    client = AzureOpenAIClient(mock=True, mock_response=FIXTURE)
    short = client.predict("short", stage="declarations")
    long = client.predict("x" * 10000, stage="declarations")
    assert long["input_tokens"] > short["input_tokens"]


def test_real_client_requires_endpoint():
    import pytest
    with pytest.raises(ValueError, match="endpoint"):
        AzureOpenAIClient(mock=False)  # no endpoint provided
