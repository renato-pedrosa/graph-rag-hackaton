import boto3
import json
import logging
from botocore.config import Config
from typing import Any, Dict, List, Optional, Union
from neo4j_graphrag.types import LLMMessage
from neo4j_graphrag.message_history import MessageHistory

import time

logger = logging.getLogger(__name__)


class Claude:
    def __init__(
        self,
        read_timeout: int = 1000,
        max_retries: int = 10,
        default_max_tokens: int = 20000,
        default_temperature: float = 0.0,
        aws_region: str = "us-west-2",
        experimenting: bool = False,
    ):
        config = Config(read_timeout=read_timeout)
        self.bedrock_client = boto3.client(service_name="bedrock-runtime", config=config, region_name=aws_region)
        self.max_retries = max_retries
        self.default_max_tokens = default_max_tokens
        self.default_temperature = default_temperature
        self.experimenting = experimenting
        self.prompts_experiment: List[Dict[str, Any]] = []

    def _experiment_wrapper(self, func) -> Optional[str]:
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            elapsed_time = end_time - start_time
            self.prompts_experiment.append(
                {
                    "prompt": args[0],
                    "response": result,
                    "model_id": args[1],
                    "time_elapsed": elapsed_time,
                }
            )
            return result

        return wrapper # type: ignore

    @property
    def experiment_history(self) -> List[Dict[str, Any]]:
        return self.prompts_experiment

    def _invoke_model(self, prompt_config: dict, model_id: str) -> Optional[str]:
        retries = 0
        while retries < self.max_retries:
            try:
                body = json.dumps(prompt_config)
                response = self.bedrock_client.invoke_model(
                    body=body,
                    modelId=model_id,
                    accept="application/json",
                    contentType="application/json",
                )
                response_body = json.loads(response["body"].read())
                if "content" in response_body and len(response_body["content"]) > 0:
                    return response_body["content"][0]["text"]
                else:
                    logger.warning(f"Unexpected response format: {response_body}")
                    return None
            except Exception as e:
                logger.error(f"Error invoking model: {e}")
                retries += 1
        logger.error(f"Failed to invoke model after {self.max_retries} retries.")
        return None

    def generate_response(
        self,
        prompt: str,
        system_prompt: str = None,  # type: ignore
        message_history: Optional[Union[List[LLMMessage], MessageHistory]] = None,  # type: ignore
        model_id: str = "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
        max_tokens: int = None,  # type: ignore
        temperature: float = None,  # type: ignore
        stop_sequences: List[str] = None,  # type: ignore
    ) -> Optional[str]:
        max_tokens = max_tokens or self.default_max_tokens
        temperature = temperature or self.default_temperature

        prompt_config = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": []}],
        }

        prompt_config["messages"][0]["content"].append({"type": "text", "text": prompt})  # type: ignore

        if message_history:
            for message in message_history:
                if message.role == "system":
                    continue
                prompt_config["messages"].append(
                    {
                        "role": message.role,
                        "content": [{"type": "text", "text": message.content}],
                    }
                )

        if system_prompt:
            prompt_config["system"] = system_prompt

        if stop_sequences:
            prompt_config["stop_sequences"] = stop_sequences # type: ignore

        return (
            self._invoke_model(prompt_config, model_id)
            if not self.experimenting
            else self._experiment_wrapper(self._invoke_model)(prompt_config, model_id) # type: ignore
        )

    def generate_stream(self, response):
        """
        Generates the response by yielding text from the API response when using streaming

        Args:
            response: The API response.

        Yields:
            str: The text extracted from the response.
        """
        for event in response:
            chunk = event.get("chunk")
            if chunk:
                data = json.loads(chunk.get("bytes").decode())
                if "content_block_delta" in data["type"] and "text" in data["delta"]:
                    text = data["delta"]["text"]
                    yield text
