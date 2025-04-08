from neo4j_graphrag.llm import LLMInterface
from typing import Optional, Any, Union, List
from neo4j_graphrag.types import LLMMessage  
from neo4j_graphrag.llm.types import LLMResponse
from neo4j_graphrag.exceptions import LLMGenerationError
from neo4j_graphrag.message_history import MessageHistory

import logging
from .claude import Claude

logger = logging.getLogger(__name__)

class NeoJSClaude(LLMInterface):
    """Interface for large language models.

    Args:
        model_name (str): The name of the language model.
        model_params (Optional[dict]): Additional parameters passed to the model when text is sent to it. Defaults to None.
        **kwargs (Any): Arguments passed to the model when for the class is initialized. Defaults to None.
    """

    def __init__(
        self,
        model_name: str,
        model_params: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ):
        logger.info("Initializing NeoJSClaude with model: %s", model_name)
        self.model_name = model_name
        self.model_params = model_params or {}
        self.claude = Claude()
        
    
    def invoke(
        self,
        input: str,
        message_history: Optional[Union[List[LLMMessage], MessageHistory]] = None,
        system_instruction: Optional[str] = None,
    ) -> LLMResponse:
        """Sends a text input to the LLM and retrieves a response.

        Args:
            input (str): Text sent to the LLM.
            message_history (Optional[Union[List[LLMMessage], MessageHistory]]): A collection previous messages,
                with each message having a specific role assigned.
            system_instruction (Optional[str]): An option to override the llm system message for this invocation.

        Returns:
            LLMResponse: The response from the LLM.

        Raises:
            LLMGenerationError: If anything goes wrong.
        """
        try: 
            logger.info("Calling Claude")
            response = self.claude.generate_response(
                prompt=input,
                system_prompt=system_instruction,
                message_history=message_history,
                model_id=self.model_name,
                max_tokens=self.model_params.get("max_tokens", 20000),
                temperature=self.model_params.get("temperature", 0.0),
            )

            if not response:
                raise LLMGenerationError("Failed to generate response from LLM.")

            logger.info("Response generated")
            return LLMResponse(content=response)

        except Exception as e:
            raise LLMGenerationError(f"Failed to generate response from LLM: {str(e)}")

    async def ainvoke(
            self,
            input: str,
            message_history: Optional[Union[List[LLMMessage], MessageHistory]] = None,
            system_instruction: Optional[str] = None,
        ) -> LLMResponse:
            """Asynchronously sends a text input to the LLM and retrieves a response.

            Args:
                input (str): Text sent to the LLM.
                message_history (Optional[Union[List[LLMMessage], MessageHistory]]): A collection previous messages,
                    with each message having a specific role assigned.
                system_instruction (Optional[str]): An option to override the llm system message for this invocation.

            Returns:
                LLMResponse: The response from the LLM.

            Raises:
                LLMGenerationError: If anything goes wrong.
            """
            try: 
                logger.info("Calling Claude")
                response = self.claude.generate_response(
                    prompt=input,
                    system_prompt=system_instruction,
                    message_history=message_history,
                    model_id=self.model_name,
                    max_tokens=self.model_params.get("max_tokens", 20000),
                    temperature=self.model_params.get("temperature", 0.0),
                )

                if not response:
                    raise LLMGenerationError("Failed to generate response from LLM.")

                logger.info("Response generated")
                return LLMResponse(content=response)

            except Exception as e:
                raise LLMGenerationError(f"Failed to generate response from LLM: {str(e)}")