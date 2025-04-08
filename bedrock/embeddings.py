""" This module provides a class to interact with the embedding model. """

import boto3
import json


class EmbeddingModel:
    """
    A class representing an embedding model.

    This class provides methods to interact with an embedding model from a specified provider.
    It allows embedding text using the model and retrieving the generated embeddings.

    Attributes:
        provider (str): The provider of the embedding model.
        model_id (str): The ID of the embedding model.
        kwargs (dict): Additional keyword arguments.
        bedrock_client (boto3.client): The Bedrock client object.
    """

    def __init__(self, provider: str = "bedrock", *kwargs):
        """
        Initialize the EmbeddingModel instance.

        Args:
            provider (str): The provider of the embedding model. Defaults to "bedrock".
            *kwargs: Additional keyword arguments.
        """
        self.provider = provider
        self.kwargs = kwargs
        self.bedrock_client = self._initialize_client()

    def _init_bedrock_client(self):
        """
        Initialize and return a Bedrock client.

        Returns:
            boto3.client: The initialized Bedrock client.
        """
        bedrock_client = boto3.client(service_name="bedrock-runtime", region_name="us-west-2")
        return bedrock_client

    def _initialize_client(self):
        """
        Initialize and return the appropriate client based on the provider.

        Returns:
            The initialized client.

        Raises:
            ValueError: If the provider is unsupported.
        """
        if self.provider == "bedrock":
            # Initialize Bedrock client
            return self._init_bedrock_client()
        else:
            raise ValueError("Unsupported provider")

    # returns an embedding vector for the given text
    def embed_text(self, text, model_id: str = "amazon.titan-embed-text-v2:0") -> list:
        """
        Embed the given text using the embedding model.

        Args:
            text (str): The text to embed.

        Returns:
            list: The embedding vector.
        """
        body = json.dumps(
            {
                "inputText": text,
            }
        )

        # Invoke model
        response = self.bedrock_client.invoke_model(
            body=body,
            modelId=model_id,
            accept="application/json",
            contentType="application/json",
        )

        response_body = json.loads(response["body"].read())
        embedding = response_body.get("embedding")

        return embedding

    def get_embedding_dimension(self):
        """
        Retrieve the embedding vector dimension.

        Returns:
            int: The dimension of the embedding vector.
        """
        return len(self.embed_text("dummy text"))