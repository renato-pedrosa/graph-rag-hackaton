

from bedrock.embeddings import EmbeddingModel
from bedrock.claude import Claude


print("Embedding")
embedding_model = EmbeddingModel(provider="bedrock")
embedding = embedding_model.embed_text(text="Hello world",model_id="amazon.titan-embed-text-v2:0")
print(embedding)

print("Claude")
answer = Claude().generate_response(prompt="Hello world", system_prompt="You are a helpful assistant.")
print(answer)
