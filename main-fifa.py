from dotenv import load_dotenv
import os
import logging

import neo4j
from neo4j_graphrag.experimental.components.text_splitters.fixed_size_splitter import FixedSizeSplitter
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline

from sample_president_nodes import return_node_labels, return_rel_types, return_prompt

from neo4j_graphrag.indexes import create_vector_index
from bedrock.neojs_embedder import NeoJSEmbedder
from bedrock.neojs_claude import NeoJSClaude

from knowledge_graph.fifa_nodes import generate_nodes

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# load neo4j credentials (and openai api key in background).
load_dotenv('.env', override=True)
NEO4J_URI = os.getenv('NEO4J_URI', '')
NEO4J_USERNAME = os.getenv('NEO4J_USERNAME', '')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD', '')

driver = neo4j.GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

ex_llm=NeoJSClaude(
    model_name="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
    model_params={
        "response_format": {"type": "json_object"}, # use json_object formatting for best results
        "temperature": 0 # turning temperature down for more deterministic results
    }
)

# create text embedder
embedder = NeoJSEmbedder(model_id="amazon.titan-embed-text-v2:0")


def create_db_vector_index():
    create_vector_index(
        driver,
        name="text_embeddings",
        label="Chunk",
        embedding_property="embedding",
        dimensions=1024,
        similarity_fn="cosine",
    )

async def main():

    await generate_nodes()

    # create_db_vector_index()

    # kg_builder_pdf = SimpleKGPipeline(
    #     llm=ex_llm,
    #     driver=driver,
    #     text_splitter=FixedSizeSplitter(chunk_size=5000, chunk_overlap=100),
    #     embedder=embedder,
    #     entities=return_node_labels(),
    #     relations=return_rel_types(),
    #     prompt_template=return_prompt(),
    #     from_pdf=True,
    # )

    # pdf_file_paths = ["fifa-samples-pdfs/fifa-world-cup.pdf"]

    # for path in pdf_file_paths:
    #     print(f"Processing : {path}")
    #     pdf_result = await kg_builder_pdf.run_async(file_path=path)
    #     print(f"Result: {pdf_result}")
    #     logger.info("Finished processing...")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
