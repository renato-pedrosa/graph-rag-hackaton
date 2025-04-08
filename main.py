from dotenv import load_dotenv
import os

import neo4j
from neo4j_graphrag.llm import OllamaLLM
from neo4j_graphrag.embeddings.ollama import OllamaEmbeddings

from neo4j_graphrag.experimental.components.text_splitters.fixed_size_splitter import FixedSizeSplitter
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline

from sample_president_nodes import return_node_labels, return_rel_types, return_prompt

from neo4j_graphrag.indexes import create_vector_index

from clients import logger

from basic_knowledge_graph import create_knowledge_graph

from knowledge_graph.graph import create_prompt_template

# load neo4j credentials (and openai api key in background).
load_dotenv('.env', override=True)
NEO4J_URI = os.getenv('NEO4J_URI', '')
NEO4J_USERNAME = os.getenv('NEO4J_USERNAME', '')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD', '')

driver = neo4j.GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

ex_llm=OllamaLLM(
    model_name="llama3.2",
    model_params={
        "response_format": {"type": "json_object"}, # use json_object formatting for best results
        "temperature": 0 # turning temperature down for more deterministic results
    }
)

#create text embedder
embedder = OllamaEmbeddings(model="nomic-embed-text")


# def create_db_vector_index():
#     create_vector_index(driver, name="text_embeddings", label="Chunk",
#                     embedding_property="embedding", dimensions=768, similarity_fn="cosine")

async def main():
    logger.info("Starting the application...")  
    
    # create_knowledge_graph()
    create_prompt_template()

    # # create_db_vector_index()
    # kg_builder_pdf = SimpleKGPipeline(
    #     llm=ex_llm,
    #     driver=driver,
    #     text_splitter=FixedSizeSplitter(chunk_size=500, chunk_overlap=100),
    #     embedder=embedder,
    #     entities=return_node_labels(),
    #     relations=return_rel_types(),
    #     prompt_template=return_prompt(),
    #     from_pdf=True,
    # )

    # pdf_file_paths = ["sample-pdfs/president.pdf"]

    # for path in pdf_file_paths:
    #     print(f"Processing : {path}")
    #     pdf_result = await kg_builder_pdf.run_async(file_path=path)
    #     print(f"Result: {pdf_result}")
    #     logger.info("Finished processing...")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

# Nodes-labels: Person | Place | Organization | ArticleOrPaper | PublicationOrJournal | Disease | Drug | GeneOrProtein | Condition | EffectOrPhenotype | Expose
# Relations: was_president_of | born_in
# Nodes: Obama | Honolulu | Biden | Trump | Washington