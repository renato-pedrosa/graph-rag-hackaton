from dotenv import load_dotenv
import os
import logging
import json

import neo4j
from bedrock.neojs_embedder import NeoJSEmbedder
from bedrock.neojs_claude import NeoJSClaude

from neo4j_graphrag.generation import RagTemplate
from neo4j_graphrag.generation.graphrag import GraphRAG
from neo4j_graphrag.retrievers import VectorRetriever
from neo4j_graphrag.retrievers import VectorCypherRetriever



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

#create text embedder
# embedder = OllamaEmbeddings(model="nomc-embed-text")
embedder = NeoJSEmbedder(model_id="amazon.titan-embed-text-v2:0")

vector_retriever = VectorRetriever(
    driver,
    index_name="text_embeddings",
    embedder=embedder,
    return_properties=["text"],
)

vc_retriever = VectorCypherRetriever(
    driver,
    index_name="text_embeddings",
    embedder=embedder,
    retrieval_query="""
        //1) Go out 2-3 hops in the entity graph and get relationships
        WITH node AS chunk
        MATCH (chunk)<-[:FROM_CHUNK]-()-[relList:!FROM_CHUNK]-{1,2}()
        UNWIND relList AS rel

        //2) collect relationships and text chunks
        WITH collect(DISTINCT chunk) AS chunks, 
        collect(DISTINCT rel) AS rels

        //3) format and return context
        RETURN '=== text ===\n' + apoc.text.join([c in chunks | c.text], '\n---\n') + '\n\n=== kg_rels ===\n' +
        apoc.text.join([r in rels | startNode(r).name + ' - ' + type(r) + '(' + coalesce(r.details, '') + ')' +  ' -> ' + endNode(r).name ], '\n---\n') AS info
        """
    )



async def main():
    rag_template = RagTemplate(template='''Answer the Question using the following Context. Only respond with information mentioned in the Context. Do not inject any speculative information not mentioned. 

    # Question:
    {query_text}
    
    # Context:
    {context}

    # Answer:
    ''', expected_inputs=['query_text', 'context'])

    v_rag  = GraphRAG(llm=ex_llm, retriever=vector_retriever, prompt_template=rag_template)
    vc_rag = GraphRAG(llm=ex_llm, retriever=vc_retriever, prompt_template=rag_template)


    q = "How is precision medicine applied to Lupus? provide in list format."
    print(f"Vector Response: \n{v_rag.search(q, retriever_config={'top_k':5}).answer}")
    print("\n===========================\n")
    print(f"Vector + Cypher Response: \n{vc_rag.search(q, retriever_config={'top_k':5}).answer}")


    q = "Can you summarize systemic lupus erythematosus (SLE)? including common effects, biomarkers, and treatments? Provide in detailed list format."

    v_rag_result = v_rag.search(q, retriever_config={'top_k': 5}, return_context=True)
    vc_rag_result = vc_rag.search(q, retriever_config={'top_k': 5}, return_context=True)

    print(f"Vector Response: \n{v_rag_result.answer}")
    print("\n===========================\n")
    print(f"Vector + Cypher Response: \n{vc_rag_result.answer}")

    for i in v_rag_result.retriever_result.items: print(json.dumps(eval(i.content), indent=1))  # noqa: E701

    vc_ls = vc_rag_result.retriever_result.items[0].content.split('\\n---\\n')
    for i in vc_ls:
        if "biomarker" in i: print(i)  # noqa: E701

    vc_ls = vc_rag_result.retriever_result.items[0].content.split('\\n---\\n')
    for i in vc_ls:
        if "treat" in i: print(i)  # noqa: E701

    q = "Can you summarize systemic lupus erythematosus (SLE)? including common effects, biomarkers, treatments, and current challenges faced by Physicians and patients? provide in list format with details for each item."
    print(f"Vector Response: \n{v_rag.search(q, retriever_config={'top_k': 5}).answer}")
    print("\n===========================\n")
    print(f"Vector + Cypher Response: \n{vc_rag.search(q, retriever_config={'top_k': 5}).answer}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())