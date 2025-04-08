
from dotenv import load_dotenv
import os
import logging
import json
import neo4j

from neo4j_graphrag.retrievers import VectorRetriever
from bedrock.neojs_embedder import NeoJSEmbedder

from neo4j_graphrag.retrievers import VectorCypherRetriever

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# load neo4j credentials (and openai api key in background).
load_dotenv('.env', override=True)
NEO4J_URI = os.getenv('NEO4J_URI', '')
NEO4J_USERNAME = os.getenv('NEO4J_USERNAME', '')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD', '')

driver = neo4j.GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

embedder = NeoJSEmbedder(model_id="amazon.titan-embed-text-v2:0")

vector_retriever = VectorRetriever(
    driver,
    index_name="text_embeddings",
    embedder=embedder,
    return_properties=["text"],
)

vector_res = vector_retriever.get_search_results(query_text = "How is precision medicine applied to Lupus?", 
                                                 top_k=3)
for i in vector_res.records:
    print("====" + json.dumps(i.data(), indent=4))


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
RETURN '=== text ===' + apoc.text.join([c in chunks | c.text], '---') + '=== kg_rels ===' +
  apoc.text.join([r in rels | startNode(r).name + ' - ' + type(r) + '(' + coalesce(r.details, '') + ')' +  ' -> ' + endNode(r).name ], '---') AS info
""",
)



vc_res = vc_retriever.get_search_results(query_text = "How is precision medicine applied to Lupus?", top_k=3)

# print output
kg_rel_pos = vc_res.records[0]["info"].find("=== kg_rels ===")
print("# Text Chunk Context:")
print(vc_res.records[0]['info'][:kg_rel_pos])
print("# KG Context From Relationships:")
print(vc_res.records[0]['info'][kg_rel_pos:])


