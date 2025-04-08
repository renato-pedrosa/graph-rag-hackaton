import os
from neo4j_graphrag.experimental.components.pdf_loader import PdfLoader
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from neo4j_graphrag.experimental.components.text_splitters.fixed_size_splitter import FixedSizeSplitter
from neo4j_graphrag.experimental.pipeline import Pipeline
from langchain_aws import ChatBedrock
import json

from clients import logger

from pyvis.network import Network

nodes_labels = []

rel_types = []

def return_rel_types():
    return rel_types

def return_node_labels():
    return nodes_labels
    

def return_prompt():
    return """
You are a journalist specialized in FIFA World Cup researcher tasks with extracting information from papers 
and structuring it in a property graph to inform further FIFA World Cup research Q&A.

Extract the entities (nodes) and specify their type from the following Input text.
Also extract the relationships between these nodes. the relationship direction goes from the start node to the end node. Make sure that Nodes with the same name are being set with the same id.


Return result as JSON using the following format:
{{"nodes": [ {{"id": "0", "label": "the type of entity", "properties": {{"name": "name of entity" }} }}],
  "relationships": [{{"type": "TYPE_OF_RELATIONSHIP", "start_node_id": "0", "end_node_id": "1", "properties": {{"details": "Description of the relationship"}} }}] }}

- Use only the information from the Input text.  Do not add any additional information.  
- If the input text is empty, return empty Json. 
- Make sure to create as many nodes and relationships as needed to offer rich FIFA World Cup context for further research.
- An AI knowledge assistant must be able to read this graph and immediately understand the context to inform detailed research questions. 
- Multiple documents will be ingested from different sources and we are using this property graph to connect information, so make sure entity types are fairly general. 

Use only fhe following nodes and relationships (if provided):
{schema}

Assign a unique ID (string) to each node, and reuse it to define relationships.
Do respect the source and target node types for relationship and
the relationship direction.

Do not return any additional information other than the JSON in it.

Examples:
{examples}

Input text:

{text}
"""




instruction = """
    You are a data scientist working for a company that is building a graph database. Your task is to extract information from data and convert it into a graph database.
    Provide a set of Nodes in the form [ENTITY_ID, TYPE, PROPERTIES] and a set of relationships in the form [ENTITY_ID_1, RELATIONSHIP, ENTITY_ID_2, PROPERTIES].
    It is important that the ENTITY_ID_1 and ENTITY_ID_2 exists as nodes with a matching ENTITY_ID. If you can't pair a relationship with a pair of nodes don't add it.
    When you find a node or relationship you want to add try to create a generic TYPE for it that  describes the entity you can also think of it as a label.
    Note that output must in dictionary type
    
    Example:
    Input : Alice lawyer and is 25 years old and Bob is her roommate since 2001. Bob works as a journalist. Alice owns a the webpage www.alice.com and Bob owns the webpage www.bob.com.
    Output : 
    { "Nodes": ["alice", "Person", {"age": 25, "occupation": "lawyer", "name":"Alice"}], ["bob", "Person", {"occupation": "journalist", "name": "Bob"}], ["alice.com", "Webpage", {"url": "www.alice.com"}], ["bob.com", "Webpage", {"url": "www.bob.com"}],
      "Edges": ["alice", "roommate", "bob", {"start": 2021}], ["alice", "owns", "alice.com", {}], ["bob", "owns", "bob.com", {}]
      }
"""

def extract_node_identifiers(json_data, key: str):
    node_identifiers = []

    # Loop through each object in the main array
    for item in json_data:
        # Check if the item has a 'Nodes' key
        if key in item:
            # Loop through each node in the Nodes array
            for node in item[key]:
                # Append the first position (identifier) to our list
                if node and len(node) > 1:
                    # check if the identifier is already in the list
                    if node[1] not in node_identifiers:
                        node_identifiers.append(node[1])

    return node_identifiers


async def generate_nodes():

    results = []
    pdf_text = await PdfLoader().run(filepath="fifa-samples-pdfs/fifa-world-cup.pdf")

    # logger.info(f"PDF text: {pdf_text.text}")

    # check if json file exists load from file
    if os.path.exists("knowledge_graph/fifa_nodes_example.json"):
        with open("knowledge_graph/fifa_nodes_example.json", "r") as f:
            results = json.load(f)

    else:
        # pipeline = Pipeline()
        text_splitter = FixedSizeSplitter(
            chunk_size=5000, chunk_overlap=200, approximate=True
        )
        # pipeline.add_component(text_splitter, "text_splitter")
        splitter_result = await text_splitter.run(text=pdf_text.text)

        logger.info(f"Splitter Chunks len: {len(splitter_result.chunks)}")

        for chunk in splitter_result.chunks:
            # Create prompt template
            prompt_template = PromptTemplate.from_template("""
            {instruction}
            Here is document. {documents}

            Return only the JSON object with the nodes and edges, do not add any additional text.
        """)

            # logger.info(f"Prompt template created: {prompt_template}")

            logger.info("Creating prompt template...")
            llm = ChatBedrock(model="anthropic.claude-3-5-sonnet-20240620-v1:0")
            llm_chain = LLMChain(prompt=prompt_template, llm=llm)

            logger.info("Calling prompt template...")

            result = llm_chain.run(instruction=instruction, documents=chunk.text)
            results.append(result)

    # logger.info(f"Prompt template result: {results}")

    node_labels = extract_node_identifiers(results, "Nodes")
    # print(f"Node labels: {node_labels}")
    rel_types = extract_node_identifiers(results, "Edges")
    # print(f"Relationship types: {rel_types}")

    # results_parsed = parser(results)
    # print(result_parsed)
    net = Network(
        notebook=True,
        cdn_resources="remote",
        bgcolor="#222222",
        font_color="white",
        height="750px",
        width="100%",
        select_menu=True,
        filter_menu=True,
    )

    nodes = []
    edges = []
    for result in results:
        # print(f"Result: {result}")
        # result_parsed = parser(result)
        # merge all nodes and edges
        nodes.extend(result.get("Nodes"))
        edges.extend(result.get("Edges"))

    # Create Nodes and Edges
    for node in nodes:
        net.add_node(node[0].lower(), label=node[0].lower(), title=str(node[2]))

    for edge in edges:
        from_node, title_edge, to_node = edge[0].lower(), edge[1], edge[2].lower()
        try:
            net.add_edge(from_node, to=to_node, title=title_edge)
        except Exception as e:
            logger.error(f"Error adding edge: {e}")
            net.add_node(to_node, label=to_node, title=str(to_node))
            net.add_edge(from_node, to=to_node, title=title_edge)
            # print(relationship)

    net.show("fifa-edges.html")

    return node_labels, rel_types

def parser(result):
    """
    Convert string to dictionary with "Nodes" and "Edges" as keys.
    """
    return json.loads(result)

