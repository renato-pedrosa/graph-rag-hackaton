nodes_labels = ["Person", "Country", "PoliticalParty", "State", "City", "University", "Company", "TVShow", "Legislation"]

rel_types = ["was_president_of", "member_of", "served_as_senator_for", "born_in", "married_to", "parent_of", "signed", "is_president_of", "born_in", "graduated_from", "served_as_senator_for", "was_vice_president_under", "running_mate", "was_president_of", "graduated_from", "owns", "hosted"]

def return_rel_types():
    return rel_types

def return_node_labels():
    return nodes_labels
    

def return_prompt():
    return '''
You are a political researcher tasks with extracting information from papers 
and structuring it in a property graph to inform further political and research Q&A.

Extract the entities (nodes) and specify their type from the following Input text.
Also extract the relationships between these nodes. the relationship direction goes from the start node to the end node. 


Return result as JSON using the following format:
{{"nodes": [ {{"id": "0", "label": "the type of entity", "properties": {{"name": "name of entity" }} }}],
  "relationships": [{{"type": "TYPE_OF_RELATIONSHIP", "start_node_id": "0", "end_node_id": "1", "properties": {{"details": "Description of the relationship"}} }}] }}

- Use only the information from the Input text.  Do not add any additional information.  
- If the input text is empty, return empty Json. 
- Make sure to create as many nodes and relationships as needed to offer rich political context for further research.
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
'''