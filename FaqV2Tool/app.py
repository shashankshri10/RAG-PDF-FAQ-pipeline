import gradio as gr
import os
import logging
from typing import List
from openai import OpenAI
from pymongo import MongoClient
from pydantic import BaseModel, Field
from dotenv import load_dotenv
load_dotenv("./.env.shared")
from envyaml import EnvYAML
config = EnvYAML("./config.yaml")



logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.INFO)

openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
class ResponseModel(BaseModel):
    response: str = Field(default_factory=lambda : "",description="Response to user query")


def weighted_reciprocal_rank(doc_lists):
    c=60
    weights=[0.50,0.50]
    
    all_documents = set()
    for doc_list in doc_lists:
        for doc in doc_list:
            all_documents.add(doc["content"])

    rrf_score_dic = {doc: 0.0 for doc in all_documents}

    for doc_list, weight in zip(doc_lists, weights):
        for rank, doc in enumerate(doc_list, start=1):
            rrf_score = weight * (1 / (rank + c))
            rrf_score_dic[doc["content"]] += rrf_score

    sorted_documents = sorted(
        rrf_score_dic.keys(), key=lambda x: rrf_score_dic[x], reverse=True
    )
    page_content_to_doc_map = {
        doc["content"]: doc for doc_list in doc_lists for doc in doc_list
    }
    sorted_docs = [
        page_content_to_doc_map[page_content] for page_content in sorted_documents
    ]

    return sorted_docs


def handler(query: str, queryV2: List[str], client_id: str, tag: str):

    
    def create_embeddings(document: str):
        try:
            response = openai_client.embeddings.create(
                input=document,
                model=config['openai']['embedding_model']
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error creating embeddings for document: {e}")

    client = MongoClient(config['database']['mongodb']['host'])
    db = client[config['database']['mongodb']['db_name']]
    collection = db[client_id]
    
    results = []
    
    for tmp in queryV2:
        semantic_results = list(collection.aggregate([
            {
                "$vectorSearch":
                {
                    "queryVector": create_embeddings(tmp),
                    "path": "vector",
                    "numCandidates": 150,
                    "limit": 10,
                    "index": "semantic-search",
                    "filter": {
                        "filter_tags": tag
                    }
                },
            },
            {
                "$project":
                    {
                        "_id": 1,
                        "content": 1,
                        "keywords": 1,
                        "score":{"$meta":"vectorSearchScore"}
                    }
            }
        ]))
        results.extend(semantic_results)
        
    semantic_results_v2 = list(collection.aggregate([
        {
            "$vectorSearch":
            {
                "queryVector": create_embeddings(query),
                "path": "vector",
                "numCandidates": 150,
                "limit": 10,
                "index": "semantic-search",
                "filter": {
                    "filter_tags": tag
                }
            },
        },
        {
            "$project":
                {
                    "_id": 1,
                    "content": 1,
                    "keywords": 1,
                    "score":{"$meta":"vectorSearchScore"}
                }
        }
    ]))
    results.extend(semantic_results_v2)
    
    keyword_results = []
    for tmp in queryV2:
        keyword_results = list(collection.aggregate([
            {'$search':{
                "index": "keyword-search",
                "compound": {
                    "must": [
                        {
                            "text": {
                                "query": tmp,
                                "path": ["keywords"],
                                "fuzzy": {"maxEdits": 2}
                            }
                        },
                        {"text": {"query": tag,"path": "filter_tags"}}
                    ]
                }}
            },
            {"$project":{"_id": 1,"content": 1,"keywords": 1,"score": {"$meta": "searchScore"}}},
            {"$limit": 15}
        ]))
        keyword_results.extend(keyword_results)
        
        
    keyword_original = list(collection.aggregate([
        {'$search':{
            "index": "keyword-search",
            "compound": {
                "must": [
                    {
                        "text": {
                            "query": query,
                            "path": ["keywords"],
                            "fuzzy": {"maxEdits": 2}
                        }
                    },
                    {"text": {"query": tag,"path": "filter_tags"}}
                ]
            }}
        },
        {"$project":{"_id": 1,"content": 1,"keywords": 1,"score": {"$meta": "searchScore"}}},
        {"$limit": 15}
    ]))
    keyword_results.extend(keyword_original)
    
    docs = [i['content'] for i in weighted_reciprocal_rank([semantic_results,keyword_results])][:8]
    results = "\n".join(docs)
    
    prompt = f"""
    Role: Content Retrival based Question Answering System
    Task: Your task is to provide the answer to the users query based on the provided information.
    Your task is of understand the query and only provide the answer using the data from provide information.

    Reference Information:
    {results}
    
    Steps To Follow:
    1. Create questions synonmyous questions from user provided query
    2. Try to figure out whether any of the results contains information which could answer any of the questions
    3. Answer the questions based on instructions provided below.

    After generation of response think within <Thinking> section what all aspects are covered in information
    section, can any of the provided information able to answer any question while like mapping durations as short-terms, long-terms,
    size as large, small, medium, lease as contract, agreement, etc into question and information. Try to think, by analytically
    thinking if some information can be manipulated to answer question.
        
    Instructions:
    1. Return ONLY information explicitly present in the context
    2. No external knowledge or assumptions allowed
    3. If information is not sufficient to answer the question, simply repond with "Sorry, I didn’t understand your question. Do you want to connect with a live agent?"
    4. If the response is detailed, make sure you explain it in detail
    5. Response must to the point and should answer the question
    """
    aux = "\n".join(queryV2)
    output = openai_client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"User query: {aux}"}
        ],
        temperature=0,
        response_format=ResponseModel
    )

    return {"RESP": output.choices[0].message.parsed.response, "DOCUMENTS": docs}


def query_expansion(text: str):
    from typing import List
    
    class DecomposeModel(BaseModel):
        query: List[str] = Field(default_factory=lambda : text, description="List of queries")
        # keywords_metadata: List[str] = Field(default_factory=lambda : text, description="synonym tags metadata")
        
    prompt = f"""
    Role: Document Query Expansion System
    Objective: Generate variations of user queries to enhance document search accuracy

    Core Capabilities:
    - Analyze raw user queries about any topic in the document
    - Reframe queries for clarity and specificity
    - Generate semantically similar questions
    - Map related terms and synonyms

    Query Generation Principles:
    - Ensure complete contextual understanding
    - Eliminate ambiguity
    - Capture entire query intent
    - Create diverse but relevant query variations
    - Maintain original query intent

    Instructions:
    - Don't generate more than 5 queries
    - Use ONLY provided context
    - Prohibit external knowledge injection
    - Maintain query precision
    - Ensure semantic accuracy
    - Preserve original intent
"""

    output = openai_client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": text}
        ],
        temperature=0,
        response_format=DecomposeModel
    )

    tmp = output.choices[0].message.parsed
    return tmp.query
# , " ".join(tmp.keywords_metadata)
def process_query(client_id, query, model, tag):
    """Process query through the API."""
    logger.info(f"Starting query processing for client_id: {client_id}")
    logger.debug(f"Query content: {query}")

    try:
        queryV2 = query_expansion(query)
        print(queryV2)
        rs = handler(query,queryV2,client_id,tag)
        return {
            "RESPONSE": rs['RESP'],
            "DOCUMENTS": rs['DOCUMENTS'],
            "QUERY": queryV2,
            # "KEYWORDS": keywords
        }
    except Exception as e:
        logger.error(f"Error in process_query: {str(e)}")
        return f"Error processing query: {str(e)}"

# Create Gradio interface
demo = gr.Interface(
    fn=process_query,
    inputs=[
        gr.Textbox(label="Collection Name", placeholder="Enter your collection name"),
        gr.Textbox(label="Query", lines=5),
        gr.Dropdown(choices=["gpt-4o"], label="Model"),
        gr.Textbox(label="Tag")
    ],
    outputs=gr.JSON(label="Response"),
    title="Query Processor",
    description="Enter query details to get a response."
)

demo.launch()