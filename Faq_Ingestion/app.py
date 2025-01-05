from pydantic import BaseModel, Field
from typing import List
import openai
from openai import OpenAI
from pymongo import MongoClient
from pymongo.operations import SearchIndexModel
# import weave
import re
import os
import logging
import json
import base64
import fitz
import time
from dotenv import load_dotenv
load_dotenv("./.env.shared")
from envyaml import EnvYAML
import gradio as gr

# weave.init("faq_tool")
config = EnvYAML("./config.yaml")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.INFO)

class OutputResponse(BaseModel):
    data: str = Field(default_factory=lambda : "",description="Data extracted from text")
    keywords: List[str] = Field(default_factory=lambda : [""],description="Keywords extracted from text")

class DataExtraction:
    class ResponseModel(BaseModel):
        result: List[OutputResponse] = Field(default_factory=lambda : [OutputResponse()])
        carry_forward: str = Field(default_factory=lambda : "",description="Data that is incomplete and can be carry forwarded to next")
    
    def __init__(self,  collection: str, model: str = config['openai']['extraction_model']) -> None:
        logger.info(f"Initializing DataExtraction with collection: {collection}, model: {model}")
        self.model = model
        self.embedding_model = config['openai']['embedding_model']
        self.embedding_dimension = 3072
        self.temperature = 0
        self.client_openai = OpenAI(
            api_key = os.getenv("OPENAI_API_KEY"),
        )
        self.retry_factor = {
            "segment_data_extraction": 0,
            "embedding_generation": 0,
        }

        self.db_uri = config['database']['mongodb']['host']
        self.db_name = config['database']['mongodb']['db_name']
        self.collection_name = collection
        self.active_db_connection = []
        self.collection_index = {
            "semantic-search": SearchIndexModel(
                definition = {
                    "fields": [
                        {
                            "type": "filter",
                            "path": "filter_tags"
                        },
                        {
                            "type": "vector",
                            "numDimensions": 3072,
                            "path": "vector",
                            "similarity": "cosine"
                        }
                    ]
                },
                type="vectorSearch",
                name="semantic-search"
            ),
            "keyword-search": SearchIndexModel(
                definition={
                    "mappings": {
                        "dynamic": False,
                        "fields": {
                            "filter_tags": [{
                                "type": "string",
                                "analyzer": "lucene.keyword"
                            }],
                            "keywords": [{
                                "type": "string",
                                "analyzer": "lucene.standard",
                                "searchAnalyzer": "lucene.standard",
                            }],
                            "content": [{
                                "type": "string",
                                "analyzer": "lucene.standard",
                                "searchAnalyzer": "lucene.standard",
                            }],
                        }
                    }
                },
                type="search",
                name="keyword-search"
            )
        }
        
    def __connection_manager(self,ops: str = 'fetch') -> None:
        logger.debug(f"Managing database connection. Operation: {ops}")
        if ops == 'fetch':
            if len(self.active_db_connection) > 0:
                self.client = self.active_db_connection.pop()

            else:
                self.client = MongoClient(self.db_uri)

            if self.client.admin.command("ping")['ok'] == 1:
                self.active_db_connection.append(self.client)
                logger.info("Successfully established database connection")
            else:
                self.client.close()
                raise Exception("MongoClient: Connection Error")

            self.db = self.client[self.db_name]
            self.collection = self.db[self.collection_name]

        elif ops == 'close':
            self.__create_index()
            if len(self.active_db_connection) > 0:
                self.client = self.active_db_connection.pop()
                self.client.close()
            else:
                raise Exception("MongoClient: Connection Error")

    # @weave.op()
    def __create_index(self,) -> None:
        vectordb_index_status = [index['name'] for index in self.collection.list_search_indexes() 
                                 if index['name'] in ['semantic-search','keyword-search']]
        
        indexes = []
        for idx in ['semantic-search','keyword-search']:
            if idx not in vectordb_index_status:
                indexes.append(self.collection_index[idx])
        
        if len(indexes) > 0:
            self.collection.create_search_indexes(models=indexes)
            
    # @weave.op()
    def __data_extraction(self,data: str):
        logger.debug("Starting data extraction process")
        try:
            output = self.client_openai.beta.chat.completions.parse(
                messages=[
                    {"role": "system",
                    "content": """
                    Role: Document Data miner, and Extraction Master agent
                    Task: Your task is to analyze the data, derive insights, and mine sections from data that refer to same base section.
                    You are required to extract subsets of data as it is from the provided text, along with that generate a list of metadata tags for
                    each extracted subset stating what it represents.
                    These tags, and chuncks should convey the brief information so that they can be used for extraction of information
                    if user ask some specific query. Also, the text chunk (present at the last sections of the provided text) that is incomplete wrt content 
                    being sliced or lacking meaning should be ignored, and stated as carry_forward. Each generated subset must have decent information related to those
                    tags, and each subset should have complete information and not just single qna.
                    
                    Instructions:
                    1) Don't use generate illogical and incomplete data, and keywords
                    2) Every subset of text extracted must be statefull, and keywords
                    3) Data for each subset must be extracted from provided text without modification, and paraphrasinh
                    4) JSON response must be in provided format
                    5) carry_forward should be prefectly extracted without modification to carry_forwarded to next subset
                    6) If carry_foward is not present, simple state it as ""
                    """},
                    {"role": "user", "content": f"Provided context: {data}\n"}
                ],
                model=self.model,
                temperature=0,
                response_format=self.ResponseModel
            )
            output = output.choices[0].message.parsed
            logger.info("Data extraction completed successfully")
            return output

        except openai.error.RateLimitError as e:
            logger.error(f"OpenAI rate limit error: {e}")
            return openai.error.RateLimitError(e)
        except openai.error.Timeout as e:
            logger.error(f"OpenAI timeout error: {e}")
            return openai.error.Timeout(e)

    @staticmethod
    def data_preprocessing(data: List[str]):
        aux_output = []
        
        for tmp in data:
            try:
                temp_data = re.sub("\n+","\n",tmp)
                temp_data = re.sub("\t+","\t",temp_data)

                aux_output.append(temp_data)
            except Exception as e:
                logger.info(msg=f"Error: {e}")
                aux_output.append(tmp)

        return aux_output

    # @weave.op()
    @staticmethod
    def create_embeddings(model: str, client_openai, document: str, dimension: int = 3072):
        try:
            response = client_openai.embeddings.create(
                input=document,
                model=model,
                dimensions=dimension
            )
            return response.data[0].embedding
        
        except openai.RateLimitError as e:
            return openai.RateLimitError(e)
        except openai.APITimeoutError as e:
            return openai.APITimeoutError(e)

    # @weave.op()
    def __process_extraction(self, data: List[str]):
        logger.info("Starting batch data extraction process")
        carry_data = None
        results = []

        preprocessed_data = DataExtraction.data_preprocessing(data)

        for data in preprocessed_data:
            logger.debug(f"Processing data chunk of length: {len(data)}")
            temp_data = data

            if carry_data is not None:
                temp_data += carry_data
                carry_data = None
            
            try:
                temp_result = self.ResponseModel()
                temp_result = self.__data_extraction(temp_data)
            
            except openai.error.RateLimitError as e:
                logger.error(msg="RateLimit Error")
                
                time.sleep(2*self.retry_factor['segment_data_extraction']+1)
                self.retry_factor['segment_data_extraction'] += 1

                temp_result = self.__data_extraction(temp_data)

            except openai.error.Timeout as e:
                logger.error(msg="Timeout Error")
                time.sleep(2)
                temp_result = self.__data_extraction(temp_data)

            print(temp_result)
            if len(temp_result.carry_forward) > 0:
                carry_data = temp_result.carry_forward

            results.extend([{"data":df.data,"keywords":df.keywords} for df in temp_result.result])

        logger.info(f"Batch processing completed. Extracted {len(results)} segments")
        return results
  
    # @weave.op()
    def ingestion_pipeline(self, tag: str, data: List[str]):
        logger.info(f"Starting ingestion pipeline for tag: {tag}")
        tmp_data = self.__process_extraction(data)
        ingestion_data = []

        for idx, tmp in enumerate(tmp_data):
            logger.debug(f"Processing item {idx+1}/{len(tmp_data)}")
            try:
                ingestion_data.append({
                    "vector": DataExtraction.create_embeddings(
                        model=self.embedding_model,
                        client_openai=self.client_openai,
                        document=tmp["data"]
                    ),
                    "keywords": ",".join(tmp["keywords"]),
                    "content": tmp["data"],
                    "filter_tags": tag
                })
            except openai.RateLimitError as e:
                time.sleep(2*self.retry_factor['embedding_generation']+1)
                self.retry_factor['embedding_generation'] += 1

            except openai.APITimeoutError as e:
                logger.error(msg="Timeout Error")
                time.sleep(2)
            except Exception as e:
                print(tmp)
                logger.error(msg=f"Error: {e}")

        logger.info(f"Ingestion pipeline completed. Processed {len(ingestion_data)} items")
        return ingestion_data
    # @weave.op()
    def vectordb_data_dump(self, data):
        logger.info(f"Starting vector database dump of {len(data)} items")
        try:
            self.__connection_manager(ops='fetch')
            self.collection.insert_many(data)
            self.__connection_manager(ops='close')
            logger.info("Vector database dump completed successfully")
            return {"status": 200, "msg": "success"}
        except Exception as e:
            logger.error(f"Vector database dump failed: {e}")
            return {"status": 500, "msg": str(e)}

def process(collection: str, path: str = "", tag: str = ""):
    logger.info(f"Processing PDF file: {path} for collection: {collection} with tag: {tag}")
    text = []
    try:
        pdf_document = fitz.open(path)
        for page_num, page in enumerate(pdf_document):
            logger.debug(f"Extracting text from page {page_num+1}")
            text.append(page.get_text())
        pdf_document.close()
        logger.info(f"Successfully extracted text from {len(text)} pages")
    except Exception as e:
        logger.error(f"Error processing PDF: {e}")
        raise

    pipeline = DataExtraction(collection=collection)
    ingestion_data = pipeline.ingestion_pipeline(tag=tag,data=text)
    result = pipeline.vectordb_data_dump(ingestion_data)
    
    if result['status'] == 200:
        logger.info("Process completed successfully")
        return {"status": "success"}
    else:
        logger.error(f"Process failed: {result.get('msg', 'Unknown error')}")
        return {"status": "Error"}

def handler(file, client_id, tag, delete_old):
    logger.info(f"Handling request for client_id: {client_id}, tag: {tag}, delete_old: {delete_old}")
    if file is None:
        logger.error("Empty file field")
        return {
            'statusCode': 400,
            'body': json.dumps({"error": "File not sent"})
        }
    
    if delete_old:
        try:
            client = MongoClient(config['database']['mongodb']['host'])
            db = client[config['database']['mongodb']['db_name']]
            collection = db[client_id]
            collection.delete_many({"filter_tags": tag})
            client.close()
        except Exception as e:
            logger.error(f"Error deleting old records: {e}")
            return {
                'statusCode': 500,
                'body': json.dumps({"error": "Error deleting old records"})
            }
    
    process(collection=client_id, path=file.name, tag=tag)
    
    try:
        return {
            "statusCode": 200,
            "body": json.dumps({
                "msg": "Ingestion Completed Successfully!"
            })
        }
    except Exception as e:
        logger.error(f"Error ingesting file: {e}")
        return {
            'statusCode': 500,
            'body': json.dumps({"error": "Error ingesting file"})
        }

demo = gr.Interface(
    fn=handler,
    inputs=[
        gr.File(label="Upload PDF", file_types=[".pdf"]),
        gr.Textbox(label="Collection Name", placeholder="Enter your collection name"),
        gr.Textbox(label="Tag", placeholder="Enter your tag here"),
        gr.Checkbox(label="Delete old version", value=False),
    ],
    outputs=gr.JSON(label="Response"),
    title="FAQ Ingestion",
    description="Upload a PDF file to be ingested into database for querying."
)
demo.launch()