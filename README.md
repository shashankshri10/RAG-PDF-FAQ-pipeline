# FAQ Chatbot Pipeline Documentation

## 1. Ingestion Pipeline

### Overview
The ingestion pipeline processes PDF documents into searchable chunks with semantic embeddings and keywords. 

### Functional Flow
1. PDF Upload → Text Extraction
2. Text Preprocessing & Chunking
3. LLM-based Data Extraction
4. Embedding Generation
5. MongoDB Storage with Dual Indexing

### Component Breakdown

#### `process(collection, path, tag)`
- Entry point for PDF processing
- Handles PDF file reading and text extraction
- Initializes pipeline components
- Returns success/error status

#### `DataExtraction` Class
1. `__data_extraction(data)`
   - Uses LLM to extract meaningful chunks
   - Generates keywords for each chunk
   - Handles content carry-forward between chunks
   - Returns structured data using `ResponseModel`

2. `data_preprocessing(data)`
   - Cleans text data
   - Removes excessive whitespace/newlines
   - Normalizes formatting

3. `create_embeddings(model, document)`
   - Generates vector embeddings using OpenAI API
   - Handles rate limiting and timeouts
   - Returns embedding vector

4. `ingestion_pipeline(tag, data)`
   - Orchestrates extraction and embedding generation
   - Combines metadata with embeddings
   - Prepares data for database insertion

5. `vectordb_data_dump(data)`
   - Manages MongoDB connections
   - Creates search indexes
   - Handles batch insertions

## 2. Query Pipeline

### Overview
The query pipeline handles user questions through hybrid search and LLM-based response generation.

### Functional Flow
1. Query Input → Query Expansion
2. Parallel Search (Semantic + Keyword)
3. Result Reranking
4. Context-based Response Generation

### Component Breakdown

#### `process_query(client_id, query, model, tag)`
- Entry point for query processing
- Orchestrates query expansion and search
- Returns formatted response

#### `query_expansion(text)`
- Generates semantic variations of query
- Uses specialized LLM prompt
- Returns list of related queries

#### `handler(query, queryV2, client_id, tag)`
1. `create_embeddings(document)`
   - Generates query embeddings
   - Matches ingestion embeddings format

2. Vector Search
   - Uses MongoDB's `$vectorSearch`
   - Filters by tag
   - Returns top semantic matches

3. Keyword Search
   - Uses MongoDB's text search
   - Implements fuzzy matching
   - Returns BM25-ranked results

#### `weighted_reciprocal_rank(doc_lists)`
- Combines results from both search methods
- Applies configurable weights
- Returns reranked document list

# Local Setup Guide

Here's the documentation breakdown:

# FAQ Chatbot Pipeline Documentation

## 1. Ingestion Pipeline

### Overview
The ingestion pipeline processes PDF documents into searchable chunks with semantic embeddings and keywords. 

### Functional Flow
1. PDF Upload → Text Extraction
2. Text Preprocessing & Chunking
3. LLM-based Data Extraction
4. Embedding Generation
5. MongoDB Storage with Dual Indexing

### Component Breakdown

#### `process(collection, path, tag)`
- Entry point for PDF processing
- Handles PDF file reading and text extraction
- Initializes pipeline components
- Returns success/error status

#### `DataExtraction` Class
1. `__data_extraction(data)`
   - Uses LLM to extract meaningful chunks
   - Generates keywords for each chunk
   - Handles content carry-forward between chunks
   - Returns structured data using `ResponseModel`

2. `data_preprocessing(data)`
   - Cleans text data
   - Removes excessive whitespace/newlines
   - Normalizes formatting

3. `create_embeddings(model, document)`
   - Generates vector embeddings using OpenAI API
   - Handles rate limiting and timeouts
   - Returns embedding vector

4. `ingestion_pipeline(tag, data)`
   - Orchestrates extraction and embedding generation
   - Combines metadata with embeddings
   - Prepares data for database insertion

5. `vectordb_data_dump(data)`
   - Manages MongoDB connections
   - Creates search indexes
   - Handles batch insertions

## 2. Query Pipeline

### Overview
The query pipeline handles user questions through hybrid search and LLM-based response generation.

### Functional Flow
1. Query Input → Query Expansion
2. Parallel Search (Semantic + Keyword)
3. Result Reranking
4. Context-based Response Generation

### Component Breakdown

#### `process_query(client_id, query, model, tag)`
- Entry point for query processing
- Orchestrates query expansion and search
- Returns formatted response

#### `query_expansion(text)`
- Generates semantic variations of query
- Uses specialized LLM prompt
- Returns list of related queries

#### `handler(query, queryV2, client_id, tag)`
1. `create_embeddings(document)`
   - Generates query embeddings
   - Matches ingestion embeddings format

2. Vector Search
   - Uses MongoDB's `$vectorSearch`
   - Filters by tag
   - Returns top semantic matches

3. Keyword Search
   - Uses MongoDB's text search
   - Implements fuzzy matching
   - Returns BM25-ranked results

#### `weighted_reciprocal_rank(doc_lists)`
- Combines results from both search methods
- Applies configurable weights
- Returns reranked document list

# Local Setup Guide

### Prerequisites
- Python 3.8+
- MongoDB instance
- OpenAI API key

### Configuration Setup
1. Create `config.yaml`:
```yaml
openai:
  extraction_model: "gpt-4"
  embedding_model: "text-embedding-3-large"

database:
  mongodb:
    host: "your_mongodb_uri"
    db_name: "your_db_name"
```

2. Create `.env.shared`:
```env
OPENAI_API_KEY=your_api_key_here
```

### Installation & Running

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Start Ingestion Interface:
```bash
python Faq_Ingestion/app.py
```
- Access at `http://localhost:7860`
- Upload PDF, set collection name and tag
- Monitor console for ingestion progress

3. Start Query Interface:
```bash
python FaqV2Tool/app.py
```
- Access at `http://localhost:7861`
- Enter collection name, query, and tag
- View structured responses with source documents

### Notes
- Ensure MongoDB is running and accessible
- Monitor API rate limits with OpenAI
- Check logs for any processing errors
- Both interfaces can run simultaneously

This setup allows for local development and testing of the complete FAQ chatbot pipeline.
### Prerequisites
- Python 3.8+
- MongoDB instance
- OpenAI API key

### Configuration Setup
1. Create `config.yaml`:
```yaml
openai:
  extraction_model: "gpt-4"
  embedding_model: "text-embedding-3-large"

database:
  mongodb:
    host: "your_mongodb_uri"
    db_name: "your_db_name"
```

2. Create `.env.shared`:
```env
OPENAI_API_KEY=your_api_key_here
```

### Installation & Running

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Start Ingestion Interface:
```bash
python Faq_Ingestion/app.py
```
- Access at `http://localhost:7860`
- Upload PDF, set collection name and tag
- Monitor console for ingestion progress

3. Start Query Interface:
```bash
python FaqV2Tool/app.py
```
- Access at `http://localhost:7861`
- Enter collection name, query, and tag
- View structured responses with source documents

### Notes
- Ensure MongoDB is running and accessible
- Monitor API rate limits with OpenAI
- Check logs for any processing errors
- Both interfaces can run simultaneously

This setup allows for local development and testing of the complete FAQ chatbot pipeline.
