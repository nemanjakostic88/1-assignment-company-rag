# 1. Setup Environment
## Create virtual environment
python -m venv venv

source venv/bin/activate  # or `venv\Scripts\activate` on Windows

## Install all dependencies
pip install -r requirements.txt

## Configure API Keys
Copy .env.sample to .env file in the root directory and set variables

OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

MONGO_DB_URL=

LANGCHAIN_TRACING_V2=true

LANGCHAIN_API_KEY=lsv2_pt_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

LANGCHAIN_PROJECT=rag-company

# 2. Enter data
python ingestion.py
# 3. Ask questions
python generation.py
# 4. See Improvement from filtering
python precision_delta.py


