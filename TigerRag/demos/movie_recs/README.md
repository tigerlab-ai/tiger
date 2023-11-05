# Movie Recommendation with EBR, RAG, and GAR

This demo project uses embeddings-based retrieval (EBR), retrieval-augmented generation (RAG), and generation-augmented retrieval (GAR) to recommend movies based on a given query.

## Prerequisites

- Python 3.9 or higher

## Setup

### Method 1: Install the Python Package

1. **Install the `tigerrag` python package**:

```
pip install git+https://github.com/tigerlab-ai/tiger.git#subdirectory=TigerRag
```

### Method 2: Git Clone the Repo

1. **Set Up a Virtual Environment** (optional but recommended):

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Clone the Repository**:

   ```bash
   git clone git@github.com:tigerrag-ai/tiger.git
   cd TigerRag/tigerrag
   pip install .
   ```

3. **Set Up OpenAI API Key**:

   Before running the script, you need to set your OpenAI API key as an environment variable. Replace `YOUR_API_KEY` with your actual API key:

   ```bash
   export OPENAI_API_KEY=YOUR_API_KEY
   ```

   Note: Ensure you have a valid OpenAI API key and have access to the desired models (e.g., 'text-davinci-003').

4. **Data Setup**:

   Ensure the data files (`labels.csv`, `movies.csv`, and `queries.csv`) are present in the `data` directory relative to the script.

5. **Run Demo**:

   Navigate to the directory containing the script and execute:

   Example: Movie Recommendations Example (EBR, RAG, GAR): 
   ```bash
   cd demos/movie_recs
   python demo_ebr.py
   python demo_rag.py
   python demo_gar.py
   ```

Refer to the demo files for explanation of what each demo does

- Always ensure you have the required credits/budget for OpenAI API calls.
- Adjust the `max_tokens` parameter in API calls as needed, based on the desired length of the generated text and budget considerations.
