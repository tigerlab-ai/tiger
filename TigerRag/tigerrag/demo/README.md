# Movie Recommendation with EBR, RAG, and GAR

This demo project uses embeddings-based retrieval (EBR), retrieval-augmented generation (RAG), and generation-augmented retrieval (GAR) to recommend movies based on a given query.

## Prerequisites

- Python 3.6 or higher
- Required libraries (listed in `requirements.txt`)

## Setup

1. **Clone the Repository**:

   ```bash
   git clone git@github.com:tigerrag-ai/tiger.git
   cd TigerRag/tigerrag
   ```

2. **Set Up a Virtual Environment** (optional but recommended):

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Required Libraries**:

   You can install the required Python libraries using the following command:
   
   ```bash
   pip install -r requirements.txt
   ```

4. **Set Up OpenAI API Key**:

   Before running the script, you need to set your OpenAI API key as an environment variable. Replace `YOUR_API_KEY` with your actual API key:

   ```bash
   export OPENAI_API_KEY=YOUR_API_KEY
   ```

   Note: Ensure you have a valid OpenAI API key and have access to the desired models (e.g., 'text-davinci-003').

5. **Data Setup**:

   Ensure the data files (`labels.csv`, `movies.csv`, and `queries.csv`) are present in the `demo` directory relative to the script.

## Running the Script

Navigate to the directory containing the script and execute:

```bash
python movie_recs/demo.py
```

The script will:
- Initialize the embeddings for movies using BERT.
- Build a FAISS index for movie embeddings.
- Display the top 5 movies for a given query using EBR.
- Calculate and display the average Recall@5 for the EBR system.
- Demonstrate the RAG approach by retrieving relevant movies and generating a summary using GPT-3.
- Demonstrate the GAR approach by augmenting the query using GPT-3 and then retrieving relevant movies.

## Expected Outputs

1. Top 5 movies for a specified query using EBR.
2. Average Recall@5 for the EBR system.
3. A summary or answer for the RAG approach using GPT-3.
4. Top movies for the GAR approach using GPT-3.

## Notes

- Always ensure you have the required credits/budget for OpenAI API calls.
- Adjust the `max_tokens` parameter in API calls as needed, based on the desired length of the generated text and budget considerations.
