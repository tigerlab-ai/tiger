# Document Q&A with EBR, RAG, and GAR

This demo project uses embeddings-based retrieval (EBR), retrieval-augmented generation (RAG), and generation-augmented retrieval (GAR) to query/search over multiple documents/websites. This demo is built using LangChain framework.

## Prerequisites

- Python 3.6 or higher
- Required libraries (listed in `requirements.txt`)

## Setup

1. **Clone the Repository**:

   ```bash
   git clone git@github.com:tigerrag-ai/tiger.git
   cd TigerRag/demos/langchin
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


## Running the Script

Navigate to the directory containing the script and execute:

```bash
python demo.py
```

The script will:
- Initialize the embeddings for documents using GPT4ALL.
- Display the retrieved passage using EBR.
- Demonstrate the RAG approach by retrieving passage and generating a summary using GPT-3.
- Demonstrate the GAR approach by augmenting the query using GPT-3 and then retrieving the passage.

## Expected Outputs

1. Retrieved passage for a specified query using EBR.
3. A summary of retrieved passage for the RAG approach using GPT-3.
4. Retrieved passage for the GAR approach using GPT-3.

## Notes

- Always ensure you have the required credits/budget for OpenAI API calls.
- Adjust the `max_tokens` parameter in API calls as needed, based on the desired length of the generated text and budget considerations.
