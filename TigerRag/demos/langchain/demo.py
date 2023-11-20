import json
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import GPT4AllEmbeddings
import openai
import os
import sys

# Please set env var OPENAI_API_KEY for GAR and RAG.

# Sample usage:
#    python demo.py
#    python demo.py -number_of_run 4


def get_documents_embeddings(documents):
    # Load documents
    loader = WebBaseLoader(documents)

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=0)
    splits = text_splitter.split_documents(loader.load())

    # Embed and store splits
    vectorstore = Chroma.from_documents(
        documents=splits, embedding=GPT4AllEmbeddings())

    return vectorstore


# EBR
def ebr(question, vectorstore):
    # Perform similarity search
    docs = vectorstore.similarity_search(question)

    return docs[0]


# RAG
def generate_answer_with_rag_gpt3(question, context, openai_text_model):
    # Retrivel Augmented Generation
    prompt = f"Context: {context} Question: {question}. Provide a summary or answer:"

    # Generation using GPT-3
    response = openai.Completion.create(
        engine=openai_text_model, prompt=prompt, max_tokens=100)
    answer = response.choices[0].text.strip()

    return answer


# GAR
def generate_answer_with_gar_gpt3(question, context, openai_text_model, vectorstore):
    # Generation Augmented Retrieval
    prompt = f"Expand on the query: {question}"

    # Generation using GPT-3
    response = openai.Completion.create(
        engine=openai_text_model, prompt=prompt, max_tokens=100)
    augmented_query = response.choices[0].text.strip()

    # Retrieval
    answer = ebr(augmented_query, vectorstore)

    return answer


def is_intstring(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


def main():
    # Run the first 5 queries by default.
    num_of_run = 5
    args = sys.argv[1:]
    if len(args) == 2 and args[0] == '-number_of_run':
        if not is_intstring(args[1]):
            sys.exit("num_of_run argument must be integers. Exit.")
        num_of_run = int(args[1])

    if not os.environ.get('OPENAI_API_KEY'):
        raise ValueError("The OPENAI_API_KEY environment variable is not set!")

    with open('documents.jsonl') as f:
        data = json.load(f)

    documents = data["documents"]

    with open('queries.jsonl') as f:
        data = json.load(f)

    for index in range(min(num_of_run, len(data["queries"]))):
        question = data["queries"][index]
        # Example usage of EBR
        vectorstore = get_documents_embeddings(documents)
        print("The following is EBR output for question: "+question)
        retrieved_context = ebr(question, vectorstore)
        print(retrieved_context)

        # Example usage of RAG
        print("The following is RAG output for question: "+question)
        print(generate_answer_with_rag_gpt3(
            question, retrieved_context, 'text-davinci-003'))

        # Example usage of GAR
        print("The following is GAR output for question: "+question)
        print(generate_answer_with_gar_gpt3(
            question, retrieved_context, 'text-davinci-003', vectorstore))


if __name__ == "__main__":
    main()
