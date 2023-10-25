import pandas as pd
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel
import faiss
import openai
import os

# openai.api_key = ""
# openai_api_key = os.environ.get('OPENAI_API_KEY')
# if not openai_api_key:
#     raise ValueError("The OPENAI_API_KEY environment variable is not set!")

# Load data
labels_df = pd.read_csv('labels.csv')
movies_df = pd.read_csv('movies.csv')
queries_df = pd.read_csv('queries.csv')

# Initialize BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def get_embedding(text):
    """Returns the BERT embedding for a given text."""
    tokens = tokenizer(text, return_tensors='pt', padding='max_length', max_length=100, truncation=True)
    with torch.no_grad():
        embeddings = model(**tokens).last_hidden_state
    return embeddings.mean(1).numpy()

# Get embeddings for movies
movies_df['embedding'] = movies_df['Description'].apply(get_embedding)
movie_embeddings = np.vstack(movies_df['embedding'])

# Build FAISS index
d = movie_embeddings.shape[1]
index = faiss.IndexFlatL2(d)
index.add(movie_embeddings)

def get_relevant_movies(query_embedding):
    """Returns the movies that are most similar to the given query embedding using FAISS."""
    D, I = index.search(query_embedding.reshape(1, -1), k=5)  # Retrieve top 5 similar movies
    relevant_movie_ids = I[0]
    return movies_df.iloc[relevant_movie_ids]

# Test with a query
query_id = 1
query_text = queries_df[queries_df['query_id'] == query_id]['query'].values[0]
query_embedding = get_embedding(query_text)
relevant_movies = get_relevant_movies(query_embedding)

print(f"Top 5 movies for query '{query_text}':")
print(relevant_movies[['ID', 'Title']])

def get_relevant_movie_ids(query_embedding, k):
    D, I = index.search(query_embedding.reshape(1, -1), k)
    return I[0]

def calculate_recall_for_query(query_id, k):
    query_text = queries_df[queries_df['query_id'] == query_id]['query'].values[0]
    query_embedding = get_embedding(query_text)
    retrieved_movie_ids = set(get_relevant_movie_ids(query_embedding, k))
    
    actual_relevant_ids = set(labels_df[labels_df['query_id'] == query_id]['product_id'].values)
    
    intersect_count = len(retrieved_movie_ids.intersection(actual_relevant_ids))
    total_relevant = len(actual_relevant_ids)
    
    recall = intersect_count / total_relevant if total_relevant > 0 else 0
    return recall

# Evaluate Recall@k
k = 5
all_recalls = [calculate_recall_for_query(query_id, k) for query_id in queries_df['query_id'].values]

average_recall = np.mean(all_recalls)
print(f"Average Recall@{k} for the EBR system: {average_recall:.4f}")

# RAG
def generate_answer_with_rag_gpt3(query, openai_text_model):
    # Retrieval
    query_embedding = get_embedding(query)
    relevant_movies = get_relevant_movies(query_embedding)
    
    # Combine movie descriptions into a single string
    descriptions = '. '.join(relevant_movies['Description'].tolist())
    prompt = f"Query: {query}. Top retrieved movies: {descriptions}. Provide a summary or answer:"
    
    # Generation using GPT-3
    response = openai.Completion.create(engine=openai_text_model, prompt=prompt, max_tokens=150)
    answer = response.choices[0].text.strip()
    
    return answer

# Example usage
print("The following are RAG output for search query: sci-fi alien invasion movie")
print(generate_answer_with_rag_gpt3("sci-fi alien invasion movie", 'text-davinci-003'))

# GAR
def retrieve_with_gar_gpt3(query, openai_text_model):
    # Generation using GPT-3
    prompt = f"Expand on the query: {query}"
    response = openai.Completion.create(engine=openai_text_model, prompt=prompt, max_tokens=100)
    augmented_query = response.choices[0].text.strip()
    
    # Retrieval
    query_embedding = get_embedding(augmented_query)
    relevant_movies = get_relevant_movies(query_embedding)
    
    return relevant_movies[['ID', 'Title']]

# Example usage
print("The following are GAR output for search query: sci-fi alien invasion movie")
print(retrieve_with_gar_gpt3("sci-fi alien invasion movie", 'text-davinci-003'))
