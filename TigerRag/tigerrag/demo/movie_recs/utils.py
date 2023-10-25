import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
import faiss
import openai
import pkg_resources

labels_path = pkg_resources.resource_filename('tigerrag', 'demo/movie_recs/labels.csv')
movies_path = pkg_resources.resource_filename('tigerrag', 'demo/movie_recs/movies.csv')
queries_path = pkg_resources.resource_filename('tigerrag', 'demo/movie_recs/queries.csv')

# Set up openai api key
# openai.api_key = ""
# openai_api_key = os.environ.get('OPENAI_API_KEY')
# if not openai_api_key:
#     raise ValueError("The OPENAI_API_KEY environment variable is not set!")

# Load data
labels_df = pd.read_csv(labels_path)
movies_df = pd.read_csv(movies_path)
queries_df = pd.read_csv(queries_path)

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

def get_relevant_movies(query_embedding, k=5):
    """Returns the movies that are most similar to the given query embedding using FAISS."""
    D, I = index.search(query_embedding.reshape(1, -1), k)
    relevant_movie_ids = I[0]
    return movies_df.iloc[relevant_movie_ids]

def generate_answer_with_rag_gpt3(query, openai_text_model, k=5):
    query_embedding = get_embedding(query)
    relevant_movies = get_relevant_movies(query_embedding, k)
    descriptions = '. '.join(relevant_movies['Description'].tolist())
    prompt = f"Query: {query}. Top retrieved movies: {descriptions}. Provide a summary or answer:"
    response = openai.Completion.create(engine=openai_text_model, prompt=prompt, max_tokens=150)
    answer = response.choices[0].text.strip()
    return answer

def retrieve_with_gar_gpt3(query, openai_text_model, k=5):
    prompt = f"Expand on the query: {query}"
    response = openai.Completion.create(engine=openai_text_model, prompt=prompt, max_tokens=100)
    augmented_query = response.choices[0].text.strip()
    query_embedding = get_embedding(augmented_query)
    relevant_movies = get_relevant_movies(query_embedding, k)
    return relevant_movies[['ID', 'Title']]
