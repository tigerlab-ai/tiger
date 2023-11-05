from tigerrag.base.loaders import TigerRAGDataFrameLoader
from tigerrag.base.models import EmbeddingModel, TigerRAGEmbeddingModel
from tigerrag.ebr.similarity_search import FaissFlatL2Search
from tigerrag.rag.retrieval_augmenters import OpenAIRetrievalAugmenter

"""
This demo loads movie-related data from CSV files, uses BERT embeddings to generate movie descriptions, 
and constructs a FAISS index for efficient similarity search. It then retrieves the top 5 movie recommendations 
for a user query and prints their IDs and titles. Additionally, it demonstrates retrieval-augmented generation 
by utilizing OpenAI's "text-davinci-003" model to augment the retrieval results, providing a summary or answer 
based on the query and retrieved movie context. This script showcases a comprehensive workflow for content-based movie 
recommendation, retrieval, and generation, incorporating embeddings, similarity search, and an external text model for 
enhanced user experiences.
"""

# Load data
trag_df_loader = TigerRAGDataFrameLoader()
labels_df = trag_df_loader.from_csv("data/labels.csv")
movies_df = trag_df_loader.from_csv("data/movies.csv")
queries_df = trag_df_loader.from_csv("data/queries.csv")

# Initialize BERT tokenizer and model
trag_bert_model = TigerRAGEmbeddingModel(EmbeddingModel.BERT)


# Prepare Embeddings with Model
movie_embeddings = trag_bert_model.get_embedding_from_series(movies_df["Description"])
embedding_dim = movie_embeddings.shape[1]


# Build FAISS index
faiss_flat_l2_search = FaissFlatL2Search(embedding_dim)
faiss_flat_l2_search.add_to_index(movie_embeddings)

# Test EBR with a query
query_id = 1
query_text = queries_df[queries_df["query_id"] == query_id]["query"].values[0]
query_embedding = trag_bert_model.get_embedding_from_text(query_text)
_, labels = faiss_flat_l2_search.search(query_embedding, k=5)
retrieved_movie_ids = labels[0]
retrieved_movies = movies_df.iloc[retrieved_movie_ids]

# Print output
print(f"Top 5 movies for query '{query_text}':")
print(retrieved_movies[["ID", "Title"]])


# Improve Generaton given Retrieval by FAISS and Embedding Model (Retrieval-Augmented Generation)
openai_text_model = "text-davinci-003"
openai_generative_retrieval_augmenter = OpenAIRetrievalAugmenter(openai_text_model)
query_id = 1
query_text = queries_df[queries_df["query_id"] == query_id]["query"].values[0]
query_embedding = trag_bert_model.get_embedding_from_text(query_text)
_, labels = faiss_flat_l2_search.search(query_embedding, k=5)
retrieved_movies = movies_df.iloc[retrieved_movie_ids]

retrieval_text = f"Movie Descriptions: {', '.join(retrieved_movies['Description'].tolist())}"
prompt = f"Query: {query_text}. Top retrieved movies: {retrieval_text}. Provide a summary or answer:"

augmented_retrieval = openai_generative_retrieval_augmenter.get_augmented_retrieval(prompt)

print(augmented_retrieval)
