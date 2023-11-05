from tigerrag.rag.base.loaders import TigerRAGDataFrameLoader
from tigerrag.rag.base.models import EmbeddingModel, TigerRAGEmbeddingModel
from tigerrag.rag.ebr.similarity_search import FaissFlatL2Search

from demo_utils import calculate_averaged_recall, calculate_single_recall

"""
The provided Python script leverages the TigerRAG library and FAISS for an embeddings-based retrieval (EBR) system. 
It loads data from CSV files, including labels, movie descriptions, and queries. 
The code initializes a BERT model to create embeddings for movie descriptions and builds a FAISS index for efficient similarity search. 
It then performs a sample query, finds the top 5 movies related to that query, calculates the recall metric, 
and calculates the average recall over all queries in the dataset, presenting the results. 
The script demonstrates how to use embeddings and similarity search for information retrieval tasks.
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

# Calculate recall metric
expected_movie_ids = labels_df[labels_df["query_id"] == query_id]["product_id"].values
recall = calculate_single_recall(retrieved_movie_ids, expected_movie_ids)

# Print output
print(f"Top 5 movies for query '{query_text}':")
print(retrieved_movies[["ID", "Title"]])
print(f"Recall: {recall}")


# Get an average recall metric over all queries
k = 5
query_ids = queries_df["query_id"]
retrieved_movie_ids_l = []
expected_movie_ids_l = []

# We calculate the recall over all queries in query_df
for query_id in query_ids:
    query_text = queries_df[queries_df["query_id"] == query_id]["query"].values[0]
    query_embedding = trag_bert_model.get_embedding_from_text(query_text)
    _, labels = faiss_flat_l2_search.search(query_embedding, k=k)
    retrieved_movie_ids = labels[0]
    expected_movie_ids = labels_df[labels_df["query_id"] == query_id]["product_id"].values
    retrieved_movie_ids_l.append(retrieved_movie_ids)
    expected_movie_ids_l.append(expected_movie_ids)


average_recall = calculate_averaged_recall(retrieved_movie_ids_l, expected_movie_ids_l)
print(f"Average Recall@{k} for the EBR system: {average_recall:.4f}")
