from tigerrag.rag.base.loaders import TigerRAGDataFrameLoader
from tigerrag.rag.base.models import EmbeddingModel, TigerRAGEmbeddingModel
from tigerrag.rag.ebr.similarity_search import FaissFlatL2Search
from tigerrag.rag.gar.query_augmenters import OpenAIQueryAugmenter

from demo_utils import calculate_single_recall

"""
Evaluating an Embeddings-Based Retrieval (EBR) System using TigerRAG and FAISS Libraries. 

This script demonstrates the use of the TigerRAG and FAISS libraries to build and evaluate an EBR system. 
It loads data from CSV files, including labels, movie descriptions, and queries, and initializes a BERT-based model for embeddings. 
The movie embeddings are used to build a FAISS index for efficient similarity search.It also showcases the concept of 
Generation-Augmented Retrieval (GAR) using OpenAI's text model. A query is augmented using OpenAI's generative capabilities, 
and the augmented query is used to retrieve the top 5 movies related to it. Recall metrics are calculated for the individual query 
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

# Augment Query using OpenAI (Generation-Augmented Retrieval)
openai_text_model = "text-davinci-003"

openai_generative_query_augmenter = OpenAIQueryAugmenter(openai_text_model)
query_id = 1
query_text = queries_df[queries_df["query_id"] == query_id]["query"].values[0]
augmented_query = openai_generative_query_augmenter.get_augmented_query(query_text)
query_embedding = trag_bert_model.get_embedding_from_text(augmented_query)
_, labels = faiss_flat_l2_search.search(query_embedding, k=5)
retrieved_movie_ids = labels[0]
retrieved_movies = movies_df.iloc[retrieved_movie_ids]

# Print output
print(f"Top 5 movies for query '{query_text}':")
print(retrieved_movies[["ID", "Title"]])

# Calculate recall metric
expected_movie_ids = labels_df[labels_df["query_id"] == query_id]["product_id"].values
recall = calculate_single_recall(retrieved_movie_ids, expected_movie_ids)
print(retrieved_movie_ids, expected_movie_ids)
print(f"Recall: {recall}")
