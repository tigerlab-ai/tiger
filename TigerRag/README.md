# RAG: Retrieval Augmented Generation

**Description**:
RAG, or Retrieval Augmented Generation, combines the power of retrieval-based models with that of transformer-based generation models. The aim is to produce coherent and precise answers by first retrieving relevant content from a vast corpus and then using a Large Language Model (LLM) to generate a response based on the retrieved content.

**Workflow**:
1. **Retrieval**: Search the corpus to retrieve potential passages or documents relevant to the given query.
2. **LLM Generation**: Use the retrieved content as context to generate a detailed and coherent response using a transformer-based model.

**Advantages**:
- Broad coverage from retrieval with coherent generation capabilities of LLMs.
- Scalability with the size of the corpus.
- Potential access to more recent information than what's embedded in pre-trained models.

---

# GAR: Generation Augmented Retrieval

**Description**:
GAR, or Generation Augmented Retrieval, inverts the workflow of RAG. Instead of using retrieval to aid generation, GAR employs generation to assist the retrieval process. The idea is to generate potential queries or augmentations to the initial question, which can then be used to better retrieve relevant passages or documents.

**Workflow**:
1. **LLM Generation**: Enhance or expand the initial query using a transformer-based model to generate potential queries or keywords.
2. **Retrieval**: Use the enhanced or expanded queries to search the corpus and retrieve relevant content.

**Advantages**:
- More precise retrieval by understanding the nuance of a query.
- Overcomes strict keyword-based limitations of traditional search.
- Potential to discover deeply relevant content that might be missed with direct queries.

---

**Dependencies**:
- Retrieval System (link to external datasets)
- Transformer-based Large Language Model (e.g., GPT-4, BERT)

**Setup & Installation**:
```
pip install -r tigerrag/requirements.txt
```
Demo:
```
cd tigerrag/demo/movie_recs
python demo.py
```

**Usage**:
- rag
- gar


