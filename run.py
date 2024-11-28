import streamlit as st
import numpy as np
import faiss
import re
from sentence_transformers import SentenceTransformer
from utils import load_chunks, load_embeddings, initialize_faiss_index, load_summarizers, summarize_text
import torch

# Load embedding model
embedding_model_name = "all-MiniLM-L6-v2"  # You can choose an appropriate model here
model = SentenceTransformer(embedding_model_name)

# Load your embeddings and chunks
chunks = load_chunks('E:\\Medical Rag\\final_chunks.txt')
chunk_embeddings = load_embeddings('E:\\Medical Rag\\chunk_embeddings.npy')

# Initialize FAISS index
index = initialize_faiss_index(chunk_embeddings)

# Load summarization models
summarizers = load_summarizers()

# Text preprocessing function
def preprocess_text(text):
    """
    Cleans up the input text by removing special characters, extra whitespace, and repetitive words.
    """
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and numbers
    text = re.sub(r"[^a-zA-Z\s]", '', text)
    # Remove extra spaces
    text = re.sub(r"\s+", ' ', text).strip()
    # Remove repeated words
    text = re.sub(r'\b(\w+)\s+\1\b', r'\1', text)  # Remove repeated consecutive words
    return text

# Summarization function with enhancements
def summarize_text(summarizer, text):
    """
    Generates a summary for a given text using the provided summarizer model and tokenizer.
    """
    model, tokenizer = summarizer
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Tokenize and generate summary
    inputs = tokenizer.encode(text, return_tensors="pt", max_length=1024, truncation=True).to(device)
    summary_ids = model.generate(
        inputs, 
        max_length=250,  # Adjust length based on input complexity
        min_length=100,   # Min length for concise summaries
        length_penalty=2.0, 
        num_beams=4,
        repetition_penalty=1.2,  # Discourage repetitive phrases
        early_stopping=True
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Streamlit app interface
st.title("SageAI Your own Medical chatbot")
st.subheader("Your own Medical chatbot")

# User input
user_input = st.text_input("Ask a medical question:")

if user_input:
    # Preprocess user input
    user_input_cleaned = preprocess_text(user_input.strip())

    # Generate embedding for the user input
    query_embedding = model.encode([user_input_cleaned])

    # Search the FAISS index for relevant chunks
    distances, indices = index.search(query_embedding, k=5)  # Retrieve top 5 results
    responses = [chunks[i] for i in indices[0]]

    # Collect all responses into a single string
    all_responses = " ".join(preprocess_text(response) for response in responses)

    # Display summarized responses
    st.write("Summarized Medical Responses:")
    
    # Now summarize the combined responses
    for name, summarizer in summarizers.items():
        combined_summary = summarize_text(summarizer, all_responses)  # Summarize the entire collected response
        st.write(combined_summary)
