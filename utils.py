import numpy as np
import faiss
import torch
import sentencepiece
from transformers import (
    BartTokenizer, BartForConditionalGeneration,
    T5Tokenizer, T5ForConditionalGeneration,
    PegasusTokenizer, PegasusForConditionalGeneration
)

def load_chunks(file_path):
    chunks = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if line.startswith('Chunk'):
                chunk = next(file).strip()
                chunks.append(chunk)
    return chunks

def load_embeddings(file_path):
    return np.load(file_path)

def initialize_faiss_index(embeddings):
    index = faiss.IndexFlatL2(embeddings.shape[1])  # Use L2 distance
    index.add(embeddings)
    return index

def load_summarizers():
    device = "cuda" if torch.cuda.is_available() else "cpu"



    

    # Load the Pegasus model for summarization
    pegasus_model = PegasusForConditionalGeneration.from_pretrained("google/pegasus-xsum").to(device)
    pegasus_tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-xsum")

    return {
        
        "Pegasus": (pegasus_model, pegasus_tokenizer)
    
    }

def summarize_text(summarizer, text):
    model, tokenizer = summarizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    inputs = tokenizer.encode(text, return_tensors="pt", max_length=1600, truncation=True).to(device)
    summary_ids = model.generate(inputs, max_length=300, min_length=80, length_penalty=2.0, num_beams=4, early_stopping=True)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)
