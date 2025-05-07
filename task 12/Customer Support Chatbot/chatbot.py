import pickle
import os
import numpy as np
import re
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.document_loaders import PyPDFLoader
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter


embedding_model = SentenceTransformer("all-mpnet-base-v2")


vector_store = {}


def chunk_text(txt, chunk_size=500, chunk_overlap=250):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(txt)



def process_pdfs(pdf_folder="data"):
    allthe_chunks = []
    
    for filename in os.listdir(pdf_folder):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, filename)
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()

            for doc in docs:
                chunks = chunk_text(doc.page_content)  
                allthe_chunks.extend(chunks)

    return allthe_chunks



EMBEDDING_FILE = "embeddings.pkl"

def save_embeddings():
    """Saves vector_store to a file so embeddings don’t regenerate every time."""
    with open(EMBEDDING_FILE, "wb") as f:
        pickle.dump(vector_store, f)

def load_embeddings():
    """Loads embeddings from the file if it exists."""
    global vector_store
    if os.path.exists(EMBEDDING_FILE):
        with open(EMBEDDING_FILE, "rb") as f:
            vector_store = pickle.load(f)
        print(f"Loaded {len(vector_store)} stored embeddings from file.")
    else:
        print("No saved embeddings found. Generating new ones.")

def setup_memory_store():
    """Loads embeddings if they exist; otherwise, processes PDFs and generates new embeddings."""
    load_embeddings()
    
    if vector_store: 
        return
    
    texts = process_pdfs() 

    if not texts:
        print("No txt extracted from PDFs.")
        return

    print(f"Generating embeddings for {len(texts)} chunks...")
    
    embeddings = embedding_model.encode(texts).tolist()

 
    for i, txt in enumerate(texts):
        vector_store[i] = {"embedding": embeddings[i], "txt": txt}

   


setup_memory_store()



import re
import random


def make_conversational(response, query):
    """Adds a conversational tone but avoids redundant phrases."""
    
    response = clean_response(response)  

    if len(response.split()) < 10:  
        return response  

    if "return" in query.lower():
        intro = random.choice(["Sure! Here’s our return policy:", "Of course! Here's how returns work:", "You can return your product under these conditions:"])
    elif "exchange" in query.lower():
        intro = random.choice(["No problem! Here's our exchange policy:", "You can exchange your product under these conditions:", "Exchanges are allowed under these terms:"])
    elif "contact" in query.lower():
        intro = ""  
    elif "shipping" in query.lower() or "order" in query.lower():
        intro = random.choice(["Your order details:", "Shipping details are as follows:"])
    else:
        intro = ""

    return f"{intro} {response}".strip()  


def clean_response(txt):
    """Cleans response by removing unnecessary headers, bullet points, and special characters."""
    txt = re.sub(r"^[^A-Za-z0-9]+", "", txt)  
    txt = txt.replace("Contact Us", "").strip()  
    txt = txt.replace("Return & Exchange Policy", "").strip()  
    txt = txt.replace("Shipping & Delivery Policy", "").strip()
    
    return txt[0].upper() + txt[1:] if txt else "No relevant info found."

def retrieve_answer(query):
    if not vector_store:
        return "No documents available to retrieve an answer."

 
    greetings = ["hello", "hi", "hey", "good morning", "good evening"]
    if query.lower() in greetings:
        return random.choice(["Hey there! How can I assist you today?", "Hello! What would you like to know?", "Hi! Feel free to ask me anything."])

    query_emb = embedding_model.encode([query])[0]
    query_vector = np.array([query_emb])

    stored_vectors = np.array([v["embedding"] for v in vector_store.values()])
    similarities = cosine_similarity(query_vector, stored_vectors)[0]

    top_indices = np.argsort(similarities)[-3:][::-1]
    retrieved_texts = [vector_store[i]["txt"] for i in top_indices]

   
    query_kw = query.lower().split()
    
    for txt in retrieved_texts:
        sence = re.split(r'(?<=[.!?])\s+', txt) 
        for sentence in sence:
            if any(keyword in sentence.lower() for keyword in query_kw):
                cleaned_sentence = clean_response(sentence)
                return make_conversational(cleaned_sentence, query) 

    return "No relevant info found."
