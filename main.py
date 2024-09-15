from bs4 import BeautifulSoup
import requests
import os
import streamlit as st
import pickle
import time
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss
from sentence_transformers import SentenceTransformer
from langchain.document_loaders import TextLoader


st.title("ScrapeBot: Web Scraper LLM 📈")
st.sidebar.title("Enter Article URLs")
urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    if(url):
        urls.append(url)
data=[]
process_url_clicked = st.sidebar.button("Process URL")
file_path = "faiss_store_openai.pkl"

main_placeholder = st.empty()
llm = SentenceTransformer('all-MiniLM-L6-v2') 


def get_data(url):
    page=requests.get(url)
    soup=(BeautifulSoup(page.text,'html.parser'))
    paragraphs = soup.find_all('p')
    for p in paragraphs:
        data.append(p.get_text())
if process_url_clicked:
    # load data
    main_placeholder.text("Data Loading...Started...✅✅✅")
    for url in urls:
        get_data(url)
    info=" ".join(data)
    with open("output.txt", "w", encoding="utf-8") as file:
        file.write(info)
    loader=TextLoader('output.txt', encoding="utf-8")
    new_data=loader.load()

    # split data
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=200
    )
    main_placeholder.text("Text Splitter...Started...✅✅✅")
    docs = text_splitter.split_documents(new_data)

    # # create embeddings and save it to FAISS index
   
    texts = [doc.page_content for doc in docs]   
    embeddings = llm.encode(texts)

    # Create and populate FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    # Save the FAISS index to a pickle file
    vectorstore_openai = {'index': index, 'docs': docs}
    with open(file_path, "wb") as f:
        pickle.dump(vectorstore_openai, f)

    main_placeholder.text("Embedding Vector Started Building...✅✅✅")
    time.sleep(2)

# Query processing
query = main_placeholder.text_input("Question: ")
if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)
            index = vectorstore['index']
            docs = vectorstore['docs']
            # Encode the query
            query_embedding = llm.encode([query])
            # Perform the search
            distances, indices = index.search(query_embedding, k=2) 
            # Retrieve the top documents
            top_docs = [docs[i] for i in indices[0]]
            # Display the results
            st.header("Answer")
            for doc in top_docs:
                st.write(doc.page_content)



