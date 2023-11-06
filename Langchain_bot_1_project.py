import os
import time
import openai
import pickle
import langchain
import streamlit as st
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

os.environ['OPENAI_API_KEY'] = open("API_KEY","r").read()

#user interface using streamlit
st.title("News Research tool ")
st.sidebar.title("News Article URLs")

urls =[]
for i in range(3):
    url = st.sidebar.text_input(f" URL {i+1} ")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_openai.pkl"

main_placeholder = st.empty()
llm = OpenAI(temperature=0.8, max_tokens=500)

if process_url_clicked:
    # data loader
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("data loading....started....")
    data = loader.load()
    # splitting the data into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\nn', '\n', '.', ' ' ],
        chunk_size=1000
    )
    main_placeholder.text("data splitting ....started....")
    docs = text_splitter.split_documents(data)
    # embedding into vector format
    embeddings = OpenAIEmbeddings()
    vectorstore_openai = FAISS.from_documents(docs,embeddings)
    main_placeholder.text("Embedding Vector....started....")
    time.sleep(2)
    # saving the FAISS file to a pickle file
    with open(file_path, "wb") as f:
        pickle.dump(vectorstore_openai,f)

query = main_placeholder.text_input("Question: ")
if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)
            chain = RetrievalQAWithSourcesChain.from_llm( llm=llm, retriever= vectorstore.as_retriever())
            result = chain({"question": query}, return_only_outputs=True)

            # display answer
            st.header("Answer")
            st.write(result["answer"])

            # display source of answer
            sources = result.get("sources", "")
            if sources:
                st.subheader("Sources: ")
                sources_list = sources.split("\n") # split sources by new line
                for source in sources_list:
                    st.write(source)