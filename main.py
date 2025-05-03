import gradio as gr
import tempfile
import os

from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import Ollama
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings


# Global state
retriever = None
qa_chain = None

def process_file(file):
    global retriever, qa_chain

    # Gradio provides 'file' as a str path, not a file object
    file_path = file.name if hasattr(file, 'name') else file  # handles string or file object

    # Load and split
    loader = PyPDFLoader(file_path) if file_path.endswith(".pdf") else TextLoader(file_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(documents)

    # Embeddings + FAISS
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    llm = OllamaLLM(model="gemma:2b", temperature=0.1, max_tokens=512)
    vectorstore = FAISS.from_documents(docs, embedding_model)
    retriever = vectorstore.as_retriever()


    # QA Chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

    return "✅ Document processed! You can now ask questions."


def ask_question(question):
    if qa_chain is None:
        return "❌ Please upload and process a document first."
    
    result = qa_chain.invoke({"query": question})
    answer = result["result"]
    return answer


# Gradio UI
with gr.Blocks() as app:
    gr.Markdown("## 📄 Local Document Q&A Chatbot")
    file_input = gr.File(label="Upload .pdf or .txt file")
    status_output = gr.Textbox(label="Status")
    question_input = gr.Textbox(label="Ask a question")
    answer_output = gr.Textbox(label="Answer")

    file_input.change(fn=process_file, inputs=file_input, outputs=status_output)
    question_input.submit(fn=ask_question, inputs=question_input, outputs=answer_output)

app.launch()
