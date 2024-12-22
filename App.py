import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import gradio as gr

# Load environment variables
load_dotenv()

# Define your GROQ API Key
groq_api_key = os.environ.get("GROQ_API_KEY") 


# Initialize the LLM
def init_llm():
    """Initialize the Groq language model."""
    llm = ChatGroq(
        temperature=0,
        groq_api_key=groq_api_key,
        model="llama-3.3-70b-versatile"
    )
    return llm


# Load PDF data and embed into ChromaDB
def load_data(data_path):
    """Load PDF data, split text, embed, and store in Chroma DB."""
    loader = DirectoryLoader(data_path, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v1")
    vector_db = Chroma.from_documents(texts, embeddings, persist_directory="./chroma_db")
    vector_db.persist()
    
    return vector_db


# Set up QA Chain
def setup_QA_chain(vector_db, llm):
    """Set up the RetrievalQA chain."""
    retriever = vector_db.as_retriever()
    
    prompt_template = """
    Use the following pieces of information to answer the user's question.
    If you don't know the answer, just say 'I don't know', don't try to make up an answer.
    Context: {context}
    Question: {question}
    Only return the helpful answer below and nothing else.
    Helpful answer:
    """
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    
    QA_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": PROMPT}
    )
    return QA_chain


# Chatbot response for Gradio
def chatbot_response(user_input, history=[]):
    """Handle user queries via Gradio ChatInterface."""
    if not user_input.strip():
        return "Please provide a valid input.", history
    
    try:
        response = qa_chain.run(user_input)
        history.append((user_input, response))
        return "",history
    except Exception as e:
        history.append((user_input, f"An error occurred: {e}"))
        return history




def main():
    global qa_chain  
    
    llm = init_llm()
    db_path = "./chroma_db"
    data_path = "./data"  # Adjust this path to your PDF folder
    
    if not os.path.exists(db_path):
        print("Database not found. Creating a new vector database...")
        vector_db = load_data(data_path)
    else:
        print("Loading existing vector database...")
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v1")
        vector_db = Chroma(persist_directory=db_path, embedding_function=embeddings)
    
    qa_chain = setup_QA_chain(vector_db, llm)
    
    with gr.Blocks(theme='Respair/Shiki@1.2.1') as app:
        chatbot=gr.ChatInterface(
            fn=chatbot_response,
            title="Mental Health AI Assistant",
            description="Ask questions regarding Mental Health Care.",
        )

    
    app.launch()


if __name__ == "__main__":
    main()
