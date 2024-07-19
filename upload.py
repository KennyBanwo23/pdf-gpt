import os
import streamlit as st
from langchain_groq import ChatGroq
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

# Load the GROQ and Google API key from the .env file
groq_api_key = os.getenv("GROQ_API_KEY")
os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY')

# Sidebar
with st.sidebar:
    st.title("Welcome to Pdf-GPT!")
    st.markdown('''
    ## About
    An LLM-powered chatbot that allows you to upload your PDF files and is capable of reading them and providing accurate and fast responses.
    It was built using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [OpenAI](https://platform.openai.com/docs/models) LLM Model
    '''
    )
    add_vertical_space(1)
    st.write('Made with Love by Kehinde Ogunbanwo')

llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

prompt = ChatPromptTemplate.from_template("""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question.
<context>
{context}
<context>
Questions: {input}
""")

def vector_embedding(uploaded_file):
    if "vectors" not in st.session_state:
        # Ensure tempDir exists
        os.makedirs("tempDir", exist_ok=True)

        # Save the uploaded file to tempDir
        with open(os.path.join("tempDir", uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        st.session_state.loader = PyPDFLoader(os.path.join("tempDir", uploaded_file.name)) # Data Ingestion
        st.session_state.docs = st.session_state.loader.load() # Document Loading
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file is not None:
    if st.button("Creating Vector Store"):
        vector_embedding(uploaded_file)
        st.write("Vector Store DB is Ready")

if "history" not in st.session_state:
    st.session_state.history = []
if "question_counter" not in st.session_state:
    st.session_state.question_counter = 1

# Function to handle question submission
def handle_question(question):
    if st.session_state.vectors is not None:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        
        response = retrieval_chain.invoke({'input': question})
        st.session_state.history.append((question, response['answer'], response['context']))
        
        st.write(f"**Question:** {question}")
        st.write(f"**Answer:** {response['answer']}")
        
        # Display relevant document chunks
        with st.expander("Document Similarity Search"):
            for i, doc in enumerate(response['context']):
                st.write(doc.page_content)
                st.write("---------------------------------------------------")
    else:
        st.warning("Please upload a PDF file and create the vector store first.")

# Input box for questions
for i in range(st.session_state.question_counter):
    question = st.text_input(f"Question {i+1}", key=f"question_{i}")

if st.button("Submit Question"):
    question = st.session_state.get(f"question_{st.session_state.question_counter - 1}")
    if question:
        handle_question(question)
        st.session_state.question_counter += 1
        st.experimental_rerun()  # Rerun the script to update the UI

# Display the conversation history
if st.session_state.history:
    st.subheader("Conversation History")
    for q, a, ctx in st.session_state.history:
        st.write(f"**Question:** {q}")
        st.write(f"**Answer:** {a}")
        with st.expander("Document Similarity Search"):
            for doc in ctx:
                st.write(doc.page_content)
                st.write("---------------------------------------------------")
