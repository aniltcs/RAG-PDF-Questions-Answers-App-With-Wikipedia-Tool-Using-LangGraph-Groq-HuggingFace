import streamlit as st
import hashlib
import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from typing import Literal, List
from langchain_core.pydantic_v1 import BaseModel, Field
from typing_extensions import TypedDict, Annotated
from langchain.schema import Document
from langgraph.graph import END, StateGraph, START
import operator

load_dotenv()

# -------------------------------
# Helper Functions
# -------------------------------
def file_hash(file):
    return hashlib.md5(file.getvalue()).hexdigest()

# -------------------------------
# Data Models
# -------------------------------
class RouteQuery(BaseModel):
    datasource: Literal["vectorstore", "wiki_search"] = Field(
        ..., description="Route question to vectorstore or Wikipedia"
    )

class GraphState(TypedDict):
    question: Annotated[List[AnyMessage], operator.add]
    generation: List[AIMessage]
    documents: List[Document]

# -------------------------------
# Node Functions
# -------------------------------
def vectorstore_node(state):
    question_messages = state.get("question", [])
    human_messages = [m for m in question_messages if isinstance(m, HumanMessage)]
    question_text = human_messages[-1].content if human_messages else ""
    docs = retriever.get_relevant_documents(question_text)
    return {"documents": docs,"question": question_messages}

def wiki_search(state):
     question_messages = state.get("question", [])
     human_messages = [m for m in question_messages if isinstance(m, HumanMessage)]
     question_text = human_messages[-1].content if human_messages else ""
     wiki_text = wikipedia_tool.invoke({"query": question_text})
     doc = Document(page_content=str(wiki_text))
     return {"documents": [doc],"question": question_messages}

def route_question(state):
     question_messages = state.get("question", [])
     human_messages = [m for m in question_messages if isinstance(m, HumanMessage)]
     question_text = human_messages[-1].content if human_messages else ""
     source = chain.invoke({"question": question_text})
     if source.datasource == "wiki_search":
        return "wiki_search"
     elif source.datasource == "vectorstore":
        return "vectorstore"

def generate_answer(state):
    question_messages = state.get("question", [])
    docs = state.get("documents", [])
    context = "\n\n".join([str(d.page_content) for d in docs])
    human_messages = [m for m in question_messages if isinstance(m, HumanMessage)]
    question_text = human_messages[-1].content if human_messages else ""
    # Build conversation memory
    conversation_history = ""
    for msg in question_messages[:-1]:  # exclude current question
        if isinstance(msg, HumanMessage):
            conversation_history += f"User: {msg.content}\n"
        elif isinstance(msg, AIMessage):
            conversation_history += f"AI: {msg.content}\n"
    prompt_text = f"""
    You are a conversation-aware assistant.
    Below is the full conversation so far:
    {conversation_history}
    User's latest question: {question_text}
    Relevant context from documents:
    {context}
    Answer the latest question using the context and conversation history when needed.
    """
    # Ensure we always get a string
    answer_text = llm.invoke(prompt_text)
    return {"generation": [AIMessage(content=answer_text.content)], "documents": docs, "question": question_messages}

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("LangGraph PDF Q&A RAG - Chat")

groq_api_key = st.text_input("Groq API Key", type="password")
hf_token = st.text_input("HuggingFace API Key", type="password")
uploaded_files = st.file_uploader("Upload PDF Files", type=["pdf"], accept_multiple_files=True)
query = st.text_input("Ask a question:")

# Session state
if "processed_hashes" not in st.session_state:
    st.session_state.processed_hashes = set()
if "vectorStore" not in st.session_state:
    st.session_state.vectorStore = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if groq_api_key and hf_token:
    os.environ["HF_TOKEN"] = hf_token
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    llm = ChatGroq(model="llama-3.1-8b-instant", groq_api_key=groq_api_key)

    # Router chain
    system_prompt = """You are an expert at routing a user question to a vectorstore or Wikipedia.
Use vectorstore for questions about Skills, Tools, Personal Info, Work Experience, Academic details.
Otherwise, use Wikipedia."""
    prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{question}")])
    structured_llm_router = llm.with_structured_output(RouteQuery)
    chain = prompt | structured_llm_router

    # Wikipedia tool
    wiki_wrapper = WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=2000)
    wikipedia_tool = WikipediaQueryRun(api_wrapper=wiki_wrapper)

    # -------------------------------
    # Handle PDF uploads
    # -------------------------------
    if uploaded_files:
        new_files = [f for f in uploaded_files if file_hash(f) not in st.session_state.processed_hashes]
        if new_files:
            for file in new_files:
                st.session_state.processed_hashes.add(file_hash(file))
            all_docs = []
            for file in new_files:
                os.makedirs("uploads", exist_ok=True)
                pdf_path = os.path.join("uploads", file.name)
                with open(pdf_path, "wb") as f:
                    f.write(file.getvalue())
                loader = PyPDFLoader(pdf_path)
                docs = loader.load()
                for doc in docs:
                    doc.metadata["file_hash"] = file_hash(file)
                all_docs.extend(docs)
            splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
            splits = splitter.split_documents(all_docs)
            st.session_state.vectorStore = FAISS.from_documents(splits, embeddings)
        if st.session_state.vectorStore:
            retriever = st.session_state.vectorStore.as_retriever()

        # Workflow
        workflow = StateGraph(GraphState)
        workflow.add_node("wiki_search", wiki_search)
        workflow.add_node("vectorstore", vectorstore_node)
        workflow.add_node("generate_answer", generate_answer)
        workflow.add_conditional_edges(
            START,
            route_question,
            {"wiki_search": "wiki_search", "vectorstore": "vectorstore"}
        )
        workflow.add_edge("vectorstore", "generate_answer")
        workflow.add_edge("wiki_search", "generate_answer")
        workflow.add_edge("generate_answer", END)

        app = workflow.compile()

        # -------------------------------
        # Handle user query
        # -------------------------------
        if query:
            input_question = [HumanMessage(content=str(query))]
            input_state = {"question": st.session_state.chat_history + input_question}
            result = app.invoke(input_state)
            # -------------------------------
            #  Display Response
            # -------------------------------
            st.subheader("Answer")
            st.write(result["generation"][0].content)
            st.subheader("Retrieved Documents")
            with st.expander("Show Retrieved Documents", expanded=False):
                doc_html = """
                <div style="
                max-height: 300px;
                overflow-y: auto;
                padding: 10px;
                border: 1px solid #ddd;
                border-radius: 5px;
                ">
                """
                for i, doc in enumerate(result["documents"], 1):
                    doc_html += f"<p><strong>Document {i}:</strong><br>{doc.page_content[:500]}...</p><hr>"
                doc_html += "</div>"
                st.markdown(doc_html, unsafe_allow_html=True)

            # Append messages to chat history
            st.session_state.chat_history.extend(input_question)
            st.session_state.chat_history.extend(result["generation"])

            # -------------------------------
            #  Display chat history
            # -------------------------------
            st.subheader("Chat History")
            # Scrollable container using HTML + CSS
            chat_html = """
            <div style="
            max-height: 400px;
            overflow-y: auto;
            padding: 10px;
            border: 1px solid #ccc;
            border-radius: 5px;">
            """
            # Build chat content
            for msg in st.session_state.chat_history:
                if isinstance(msg, HumanMessage):
                    chat_html += f"<p><strong>You:</strong> {msg.content}</p>"
                elif isinstance(msg, AIMessage):
                    chat_html += f"<p><strong>AI:</strong> {msg.content}</p>"
            chat_html += "</div>"
            # Display the HTML block
            st.markdown(chat_html, unsafe_allow_html=True)
