
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.llms import Ollama
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import datetime
import time
import torch
import tensorflow as tf
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import pandas as pd
import json
import re
from typing import Dict, List
import os
from difflib import SequenceMatcher

# Initialize Metrics Calculator
class PhysicsMetricsCalculator:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.embeddings_model = SentenceTransformer('all-mpnet-base-v2').to(self.device)
        print(f"Initialized PhysicsMetricsCalculator with embeddings model on {self.device}.")


        

    def calculate_metrics(self, question: str, response: str, context: str, response_time: float) -> Dict:
        print(f"Calculating metrics for question: '{question}' and response: '{response}'")
        metrics = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'response_time': response_time,
            'response_length': len(response.split()),
            'question_similarity': float(self._calculate_similarity(question, response)),
            'context_similarity': float(self._calculate_similarity(context, response)),
            'equations_count': len(self._find_equations(response)),
            'units_count': len(self._find_units(response)),
            'steps_count': len(self._find_steps(response))
        }
        print(f"Calculated metrics: {metrics}")
        return metrics
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        print(f"Calculating similarity between '{text1}' and '{text2}'")
        emb1 = self.embeddings_model.encode([text1], convert_to_tensor=True)
        emb2 = self.embeddings_model.encode([text2], convert_to_tensor=True)
        similarity = cosine_similarity(emb1.cpu().numpy(), emb2.cpu().numpy())[0][0]
        print(f"Calculated similarity: {similarity}")
        return similarity

     
    
    def _find_equations(self, text: str) -> List[str]:
        equations = re.findall(r'\b[A-Za-z]+\s*=\s*.*[0-9a-zA-Z+\-*/^()]+', text)
        print(f"Found equations: {equations}")
        return equations

    
    def _find_units(self, text: str) -> List[str]:
        units = re.findall(r'\b(m/s|kg|N|J|W|Pa|Hz|V|\u03a9|\u00b0C|K)\b', text)
        print(f"Found units: {units}")
        return units
    
    def _find_steps(self, text: str) -> List[str]:
        steps = re.findall(r'step|:|\d\)', text.lower())
        print(f"Found steps: {steps}")
        return steps

# Initialize session state
def init_session_state():
    print("Initializing session state.")
    if 'conversations' not in st.session_state:
        st.session_state.conversations = {}
        print("Initialized 'conversations' in session state.")
    if 'active_conversation' not in st.session_state:
        st.session_state.active_conversation = None
        print("Initialized 'active_conversation' in session state.")
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
        print("Initialized 'conversation_history' in session state.")
    if 'metrics_calculator' not in st.session_state:
        st.session_state.metrics_calculator = PhysicsMetricsCalculator()
        print("Initialized 'metrics_calculator' in session state.")
    if 'evaluation_metrics' not in st.session_state:
        st.session_state.evaluation_metrics = []
        print("Initialized 'evaluation_metrics' in session state.")

# Add custom CSS styles
def add_custom_css():
    print("Adding custom CSS.")
    st.markdown("""
        <style>
            .sidebar-title { font-size: 24px; font-weight: bold; color: #4CAF50; }
            .history-container { max-height: 400px; overflow-y: auto; }
            .history-entry { margin-bottom: 10px; padding: 10px; border-radius: 6px; background-color: #f0f8ff; }
            .history-question { font-weight: bold; color: #1f77b4; }
            .history-answer { color: #333333; }
            .metric-card { padding: 1rem; border-radius: 0.5rem; margin: 0.5rem; background-color: #f8f9fa; }
            .metric-value { font-size: 1.5rem; font-weight: bold; color: #4CAF50; }
            .upload-container { margin-bottom: 20px; }
            .chat-container { margin-top: 20px; }
        </style>
    """, unsafe_allow_html=True)


def get_pdf_text(pdf_docs):
    print("Extracting text from uploaded PDFs.")
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            extracted_text = page.extract_text()
            text += extracted_text
            print(f"Extracted text from page: {extracted_text[:100]}...")
    return text
  
def get_text_chunks(text):
    print("Splitting text into chunks using RecursiveCharacterTextSplitter.")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    print(f"Created {len(chunks)} text chunks using RecursiveCharacterTextSplitter.")
    return chunks

def get_vector_store(text_chunks):
    print("Creating vector store from text chunks using ChromaDB.")
    embeddings = SentenceTransformerEmbeddings(model_name="all-mpnet-base-v2")
    vector_store = Chroma.from_texts(text_chunks, embedding=embeddings, persist_directory="chroma_index")
    vector_store.persist()
    print("Vector store saved locally as 'chroma_index'.")

# Initialize Ollama conversational chain
def get_conversational_chain():
    print("Initializing conversational chain.")
    prompt_template = """

You are an expert physics tutor having a continuous conversation with a student. Use the following context to give better answers:

    CONVERSATION HISTORY:
    {context}

    CURRENT QUESTION:
    {question}

    1. SOLUTION STRUCTURE:
       - First, clearly state the given information
       - List what we need to find
       - Show all relevant formulas
       - Solve step by step with clear explanations
       - State the final answer with units

    2. MATHEMATICAL CLARITY:
       - Write all equations using proper mathematical notation
       - Explain each mathematical step clearly
       - Use '=' to show each step of calculations

    3. CONCEPT EXPLANATION:
       - Define any physics concepts used
       - Explain why specific formulas are chosen
       - Include relevant diagrams when needed

    Respond in whatever the student asks, and provide detailed explanations for each question.

    """

# Set up GPU usage for TensorFlow (if needed)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
            tf.config.experimental.set_memory_growth(gpus[0], True)
        except RuntimeError as e:
            print(e)

    # Initialize Ollama model (assuming it supports TensorFlow)
    model = Ollama(model="llama3:latest")
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    print("Conversational chain initialized.")
    return chain

def user_input(user_question):
    print(f"User input received: {user_question}")
    start_time = time.time()
    embeddings = SentenceTransformerEmbeddings(model_name="all-mpnet-base-v2")
    new_db = Chroma(persist_directory="chroma_index", embedding_function=embeddings)
    print("Loaded ChromaDB index.")

    # Using short-term memory (rolling context)
    context_history = "\n".join([f"Q: {entry['question']}\nA: {entry['response']}" 
                                for entry in st.session_state.conversation_history[-5:]])
    combined_query = f"{context_history}\nQ: {user_question}" if context_history else user_question

    docs = new_db.similarity_search(combined_query)
    print(f"Found {len(docs)} documents for the query.")

    if docs:
        chain = get_conversational_chain()
        response = chain({
            "input_documents": docs,
            "context": context_history,
            "question": user_question
        }, return_only_outputs=True)
        response_text = response['output_text']
        print(f"Generated response: {response_text}")

        end_time = time.time()

        # Calculate and store metrics
        metrics = st.session_state.metrics_calculator.calculate_metrics(
            question=user_question,
            response=response_text,
            context=docs[0].page_content,
            response_time=end_time - start_time
        )
        st.session_state.evaluation_metrics.append(metrics)
        print("Metrics calculated and stored.")

        # Update conversation state
        active_conv = st.session_state.conversations.get(st.session_state.active_conversation)
        if active_conv:
            if active_conv['title'] == "Untitled Conversation":
                active_conv['title'] = user_question[:50] + "..."
            
            active_conv['history'].append({
                "question": user_question,
                "response": response_text,
                "metrics": metrics
            })
            st.session_state.conversation_history.append({
                "question": user_question,
                "response": response_text
            })
            print("Conversation history updated.")
        
        st.markdown(f"<div class='output-box'>**Reply:** {response_text}</div>", unsafe_allow_html=True)
    else:
        print("No relevant context found in the uploaded documents.")
        st.error("No relevant context found in the uploaded documents.")


#---------------------------------#
def evaluate_benchmark():
    print("Evaluating benchmark questions.")
    with open("benchmark_questions.json", "r") as file:
        benchmark_questions = json.load(file)
    embeddings = SentenceTransformerEmbeddings(model_name="all-mpnet-base-v2")
    new_db = Chroma(persist_directory="chroma_index", embedding_function=embeddings)

    results = []
    for item in benchmark_questions:
        start_time = time.time()
        docs = new_db.similarity_search(item["question"])
        
        if docs:
            chain = get_conversational_chain()
            response = chain({
                "input_documents": docs,
                "context": "",
                "question": item["question"]
            }, return_only_outputs=True)
            end_time = time.time()

            response_text = response['output_text']
            metrics = st.session_state.metrics_calculator.calculate_metrics(
                question=item["question"],
                response=response_text,
                context=docs[0].page_content,
                response_time=end_time - start_time
            )
            
            # Calculate similarity score and determine correctness
            is_correct = is_answer_correct(item["expected_answer"], response_text)

            result = {
                "question": item["question"],
                "expected_answer": item["expected_answer"],
                "actual_answer": response_text,
                "is_correct": is_correct,
                "metrics": metrics
            }
            results.append(result)
            print(f"Evaluated benchmark question: {item['question']}, Correct: {is_correct}")
    
    return results

def is_answer_correct(expected, actual):
    similarity = SequenceMatcher(None, expected.lower(), actual.lower()).ratio()
    return similarity > 0.6



# Start new conversation
def start_new_conversation():
    print("Starting new conversation.")
    conversation_id = str(len(st.session_state.conversations) + 1)
    st.session_state.conversations[conversation_id] = {
        "title": "Untitled Conversation",
        "history": []
    }
    st.session_state.active_conversation = conversation_id
    st.session_state.conversation_history = []
    print(f"New conversation started with ID: {conversation_id}")

def render_metrics_dashboard():
    print("Rendering metrics dashboard.")
    st.title("Evaluation Metrics Dashboard")
    
    if st.session_state.evaluation_metrics:
        df = pd.DataFrame(st.session_state.evaluation_metrics)
        print("Metrics dataframe created.")
        
        # Response Time Analysis
        st.subheader("Response Time Analysis")
        fig_time = px.line(df, x='timestamp', y='response_time', 
                          title='Response Time Trend')
        st.plotly_chart(fig_time)
        print("Response Time Analysis chart rendered.")
        
        # Content Quality Metrics
        st.subheader("Content Quality Metrics")
        fig_quality = px.line(df, x='timestamp', 
                            y=['question_similarity', 'context_similarity'],
                            title='Response Quality Trends')
        st.plotly_chart(fig_quality)
        print("Content Quality Metrics chart rendered.")
        
        # Physics Content Analysis
        st.subheader("Physics Content Metrics")
        fig_physics = px.bar(df, x='timestamp',
                           y=['equations_count', 'units_count', 'steps_count'],
                           title='Physics Content Analysis')
        st.plotly_chart(fig_physics)
        print("Physics Content Metrics chart rendered.")
        
        # Export metrics
        if st.button("Export Metrics"):
            try:
                output_path = os.path.join(os.getcwd(), "evaluation_metrics.csv")
                df.to_csv(output_path, index=False)
                st.success(f"Metrics exported to {output_path}")
                print(f"Metrics exported to '{output_path}'.")
            except Exception as e:
                st.error(f"Failed to export metrics: {e}")
                print(f"Failed to export metrics: {e}")
    else:
        print("No evaluation data available yet.")
        st.info("No evaluation data available yet. Start some conversations to generate metrics.")


# Main function
def main():
    print("Running main function.")
    st.set_page_config(page_title="Physics Chatbot", page_icon="📚", layout="wide")
    init_session_state()
    add_custom_css()

    # Sidebar navigation
    st.sidebar.markdown("<div class='sidebar-title'>Navigation</div>", unsafe_allow_html=True)
    page = st.sidebar.radio("", ["Chat", "Evaluation Dashboard"])
    
    if page == "Chat":
        with st.sidebar:
            st.markdown("<div class='sidebar-title'>Conversations</div>", unsafe_allow_html=True)
            for conv_id, conv_data in st.session_state.conversations.items():
                if st.button(conv_data["title"], key=conv_id):
                    st.session_state.active_conversation = conv_id
                    print(f"Switched to conversation ID: {conv_id}")

            if st.button("Start New Conversation"):
                start_new_conversation()

            if st.button("Clear All Conversations", key="clear_all"):
                st.session_state.conversations.clear()
                st.session_state.active_conversation = None
                st.session_state.evaluation_metrics = []
                print("All conversations cleared.")

        st.title("Physics PDF Chatbot")
        st.subheader("📂 Upload your Physics PDF Files:")
        pdf_docs = st.file_uploader("Upload PDFs", accept_multiple_files=True, key="pdf_uploader", help="Upload one or more Physics textbooks or notes in PDF format for analysis.")
        
        if pdf_docs and st.button("Process Files", key="process_files"):
            with st.spinner("Extracting and processing the PDFs..."):
                start_new_conversation()
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Files processed successfully!")
                print("PDF files processed successfully.")

        if st.session_state.active_conversation:
            active_conv = st.session_state.conversations[st.session_state.active_conversation]
            st.markdown(f"### {active_conv['title']}")
            
            st.markdown("<div class='history-container chat-container'>", unsafe_allow_html=True)
            for entry in active_conv["history"]:
                st.markdown(
                    f"<div class='history-entry'>"
                    f"<span class='history-question'>Q:</span> {entry['question']}<br>"
                    f"<span class='history-answer'>A:</span> {entry['response']}</div>",
                    unsafe_allow_html=True
                )
            st.markdown("</div>", unsafe_allow_html=True)

            user_question = st.text_input("Enter your physics question based on the uploaded PDFs:", key="user_question")
            if st.button("Ask", key="ask_button"):
                if user_question:
                    with st.spinner("Generating response..."):
                        user_input(user_question)
        else:
            print("No active conversation selected.")
            st.markdown("No active conversation selected. Start a new conversation or select an existing one.")
    
    else:
        render_metrics_dashboard()

if __name__ == "__main__":
    main()
