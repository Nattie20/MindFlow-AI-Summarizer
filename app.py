import streamlit as st
import os
import tempfile
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate

# GLOBAL UI & STYLING CONFIGURATION
#===================================

st.set_page_config(page_title="MindFlow AI", layout="wide" , page_icon="💡")

# Injecting specialized CSS to enhance User Experience (UX)
st.markdown("""
    <style>
    div.stButton > button:first-child {
        background-color: #5DADE2; 
        color: white;
        transition: all 0.3s ease; 
    }
    div.stButton > button:first-child:hover {
        background-color: #2E86C1; 
        border-color: #2E86C1;
        color: #FFFFFF;
    }
    </style>
    """, unsafe_allow_html=True)

# Custom Branding Header: Using HTML/CSS for advanced typography and branding alignment
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Fredoka+One&family=Montserrat:wght@400;700&display=swap');
    </style>
    <div style='text-align: center; margin-bottom: 20px;'>
        <h1 style='font-family: "Fredoka One", cursive; font-size: 60px; color: #5D6D7E; letter-spacing: 2px; margin-bottom: 0px;'>
            MindFlow <span style='color: #85C1E9;'>AI</span> <br>
            <span style='text-align: center; font-family: "Segoe UI";font-size: 30px; color: #666;'>Driven Assistant Summarization</span>
        </h1>
    </div>
    """, unsafe_allow_html=True)

# BACKEND & MODEL INITIALIZATION
#====================================


# Setting the GOOGLE_API_KEY
os.environ["GOOGLE_API_KEY"] = os.environ.get("GOOGLE_API_KEY", "")

# Initialize Google Gemini Model
# Temperature 0.01 is utilized to minimize variance and ensure factual consistency 
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.01)


# SIDEBAR: Audience Type Controls 
with st.sidebar:
    st.title("User Settings")
    st.markdown("### **Target Audience:**")
    audience_type = st.radio("", ["Beginner", "Expert"])
    st.info(f"Targeting: {audience_type} level.")

# Data Ingestion Layer 
# Drag and drop a pdf OR paste a text manually   
st.header("Input Source")
tab1, tab2 = st.tabs(["📄 Upload PDF", "✍️ Paste Text"])

full_text = ""

# Handling PDF uploads using LangChain loaders and temporary disk storage
with tab1:
    uploaded_file = st.file_uploader("Upload PDF Document", type="pdf")
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        # Extracting semantic content from PDF
        loader = PyPDFLoader(tmp_path)
        pages = loader.load()
        # Merging PDF pages
        full_text = " ".join([page.page_content for page in pages])
        # Ensuring local storage cleanup
        os.remove(tmp_path) 

# Handling direct text input
with tab2:
    manual_text = st.text_area("Paste your article or text here:", height=300)
    if manual_text:
        full_text = manual_text

# PROCESSING PIPELINE: Summarization & Evaluation
if st.button("Generate & Evaluate"):
    if full_text.strip():
        with st.spinner("Processing..."):
            # SEMANTIC CHUNKING PHASE
            # Recursive splitting ensures text segments stay within LLM context windows
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=300)
            chunks = text_splitter.split_text(full_text)
            text_to_process = " ".join(chunks[:2])

            # TAILORED SUMMARIZATION PHASE 
            #Utilizing a persona-driven Prompt Template for audience-specific output
            summary_prompt = PromptTemplate.from_template("""
                [STRICT AUDIT MODE: ZERO HALLUCINATION TOLERANCE]
            You are an elite expert. Your ONLY source of truth is the provided text. 
            CRITICAL RESTRICTION: 
            If a concept (like 'Regularization', 'Overfitting', or 'Lasso') is NOT explicitly mentioned in the source text, 
            you are FORBIDDEN from mentioning it, even if it is factually related to the topic. 
            Failure to follow this will result in an inaccurate evaluation.
            
            Act as an elite educational consultant and technical expert. Your goal is to transform complex information into a high-quality summary perfectly tailored for a {audience} audience.

            Target Persona:
            - If Audience is "Beginner": You are a supportive teacher. Use simple analogies, avoid technical jargon unless explained, and focus on the "Big Picture" and "Why it matters." Use friendly, encouraging tone and clear bullet points.
            - If Audience is "Expert": You are a senior researcher. Use precise academic terminology, focus on methodology, data results, and nuanced conclusions. Maintain high information density and professional tone.

            Task Instructions:
            1. Core Essence**: Extract the most critical information without losing the original context.
            2. Structural Integrity: Organize the output with clear headers (e.g., "Overview", "Key Findings", "Implications").
            3. Contextual Adaptation: 
            - For Beginners: Include a "Simple Definition" section for complex terms.
            - For Experts: Include a "Technical Highlights" section focusing on metrics or logic.
            4. Faithfulness: Ensure ( 100% ) accuracy to the source text; do not hallucinate or add external information[cite: 56].
            CONSTRAINTS:
            - STRICT ADHERENCE: Do NOT include any information, concepts, or terms that are NOT present in the source text. 
            - NO OUTSIDE KNOWLEDGE: Even if you know more about the topic, ignore it. 
            - FORBIDDEN TOPICS: If the source text does not mention things like 'Regularization' or 'Overfitting', you MUST NOT mention them.
            - AUDIENCE ADAPTATION: 
                - If {audience} is Beginner: Explain ONLY the concepts in the text using simple analogies.
                - If {audience} is Expert: Focus ONLY on the technical details provided in the text.
            Source Text:
            {text}
            Final Output Requirements:
            - Format: Professional Markdown.
            - Language: Clear and Concise English[cite: 6].
            - Accuracy: Maintain strict adherence to the facts provided in the document[cite: 56].
            """)
            
            summary_chain = summary_prompt | llm
            summary_output = summary_chain.invoke({"audience": audience_type, "text": text_to_process})
            
            st.subheader(f"📝 Summary for {audience_type}")
            st.markdown(summary_output.content)
            
            st.divider()

            # AUTOMATED AI-AS-A-JUDGE EVALUATION PHASE
            # Implementing a secondary LLM chain to audit the quality of the generated summary
            eval_prompt = PromptTemplate.from_template("""
            As an AI Auditor, evaluate the summary against the source text.
            Return a Markdown table with scores (1-5) and justifications.
            
            | Criterion | Score | Justification |
            | :--- | :--- | :--- |
            | Faithfulness | | |
            | Coherence | | |
            | Audience Alignment | | |
            
            Level: {level}
            Source: {source}
            Summary: {summary}
            """)
            
            eval_chain = eval_prompt | llm
            eval_output = eval_chain.invoke({
                "level": audience_type, 
                "source": text_to_process[:4000], 
                "summary": summary_output.content
            })

            st.subheader("📊 Automated Quality Evaluation")
            st.markdown(eval_output.content)
    else:
        st.warning("Please upload a PDF or paste some text first!")
