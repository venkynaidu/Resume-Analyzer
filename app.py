import streamlit as st
from PyPDF2 import PdfReader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

# 1. Define the structured output format using Pydantic
class ResumeAnalysis(BaseModel):
    match_percentage: int = Field(description="ATS match score from 0 to 100 based on the job description")
    matched_skills: list[str] = Field(description="Keywords and skills found in the resume that match the job description")
    missing_skills: list[str] = Field(description="Important keywords and skills from the job description missing in the resume")
    recommendations: list[str] = Field(description="3 actionable bullet points to improve the resume for this job role")

# Set up Streamlit Page Configuration
st.set_page_config(page_title="AI Resume Analyzer", page_icon="📄", layout="wide")

st.title("📄 AI Resume Analyzer by Venkys.AI")
st.write("Upload your resume and a job description to check your ATS compatibility.")

# Sidebar for API Key input
with st.sidebar:
    st.header("Configuration")
    gemini_api_key = st.text_input("Enter your Google Gemini API Key", type="password")
    st.markdown("[Get a free API key here](https://aistudio.google.com/)")

# Main Interface: Two-column layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Upload Resume")
    uploaded_file = st.file_uploader("Upload your Resume (PDF format only)", type=["pdf"])

with col2:
    st.subheader("2. Job Description")
    job_description = st.text_area("Paste the Target Job Description Here", height=200)

# Process Analysis when button is clicked
if st.button("Analyze Resume", type="primary"):
    if not gemini_api_key:
        st.error("Please enter your Gemini API Key in the sidebar.")
    elif not uploaded_file:
        st.error("Please upload a resume PDF.")
    elif not job_description.strip():
        st.error("Please paste a job description.")
    else:
        with st.spinner("Extracting text and analyzing with Gemini..."):
            try:
                # Extract text from PDF
                reader = PdfReader(uploaded_file)
                resume_text = ""
                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        resume_text += text
                
                # Initialize LangChain Gemini model (UPDATED TO CURRENT WORKING MODEL)
                llm = ChatGoogleGenerativeAI(
                    model="gemini-3.7-flash",    # Updated to the active production model to fix 404
                    google_api_key=gemini_api_key,
                    temperature=0.2,
                    version="v1"                 # Forces production API pipeline
                )
                
                # Set up structured JSON parser
                parser = JsonOutputParser(pydantic_object=ResumeAnalysis)
                
                # Create the prompt template
                prompt = ChatPromptTemplate.from_messages([
                    ("system", "You are an expert ATS (Applicant Tracking System) optimizer. "
                               "Compare the candidate's resume text against the provided job description. "
                               "Provide unbiased assessment. "
                               "You must return your output strictly in JSON format matching the schema instructions.\n{format_instructions}"),
                    ("human", "RESUME:\n{resume}\n\nJOB DESCRIPTION:\n{job_description}")
                ])
                
                # Chain together: Prompt -> LLM -> JSON Parser
                chain = prompt | llm | parser
                
                # Execute the chain
                result = chain.invoke({
                    "resume": resume_text,
                    "job_description": job_description,
                    "format_instructions": parser.get_format_instructions()
                })
                
                # --- RENDER RESULTS ---
                st.success("Analysis Complete!")
                st.divider()
                
                # Display ATS Metric Score Card
                score = result.get("match_percentage", 0)
                if score >= 75:
                    st.balloons()
                    st.metric(label="ATS Match Score", value=f"{score}%", delta="Strong Match")
                elif score >= 50:
                    st.metric(label="ATS Match Score", value=f"{score}%", delta="Needs Improvement", delta_color="off")
                else:
                    st.metric(label="ATS Match Score", value=f"{score}%", delta="Weak Match", delta_color="inverse")
                
                # Display Lists
                res_col1, res_col2 = st.columns(2)
                with res_col1:
                    st.subheader("✅ Matched Skills & Keywords")
                    for skill in result.get("matched_skills", []):
                        st.markdown(f"- {skill}")
                        
                with res_col2:
                    st.subheader("❌ Missing Critical Keywords")
                    for skill in result.get("missing_skills", []):
                        st.markdown(f"- <span style='color:#ff4b4b'>**{skill}**</span>", unsafe_allow_html=True)
                
                st.subheader("💡 Recommendations to Optimize Your Resume")
                for rec in result.get("recommendations", []):
                    st.markdown(f"* {rec}")
                    
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
