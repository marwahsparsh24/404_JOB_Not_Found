# 🧠 404: Job Not Found – AI-Powered Resume Feedback & Job Matching Assistant

A multi-agent, LLM-powered platform designed to help job seekers optimize their resumes, evaluate ATS compatibility, and discover better job matches with tailored feedback. Built using OpenAI, ChromaDB, MongoDB, and Streamlit.

---

## 🚀 Overview

**404: Job Not Found** is an AI-driven assistant that simulates how Applicant Tracking Systems (ATS) process resumes. It provides personalized feedback based on formatting compliance, ATS parsing errors, keyword matching, similarity scores, and even suggests better-fit jobs with cover letter generation.

### 🧩 Key Features

- **Multi-Agent Architecture (CrewAI)**: Each agent handles a distinct step—formatting check, ATS parsing, scoring, job matching, and final feedback.
- **ATS Score Evaluation**: Combines keyword overlap, cosine similarity, and LLM reasoning for comprehensive scoring.
- **Job Recommendation Engine**: Recommends top-matching jobs and generates personalized cover letters.
- **Streamlit Frontend**: Interactive UI to upload resumes, job descriptions, and view feedback in real-time.
- **Persistent Storage**: Uses MongoDB for structured data and ChromaDB for vector embeddings.

---

## 🏗️ Tech Stack

| Layer | Tools |
|------|-------|
| Language Models | OpenAI (GPT-4) |
| Orchestration | LangChain, CrewAI |
| Embedding Storage | ChromaDB |
| Data Storage | MongoDB |
| Interface | Streamlit |
| Resume Parsing | PyMuPDF, Python |

---

## 🧠 System Architecture

🧠 **Multi-Agent Workflow**
The system processes resumes and job descriptions through the following agents, each performing a specific function:

📤 **Resume/JD Upload**
➜ User uploads their resume and job description via the Streamlit interface.

🧾 **Formatting Agent**
➜ Checks layout, consistency, fonts, bullet styles, and sectioning to ensure ATS-compliant formatting.

🔍 **Parsing Agent**
➜ Simulates ATS parsing behavior, identifies extraction issues, and highlights missing or misclassified sections.

📊 **ATS Scoring Agent**
➜ Calculates resume-job match score using a hybrid method:
• Cosine similarity of embeddings (ChromaDB)
• Keyword overlap analysis
• LLM-based reasoning to assess relevance

💼 **Job Matcher Agent**
➜ Searches for better-matching job descriptions and suggests top recommendations from live or stored job listings.

📝 **Feedback Generator Agent**
➜ Provides a detailed report with improvement suggestions, formatting issues, ATS compatibility score, and personalized cover letter.

## 📝 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/marwahsparsh24/404_JOB_Not_Found.git
cd 404_JOB_Not_Found
```

### 2. Install required libraries

```bash
pip install -r requirements.txt
```
### 3. Configure Environment Variables

Create a .env file and add:

env
```bash
OPENAI_API_KEY=your_key
MONGODB_URI=your_mongo_uri
CHROMADB_PERSIST_DIR=your_chroma_dir
```

### 4. Launch the App

```bash
streamlit run Main.py
```
