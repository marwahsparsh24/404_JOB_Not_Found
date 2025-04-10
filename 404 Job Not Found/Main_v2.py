import streamlit as st
import os
import pandas as pd
from Final import (
    extract_resume_text,
    get_job_recommendations,
    get_resume_feedback,
    run_formatting_compliance_last,
    ats_parsing_error_detection_agent,
    run_ats_scoring_with_embedding,
    save_uploaded_file,
    upload_rejected_resume_local_to_df,
    get_formatted_df,
    get_plain_df
)

# --- Page Config ---
st.set_page_config(page_title="Career Coach AI", layout="centered")
st.title("🎓 404: Job Not Found")
st.markdown("Get resume tips, score your resume, and find jobs that fit **you**.")

# --- Upload Once for All Tools ---
st.sidebar.header("📤 Upload Resume & JD")

resume_file = st.sidebar.file_uploader("📄 Upload Resume (PDF or DOCX)", type=["pdf", "docx"], key="resume")
jd_text = st.sidebar.text_area("📃 Paste Job Description", height=200)
submit_uploaded_files = st.sidebar.button("✅ Submit & Store")

# Handle resume extraction and session state
if resume_file:
    st.session_state["resume_text"] = extract_resume_text(resume_file)
    st.session_state["resume_bytes"] = resume_file.read()
    st.session_state["resume_filename"] = resume_file.name

# Store JD text in session
if jd_text:
    st.session_state["jd_text"] = jd_text.strip()

resume_ready = "resume_text" in st.session_state and st.session_state["resume_text"].strip() != ""
jd_ready = "jd_text" in st.session_state and st.session_state["jd_text"].strip() != ""

# === Submit Button Logic ===
if submit_uploaded_files:
    if resume_ready and jd_ready:
        uploaded_file_name = save_uploaded_file(
            st.session_state["resume_filename"],
            st.session_state["resume_bytes"]
        )

        upload_result = upload_rejected_resume_local_to_df(
            resume_filename=uploaded_file_name,
            job_description_text=st.session_state["jd_text"],
            application_status="rejected"
        )

        st.sidebar.success("✅ Upload complete and data stored!")
        st.sidebar.markdown(f"**Resume file:** {uploaded_file_name}")
    else:
        st.sidebar.warning("⚠️ Please upload both resume and job description before submitting.")

# --- Sidebar Navigation ---
menu = st.sidebar.radio(
    "📚 What would you like to do?",
    [
        "🏁 Home",
        "📐 Resume Format Checker",
        "🔍 Resume Parsing Checker",
        "📊 Resume Score Estimator",
        "💼 Job Finder",
        "🧠 Resume Feedback & Suggestions"
    ]
)

# --- Home Page ---
if menu == "🏁 Home":
    st.subheader("👋 Welcome!")
    st.markdown("""
        **Career Coach AI** helps you:
        - Fix resume formatting issues ✅
        - Check if your resume is ATS-friendly ✅
        - Get a resume score ✅
        - Get job recommendations based on your resume ✅
        - See personalized suggestions to improve your chances ✅

        Upload your resume and job description on the sidebar to get started!
    """)

# --- Resume Format Checker ---
elif menu == "📐 Resume Format Checker":
    st.header("📐 Resume Format Compliance Checker")
    if resume_ready:
        result = run_formatting_compliance_last(get_formatted_df())
        st.subheader("🧾 Format Check Report:")
        st.markdown(result)
    else:
        st.warning("Please upload your resume on the sidebar.")

# --- Resume Parsing Checker ---
elif menu == "🔍 Resume Parsing Checker":
    st.header("🔍 ATS Parsing Error Checker")
    if resume_ready:
        result = ats_parsing_error_detection_agent(get_plain_df())
        st.subheader("🔎 Parsing Feedback:")
        st.markdown(result)
    else:
        st.warning("Please upload your resume on the sidebar.")

# --- Resume Score Estimator ---
elif menu == "📊 Resume Score Estimator":
    st.header("📊 Resume vs JD Scoring")
    if resume_ready and jd_ready:
        result = run_ats_scoring_with_embedding()
        st.subheader("📈 ATS Compatibility Score:")
        st.markdown(result)
    else:
        st.warning("Please upload both resume and job description on the sidebar.")

# --- Resume Feedback ---
elif menu == "🧠 Resume Feedback & Suggestions":
    st.header("🧠 Get Resume Feedback & Suggestions")
    if resume_ready:
        with st.spinner("Analyzing your resume..."):
            feedback = get_resume_feedback(st.session_state["resume_text"])
        st.subheader("✅ Expert Resume Feedback:")
        st.markdown(feedback)
    else:
        st.warning("Please upload your resume on the sidebar.")

# --- Job Finder ---
elif menu == "💼 Job Finder":
    st.header("💼 Find Jobs That Match Your Resume")
    if resume_ready:
        with st.spinner("Finding jobs tailored to your resume..."):
            results = get_job_recommendations(st.session_state["resume_text"])
        st.subheader("✅ Jobs Recommended for You:")
        st.markdown(results, unsafe_allow_html=True)
    else:
        st.warning("Please upload your resume on the sidebar.")
