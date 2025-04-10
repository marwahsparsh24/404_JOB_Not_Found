import streamlit as st
import pandas as pd
import base64
import os
import pymupdf
from io import BytesIO
from final_v4 import (
    extract_resume_text,
    get_job_recommendations,
    get_resume_feedback,
    run_formatting_compliance_last,
    ats_parsing_error_detection_agent,
    run_ats_scoring_with_embedding,
    save_uploaded_file,
    upload_rejected_resume_local_to_df,
    generate_cover_letter
)

st.set_page_config(page_title="404: Job Not Found", layout="wide")


def get_base64_image(image_path):
    with open(image_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()
# Custom Blurred Background with Overlay Text
image_base64 = get_base64_image("images/image2.jpeg")

st.markdown(f"""
    <style>
    .background-container {{
        position: relative;
        width: 100%;
        height: 300px;
        display: flex;
        align-items: center;
        justify-content: center;
        overflow: hidden;
        margin-bottom: 20px;
    }}
    .background-container::before {{
        content: "";
        background: url("data:image/jpeg;base64,{image_base64}") no-repeat center center;
        background-size: cover;
        filter: blur(9px);
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 300px;
        z-index: 0;
        opacity: 0.8;
    }}
    .overlay-text {{
        position: relative;
        z-index: 1;
        text-align: center;
    }}
    .overlay-text h1 {{
        font-size: 76px;
        font-weight: 900;
        font-family: 'Amasis MT Pro Black', serif;
        color: black;
        margin: 0;
    }}
    .overlay-text h4 {{
        font-size: 32px;
        font-weight: 500;
        font-family: 'Amasis MT Pro Black', serif;
        color: black;
        margin-top: 10px;
    }}
    .upload-box {{
        border: 2px dashed #6c757d;
        border-radius: 12px;
        padding: 30px 20px 20px;
        background-color: #1e1e1e;
        text-align: center;
        transition: all 0.3s ease-in-out;
    }}
    .upload-box:hover {{
        border-color: #339af0;
        background-color: #2b2b2b;
    }}
    .upload-icon {{
        font-size: 40px;
        color: #339af0;
        margin-bottom: 10px;
    }}
    .upload-title {{
        font-size: 20px;
        font-weight: bold;
        color: white;
        margin-bottom: 5px;
    }}
    .upload-subtext {{
        font-size: 14px;
        color: #bbbbbb;
        margin-bottom: 15px;
    }}
    .file-uploader-wrapper section[data-testid="stFileUploader"] {{
        display: flex;
        justify-content: center;
        margin-top: -10px;
    }}
    textarea {{
        font-size: 18px !important;
        color: white !important;
        text-align: center !important;
        padding: 25px !important;
        height: 260px !important;
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
    }}
    .stTextArea label {{
        display: none !important;
    }}
    div.stButton > button {{
        display: block;
        margin: 30px auto;
        background-color: #4CAF50;
        color: white;
        font-size: 20px;
        padding: 12px 30px;
        border-radius: 8px;
        transition: 0.3s;
    }}
    div.stButton > button:hover {{
        background-color: #45a049;
    }}
    </style>

    <div class="background-container">
        <div class="overlay-text">
            <h1>404: Job Not Found</h1>
            <h4>99% Hard Work, 1% AI</h4>
        </div>
    </div>
""", unsafe_allow_html=True)

# Upload Resume and JD section
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
        <div class="upload-box">
            <div class="upload-icon">📄</div>
            <div class="upload-title">Drag and drop file here</div>
            <div class="upload-subtext">Limit 200MB per file • PDF, DOCX</div>
        </div>
    """, unsafe_allow_html=True)
    resume_file = st.file_uploader("Resume Upload", type=["pdf", "docx"], label_visibility="collapsed")

with col2:
    jd_input = st.text_area(
        label="Job Description",
        placeholder="Paste Job Description here...",
        height=300,
        label_visibility="collapsed"
    )
# Submit Button
if st.button("Submit Resume & JD"):
    if resume_file and jd_input.strip():
        # Read file only once
        file_bytes = resume_file.read()

        # Save to disk first
        save_uploaded_file(resume_file.name, file_bytes)
        resume_path = os.path.join("data", "inputdata", resume_file.name)

        # Pass raw bytes to PyMuPDF and docx
        from io import BytesIO
        file_buffer = BytesIO(file_bytes)

        # Set session values
        st.session_state.resume_text = extract_resume_text(file_bytes, resume_file.name)
        st.session_state.jd_text = jd_input.strip()  # optional if you want to display/use later

        # Store to DBs
        upload_rejected_resume_local_to_df(resume_file.name, st.session_state.jd_text, "rejected")

        st.success("✅ Upload complete! Now run individual agents.")
    else:
        st.warning("⚠️ Please upload both resume and job description.")

# # Submit Button
# if st.button("Submit Resume & JD"):
#     if resume_file and jd_input.strip():
#         st.session_state.resume_text = extract_resume_text(resume_file)
#         st.session_state.resume_formatted_text = extract_formatted_text(resume_path)
#         st.session_state.resume_plain_text = extract_plain_text(resume_file)
#         st.session_state.jd_text = jd_input.strip()
#         st.session_state.resume_file_name = resume_file.name
#
#         # Save file to disk
#         file_bytes = resume_file.read()
#         save_uploaded_file(resume_file.name, file_bytes)  # <- backend handles path
#
#         # Store with "rejected" status
#         upload_rejected_resume_local_to_df(resume_file.name, st.session_state.jd_text, "rejected")
#
#         st.success("✅ Upload complete! Now run individual agents.")
#     else:
#         st.warning("⚠️ Please upload both resume and job description.")

# Agent sections
# Agent sections
# -- AGENT CONFIG --
agents = [
    ("format", "Format Checker", "images/format.png"),
    ("parser", "Resume Parser", "images/parser.png"),
    ("ats", "ATS Estimator", "images/ATS.jpeg"),
    ("recommend", "Job Recommender", "images/job_recommender.jpeg"),
    ("feedback", "Resume Feedback", "images/Feedback.jpeg"),
    ("cover", "Cover Letter Generator", "images/cover_letter.jpeg"),
    ("tailored", "Tailored Resume", "images/tailored_resume.jpeg"),
    ("chatbot", "Chatbot", "images/chatbot.jpeg")
]

# --- CSS ---
st.markdown("""
<style>
.agent-card.selected {
    background-color: #bbf7d0;
    border-color: #4ade80;
}
.agent-card img {
    width: 140px;
    height: 140px;
    object-fit: cover;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

# --- Init state
if "active_agent" not in st.session_state:
    st.session_state.active_agent = None

for i in range(0, len(agents), 4):
    row_agents = agents[i:i + 4]
    cols = st.columns(4)
    for col, (agent_id, label, image_path) in zip(cols, row_agents):
        with col:
            selected = st.session_state.active_agent == agent_id
            st.markdown(
                f"<div class='agent-card {'selected' if selected else ''}'>", unsafe_allow_html=True
            )
            st.image(image_path, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

            if st.button(label, key=f"trigger_{agent_id}"):
                st.session_state.active_agent = agent_id

# --- Output display
if st.session_state.active_agent:
    st.markdown("---")
    agent_id = st.session_state.active_agent

    if agent_id == "format":
        result = run_formatting_compliance_last()
        st.markdown("### 🧾 Format Checker")
        st.markdown(f"<div style='text-align:left; font-size:16px; color:white;'>{result}</div>", unsafe_allow_html=True)

    elif agent_id == "parser":
        result = ats_parsing_error_detection_agent()
        st.markdown("### 🧬 Resume Parser")
        st.markdown(f"<div style='text-align:left; font-size:16px; color:white;'>{result}</div>", unsafe_allow_html=True)

    elif agent_id == "ats":
        result = run_ats_scoring_with_embedding()
        st.markdown("### 📊 ATS Estimator")
        st.markdown(f"<div style='text-align:left; font-size:16px; color:white;'>{result}</div>", unsafe_allow_html=True)

    # elif agent_id == "recommend":
    #     result = get_job_recommendations(st.session_state.get("resume_text", ""))
    #     st.markdown("### 💼 Job Recommender")
    #     st.markdown(f"<div style='text-align:left; font-size:16px; color:white;'>{result}</div>", unsafe_allow_html=True)

    elif agent_id == "recommend":
        result = get_job_recommendations(st.session_state.get("resume_text", ""))
        st.markdown("### 💼 Job Recommender")
        st.markdown(result, unsafe_allow_html=True)

    elif agent_id == "feedback":
        result = get_resume_feedback(st.session_state.get("resume_text", ""))
        st.markdown("### ✅ Resume Feedback")
        st.markdown(f"<div style='text-align:left; font-size:16px; color:white;'>{result}</div>", unsafe_allow_html=True)


    elif agent_id == "cover":
        cover_letter_text, docx_path = generate_cover_letter()

        if not docx_path or not os.path.exists(docx_path):
            st.error("❌ Failed to generate DOCX file.")
        else:
            st.markdown("### ✍️ Cover Letter Generator")
            st.markdown(f"<div style='text-align:left; font-size:16px; color:white;'>{cover_letter_text}</div>",
                        unsafe_allow_html=True)

            with open(docx_path, "rb") as f:
                st.download_button(
                    label="📥 Download Cover Letter (.docx)",
                    data=f,
                    file_name="Cover_Letter.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                )


    elif agent_id == "tailored":
        st.markdown("### 🎯 Tailored Resume Generator")
        st.info("🛠 Coming soon!")

    elif agent_id == "chatbot":
        st.markdown("### 🤖 Chatbot")
        st.info("🛠 Coming soon!")