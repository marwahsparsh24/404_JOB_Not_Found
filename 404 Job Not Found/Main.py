import streamlit as st
import pandas as pd
import base64
import os
from final_v4 import (
    extract_resume_text,
    get_job_recommendations,
    get_resume_feedback,
    run_formatting_compliance_last,
    ats_parsing_error_detection_agent,
    run_ats_scoring_with_embedding,
    save_uploaded_file,
    upload_rejected_resume_local_to_df,
    get_chatbot_response,
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

    # elif agent_id == "ats":
    #     result = run_ats_scoring_with_embedding()
    #     score = result["score"]
    #     reasoning = result.get("reasoning", "")
    #
    #     st.markdown("### 📊 ATS Score Estimator")
    #
    #     # Circle Progress CSS + Score Display
    #     score_html = f"""
    #     <style>
    #     .circle-container {{
    #         width: 220px;
    #         height: 220px;
    #         border-radius: 50%;
    #         background: conic-gradient(#3b82f6  {int(score * 3.6)}deg, #1e293b 0deg);
    #         position: relative;
    #         margin: 30px auto;
    #         box-shadow: 0 0 20px rgba(0,0,0,0.4);
    #     }}
    #
    #     .score-center {{
    #         position: absolute;
    #         top: 50%;
    #         left: 50%;
    #         transform: translate(-50%, -50%);
    #         font-size: 72px;
    #         font-weight: bold;
    #         color: white;
    #         font-family: Arial, sans-serif;
    #     }}
    #     </style>
    #
    #     <div class="circle-container">
    #         <div class="score-center">{int(score)}%</div>
    #     </div>
    #     """
    #
    #     st.markdown(score_html, unsafe_allow_html=True)
    #
    #     # Feedback
    #     st.markdown("#### 💡 Reasoning & Feedback")
    #     st.markdown(
    #         f"<div style='font-size:16px; color:white; text-align:left; line-height:1.6'>{reasoning}</div>",
    #         unsafe_allow_html=True
    #     )
    elif agent_id == "ats":
        result = run_ats_scoring_with_embedding()
        score = result["score"]
        reasoning = result.get("reasoning", {})
        resume_file = result.get("resume_file", "N/A")

        st.markdown("### 📊 ATS Estimator")

        # Circular Score Display (with custom color)
        score_html = f"""
        <style>
        .circle-container {{
            width: 220px;
            height: 220px;
            border-radius: 50%;
            background: conic-gradient(#3b82f6 {int(score * 3.6)}deg, #1e293b 0deg);
            position: relative;
            margin: 30px auto;
            box-shadow: 0 0 15px rgba(0,0,0,0.4);
        }}
        .score-center {{
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            font-size: 64px;
            font-weight: bold;
            color: white;
            font-family: Arial, sans-serif;
        }}
        </style>
        <div class="circle-container">
            <div class="score-center">{int(score)}%</div>
        </div>
        """
        st.markdown(score_html, unsafe_allow_html=True)

        # Resume File Info
        st.markdown(f"📄 **Resume:** `{resume_file}`", unsafe_allow_html=True)

        # Collapsible Sections for Reasoning
        st.markdown("### 💡 Reasoning & Feedback")

        with st.expander("🧠 Keyword Overlap"):
            st.markdown(reasoning.get("Keyword Overlap", "No information available."), unsafe_allow_html=True)

        with st.expander("📝 Formatting Issues"):
            st.markdown(reasoning.get("Formatting Issues", "No information available."), unsafe_allow_html=True)

        with st.expander("📄 Missing Sections"):
            st.markdown(reasoning.get("Missing Sections", "No information available."), unsafe_allow_html=True)

        with st.expander("✅ Conclusion"):
            st.markdown(reasoning.get("Conclusion", "No information available."), unsafe_allow_html=True)


    elif agent_id == "recommend":
        result = get_job_recommendations(st.session_state.get("resume_text", ""))
        st.markdown("### 💼 Job Recommender")

        st.markdown("""
        <style>
        .job-card {
            background-color: #1f2937;
            padding: 20px;
            margin-bottom: 20px;
            border-radius: 16px;
            color: white;
            box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        }
        .job-title {
            font-size: 20px;
            font-weight: bold;
            margin-bottom: 6px;
        }
        .match-circle {
            float: right;
            background-color: #4ade80;
            color: black;
            font-weight: 700;
            border-radius: 50%;
            width: 70px;
            height: 70px;
            text-align: center;
            line-height: 60px;
            font-size: 18px;
            margin-left: 10px;
        }
        .job-meta {
            font-size: 14px;
            color: #d1d5db;
            margin-bottom: 10px;
        }
        .job-desc {
            font-size: 15px;
            color: white;
        }
        </style>
        """, unsafe_allow_html=True)

        # Parse and rebuild output from your get_job_recommendations()
        job_blocks = result.split("**[")
        for block in job_blocks[1:]:  # skip the empty one before first match
            title_line, rest = block.split("**", 1)
            title, company = title_line.split("](", 1)
            company = company.split(")")[1].strip(" -")

            match_line = [line for line in rest.split("\n") if "Match Score:" in line]
            score = match_line[0].split(":")[1].strip().replace("%", "") if match_line else "N/A"

            description_lines = [line for line in rest.split("\n") if "Description:" in line]
            desc = description_lines[0].split(":", 1)[-1].strip() if description_lines else ""

            meta_lines = [line for line in rest.split("\n") if "Location:" in line or "Posted:" in line]
            meta_info = "<br>".join(meta_lines)

            job_html = f"""
            <div class="job-card">
                <div class="job-title">
                    <a href="{title_line.split('](')[-1]}" style="color:#60a5fa" target="_blank">{title}</a>
                    <div class="match-circle">{score}%</div>
                </div>
                <div class="job-meta">{company}<br>{meta_info}</div>
                <div class="job-desc">{desc}</div>
            </div>
            """

            st.markdown(job_html, unsafe_allow_html=True)


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
                )  # ✅ Closing parenthesis added here


    elif agent_id == "tailored":
        st.markdown("### 🎯 Tailored Resume Generator")
        st.info("🛠 Coming soon!")


    elif agent_id == "chatbot":

        st.markdown("### 🤖 Career Chatbot")

        user_query = st.text_input("Ask your question here:")

        if user_query:
            reply = get_chatbot_response(user_query)

            st.markdown(f"<div style='text-align:left; font-size:16px; color:white;'>{reply}</div>",
                        unsafe_allow_html=True)


