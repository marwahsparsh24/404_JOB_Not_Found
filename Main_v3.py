# Full Streamlit App: 404 Job Not Found
import streamlit as st
import pandas as pd
import base64
import os
import re
import json
from streamlit_echarts import st_echarts


from final_v4 import (
    extract_resume_text,
    get_job_recommendations,
    get_resume_feedback,
    run_formatting_compliance_last,
    run_ats_scoring_with_embedding,
    save_uploaded_file,
    upload_rejected_resume_local_to_df,
    get_chatbot_response,
    generate_cover_letter
)

st.set_page_config(page_title="404: Job Not Found", layout="wide")

def get_download_link(file_path, label="📄 Download here"):
    with open(file_path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    return f'<a href="data:application/octet-stream;base64,{b64}" download="{os.path.basename(file_path)}">{label}</a>'


def get_base64_image(image_path):
    with open(image_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Load Header Background
image_base64 = get_base64_image("images/image2.jpeg")

# Inject CSS
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
        height: 260px;
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
    textarea.stTextArea > div > textarea {{
        height: 260px !important;
        font-size: 18px;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 0 10px rgba(0,0,0,0.2);
        background-color: #f9f9f9;
    }}
    div.stButton > button {{
        background-color: transparent;
        color: white;
        border: 2px solid #00cc66;
        transition: background-color 0.3s ease;
        font-size: 16px;
        padding: 8px 20px;
        border-radius: 8px;
    }}
    div.stButton > button:hover {{
        background-color: #00cc66;
        color: white;
    }}
    </style>
    <div class="background-container">
        <div class="overlay-text">
            <h1>404: Job Not Found</h1>
            <h4>99% AI, 1% Luck</h4>
        </div>
    </div>
""", unsafe_allow_html=True)

# Sidebar selector
# --- Sidebar Selector with session state ---
with st.sidebar:
    st.header("📎 Tools")

    # Only set default the first time
    if "selected_tool" not in st.session_state:
        st.session_state.selected_tool = "Home"

    selected_tool = st.radio(
        "Choose functionality:",
        ["Home", "Resume Format Checker", "Generate Cover Letter", "Career Chat Bot"],
        index=["Home", "Resume Format Checker", "Generate Cover Letter", "Career Chat Bot"].index(
            st.session_state.selected_tool
        ),
        key="tool_selector"
    )


# Main layout
if selected_tool == "Home":
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
        jd_input = st.text_area("Paste Job Description", height=260)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        submit_clicked = st.button("Submit Resume & JD", use_container_width=True)

    if submit_clicked:
        if resume_file and jd_input.strip():
            file_bytes = resume_file.read()
            save_uploaded_file(resume_file.name, file_bytes)

            st.session_state.resume_text = extract_resume_text(file_bytes, resume_file.name)
            st.session_state.jd_text = jd_input.strip()

            upload_rejected_resume_local_to_df(resume_file.name, st.session_state.jd_text, "rejected")

            st.success("✅ Upload complete! Showing results below...")

            ats_result = run_ats_scoring_with_embedding()
            feedback_result = get_resume_feedback(st.session_state.resume_text)
            job_output , cover_letter_paths = get_job_recommendations(st.session_state.resume_text)

            # --- ATS Score Placeholder ---
            with st.container():
                st.markdown("### 📊 ATS Score")

                # Create columns for score and keywords
                col1, col2 = st.columns([1, 2])

                with col1:
                    score_value = int(ats_result["score"])

                    # Gauge color based on score
                    if score_value < 60:
                        color = "#f87171"  # red
                    elif score_value < 80:
                        color = "#facc15"  # yellow
                    else:
                        color = "#4ade80"  # green

                    option = {
                        "series": [
                            {
                                "type": "gauge",
                                "startAngle": 90,
                                "endAngle": -270,
                                "min": 0,
                                "max": 100,
                                "progress": {
                                    "show": True,
                                    "width": 30,
                                    "itemStyle": {
                                        "color": color
                                    }
                                },
                                "axisLine": {
                                    "lineStyle": {
                                        "width": 30,
                                        "color": [[1, "#374151"]]  # gray background
                                    }
                                },
                                "pointer": {"show": False},
                                "axisTick": {"show": False},
                                "splitLine": {"show": False},
                                "axisLabel": {"show": False},
                                "anchor": {"show": False},
                                "detail": {
                                    "valueAnimation": True,
                                    "formatter": "{value}%",
                                    "fontSize": 64,
                                    "offsetCenter": [0, "0%"],
                                    "color": "#bbf7d0"
                                },
                                "title": {"show": False},
                                "data": [{"value": score_value}]
                            }
                        ]
                    }

                    st_echarts(options=option, height="300px")

                with col2:
                    matched = ats_result.get("keywords", {}).get("matched", [])
                    unmatched = ats_result.get("keywords", {}).get("unmatched", [])

                    st.markdown(f"""
                        <div style='background-color:#1f2937; padding:20px; border-radius:12px; color:white; height:100%;'>
                            <div style='margin-bottom:15px;'>
                                <h4 style='color:#86efac; margin-bottom:8px;'>✅ Matched Keywords ({len(matched)})</h4>
                                <div style='display:flex; flex-wrap:wrap; gap:6px;'>
                                    {''.join([f"<span style='background:#374151; padding:4px 10px; border-radius:12px; font-size:14px;'>{kw}</span>" for kw in matched]) or "<span style='color:#9ca3af;'>No keywords matched</span>"}
                                </div>
                            </div>
                            <div>
                                <h4 style='color:#fca5a5; margin-bottom:8px;'>❌ Missing Keywords ({len(unmatched)})</h4>
                                <div style='display:flex; flex-wrap:wrap; gap:6px;'>
                                    {''.join([f"<span style='background:#374151; padding:4px 10px; border-radius:12px; font-size:14px; opacity:0.8;'>{kw}</span>" for kw in unmatched]) or "<span style='color:#9ca3af;'>All keywords matched!</span>"}
                                </div>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)

            # --- Resume Feedback Placeholder ---
            with st.container():
                st.markdown("### ✅ Resume Feedback")
                st.markdown(f"""
                    <div style='background-color:#fef9c3; color:#1e1e1e; padding:15px; border-radius:10px; font-size:15px;'>
                        {feedback_result}
                    </div>
                """, unsafe_allow_html=True)

            with st.container():
                st.markdown("### 💼 Job Recommendations")
                if not job_output:
                    st.warning("❌ No jobs found.")
                else:
                    job_listings = [job for job in job_output.split('\n---\n') if job.strip()]

                    for i, job in enumerate(job_listings):
                        file_path = cover_letter_paths[i] if cover_letter_paths and i < len(
                            cover_letter_paths) else None

                        # Open styled container for job card
                        st.markdown(
                            "<div style='background-color:#1e293b; padding:20px; border-radius:12px; color:white; font-size:15px; margin-bottom:20px;'>",
                            unsafe_allow_html=True
                        )

                        # ✅ Render job description (as Markdown, not raw HTML)
                        st.markdown(job, unsafe_allow_html=True)

                        # ✅ Render download button below job (inside same container)
                        if file_path and os.path.exists(file_path):
                            with open(file_path, "rb") as f:
                                file_data = f.read()
                                file_hash = hash(file_data)

                            with open(file_path, "rb") as f:
                                st.markdown(get_download_link(file_path, label="📎 Download Cover Letter"), unsafe_allow_html=True)


                        # Close styled container
                        st.markdown("</div>", unsafe_allow_html=True)


elif selected_tool == "Resume Format Checker":
    result = run_formatting_compliance_last()
    st.subheader("🧾 Format Checker Output")
    st.markdown(f"""
        <div style='background-color:#fef9c3; color:#1e1e1e; padding:15px; border-radius:10px; font-size:15px;'>
            {result}
        </div>
    """, unsafe_allow_html=True)

elif selected_tool == "Generate Cover Letter":
    cover_letter_text, docx_path = generate_cover_letter()
    if cover_letter_text:
        st.subheader("✍️ Generated Cover Letter")
        st.markdown(f"""
            <div style='background-color:#dcfce7; color:#1e1e1e; padding:15px; border-radius:10px; font-size:15px;'>
                {cover_letter_text}
            </div>
        """, unsafe_allow_html=True)
        if docx_path and os.path.exists(docx_path):
            with open(docx_path, "rb") as f:
                st.download_button(
                    label="📥 Download Cover Letter (.docx)",
                    data=f,
                    file_name="Cover_Letter.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                )

elif selected_tool == "Career Chat Bot":
    st.subheader("💬 Career Chat Bot")
    user_query = st.text_input("Ask me anything about your resume, job search, or career:")
    if user_query:
        response = get_chatbot_response(user_query)
        st.markdown(f"""
            <div style='background-color:#f0f9ff; color:#1e1e1e; padding:15px; border-radius:10px; font-size:15px;'>
                {response}
            </div>
        """, unsafe_allow_html=True)
