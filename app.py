import io
import json
import time
import zipfile
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import streamlit as st
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# ---------- Schemas ----------
class ResumeInfo(BaseModel):
    technical_skills: List[str] = Field(default_factory=list)
    soft_skills: List[str] = Field(default_factory=list)
    experience: List[str] = Field(default_factory=list)

class JobRequirements(BaseModel):
    must_have_technical_skills: List[str] = Field(default_factory=list)
    must_have_soft_skills: List[str] = Field(default_factory=list)
    required_experience: List[str] = Field(default_factory=list)

class GapAnalysis(BaseModel):
    met_requirements: List[str] = Field(default_factory=list)
    missing_or_weak: List[str] = Field(default_factory=list)
    notes: Optional[str] = None

# ---------- Prompts ----------
resume_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a precise resume analysis agent. Extract only what's present."),
    ("human", "Resume:\n\n{resume}\n\nReturn JSON keys: technical_skills, soft_skills, experience.")
])

job_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a job description analysis agent. Extract explicit requirements only."),
    ("human", "Job Description:\n\n{job_desc}\n\nReturn JSON keys: must_have_technical_skills, must_have_soft_skills, required_experience.")
])

gap_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert job application coach. Compare and be candid."),
    ("human",
     "Resume Information (JSON):\n{resume_info}\n\n"
     "Job Requirements (JSON):\n{job_requirements}\n\n"
     "Return JSON keys: met_requirements, missing_or_weak, notes.")
])

cover_prompt = ChatPromptTemplate.from_messages([
    ("system", "Write a concise, professional, tailored cover letter (<=200 words). Avoid fluff."),
    ("human",
     "Resume Information (JSON):\n{resume_info}\n\n"
     "Job Requirements (JSON):\n{job_requirements}\n\n"
     "Gap Analysis (JSON):\n{gap_analysis}\n\n"
     "Write the cover letter only—no preamble.")
])

# ---------- Helpers ----------
def read_text_from_pdf(file_bytes: bytes) -> str:
    # Lightweight PDF text extractor using pypdf
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(file_bytes))
    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            pages.append("")
    return "\n".join(pages).strip()

def load_text_file(upload) -> str:
    return upload.read().decode("utf-8", errors="ignore")

def load_resume_or_jd(upload) -> str:
    name = upload.name.lower()
    if name.endswith(".pdf"):
        return read_text_from_pdf(upload.read())
    return load_text_file(upload)

@st.cache_resource(show_spinner=False)
def get_llm(model_name: str, temperature: float):
    api_key = st.secrets.get("OPENAI_API_KEY", None)
    if not api_key:
        st.warning("OPENAI_API_KEY missing in Secrets. Using environment if present.")
    return ChatOpenAI(model=model_name, temperature=temperature)

def structured_chain(prompt: ChatPromptTemplate, llm, schema):
    return prompt | llm.with_structured_output(schema)

def text_chain(prompt: ChatPromptTemplate, llm):
    from langchain_core.output_parsers import StrOutputParser
    return prompt | llm | StrOutputParser()

def run_pipeline_single(resume_text: str, job_desc_text: str, model_name: str, temperature: float):
    llm = get_llm(model_name, temperature)

    resume_chain = structured_chain(resume_prompt, llm, ResumeInfo)
    resume_info: ResumeInfo = resume_chain.invoke({"resume": resume_text})

    job_chain = structured_chain(job_prompt, llm, JobRequirements)
    job_requirements: JobRequirements = job_chain.invoke({"job_desc": job_desc_text})

    gap_chain = structured_chain(gap_prompt, llm, GapAnalysis)
    gap_analysis: GapAnalysis = gap_chain.invoke({
        "resume_info": resume_info.model_dump_json(),
        "job_requirements": job_requirements.model_dump_json(),
    })

    cover_chain = text_chain(cover_prompt, llm)
    cover_letter: str = cover_chain.invoke({
        "resume_info": resume_info.model_dump_json(),
        "job_requirements": job_requirements.model_dump_json(),
        "gap_analysis": gap_analysis.model_dump_json(),
    })

    return resume_info, job_requirements, gap_analysis, cover_letter

def run_pipeline_batch(
    resumes: List[Tuple[str, str]],  # (filename, text)
    job_desc_text: str,
    model_name: str,
    temperature: float,
    delay_s: float = 0.0
):
    """Process many resumes against ONE job description."""
    rows = []
    cover_zip_buffer = io.BytesIO()
    with zipfile.ZipFile(cover_zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        progress = st.progress(0.0, text="Processing resumes…")
        for i, (fname, rtext) in enumerate(resumes, start=1):
            try:
                resume_info, job_requirements, gap_analysis, cover_letter = run_pipeline_single(
                    rtext, job_desc_text, model_name, temperature
                )
                rows.append({
                    "resume_file": fname,
                    "tech_skills": ", ".join(resume_info.technical_skills),
                    "soft_skills": ", ".join(resume_info.soft_skills),
                    "experience": "; ".join(resume_info.experience),
                    "must_have_tech": ", ".join(job_requirements.must_have_technical_skills),
                    "must_have_soft": ", ".join(job_requirements.must_have_soft_skills),
                    "required_experience": "; ".join(job_requirements.required_experience),
                    "met_requirements": "; ".join(gap_analysis.met_requirements),
                    "missing_or_weak": "; ".join(gap_analysis.missing_or_weak),
                })
                # add cover letter file to zip
                leaf = Path(fname).with_suffix(".txt").name
                zf.writestr(f"cover_letters/{leaf}", cover_letter.strip())
            except Exception as e:
                rows.append({
                    "resume_file": fname,
                    "tech_skills": "",
                    "soft_skills": "",
                    "experience": "",
                    "must_have_tech": "",
                    "must_have_soft": "",
                    "required_experience": "",
                    "met_requirements": "",
                    "missing_or_weak": f"ERROR: {e}",
                })
            progress.progress(i / len(resumes), text=f"Processed {i}/{len(resumes)}")
            if delay_s > 0:
                time.sleep(delay_s)

    csv_bytes = io.BytesIO()
    pd.DataFrame(rows).to_csv(csv_bytes, index=False)
    csv_bytes.seek(0)

    cover_zip_buffer.seek(0)
    return csv_bytes, cover_zip_buffer

# ---------- UI ----------
st.set_page_config(page_title="Job Application Multi-Agent", page_icon="🧩", layout="wide")
st.title("🧩 Job Application Assistant — Multi-Agent")
st.caption("Parse resumes (TXT/PDF), analyze job descriptions, find gaps, and generate tailored cover letters. Supports single and batch (100+ resumes).")

with st.sidebar:
    st.subheader("Model Settings")
    model = st.selectbox(
        "Model",
        options=["gpt-4o", "o4-mini", "gpt-4.1-mini"],
        index=0
    )
    temperature = st.slider("Creativity (temperature)", 0.0, 1.0, 0.4, 0.1)
    st.markdown("---")
    st.subheader("Batch Controls")
    rate_delay = st.slider("Delay between resumes (sec)", 0.0, 2.0, 0.0, 0.1,
                           help="Use a small delay if you hit rate limits on large batches.")
    st.markdown("**Secrets** → add `OPENAI_API_KEY` in Streamlit Cloud")

tabs = st.tabs(["Single Resume", "Batch (100+ resumes)"])

# -------- Single Resume Tab --------
with tabs[0]:
    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.subheader("Resume (TXT or PDF)")
        resume_upload = st.file_uploader("Upload resume", type=["txt", "pdf"], key="res_single")
        resume_text = st.text_area("…or paste resume", value="", height=220, placeholder="Paste plain-text resume")

        if resume_upload:
            try:
                resume_text = load_resume_or_jd(resume_upload)
            except Exception as e:
                st.error(f"Failed to read resume: {e}")

    with col2:
        st.subheader("Job Description (TXT or PDF)")
        jd_upload = st.file_uploader("Upload job description", type=["txt", "pdf"], key="jd_single")
        job_desc_text = st.text_area("…or paste job description", value="", height=220, placeholder="Paste the JD here")

        if jd_upload:
            try:
                job_desc_text = load_resume_or_jd(jd_upload)
            except Exception as e:
                st.error(f"Failed to read JD: {e}")

    run_single = st.button("Run pipeline (Single)", type="primary", use_container_width=True)

    if run_single:
        if not resume_text.strip() or not job_desc_text.strip():
            st.error("Please provide both a resume and a job description.")
            st.stop()
        with st.spinner("Analyzing…"):
            resume_info, job_requirements, gap_analysis, cover_letter = run_pipeline_single(
                resume_text, job_desc_text, model, temperature
            )
        st.success("Done!")

        cA, cB, cC = st.columns(3)
        with cA:
            st.markdown("### Resume Info")
            st.json(resume_info.model_dump(), expanded=False)
        with cB:
            st.markdown("### Job Requirements")
            st.json(job_requirements.model_dump(), expanded=False)
        with cC:
            st.markdown("### Gap Analysis")
            st.json(gap_analysis.model_dump(), expanded=False)

        st.markdown("### Tailored Cover Letter")
        st.write(cover_letter.strip())

        # Downloads
        artifacts = {
            "resume_info.json": json.dumps(resume_info.model_dump(), indent=2).encode("utf-8"),
            "job_requirements.json": json.dumps(job_requirements.model_dump(), indent=2).encode("utf-8"),
            "gap_analysis.json": json.dumps(gap_analysis.model_dump(), indent=2).encode("utf-8"),
            "cover_letter.txt": cover_letter.strip().encode("utf-8"),
        }
        st.markdown("#### Download Artifacts")
        d1, d2, d3, d4 = st.columns(4)
        for (name, data), holder in zip(artifacts.items(), [d1, d2, d3, d4]):
            with holder:
                st.download_button(
                    label=f"Download {name}",
                    data=data,
                    file_name=name,
                    mime="application/json" if name.endswith(".json") else "text/plain",
                    use_container_width=True,
                )

# -------- Batch Tab --------
with tabs[1]:
    st.subheader("Upload many resumes (TXT/PDF). Choose ONE JD (TXT/PDF).")
    resume_files = st.file_uploader("Resumes", type=["txt", "pdf"], accept_multiple_files=True, key="res_batch")
    jd_batch = st.file_uploader("Job Description", type=["txt", "pdf"], key="jd_batch")

    run_batch = st.button("Run pipeline (Batch)", type="primary", use_container_width=True)

    if run_batch:
        if not resume_files or not jd_batch:
            st.error("Please upload resumes and one job description.")
            st.stop()

        # read JD once
        try:
            jd_text = load_resume_or_jd(jd_batch)
        except Exception as e:
            st.error(f"Failed to read JD: {e}")
            st.stop()

        # read all resumes
        parsed_resumes: List[Tuple[str, str]] = []
        for f in resume_files:
            try:
                text = load_resume_or_jd(f)
                parsed_resumes.append((f.name, text))
            except Exception as e:
                parsed_resumes.append((f.name, f""))  # mark as empty to still include in CSV
                st.warning(f"Failed to parse {f.name}: {e}")

        if len(parsed_resumes) == 0:
            st.error("No valid resumes to process.")
            st.stop()

        with st.spinner(f"Processing {len(parsed_resumes)} resumes…"):
            csv_bytes, zip_bytes = run_pipeline_batch(
                parsed_resumes, jd_text, model, temperature, delay_s=rate_delay
            )

        st.success("Batch complete!")

        st.markdown("#### Download Results")
        st.download_button(
            "Download summary.csv",
            data=csv_bytes,
            file_name="summary.csv",
            mime="text/csv",
            use_container_width=True
        )
        st.download_button(
            "Download cover_letters.zip",
            data=zip_bytes,
            file_name="cover_letters.zip",
            mime="application/zip",
            use_container_width=True
        )

st.caption("Tip: For 100+ resumes, start with o4-mini and a small delay if you hit rate limits.")
