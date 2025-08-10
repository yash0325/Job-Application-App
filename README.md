# Job Application Assistant — Multi-Agent (Streamlit)
Analyze resumes, parse job descriptions, identify skill gaps, and generate tailored cover letters — at scale.
Supports TXT & PDF resumes, single or batch (100+) processing, with downloadable artifacts.

# ✨ Features
Multi-agent chain: Resume parsing → JD parsing → Gap analysis → Cover letter generation

Batch mode: Process 100+ resumes against one JD with progress & exports

PDF + TXT support: Uses pypdf for text extraction (works best with digital PDFs)

Structured outputs: Validated JSON for each step; CSV roll-up; ZIP of cover letters

Configurable LLMs: Choose gpt-4o, o4-mini, or gpt-4.1-mini in the UI

Rate-limit friendly: Optional delay between resumes in batch mode

# 🧱 Tech Stack
Streamlit, LangChain (langchain-openai)

OpenAI API

Pydantic v2 (structured outputs)

pypdf, pandas

# UI Preview
<img width="1918" height="861" alt="image" src="https://github.com/user-attachments/assets/a25adf4d-3556-4658-8eb7-4abb0edb732b" />

