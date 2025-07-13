import streamlit as st
import pandas as pd
import fitz  # PyMuPDF
import re
from sentence_transformers import SentenceTransformer, util

# Load Model & Data

SKILL_DB = {
    'python', 'java', 'c++', 'sql', 'machine learning', 'deep learning', 'data analysis',
    'communication', 'leadership', 'pandas', 'numpy', 'hadoop', 'spark', 'excel', 'linux',
    'matplotlib', 'tensorflow', 'scikit-learn', 'project management', 'statistics',
    'problem solving', 'teamwork', 'nlp', 'cloud computing'
}


@st.cache_data
def load_data():
    jd_df = pd.read_csv("JOB_POSTS_DATASET.csv")
    resume_df = pd.read_csv("RESUME_SCREENING_DATASET.csv")
    jd_df.fillna("", inplace=True)
    resume_df.fillna("", inplace=True)
    return jd_df, resume_df

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

#  Helper Functions
def extract_text_from_pdf(uploaded_file):
    with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
        return "".join(page.get_text() for page in doc)

def get_ada_score(resume_text, jd_text, model):
    resume_emb = model.encode([resume_text], convert_to_tensor=True)
    jd_emb = model.encode([jd_text], convert_to_tensor=True)
    return round(util.cos_sim(resume_emb, jd_emb).item() * 100, 2)

import dateutil.parser as dparser
from datetime import datetime

def extract_resume_info(text):
    text = text.lower()
    exp = 0
    try:
        matches = re.findall(r'([a-zA-Z]+\s\d{4})\s*[-–]\s*([a-zA-Z]+\s\d{4}|present)', text)
        for start, end in matches:
            start_date = dparser.parse(start, fuzzy=True)
            end_date = datetime.now() if 'present' in end else dparser.parse(end, fuzzy=True)
            exp += (end_date - start_date).days / 365.0

        if exp == 0:
            direct_exp = re.search(r'(\d+)\s*(\+)?\s*years', text)
            if direct_exp:
                exp = int(direct_exp.group(1))
        
        return int(round(exp))
    except:
        return 0


def get_cleaned_titles(jd_df, resume_df):
    jd_titles = jd_df['Title'].dropna().astype(str)
    resume_titles = resume_df['last_job_title'].dropna().astype(str)
    all_titles = pd.Series(list(jd_titles) + list(resume_titles))
    all_titles = all_titles[all_titles.str.len() < 50]
    cleaned = all_titles.str.lower().str.replace(r"[^a-zA-Z\s]", "", regex=True)
    cleaned = cleaned.str.replace(r"\bit department\b", "", regex=True)
    cleaned = cleaned.str.replace(r"\b\w+\s+(branch manager|developer|engineer|analyst|programmer)\b", r"\1", regex=True)
    cleaned = cleaned.str.replace(r"\s+", " ", regex=True).str.strip().str.title()
    return sorted(cleaned.drop_duplicates())

#  Feedback System
import re
import spacy.cli
spacy.cli.download("en_core_web_sm")
import spacy

nlp = spacy.load("en_core_web_sm")

def extract_skills_from_text(text, skill_db):
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text.lower())
    found_skills = set()

    for skill in skill_db:
        if all(word in text for word in skill.split()):
            found_skills.add(skill)

    return list(found_skills)



# Final feedback function
def build_feedback(score, resume_text, jd_text, resume_exp, jd_exp):
    strengths = []
    improvements = []

    jd_skills = extract_skills_from_text(jd_text, SKILL_DB)
    resume_skills = extract_skills_from_text(resume_text, SKILL_DB)

    matched_skills = sorted(set(jd_skills) & set(resume_skills))
    missing_skills = sorted(set(jd_skills) - set(resume_skills))

    if matched_skills:
        strengths.append(f"✅ Skills matched: {', '.join(matched_skills)}")
    if missing_skills:
        improvements.append(f"⚠️ Missing key JD skills: {', '.join(missing_skills)}")

    # Experience check
    if resume_exp < jd_exp:
        improvements.append(f"⚠️ Experience below expected `{jd_exp}+ yrs` (You have {resume_exp})")
    else:
        strengths.append(f"✅ Experience meets or exceeds `{jd_exp}+ yrs`")

    # Resume length
    word_count = len(resume_text.split())
    if word_count < 150:
        improvements.append("⚠️ Resume is too short; consider adding more projects, achievements, or responsibilities.")
    else:
        strengths.append("✅ Resume length looks sufficient.")

    # Measurable result check
    if not re.search(r"\d+|managed|led|achieved|built|delivered|improved", resume_text.lower()):
        improvements.append("⚠️ Add measurable results or action words like 'led', 'achieved', etc.")

    # Insight
    if score < 30:
        insight = "⬇️ Resume needs major improvements."
    elif score < 60:
        insight = "⚠️ Resume is on the right path. Improve JD matching and content."
    else:
        insight = "✅ Great job! Resume aligns well with job expectations."

    if not strengths:
        strengths.append("Structure looks okay, but resume lacks strong skill alignment.")

    return strengths, improvements, insight



# Streamlit App 
st.set_page_config(page_title="Smart Resume Review", layout="centered")
st.title("Resume Screening System")

jd_df, resume_df = load_data()
model = load_model()
job_roles = get_cleaned_titles(jd_df, resume_df)

uploaded_resume = st.file_uploader("📄 Upload Your Resume (PDF)", type=["pdf"])
selected_role = st.selectbox("Select or Type Job Role", job_roles)

if uploaded_resume and selected_role:
    with st.spinner("Analyzing your resume..."):
        resume_text = extract_text_from_pdf(uploaded_resume)

        resume_exp = extract_resume_info(resume_text)
        jd_matches = jd_df[jd_df['Title'].str.contains(selected_role, case=False, na=False)]

        if jd_matches.empty:
            st.warning("⚠️ No direct match found. Showing general job posts.")
            jd_matches = jd_df.sample(2)

        for idx, row in enumerate(jd_matches.itertuples(), 1):
            # Choose better JD column
            jd_text = getattr(row, "jobpost", "") or getattr(row, "eligibility", "")

            company = getattr(row, "Company", "Unknown Company")
            jd_exp = 2 if "senior" not in row.Title.lower() else 5
            score = get_ada_score(resume_text, jd_text, model)
            strengths, improvements, insight = build_feedback(score, resume_text, jd_text, resume_exp, jd_exp)

            # Report Section
            st.markdown(f"---\n### {idx}. {row.Title} at **{company}**")
            st.markdown(f"#### ADA Score: `{score} / 100`")
            st.progress(score / 100)
            st.markdown(f"📊 **Score Insight:** {insight}")

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("✅ **Strengths**")
                for point in strengths:
                    st.markdown(f"- {point}")

            with col2:
                st.markdown("❌ **Needs Improvement**")
                for point in improvements:
                    st.markdown(f"- {point}")

        #  Recommended Jobs with High ADA Score 
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np

        with st.spinner(" Finding jobs where your resume fits well..."):

            sampled_df = jd_df.sample(500, random_state=42)
            jd_texts = sampled_df['jobpost'].fillna("").tolist()

            # 🔹 Embed all JDs at once
            jd_embeddings = model.encode(jd_texts, batch_size=32, show_progress_bar=False)
            resume_emb = model.encode([resume_text])[0].reshape(1, -1)

            # 🔹 Bulk similarity
            similarities = cosine_similarity(resume_emb, jd_embeddings)[0] * 100
            sampled_df['score'] = similarities

            recommended_jobs = sampled_df[sampled_df['score'] >= 70].sort_values(by='score', ascending=False).head(5)

            if not recommended_jobs.empty:
                st.markdown("## 🌟 Recommended Jobs for Your Resume (ADA ≥ 70)")
                for i, row in enumerate(recommended_jobs.itertuples(), 1):
                    st.markdown(f"**{i}. {row.Title}** at **{row.Company}**")
                    st.markdown(f"🔎 ADA Score: `{round(row.score, 2)} / 100`")
                    st.markdown("---")
            else:
                st.info("❌ No strong matches (ADA ≥ 70) found. Try improving your resume.")

